// Copyright 2026 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! S14 — the concurrency suite for the local operation log (roadmap/088
//! Stage 7, decision D11).
//!
//! These four assertions are the proof of the PRD's central claim: that moving
//! op heads out of the server's compare-and-swap and into `<op_heads>/heads/`
//! gives Vex *upstream jj's* concurrency contract rather than a weaker one.
//!
//! * (a) **No lost head.** Two concurrent commands that each commit an
//!   operation leave both heads on disk, and the next read merges them.
//! * (b) **Add before remove.** An interrupted head update leaves an *extra*
//!   head, never zero; and a removal that finds nothing succeeds.
//! * (c) **A real lock.** N concurrent readers of a divergent repository
//!   produce exactly one merge head, not N.
//! * (d) **A commit cannot fail.** Hammering one repository from several
//!   processes produces zero write errors: no local path may refuse a head
//!   write for concurrency reasons.
//!
//! (a), (c) and (d) are genuinely multi-process — separate processes are the
//! only way to exercise the `FileLock` and to make two `readdir`/`rename`
//! sequences actually interleave. They work by re-executing this test binary
//! with `--exact s14_child --ignored` and selecting the child's behaviour with
//! environment variables, and by rendezvousing on the filesystem so the writes
//! collide instead of merely being issued near each other.
//!
//! The op *store* here is [`SimpleOpStore`], not `VexOpStore`: the subject of
//! every assertion is the op-**heads** store, and pairing the two is sound
//! because `VexOpHeadsStore` treats head ids purely as hex and never
//! interprets their length.
//!
//! Unix only: the harness relies on `FileLock` and directory semantics that
//! this suite is not meant to characterise on Windows.

#![cfg(unix)]

use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use jj_lib::backend::CommitId;
use jj_lib::backend::MillisSinceEpoch;
use jj_lib::backend::Timestamp;
use jj_lib::object_id::ObjectId as _;
use jj_lib::op_heads_store::OpHeadResolutionError;
use jj_lib::op_heads_store::OpHeadsStore as _;
use jj_lib::op_heads_store::OpHeadsStoreError;
use jj_lib::op_heads_store::resolve_op_heads;
use jj_lib::op_store::OpStore;
use jj_lib::op_store::OpStoreError;
use jj_lib::op_store::Operation as OperationData;
use jj_lib::op_store::OperationId;
use jj_lib::op_store::OperationMetadata;
use jj_lib::op_store::RootOperationData;
use jj_lib::op_store::TimestampRange;
use jj_lib::op_store::ViewId;
use jj_lib::operation::Operation;
use jj_lib::simple_op_store::SimpleOpStore;
use jj_lib::vex::VexObjectReadMode;
use jj_lib::vex::VexRepoConfig;
use jj_lib::vex_op_heads_store::VexOpHeadsStore;
use pollster::FutureExt as _;

/// Readers in assertion (c) and hammering writers in assertion (d).
const CHILDREN: usize = 8;
/// Sequential operations each hammering child commits in assertion (d).
const HAMMER_OPS: usize = 20;
/// Nothing in this suite may wedge CI: every wait is bounded, and blowing the
/// bound kills the children and fails loudly.
const WATCHDOG: Duration = Duration::from_secs(60);
/// Polling interval. Used only for liveness — no assertion depends on it.
const POLL: Duration = Duration::from_millis(2);

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn op_heads_dir(root: &Path) -> PathBuf {
    root.join("op_heads")
}

fn op_store_dir(root: &Path) -> PathBuf {
    root.join("op_store")
}

fn rendezvous_dir(root: &Path) -> PathBuf {
    root.join("rendezvous")
}

fn results_dir(root: &Path) -> PathBuf {
    root.join("results")
}

/// The endpoint is unroutable on purpose: if any of this reached the backend it
/// would fail or stall rather than quietly pass.
fn test_config() -> VexRepoConfig {
    VexRepoConfig {
        endpoint: "http://127.0.0.1:1".to_string(),
        tenant_id: "tenant-id".to_string(),
        tenant_slug: "acme".to_string(),
        repo_id: "repo-id".to_string(),
        repo_slug: "widget".to_string(),
        repository_scope_kind: Some("repository".to_string()),
        virtual_repository_id: None,
        backing_repo_slug: None,
        virtual_root_path: None,
        virtual_mounts: Vec::new(),
        access_token: None,
        local_writes: false,
        object_read_mode: VexObjectReadMode::NativeOnly,
    }
}

fn root_data() -> RootOperationData {
    RootOperationData {
        root_commit_id: CommitId::new(vec![0; 32]),
    }
}

/// Open the heads store over an existing repository. `init` is idempotent
/// (`create_dir_all`), so parent and children open the same directory the same
/// way, and `heads/` always exists — no bootstrap ever runs in this suite.
fn heads_store(root: &Path) -> VexOpHeadsStore {
    VexOpHeadsStore::init(test_config(), &op_heads_dir(root)).unwrap()
}

fn op_store(root: &Path) -> Arc<dyn OpStore> {
    Arc::new(SimpleOpStore::load(&op_store_dir(root), root_data()))
}

/// A repository laid out for the harness: an initialised op store, an
/// initialised op-heads store, and the rendezvous/result directories the
/// children write into.
fn init_repo() -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(op_store_dir(root)).unwrap();
    SimpleOpStore::init(&op_store_dir(root), root_data()).unwrap();
    heads_store(root);
    fs::create_dir_all(rendezvous_dir(root)).unwrap();
    fs::create_dir_all(results_dir(root)).unwrap();
    temp
}

fn root_view_id(ops: &Arc<dyn OpStore>) -> ViewId {
    ops.read_operation(ops.root_operation_id())
        .block_on()
        .unwrap()
        .view_id
}

/// An operation whose content — and therefore whose id — is determined by
/// `description`. Distinct descriptions are what make "exactly one new
/// operation object" in assertion (c) a real assertion: identical merge
/// operations from N children would collapse to one object by content
/// addressing alone, proving nothing about the lock.
fn operation_data(view_id: ViewId, parents: Vec<OperationId>, description: &str) -> OperationData {
    let timestamp = Timestamp {
        timestamp: MillisSinceEpoch(0),
        tz_offset: 0,
    };
    OperationData {
        view_id,
        parents,
        metadata: OperationMetadata {
            time: TimestampRange {
                start: timestamp,
                end: timestamp,
            },
            description: description.to_string(),
            hostname: "s14".to_string(),
            username: "s14".to_string(),
            is_snapshot: false,
            workspace_name: None,
            attributes: BTreeMap::new(),
        },
        commit_predecessors: Some(BTreeMap::new()),
    }
}

fn heads_of(store: &VexOpHeadsStore) -> HashSet<OperationId> {
    store
        .get_op_heads()
        .block_on()
        .unwrap()
        .into_iter()
        .collect()
}

fn stored_operation_ids(root: &Path) -> HashSet<String> {
    fs::read_dir(op_store_dir(root).join("operations"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// `resolve_op_heads` is generic over one error type that all three failure
/// modes convert into.
#[derive(Debug)]
enum ResolveError {
    Resolution(OpHeadResolutionError),
    Heads(OpHeadsStoreError),
    Store(OpStoreError),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resolution(err) => write!(f, "{err}"),
            Self::Heads(err) => write!(f, "{err}"),
            Self::Store(err) => write!(f, "{err}"),
        }
    }
}

impl From<OpHeadResolutionError> for ResolveError {
    fn from(err: OpHeadResolutionError) -> Self {
        Self::Resolution(err)
    }
}

impl From<OpHeadsStoreError> for ResolveError {
    fn from(err: OpHeadsStoreError) -> Self {
        Self::Heads(err)
    }
}

impl From<OpStoreError> for ResolveError {
    fn from(err: OpStoreError) -> Self {
        Self::Store(err)
    }
}

/// Resolve the repository to a single operation, merging divergent heads into
/// a new operation tagged with `tag` so each caller's merge is a distinct
/// object.
fn resolve_to_single_head(
    store: &VexOpHeadsStore,
    ops: &Arc<dyn OpStore>,
    tag: &str,
) -> Result<Operation, ResolveError> {
    resolve_op_heads(store, ops, async |heads: Vec<Operation>| {
        let parents = heads.iter().map(|op| op.id().clone()).collect();
        let data = operation_data(
            heads[0].view_id().clone(),
            parents,
            &format!("merge of {} heads by {tag}", heads.len()),
        );
        let id = ops.write_operation(&data).await?;
        Ok(Operation::new(ops.clone(), id, data))
    })
    .block_on()
}

// ---------------------------------------------------------------------------
// Multi-process harness
// ---------------------------------------------------------------------------

/// Spawn `n` children in `role`, release them together, and wait for them all.
fn run_children(root: &Path, role: &str, n: usize) {
    let mut children: Vec<Child> = (0..n)
        .map(|idx| {
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "s14_child", "--ignored", "--nocapture"])
                .env("VEX_S14_ROOT", root)
                .env("VEX_S14_ROLE", role)
                .env("VEX_S14_IDX", idx.to_string())
                .spawn()
                .unwrap()
        })
        .collect();
    wait_for_ready(root, &mut children);
    fs::write(rendezvous_dir(root).join("go"), "").unwrap();
    reap(children);
}

/// Block until every child has announced itself, so the `go` file releases them
/// all at once and their writes genuinely collide.
fn wait_for_ready(root: &Path, children: &mut [Child]) {
    let deadline = Instant::now() + WATCHDOG;
    loop {
        let ready = (0..children.len())
            .filter(|idx| rendezvous_dir(root).join(format!("ready-{idx}")).exists())
            .count();
        if ready == children.len() {
            return;
        }
        // A child that died before the rendezvous would otherwise burn the
        // whole watchdog; fail immediately with the exit status instead.
        for (idx, child) in children.iter_mut().enumerate() {
            if let Some(status) = child.try_wait().unwrap() {
                kill_all(children);
                panic!("child {idx} exited with {status} before reaching the rendezvous");
            }
        }
        assert!(
            Instant::now() < deadline,
            "only {ready}/{} children reached the rendezvous within {WATCHDOG:?}",
            children.len()
        );
        std::thread::sleep(POLL);
    }
}

fn kill_all(children: &mut [Child]) {
    for child in children.iter_mut() {
        drop(child.kill());
        drop(child.wait());
    }
}

/// Wait for every child under the watchdog and require a clean exit from each.
fn reap(mut children: Vec<Child>) {
    let deadline = Instant::now() + WATCHDOG;
    let mut statuses = vec![None; children.len()];
    loop {
        let mut pending = false;
        for (idx, child) in children.iter_mut().enumerate() {
            if statuses[idx].is_some() {
                continue;
            }
            match child.try_wait().unwrap() {
                Some(status) => statuses[idx] = Some(status),
                None => pending = true,
            }
        }
        if !pending {
            break;
        }
        if Instant::now() >= deadline {
            kill_all(&mut children);
            panic!("children did not finish within {WATCHDOG:?}; killed");
        }
        std::thread::sleep(POLL);
    }
    for (idx, status) in statuses.into_iter().enumerate() {
        let status = status.unwrap();
        assert!(status.success(), "child {idx} exited with {status}");
    }
}

/// Every child's report, in spawn order.
fn child_results(root: &Path, n: usize) -> Vec<String> {
    (0..n)
        .map(|idx| {
            let path = results_dir(root).join(format!("result-{idx}"));
            fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("child {idx} wrote no result ({err}): {path:?}"))
        })
        .collect()
}

fn assert_all_ok(results: &[String]) {
    for (idx, result) in results.iter().enumerate() {
        for line in result.lines() {
            assert_eq!(line, "ok", "child {idx} reported a failed write: {result}");
        }
    }
}

/// D10, stated as a string search as well as a status check: no local path may
/// return a *concurrency refusal*. Even a write that somehow succeeded while
/// reporting a CAS conflict would be a Stage 7 regression.
fn assert_no_concurrency_refusals(results: &[String]) {
    for (idx, result) in results.iter().enumerate() {
        let lowered = result.to_lowercase();
        for needle in ["cas conflict", "conflict", "refus"] {
            assert!(
                !lowered.contains(needle),
                "child {idx} reported a concurrency refusal ({needle:?}): {result}"
            );
        }
    }
}

fn rendezvous(root: &Path, idx: usize) {
    let dir = rendezvous_dir(root);
    fs::write(dir.join(format!("ready-{idx}")), "").unwrap();
    let go = dir.join("go");
    let deadline = Instant::now() + WATCHDOG;
    while !go.exists() {
        assert!(
            Instant::now() < deadline,
            "the parent never released the rendezvous"
        );
        std::thread::sleep(POLL);
    }
}

/// The child body. Never run by an ordinary test pass: it is `#[ignore]`d and
/// only reachable through `run_children`, which supplies the environment.
#[test]
#[ignore = "spawned as a child by the S14 concurrency harness"]
fn s14_child() {
    let root = PathBuf::from(std::env::var_os("VEX_S14_ROOT").expect("VEX_S14_ROOT"));
    let role = std::env::var("VEX_S14_ROLE").expect("VEX_S14_ROLE");
    let idx: usize = std::env::var("VEX_S14_IDX").unwrap().parse().unwrap();

    let report = match role.as_str() {
        "commit" => child_commit(&root, idx),
        "resolve" => child_resolve(&root, idx),
        "hammer" => child_hammer(&root, idx),
        other => panic!("unknown child role {other:?}"),
    };
    fs::write(results_dir(&root).join(format!("result-{idx}")), report).unwrap();
}

/// (a): read the head the command started from, *then* rendezvous, then commit
/// on top of it. Reading first is what makes the two children genuine siblings
/// rather than a sequence.
fn child_commit(root: &Path, idx: usize) -> String {
    let store = heads_store(root);
    let ops = op_store(root);
    let parents = store.get_op_heads().block_on().unwrap();
    let view_id = root_view_id(&ops);

    rendezvous(root, idx);

    let data = operation_data(view_id, parents.clone(), &format!("commit by child {idx}"));
    let new_id = ops.write_operation(&data).block_on().unwrap();
    match store.update_op_heads(&parents, &new_id).block_on() {
        Ok(()) => "ok".to_string(),
        Err(err) => format!("err: {err}"),
    }
}

/// (c): an ordinary read of a divergent repository, which is what
/// `resolve_op_heads` performs on every command.
fn child_resolve(root: &Path, idx: usize) -> String {
    let store = heads_store(root);
    let ops = op_store(root);

    rendezvous(root, idx);

    match resolve_to_single_head(&store, &ops, &format!("child {idx}")) {
        Ok(_) => "ok".to_string(),
        Err(err) => format!("err: {err}"),
    }
}

/// (d): a sustained stream of commits on this child's own lineage. Every
/// child's first update tries to retire the same root head, so exactly one
/// removal finds the file and the other seven must succeed anyway.
fn child_hammer(root: &Path, idx: usize) -> String {
    let store = heads_store(root);
    let ops = op_store(root);
    let view_id = root_view_id(&ops);

    rendezvous(root, idx);

    let mut parents = vec![ops.root_operation_id().clone()];
    let mut report = Vec::with_capacity(HAMMER_OPS);
    for seq in 0..HAMMER_OPS {
        let data = operation_data(
            view_id.clone(),
            parents.clone(),
            &format!("child {idx} operation {seq}"),
        );
        let new_id = ops.write_operation(&data).block_on().unwrap();
        report.push(match store.update_op_heads(&parents, &new_id).block_on() {
            Ok(()) => "ok".to_string(),
            Err(err) => format!("err: {err}"),
        });
        parents = vec![new_id];
    }
    report.join("\n")
}

// ---------------------------------------------------------------------------
// (a) No lost head
// ---------------------------------------------------------------------------

/// Two concurrent same-repository commands, each committing an operation on
/// top of the same head. Both heads must survive, and the next read must merge
/// them.
///
/// This is the assertion the pre-Stage-7 store could not pass. Heads then lived
/// in the flat `vex-local-heads` file, rewritten whole by
/// `write_local_heads`: each writer serialized *its own* view of the head set
/// and renamed it over the file, so the second writer's rename silently
/// discarded the first writer's head. The operation stayed in the object store
/// but became unreachable — a committed operation lost. It passes now because
/// a head is a *file named after itself*: two writers create two different
/// files, and neither can overwrite the other's.
#[test]
fn s14_a_two_concurrent_commits_leave_both_heads() {
    let temp = init_repo();
    let root = temp.path();
    let store = heads_store(root);
    let ops = op_store(root);
    let root_op_id = ops.root_operation_id().clone();
    store.update_op_heads(&[], &root_op_id).block_on().unwrap();

    run_children(root, "commit", 2);
    assert_all_ok(&child_results(root, 2));

    let heads = heads_of(&store);
    assert_eq!(
        heads.len(),
        2,
        "both concurrently committed heads must survive: {heads:?}"
    );
    assert!(!heads.contains(&root_op_id), "the old head was retired");

    // And the next read merges them, rather than picking one.
    let merged = resolve_to_single_head(&store, &ops, "parent").unwrap();
    assert_eq!(
        merged.parent_ids().iter().cloned().collect::<HashSet<_>>(),
        heads,
        "the merge operation's parents are exactly the two concurrent heads"
    );
    assert_eq!(heads_of(&store), HashSet::from([merged.id().clone()]));
}

// ---------------------------------------------------------------------------
// (b) Add before remove
// ---------------------------------------------------------------------------

/// An update that fails partway through must leave an *extra* head, never
/// zero. Zero heads is unrecoverable — jj reads the repository as having no
/// operations at all — whereas an extra head is merged on the next load.
///
/// The interruption is forced rather than simulated: the old head's name is
/// occupied by a directory, so the removal step returns a hard error (`EISDIR`
/// / `EPERM`) instead of the tolerated `NotFound`. The update therefore fails,
/// and the assertion is that the new head is on disk anyway — which can only
/// be true if the add happened before the remove.
#[test]
fn s14_b_an_interrupted_update_leaves_an_extra_head_never_zero() {
    let temp = init_repo();
    let root = temp.path();
    let store = heads_store(root);
    let old_id = OperationId::new(vec![0x11; 32]);
    let new_id = OperationId::new(vec![0x22; 32]);
    store.update_op_heads(&[], &old_id).block_on().unwrap();

    let head_path = op_heads_dir(root).join(HEADS_SUBDIR).join(old_id.hex());
    fs::remove_file(&head_path).unwrap();
    fs::create_dir(&head_path).unwrap();
    fs::write(head_path.join("keep"), "").unwrap();

    let err = store
        .update_op_heads(std::slice::from_ref(&old_id), &new_id)
        .block_on()
        .unwrap_err();
    assert!(
        matches!(err, OpHeadsStoreError::Write { .. }),
        "the removal failed, not the add: {err:?}"
    );

    let heads = heads_of(&store);
    assert!(
        heads.contains(&new_id),
        "the new head was written before the removal was attempted: {heads:?}"
    );
    assert!(
        heads.len() >= 2,
        "an interrupted update leaves an extra head, never fewer: {heads:?}"
    );
}

/// The other half of add-before-remove: retiring a head that is already gone —
/// because a concurrent process retired it first — is a success, not an error.
/// Without this, every loser of a race would surface a spurious failure to the
/// user for an operation that was in fact recorded.
#[test]
fn s14_b_a_removal_that_finds_nothing_succeeds() {
    let temp = init_repo();
    let root = temp.path();
    let store = heads_store(root);
    let missing = OperationId::new(vec![0x33; 32]);
    let new_id = OperationId::new(vec![0x44; 32]);

    store
        .update_op_heads(std::slice::from_ref(&missing), &new_id)
        .block_on()
        .unwrap();

    assert_eq!(heads_of(&store), HashSet::from([new_id]));
}

/// Name of the subdirectory `VexOpHeadsStore` delegates head storage to. Kept
/// local to the test so the suite does not depend on a production constant
/// being made public.
const HEADS_SUBDIR: &str = "heads";

// ---------------------------------------------------------------------------
// (c) A real lock
// ---------------------------------------------------------------------------

/// Eight processes read a divergent repository at the same instant. Exactly one
/// merge must result.
///
/// This is the PRD 093 merge storm reproduced entirely locally. It passes only
/// because `lock()` returns a real `FileLock` on `heads/lock`: the readers
/// serialize on it, the first one merges, and the other seven then observe a
/// single head and return it without writing anything. With the previous
/// no-op lock, all eight would have merged concurrently and written eight
/// competing merge operations — leaving the repository more divergent than it
/// started, which is exactly the divergence-storm failure mode.
///
/// That is not a claim, it is a measurement: replacing `VexOpHeadsStore::lock`
/// with a no-op lock and re-running this test leaves 8 heads and 8 merge
/// operations.
#[test]
fn s14_c_eight_concurrent_readers_produce_exactly_one_merge() {
    let temp = init_repo();
    let root = temp.path();
    let store = heads_store(root);
    let ops = op_store(root);
    let view_id = root_view_id(&ops);
    let root_op_id = ops.root_operation_id().clone();

    // Force divergence: two sibling operations, both recorded as heads.
    let mut divergent = Vec::new();
    for side in ["left", "right"] {
        let data = operation_data(view_id.clone(), vec![root_op_id.clone()], side);
        let id = ops.write_operation(&data).block_on().unwrap();
        store.update_op_heads(&[], &id).block_on().unwrap();
        divergent.push(id);
    }
    let before = stored_operation_ids(root);
    assert_eq!(before.len(), 2);
    assert_eq!(heads_of(&store).len(), 2);

    run_children(root, "resolve", CHILDREN);
    assert_all_ok(&child_results(root, CHILDREN));

    let heads = heads_of(&store);
    assert_eq!(
        heads.len(),
        1,
        "{CHILDREN} concurrent readers must converge on one head, not {}: {heads:?}",
        heads.len()
    );
    let after = stored_operation_ids(root);
    let new_objects: Vec<_> = after.difference(&before).collect();
    assert_eq!(
        new_objects.len(),
        1,
        "exactly one merge operation may be written by {CHILDREN} readers, got {new_objects:?}"
    );
    let merge_id = heads.into_iter().next().unwrap();
    assert_eq!(merge_id.hex(), *new_objects[0]);
    let merge = ops.read_operation(&merge_id).block_on().unwrap();
    assert_eq!(
        merge.parents.iter().cloned().collect::<HashSet<_>>(),
        divergent.into_iter().collect::<HashSet<_>>(),
        "the single merge has both divergent heads as parents"
    );
}

// ---------------------------------------------------------------------------
// (d) A commit cannot fail
// ---------------------------------------------------------------------------

/// Eight processes commit twenty operations each into one repository, released
/// together. Every single head write must succeed.
///
/// This is D10 as an executable statement: an operation that has been committed
/// cannot fail to be recorded, so no local code path may answer a head write
/// with a concurrency refusal. Contention is real here — all eight children
/// race to retire the same root head on their first write, and then interleave
/// 160 writes through one directory.
#[test]
fn s14_d_hammering_one_repo_from_eight_processes_never_refuses_a_write() {
    let temp = init_repo();
    let root = temp.path();
    let store = heads_store(root);
    let ops = op_store(root);
    let root_op_id = ops.root_operation_id().clone();
    store.update_op_heads(&[], &root_op_id).block_on().unwrap();

    run_children(root, "hammer", CHILDREN);

    let results = child_results(root, CHILDREN);
    for (idx, result) in results.iter().enumerate() {
        assert_eq!(
            result.lines().count(),
            HAMMER_OPS,
            "child {idx} did not report every operation"
        );
    }
    assert_all_ok(&results);
    assert_no_concurrency_refusals(&results);

    let heads = heads_of(&store);
    assert!(!heads.is_empty(), "the repository still has heads");
    assert!(
        !heads.contains(&root_op_id),
        "the root head was retired exactly once, by whichever child got there first"
    );

    // And the divergence the hammering produced is ordinary, resolvable
    // divergence: one read converges it.
    let merged = resolve_to_single_head(&store, &ops, "parent").unwrap();
    assert_eq!(heads_of(&store), HashSet::from([merged.id().clone()]));
}
