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

//! Metric S10 for roadmap/088 D9: auto fast-forward matches jj and never
//! clobbers.
//!
//! Three cases, plus the guard the PRD demands:
//!
//! (a) an unmoved local bookmark whose server target advanced is
//!     fast-forwarded, and the advance is reported;
//! (b) a local bookmark that has moved is **not** fast-forwarded and becomes a
//!     conflicted bookmark, exactly as jj's three-way merge produces;
//! (c) no probe, in any case, changes `@` or the working-copy commit;
//! (d) no probe path can write `.jj/working_copy/checkout` — the file the 093
//!     sibling-operation incident had to be repaired by hand.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::LazyLock;
use std::sync::Mutex;

use jj_lib::backend::CommitId;
use jj_lib::object_id::ObjectId as _;
use jj_lib::op_store::RefTarget;
use jj_lib::op_store::RemoteRef;
use jj_lib::op_store::RemoteRefState;
use jj_lib::ref_name::RefName;
use jj_lib::ref_name::RefNameBuf;
use jj_lib::ref_name::RemoteName;
use jj_lib::ref_name::RemoteRefSymbol;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::Repo as _;
use jj_lib::vex_ref_sync::BookmarkUpdate;
use jj_lib::vex_ref_sync::RefSyncError;
use jj_lib::vex_ref_sync::RefSyncReport;
use jj_lib::vex_ref_sync::ServerRefSource;
use jj_lib::vex_ref_sync::VEX_REMOTE;
use jj_lib::vex_ref_sync::fast_forward_tracked_bookmarks;
use jj_lib::vex_ref_sync::sync_tracked_bookmarks;
use pollster::FutureExt as _;
use testutils::TestWorkspace;
use testutils::write_random_commit;
use testutils::write_random_commit_with_parents;

/// `VEX_NO_REFRESH` is process-global, so the opt-out test serializes against
/// anything else that reads it.
static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn remote() -> &'static RemoteName {
    RemoteName::new(VEX_REMOTE)
}

fn bookmark(name: &str) -> &RefName {
    RefName::new(name)
}

fn server_targets(entries: &[(&str, &CommitId)]) -> BTreeMap<RefNameBuf, CommitId> {
    entries
        .iter()
        .map(|(name, id)| (RefNameBuf::from(*name), (*id).clone()))
        .collect()
}

/// A server whose answers are fixed, and which counts how often it is asked —
/// so a test can assert that a suppressed pass performs no fetch at all.
struct FakeServer {
    targets: BTreeMap<RefNameBuf, CommitId>,
    calls: RefCell<usize>,
}

impl FakeServer {
    fn new(targets: BTreeMap<RefNameBuf, CommitId>) -> Self {
        Self {
            targets,
            calls: RefCell::new(0),
        }
    }
}

impl ServerRefSource for FakeServer {
    fn bookmark_targets(
        &self,
        names: &[RefNameBuf],
    ) -> Result<BTreeMap<RefNameBuf, CommitId>, RefSyncError> {
        *self.calls.borrow_mut() += 1;
        Ok(self
            .targets
            .iter()
            .filter(|(name, _)| names.contains(name))
            .map(|(name, id)| (name.clone(), id.clone()))
            .collect())
    }
}

/// A workspace with `main` tracked at `base` both locally and as `main@vex`,
/// plus the two commits a divergence needs. Returns
/// `(workspace, repo, base, server_advance, local_advance)`.
fn workspace_tracking_main() -> (TestWorkspace, std::sync::Arc<ReadonlyRepo>, [CommitId; 3]) {
    let test_workspace = TestWorkspace::init();
    let repo = test_workspace.repo.clone();

    let mut tx = repo.start_transaction();
    let base = write_random_commit(tx.repo_mut());
    let server_advance = write_random_commit_with_parents(tx.repo_mut(), &[&base]);
    let local_advance = write_random_commit_with_parents(tx.repo_mut(), &[&base]);
    tx.repo_mut()
        .set_local_bookmark_target(bookmark("main"), RefTarget::normal(base.id().clone()));
    tx.repo_mut().set_remote_bookmark(
        RemoteRefSymbol {
            name: bookmark("main"),
            remote: remote(),
        },
        RemoteRef {
            target: RefTarget::normal(base.id().clone()),
            state: RemoteRefState::Tracked,
        },
    );
    let repo = tx.commit("set up a tracked bookmark").block_on().unwrap();

    let ids = [
        base.id().clone(),
        server_advance.id().clone(),
        local_advance.id().clone(),
    ];
    (test_workspace, repo, ids)
}

/// S10 (a). The trivial resolution of `[local, remote] - [known_remote]`.
#[test]
fn an_unmoved_bookmark_fast_forwards_and_is_reported() {
    let (_workspace, repo, [base, server_advance, _]) = workspace_tracking_main();

    let mut tx = repo.start_transaction();
    let report = fast_forward_tracked_bookmarks(
        tx.repo_mut(),
        remote(),
        &server_targets(&[("main", &server_advance)]),
    )
    .unwrap();

    // The local bookmark moved to the server's target...
    assert_eq!(
        tx.repo().view().get_local_bookmark(bookmark("main")),
        &RefTarget::normal(server_advance.clone())
    );
    // ...and so did the base for the next merge, which is what keeps this a
    // three-way merge rather than a growing pile of divergence.
    assert_eq!(
        tx.repo()
            .get_remote_bookmark(RemoteRefSymbol {
                name: bookmark("main"),
                remote: remote(),
            })
            .target,
        RefTarget::normal(server_advance.clone())
    );

    // Rule 4: the advance is reported, naming both ends.
    assert_eq!(
        report.outcomes[0].update,
        BookmarkUpdate::FastForwarded {
            from: Some(base.hex()),
            to: server_advance.hex(),
        }
    );
    let lines = report.summary_lines();
    assert_eq!(lines.len(), 1);
    assert!(lines[0].starts_with("Fast-forwarded bookmark main: "));
    assert!(lines[0].contains(&base.hex()[..12]));
    assert!(lines[0].contains(&server_advance.hex()[..12]));
}

/// S10 (a), negative half: re-running the pass against the target it already
/// adopted is not reported as another advance.
#[test]
fn a_bookmark_that_is_already_at_the_server_target_reports_nothing() {
    let (_workspace, repo, [_, server_advance, _]) = workspace_tracking_main();
    let targets = server_targets(&[("main", &server_advance)]);

    let mut tx = repo.start_transaction();
    fast_forward_tracked_bookmarks(tx.repo_mut(), remote(), &targets).unwrap();
    let report = fast_forward_tracked_bookmarks(tx.repo_mut(), remote(), &targets).unwrap();

    assert_eq!(report.outcomes[0].update, BookmarkUpdate::Unchanged);
    assert!(!report.changed());
    assert!(report.summary_lines().is_empty());
}

/// S10 (b). A local bookmark that has moved is never fast-forwarded: the
/// merge is non-trivial, so jj produces a conflicted bookmark and the local
/// work is still in it.
#[test]
fn a_moved_local_bookmark_becomes_conflicted_not_clobbered() {
    let (_workspace, repo, [base, server_advance, local_advance]) = workspace_tracking_main();

    // The user moves `main` locally; the server moved it elsewhere.
    let mut tx = repo.start_transaction();
    tx.repo_mut()
        .set_local_bookmark_target(bookmark("main"), RefTarget::normal(local_advance.clone()));
    let report = fast_forward_tracked_bookmarks(
        tx.repo_mut(),
        remote(),
        &server_targets(&[("main", &server_advance)]),
    )
    .unwrap();

    let after = tx
        .repo()
        .view()
        .get_local_bookmark(bookmark("main"))
        .clone();
    assert!(after.has_conflict(), "expected a conflicted bookmark");
    // Not clobbered: the local move is still a term of the conflict, and so is
    // the server's, with the merge base removed.
    let adds: Vec<&CommitId> = after.added_ids().collect();
    assert!(adds.contains(&&local_advance));
    assert!(adds.contains(&&server_advance));
    assert_eq!(after.removed_ids().collect::<Vec<_>>(), vec![&base]);

    // Rule 2: a conflicted bookmark is reported, not an error.
    assert_eq!(
        report.outcomes[0].update,
        BookmarkUpdate::Conflicted {
            adds: 2,
            removes: 1
        }
    );
    assert_eq!(report.conflicted().collect::<Vec<_>>(), vec!["main"]);
    assert!(report.summary_lines()[0].contains("conflicted"));

    // The base still advanced, so the same divergence is not re-reported for
    // ever — exactly what `git.rs`'s `import_refs_inner` does.
    assert_eq!(
        tx.repo()
            .get_remote_bookmark(RemoteRefSymbol {
                name: bookmark("main"),
                remote: remote(),
            })
            .target,
        RefTarget::normal(server_advance)
    );
}

/// S10 (b), the other half of "never an overwrite": when only the local
/// bookmark moved, the pass leaves it exactly where the user put it.
#[test]
fn a_moved_local_bookmark_is_untouched_when_the_server_did_not_move() {
    let (_workspace, repo, [base, _, local_advance]) = workspace_tracking_main();

    let mut tx = repo.start_transaction();
    tx.repo_mut()
        .set_local_bookmark_target(bookmark("main"), RefTarget::normal(local_advance.clone()));
    let report = fast_forward_tracked_bookmarks(
        tx.repo_mut(),
        remote(),
        &server_targets(&[("main", &base)]),
    )
    .unwrap();

    assert_eq!(
        tx.repo().view().get_local_bookmark(bookmark("main")),
        &RefTarget::normal(local_advance)
    );
    assert_eq!(report.outcomes[0].update, BookmarkUpdate::Unchanged);
}

/// An untracked remote bookmark is one the user chose not to follow, and an
/// unknown name is not ours to create.
#[test]
fn only_tracked_bookmarks_are_considered() {
    let (_workspace, repo, [_, server_advance, _]) = workspace_tracking_main();

    let mut tx = repo.start_transaction();
    tx.repo_mut().untrack_remote_bookmark(RemoteRefSymbol {
        name: bookmark("main"),
        remote: remote(),
    });
    let before = tx
        .repo()
        .view()
        .get_local_bookmark(bookmark("main"))
        .clone();
    let report = fast_forward_tracked_bookmarks(
        tx.repo_mut(),
        remote(),
        &server_targets(&[("main", &server_advance), ("brand-new", &server_advance)]),
    )
    .unwrap();

    assert!(report.outcomes.is_empty());
    assert_eq!(
        tx.repo().view().get_local_bookmark(bookmark("main")),
        &before
    );
    assert!(
        tx.repo()
            .view()
            .get_local_bookmark(bookmark("brand-new"))
            .is_absent()
    );
}

/// S10 (c). The whole sync path, end to end, against a real workspace: the
/// bookmark moves and `@` does not.
#[test]
fn no_probe_path_changes_the_working_copy_commit() {
    let (workspace, repo, [_, server_advance, _]) = workspace_tracking_main();
    let wc_before = repo.view().wc_commit_ids().clone();
    assert!(
        !wc_before.is_empty(),
        "the fixture must have a working copy"
    );
    let checkout = workspace
        .workspace
        .workspace_root()
        .join(".jj")
        .join("working_copy")
        .join("checkout");
    let checkout_before = std::fs::read(&checkout).unwrap();

    let source = FakeServer::new(server_targets(&[("main", &server_advance)]));
    let (report, repo) = sync_tracked_bookmarks(&repo, &source).block_on().unwrap();

    assert!(report.changed(), "the bookmark should have moved");
    assert_eq!(
        repo.view().get_local_bookmark(bookmark("main")),
        &RefTarget::normal(server_advance)
    );
    // `@` and the working-copy commit are untouched, in the view...
    assert_eq!(repo.view().wc_commit_ids(), &wc_before);
    // ...and on disk.
    assert_eq!(std::fs::read(&checkout).unwrap(), checkout_before);
}

/// S10 (c), for the case that produces a conflicted bookmark: divergence must
/// not move the working copy either.
#[test]
fn a_conflicting_sync_does_not_change_the_working_copy_commit() {
    let (workspace, repo, [_, server_advance, local_advance]) = workspace_tracking_main();
    let mut tx = repo.start_transaction();
    tx.repo_mut()
        .set_local_bookmark_target(bookmark("main"), RefTarget::normal(local_advance));
    let repo = tx.commit("move main locally").block_on().unwrap();
    let wc_before = repo.view().wc_commit_ids().clone();
    let checkout = workspace
        .workspace
        .workspace_root()
        .join(".jj")
        .join("working_copy")
        .join("checkout");
    let checkout_before = std::fs::read(&checkout).unwrap();

    let source = FakeServer::new(server_targets(&[("main", &server_advance)]));
    let (report, repo) = sync_tracked_bookmarks(&repo, &source).block_on().unwrap();

    assert_eq!(report.conflicted().collect::<Vec<_>>(), vec!["main"]);
    assert_eq!(repo.view().wc_commit_ids(), &wc_before);
    assert_eq!(std::fs::read(&checkout).unwrap(), checkout_before);
}

/// D9 rule 5: the opt-out that disables the probe disables this too, and does
/// so before any network contact.
#[test]
fn the_no_refresh_opt_out_suppresses_the_pass_entirely() {
    let _guard = ENV_LOCK.lock().unwrap();
    let (_workspace, repo, [base, server_advance, _]) = workspace_tracking_main();
    let source = FakeServer::new(server_targets(&[("main", &server_advance)]));

    unsafe {
        std::env::set_var("VEX_NO_REFRESH", "1");
    }
    let result = sync_tracked_bookmarks(&repo, &source).block_on();
    unsafe {
        std::env::remove_var("VEX_NO_REFRESH");
    }
    let (report, repo) = result.unwrap();

    assert!(report.suppressed);
    assert!(!report.changed());
    assert_eq!(
        *source.calls.borrow(),
        0,
        "a suppressed pass must not fetch"
    );
    assert_eq!(
        repo.view().get_local_bookmark(bookmark("main")),
        &RefTarget::normal(base)
    );
}

/// S10 (d), the structural half. The guard the PRD demands is not "this run
/// happened not to write the checkout file" — it is that no probe path *can*.
///
/// The sync path takes a `ReadonlyRepo` and nothing else: there is no
/// `Workspace`, no `WorkingCopy`, and no path under `.jj/working_copy` in
/// reach. This test reads the two files that make up the probe path and fails
/// if either ever acquires one, which is what the incident recovery — hand
/// editing `.jj/working_copy/checkout` — cost.
#[test]
fn no_probe_path_can_write_the_working_copy_checkout_file() {
    // Comments legitimately mention the working copy (that is the rule being
    // documented), so the scan is over code lines only.
    let forbidden = [
        "working_copy",
        "LocalWorkingCopy",
        "checkout",
        "check_out",
        "set_wc_commit",
        "start_mutation",
        ".edit(",
    ];
    for (path, source) in [
        (
            "jj/lib/src/vex_ref_sync.rs",
            include_str!("../src/vex_ref_sync.rs"),
        ),
        (
            "jj/lib/src/vex_freshness.rs",
            include_str!("../src/vex_freshness.rs"),
        ),
    ] {
        for (number, line) in source.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            for needle in forbidden {
                assert!(
                    !code.contains(needle),
                    "{path}:{} reaches the working copy (`{needle}`); the probe path must not \
                     be able to write .jj/working_copy/checkout: {line}",
                    number + 1
                );
            }
        }
    }
    // `wc_commit_ids` is the one working-copy *read* the path performs, and it
    // exists only to assert that nothing moved. Name it explicitly so the scan
    // above staying green keeps meaning something.
    assert!(include_str!("../src/vex_ref_sync.rs").contains("wc_commit_ids"));
}

/// The runtime half of the same guard: the merge refuses to hand back a
/// transaction whose working-copy commits moved away from the operation it
/// started at, so a future change that introduces one fails loudly instead of
/// silently editing what the user is working on. Simulated here by moving `@`
/// in the transaction the pass is handed.
#[test]
fn the_merge_refuses_a_transaction_that_moved_the_working_copy() {
    let (_workspace, repo, [_, server_advance, local_advance]) = workspace_tracking_main();

    let mut tx = repo.start_transaction();
    let workspace_name = repo.view().wc_commit_ids().keys().next().unwrap().clone();
    let moved = tx
        .repo()
        .store()
        .get_commit_async(&local_advance)
        .block_on()
        .unwrap();
    tx.repo_mut()
        .edit(workspace_name, &moved)
        .block_on()
        .unwrap();

    let result = fast_forward_tracked_bookmarks(
        tx.repo_mut(),
        remote(),
        &server_targets(&[("main", &server_advance)]),
    );

    assert!(matches!(result, Err(RefSyncError::WorkingCopyMoved)));
}

/// A report that changed nothing is not worth a line of output, and the
/// summary is what the command prints.
#[test]
fn an_empty_report_prints_nothing() {
    assert!(RefSyncReport::default().summary_lines().is_empty());
    assert!(RefSyncReport::suppressed().summary_lines().is_empty());
}
