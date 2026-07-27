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

//! Vex operation-head storage (roadmap/088 Stage 7).
//!
//! Op heads are **local**. This store keeps them in `<op_heads>/heads/` by
//! delegating to [`SimpleOpHeadsStore`], jj's own directory-per-head scheme,
//! and never contacts the backend to read or write a head.
//!
//! Delegating rather than reimplementing is the point (D11). The upstream store
//! already has the two properties this stage exists to guarantee:
//!
//! - **Add before remove.** [`SimpleOpHeadsStore::update_op_heads`] writes the
//!   new head file first and only then removes the old ones, and a removal that
//!   finds nothing succeeds. An interruption therefore leaves an *extra* head,
//!   which jj merges on the next load — never zero heads, which is
//!   unrecoverable.
//! - **A real lock.** [`SimpleOpHeadsStore::lock`] returns a `FileLock` on
//!   `heads/lock`. The no-op lock this store used to return is what let N
//!   concurrent readers each commit a competing merge operation.
//!
//! The two stores share the `op_heads` directory without colliding: this file's
//! legacy markers (`vex-local-heads`, `vex-pending-publish`, `vex-server-heads`)
//! are flat files at the top level, while heads live one level down in
//! `heads/`, whose `lock` file is skipped by the reader because it is not hex.
//! Vex's 32-byte sha256 operation ids are written and read purely as hex, so
//! the length difference from upstream's 64-byte ids does not matter.
//!
//! **No local path may refuse a head write for concurrency reasons.** An
//! operation that has been committed cannot fail to be recorded (D10).

#![expect(missing_docs)]

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;

use crate::backend::BackendInitError;
use crate::object_id::ObjectId as _;
use crate::op_heads_store::OpHeadsStore;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_heads_store::OpHeadsStoreLock;
use crate::op_store::OperationId;
use crate::simple_op_heads_store::SimpleOpHeadsStore;
use crate::vex::VexClient;
use crate::vex::VexRepoConfig;
use crate::vex_op_head_delta::classify_refusal;

const ID_LENGTH: usize = 32;

/// Subdirectory of the op-heads store where heads actually live. Its existence
/// is the permanent guard on the one-time bootstrap: once it is there, nothing
/// in this file may ever contact the backend for op heads again.
const HEADS_DIR: &str = "heads";
/// Staging directory for the one-time bootstrap. Renamed onto [`HEADS_DIR`],
/// which is the commit point.
const HEADS_TMP_DIR: &str = "heads.tmp";

/// Default budget for the single backend read a pre-Stage-7 clone may perform
/// while seeding its local heads directory. Tunable with
/// `VEX_OP_HEADS_BOOTSTRAP_MS`.
const DEFAULT_BOOTSTRAP_MS: u64 = 10_000;

fn bootstrap_budget() -> Duration {
    let millis = std::env::var("VEX_OP_HEADS_BOOTSTRAP_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .unwrap_or(DEFAULT_BOOTSTRAP_MS);
    Duration::from_millis(millis)
}

/// Compatibility escape (roadmap/088 §0.4). With `VEX_PUBLISH_OP_LOG=1` a
/// successful *local* head write is additionally mirrored to the server's
/// `CommitOperation` CAS, so a mixed fleet of old and new clients can share a
/// repository during the rollout.
///
/// It restores *publishing*, nothing else: the deferred queue, the registration
/// fold and the client-side conflict classifier are deleted outright, and a
/// failed publish is logged rather than returned. Local storage stays
/// authoritative in both modes.
fn publish_op_log_escape() -> bool {
    static ESCAPE: OnceLock<bool> = OnceLock::new();
    *ESCAPE.get_or_init(|| {
        matches!(
            std::env::var("VEX_PUBLISH_OP_LOG").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    })
}

fn to_content_id(id: &OperationId) -> Result<jj_backend_types::ContentId, OpHeadsStoreError> {
    let bytes = id.to_bytes();
    if bytes.len() != ID_LENGTH {
        return Err(OpHeadsStoreError::Write {
            new_op_id: id.clone(),
            source: Box::new(std::io::Error::other(format!(
                "invalid operation id length: expected {ID_LENGTH}, got {}",
                bytes.len()
            ))),
        });
    }
    let mut content_bytes = [0; ID_LENGTH];
    content_bytes.copy_from_slice(&bytes);
    Ok(jj_backend_types::ContentId::from_bytes(content_bytes))
}

fn is_root_operation_id(id: &OperationId) -> bool {
    id.to_bytes().iter().all(|byte| *byte == 0)
}

fn read_error(err: impl std::error::Error + Send + Sync + 'static) -> OpHeadsStoreError {
    OpHeadsStoreError::Read(Box::new(err))
}

#[derive(Debug)]
pub struct VexOpHeadsStore {
    client: VexClient,
    /// The `op_heads` store directory: home of `heads/` and of the legacy
    /// bookkeeping markers the one-time bootstrap reads.
    store_dir: PathBuf,
    /// Head storage: `<store_dir>/heads`.
    simple: SimpleOpHeadsStore,
    /// When true, local-write mode is active (READ_ONLY CI runner). Since
    /// Stage 7 every mode stores heads locally, so this no longer selects a
    /// storage scheme; it means "never contact the backend at all", which now
    /// covers only the one-time bootstrap read and the compatibility escape.
    local_writes: bool,
    /// Per-process memo for [`Self::ensure_heads_dir`], so the guard costs one
    /// `stat` per process rather than one per call.
    bootstrapped: Arc<OnceLock<()>>,
}

impl Clone for VexOpHeadsStore {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            simple: SimpleOpHeadsStore::load(&self.store_dir),
            store_dir: self.store_dir.clone(),
            local_writes: self.local_writes,
            bootstrapped: Arc::clone(&self.bootstrapped),
        }
    }
}

impl VexOpHeadsStore {
    pub fn name_static() -> &'static str {
        "vex_op_heads_store"
    }

    pub fn init(config: VexRepoConfig, store_path: &Path) -> Result<Self, BackendInitError> {
        let local_writes = config.local_writes;
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        // Deliberately not `SimpleOpHeadsStore::init`: that calls `create_dir`,
        // which fails when the directory exists, and `<repo>/op_heads` has
        // already been created by the caller (`ReadonlyRepo::init`) before this
        // initializer runs.
        let heads_dir = store_path.join(HEADS_DIR);
        fs::create_dir_all(&heads_dir).map_err(|err| BackendInitError(err.into()))?;
        Ok(Self::at(client, store_path, local_writes))
    }

    pub fn load(store_path: &Path) -> Result<Self, crate::backend::BackendLoadError> {
        let client = VexClient::from_store_path(store_path)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))?;
        let local_writes = client.local_writes();
        Ok(Self::at(client, store_path, local_writes))
    }

    fn at(client: VexClient, store_path: &Path, local_writes: bool) -> Self {
        Self {
            client,
            simple: SimpleOpHeadsStore::load(store_path),
            store_dir: store_path.to_path_buf(),
            local_writes,
            bootstrapped: Arc::new(OnceLock::new()),
        }
    }

    fn heads_dir(&self) -> PathBuf {
        self.store_dir.join(HEADS_DIR)
    }

    /// Make sure `heads/` exists, seeding it once for a repository cloned
    /// before Stage 7.
    ///
    /// Idempotent, at most once per process per store (the `bootstrapped`
    /// memo), and at most once *ever* per repository (the existence of
    /// `heads/`). That second guard is what keeps this from becoming an ongoing
    /// server dependency: a repo that has the directory never reaches the seed
    /// logic, so no code path in this file can contact the backend for op heads
    /// again.
    async fn ensure_heads_dir(&self) -> Result<(), OpHeadsStoreError> {
        if self.bootstrapped.get().is_some() {
            return Ok(());
        }
        if !self.heads_dir().is_dir() {
            self.bootstrap_heads_dir().await?;
        }
        let _ = self.bootstrapped.set(());
        Ok(())
    }

    /// Seed `heads/` from whatever this repository already knows.
    ///
    /// Every source contributes: the set is a union, never narrowed to one id.
    /// Narrowing is the silent head loss D11 forbids, which is also why the
    /// PRD's "construct a fresh log from refs" instruction (right for `vex
    /// clone`, which has no history to lose) is deliberately not reused here.
    async fn bootstrap_heads_dir(&self) -> Result<(), OpHeadsStoreError> {
        let mut seed: Vec<OperationId> = Vec::new();
        let add = |id: OperationId, seed: &mut Vec<OperationId>| {
            if !seed.contains(&id) {
                seed.push(id);
            }
        };

        // (a) Heads a pre-Stage-7 client recorded locally. Covers every
        // `local_writes` repo and every repo in a deferred-publish session,
        // which is what the fleet was running by default.
        match crate::vex_publish::read_local_heads(&self.store_dir) {
            Ok(Some(ids)) => {
                for id in ids {
                    add(id, &mut seed);
                }
            }
            Ok(None) => {}
            Err(err) => tracing::warn!(error = %err, "unusable local-heads marker while seeding"),
        }
        // (b) The tip of a queue left behind by a deferred-publish session.
        match crate::vex_publish::read_pending_publish(&self.store_dir) {
            Ok(Some(chain)) => match chain.tip_ids() {
                Ok(ids) => {
                    for id in ids {
                        add(crate::vex_publish::op_id_from_content_id(&id), &mut seed);
                    }
                }
                Err(err) => tracing::warn!(error = %err, "unusable queue marker while seeding"),
            },
            Ok(None) => {}
            Err(err) => tracing::warn!(error = %err, "unusable queue marker while seeding"),
        }
        // (c) Heads a v1 server-heads marker recorded.
        match crate::vex_publish::read_server_heads(&self.store_dir) {
            Ok(Some(marker)) => match marker.head_ids() {
                Ok(ids) => {
                    for id in ids {
                        add(crate::vex_publish::op_id_from_content_id(&id), &mut seed);
                    }
                }
                Err(err) => {
                    tracing::warn!(error = %err, "unusable server-heads marker while seeding");
                }
            },
            Ok(None) => {}
            Err(err) => tracing::warn!(error = %err, "unusable server-heads marker while seeding"),
        }
        // (d) Nothing local to go on: read the server's heads exactly once,
        // under a bounded deadline. `local_writes` repos skip even this — they
        // are forbidden from contacting the backend at all.
        if seed.is_empty() && !self.local_writes {
            match self.client.get_op_heads_within(bootstrap_budget()) {
                Ok(Some(ids)) => {
                    // ALL returned ids, each as its own head. A divergent
                    // repository must not be collapsed on the way in.
                    for id in ids {
                        add(crate::vex_publish::op_id_from_content_id(&id), &mut seed);
                    }
                }
                Ok(None) => tracing::warn!("op-head bootstrap read exceeded its budget"),
                Err(err) => tracing::warn!(error = %err, "op-head bootstrap read failed"),
            }
        }
        // (e) Still nothing. Seeding an empty directory would tell jj this
        // repository has no operations at all, which is worse than failing.
        if seed.is_empty() {
            return Err(read_error(std::io::Error::other(format!(
                "cannot start a local operation log for {}: no local operation heads were \
                 recorded and none could be read from the backend; re-run `vex clone` to \
                 create a fresh workspace",
                self.store_dir.display()
            ))));
        }
        self.write_seed(&seed)
    }

    /// Write the seed into `heads.tmp` and rename it onto `heads`.
    ///
    /// The rename is the commit point, so an interrupted bootstrap leaves a
    /// stray `heads.tmp` and re-runs from scratch rather than leaving a
    /// half-seeded `heads/` that would look authoritative.
    fn write_seed(&self, seed: &[OperationId]) -> Result<(), OpHeadsStoreError> {
        let staging = self.store_dir.join(HEADS_TMP_DIR);
        if staging.exists() {
            fs::remove_dir_all(&staging).map_err(read_error)?;
        }
        fs::create_dir_all(&staging).map_err(read_error)?;
        for id in seed {
            fs::write(staging.join(id.hex()), "").map_err(read_error)?;
        }
        if let Ok(handle) = fs::File::open(&staging) {
            drop(handle.sync_all());
        }
        match fs::rename(&staging, self.heads_dir()) {
            Ok(()) => {
                tracing::info!(
                    heads = seed.len(),
                    "seeded the local operation-head directory from this repository's existing \
                     state"
                );
                Ok(())
            }
            Err(err) if self.heads_dir().is_dir() => {
                // Another process bootstrapped concurrently and won. Its seed
                // came from the same sources, so discard ours.
                tracing::debug!(error = %err, "lost the bootstrap race; keeping the other seed");
                drop(fs::remove_dir_all(&staging));
                Ok(())
            }
            Err(err) => Err(read_error(err)),
        }
    }

    /// Mirror a head that is already stored locally to the server's CAS, under
    /// the `VEX_PUBLISH_OP_LOG=1` escape. Best effort by construction: the
    /// caller logs any error and keeps the local head.
    async fn publish_head_best_effort(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        // The root operation is synthetic and never written to the remote
        // object store, so there is nothing to publish for it.
        if is_root_operation_id(new_id) {
            return Ok(());
        }
        let expected = old_ids
            .iter()
            .filter(|id| !is_root_operation_id(id))
            .map(to_content_id)
            .collect::<Result<Vec<_>, _>>()?;
        let new_content_id = to_content_id(new_id)?;
        let response = self
            .client
            .commit_op_heads(&expected, &new_content_id, &new_content_id)
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        if response.ok {
            Ok(())
        } else {
            Err(OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(classify_refusal(&response)),
            })
        }
    }

    /// The write path, with the escape decision passed in so it can be
    /// exercised without mutating the process environment.
    async fn update_op_heads_publishing(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
        publish: bool,
    ) -> Result<(), OpHeadsStoreError> {
        self.ensure_heads_dir().await?;
        self.simple.update_op_heads(old_ids, new_id).await?;
        if publish
            && let Err(err) = self.publish_head_best_effort(old_ids, new_id).await
        {
            tracing::warn!(
                error = %err,
                "VEX_PUBLISH_OP_LOG publish failed; the local head stands"
            );
        }
        Ok(())
    }
}

#[async_trait]
impl OpHeadsStore for VexOpHeadsStore {
    fn name(&self) -> &str {
        Self::name_static()
    }

    /// Record `new_id` as a head and retire `old_ids`.
    ///
    /// There is deliberately no root-operation short circuit here. The bootstrap
    /// write `ReadonlyRepo::init` performs is exactly `update_op_heads(&[],
    /// root_op_id)`, and swallowing it leaves `heads/` empty, so a freshly
    /// initialised repository would resolve no heads at all.
    ///
    /// This cannot fail for concurrency reasons. `SimpleOpHeadsStore` adds
    /// before it removes and tolerates a removal that finds nothing, so a
    /// racing writer costs an extra head that jj merges — never a refusal, and
    /// never a lost operation.
    async fn update_op_heads(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        self.update_op_heads_publishing(
            old_ids,
            new_id,
            publish_op_log_escape() && !self.local_writes,
        )
        .await
    }

    /// Read the heads out of `heads/`. Zero backend calls, in every mode.
    async fn get_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        self.ensure_heads_dir().await?;
        self.simple.get_op_heads().await
    }

    /// A real `FileLock` on `heads/lock`.
    ///
    /// This is what makes `resolve_op_heads` safe under concurrency: N readers
    /// that each find the repository divergent take this lock in turn, so one
    /// writes the merge operation and the rest observe it, instead of all N
    /// committing competing merges.
    async fn lock(&self) -> Result<Box<dyn OpHeadsStoreLock + '_>, OpHeadsStoreError> {
        self.ensure_heads_dir().await?;
        self.simple.lock().await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use futures::executor::block_on;

    use super::*;
    use crate::vex::VexObjectReadMode;
    use crate::vex_publish::ServerHeadsMarker;
    use crate::vex_publish::VexDurability;

    /// The configured endpoint is unroutable, so any backend round trip either
    /// fails outright or costs a connection refusal — a test that succeeds
    /// without one proves the path is local.
    fn test_config(local_writes: bool) -> VexRepoConfig {
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
            local_writes,
            durability: VexDurability::Sync,
            object_read_mode: VexObjectReadMode::NativeOnly,
        }
    }

    /// A store over an already-initialised repo: `heads/` exists, so nothing
    /// here ever enters the bootstrap.
    fn store_at(dir: &Path, local_writes: bool) -> VexOpHeadsStore {
        VexOpHeadsStore::init(test_config(local_writes), dir).unwrap()
    }

    /// A store constructed straight over `dir`, without the `vex.json` lookup
    /// `VexOpHeadsStore::load` performs. Used both to reopen an initialised
    /// repo and to model a *pre-Stage-7* one, where the directory exists but
    /// `heads/` does not.
    fn store_over(dir: &Path, local_writes: bool) -> VexOpHeadsStore {
        fs::create_dir_all(dir).unwrap();
        let client = VexClient::from_config(test_config(local_writes)).unwrap();
        VexOpHeadsStore::at(client, dir, local_writes)
    }

    fn op_id(byte: u8) -> OperationId {
        OperationId::new(vec![byte; ID_LENGTH])
    }

    fn root_op_id() -> OperationId {
        OperationId::new(vec![0; ID_LENGTH])
    }

    fn content_id(byte: u8) -> jj_backend_types::ContentId {
        jj_backend_types::ContentId::from_bytes([byte; ID_LENGTH])
    }

    fn heads_of(store: &VexOpHeadsStore) -> HashSet<OperationId> {
        block_on(store.get_op_heads()).unwrap().into_iter().collect()
    }

    fn head_files(dir: &Path) -> HashSet<String> {
        fs::read_dir(dir.join(HEADS_DIR))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    /// The bootstrap write `ReadonlyRepo::init` performs. The old store
    /// short-circuited it, which under local storage would leave `heads/` empty
    /// and the repository with no resolvable operation at all.
    #[test]
    fn root_operation_is_written_as_a_head_not_swallowed() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);

        block_on(store.update_op_heads(&[], &root_op_id())).unwrap();

        assert_eq!(heads_of(&store), HashSet::from([root_op_id()]));
        assert_eq!(head_files(temp.path()), HashSet::from(["0".repeat(64)]));
    }

    #[test]
    fn update_op_heads_adds_the_new_head_before_removing_old_ones() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        block_on(store.update_op_heads(&[], &op_id(1))).unwrap();

        block_on(store.update_op_heads(&[op_id(1)], &op_id(2))).unwrap();
        assert_eq!(heads_of(&store), HashSet::from([op_id(2)]));

        // The ordering property itself: with the old head already gone the
        // removal is a no-op, and the add still happened, so the repository is
        // never left with zero heads.
        fs::remove_file(temp.path().join(HEADS_DIR).join(op_id(2).hex())).unwrap();
        fs::write(temp.path().join(HEADS_DIR).join(op_id(5).hex()), "").unwrap();
        block_on(store.update_op_heads(&[op_id(2)], &op_id(6))).unwrap();
        assert_eq!(heads_of(&store), HashSet::from([op_id(5), op_id(6)]));
    }

    #[test]
    fn removing_a_head_that_is_already_gone_succeeds() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        block_on(store.update_op_heads(&[op_id(9)], &op_id(1))).unwrap();
        assert_eq!(heads_of(&store), HashSet::from([op_id(1)]));
    }

    /// Two independent stores over one directory, committing at the same time.
    /// Both heads must survive: this is the S14(a) property in-process, and it
    /// is what the pre-Stage-7 whole-file rewrite of `vex-local-heads` could
    /// not provide, because the second writer replaced the first writer's file.
    #[test]
    fn two_stores_on_one_dir_each_committing_leave_both_heads() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().to_path_buf();
        let seed = store_at(&dir, false);
        block_on(seed.update_op_heads(&[], &root_op_id())).unwrap();

        let handles: Vec<_> = [op_id(1), op_id(2)]
            .into_iter()
            .map(|id| {
                let dir = dir.clone();
                std::thread::spawn(move || {
                    let store = store_over(&dir, false);
                    block_on(store.update_op_heads(&[root_op_id()], &id)).unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        let heads = heads_of(&seed);
        assert!(heads.contains(&op_id(1)), "{heads:?}");
        assert!(heads.contains(&op_id(2)), "{heads:?}");
    }

    #[test]
    fn lock_is_held_and_excludes_a_second_locker() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().to_path_buf();
        let store = store_at(&dir, false);
        let held = block_on(store.lock()).unwrap();

        let (sender, receiver) = std::sync::mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let other = store_over(&dir, false);
            let lock = block_on(other.lock()).unwrap();
            let _ = sender.send(());
            drop(lock);
        });
        assert!(
            receiver
                .recv_timeout(std::time::Duration::from_millis(300))
                .is_err(),
            "a second locker must not get in while the first holds the lock"
        );

        drop(held);
        receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the lock is released on drop");
        waiter.join().unwrap();
    }

    /// `local_writes` is no longer a storage scheme. Keeping a second
    /// authoritative head store behind a per-process boolean — the same
    /// `.jj/repo` can be opened with it both set and unset — is a head-loss
    /// vector by construction.
    #[test]
    fn local_writes_uses_the_same_heads_directory_as_every_other_mode() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), true);
        block_on(store.update_op_heads(&[], &op_id(3))).unwrap();

        assert_eq!(head_files(temp.path()), HashSet::from([op_id(3).hex()]));
        assert!(
            !temp
                .path()
                .join(crate::vex_publish::LOCAL_HEADS_FILE)
                .exists(),
            "the flat local-heads file is never written again"
        );
        // And the same directory is what a non-local_writes store reads.
        let other = store_over(temp.path(), false);
        assert_eq!(heads_of(&other), HashSet::from([op_id(3)]));
    }

    #[test]
    fn bootstrap_seeds_every_id_from_vex_local_heads_without_a_client() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        fs::write(
            temp.path().join(crate::vex_publish::LOCAL_HEADS_FILE),
            format!("{}\n{}\n", content_id(1), content_id(2)),
        )
        .unwrap();
        let store = store_over(temp.path(), false);

        assert_eq!(heads_of(&store), HashSet::from([op_id(1), op_id(2)]));
        assert!(!temp.path().join(HEADS_TMP_DIR).exists());
    }

    #[test]
    fn bootstrap_seeds_all_server_head_ids_not_just_one() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        // A v1 marker, as a pre-Stage-7 client left it, recording a divergent
        // repository. Narrowing this to one id is the silent head loss.
        fs::write(
            temp.path().join(crate::vex_publish::SERVER_HEADS_FILE),
            format!(
                r#"{{"v":1,"heads":["{}","{}","{}"],"updated_unix":1}}"#,
                content_id(4),
                content_id(5),
                content_id(6)
            ),
        )
        .unwrap();
        let store = store_over(temp.path(), false);

        assert_eq!(
            heads_of(&store),
            HashSet::from([op_id(4), op_id(5), op_id(6)])
        );
    }

    /// The seed is a union of local sources, and it is complete before any RPC
    /// is considered — step (d) only runs when nothing local was found. The
    /// unroutable endpoint proves it: a bootstrap that consulted the backend
    /// here would have to fail or stall.
    #[test]
    fn bootstrap_prefers_local_heads_over_any_rpc() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        fs::write(
            temp.path().join(crate::vex_publish::LOCAL_HEADS_FILE),
            format!("{}\n", content_id(1)),
        )
        .unwrap();
        fs::write(
            temp.path().join(crate::vex_publish::PENDING_PUBLISH_FILE),
            format!(
                r#"{{"v":2,"base_heads":["{}"],"ops":[{{"op":"{}"}},{{"op":"{}"}}]}}"#,
                content_id(1),
                content_id(7),
                content_id(8)
            ),
        )
        .unwrap();
        let store = store_over(temp.path(), false);

        let started = std::time::Instant::now();
        let heads = heads_of(&store);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the bootstrap must not have gone near the network"
        );
        assert_eq!(
            heads,
            HashSet::from([op_id(1), op_id(8)]),
            "the local head and the queue tip, unioned"
        );
    }

    /// The permanent guard. Once `heads/` exists no later load may consult the
    /// backend or the legacy markers again — the directory is the whole truth.
    #[test]
    fn bootstrap_never_runs_twice_once_heads_dir_exists() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        fs::write(
            temp.path().join(crate::vex_publish::LOCAL_HEADS_FILE),
            format!("{}\n", content_id(1)),
        )
        .unwrap();
        let first = store_over(temp.path(), false);
        assert_eq!(heads_of(&first), HashSet::from([op_id(1)]));

        // The repository moves on, and the stale marker is deliberately left
        // behind (Stage 9 deletes the marker family).
        block_on(first.update_op_heads(&[op_id(1)], &op_id(2))).unwrap();

        // A brand new store, with no in-process memo, must serve the directory
        // and must not re-seed from the marker.
        let second = store_over(temp.path(), false);
        assert_eq!(heads_of(&second), HashSet::from([op_id(2)]));
        assert!(
            temp.path()
                .join(crate::vex_publish::LOCAL_HEADS_FILE)
                .exists(),
            "the legacy marker is left alone, and ignored"
        );
    }

    #[test]
    fn bootstrap_with_no_local_source_and_no_server_errors_rather_than_seeding_empty() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_over(temp.path(), true);

        let error = block_on(store.get_op_heads()).unwrap_err();
        assert!(
            error.to_string().contains("operation")
                || format!("{error:?}").contains("vex clone"),
            "{error:?}"
        );
        assert!(
            !temp.path().join(HEADS_DIR).exists(),
            "an empty heads directory would tell jj the repo has no operations"
        );
    }

    #[test]
    fn an_interrupted_bootstrap_leaves_no_heads_dir_and_reruns() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp.path()).unwrap();
        // A crash between "staged the seed" and "renamed it into place".
        let staging = temp.path().join(HEADS_TMP_DIR);
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join(op_id(9).hex()), "").unwrap();
        assert!(!temp.path().join(HEADS_DIR).exists());

        fs::write(
            temp.path().join(crate::vex_publish::LOCAL_HEADS_FILE),
            format!("{}\n", content_id(1)),
        )
        .unwrap();
        let store = store_over(temp.path(), false);

        assert_eq!(
            heads_of(&store),
            HashSet::from([op_id(1)]),
            "the re-run seeds from the sources, not from the abandoned staging dir"
        );
        assert!(!staging.exists());
    }

    /// S1, counter-asserted: an ordinary read/write session touches the client
    /// zero times. Any RPC against this endpoint would fail, so completing at
    /// all is the proof.
    #[test]
    fn get_op_heads_makes_zero_backend_calls() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        let mut parent = root_op_id();
        block_on(store.update_op_heads(&[], &parent)).unwrap();
        for byte in 1..8 {
            let next = op_id(byte);
            block_on(store.update_op_heads(std::slice::from_ref(&parent), &next)).unwrap();
            assert_eq!(heads_of(&store), HashSet::from([next.clone()]));
            parent = next;
        }
    }

    /// The escape is off by default, and when on it can only ever add a server
    /// round trip — it can never turn a durable local write into a failure.
    #[test]
    fn publish_op_log_escape_publishes_but_never_fails_the_write() {
        assert!(
            !publish_op_log_escape(),
            "the escape must be opt-in; the fleet default is local-only"
        );

        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        // `publish = true` against an unroutable endpoint: the publish attempt
        // is made and fails, and the write still succeeds.
        block_on(store.update_op_heads_publishing(&[], &op_id(1), true)).unwrap();
        block_on(store.update_op_heads_publishing(&[op_id(1)], &op_id(2), true)).unwrap();
        assert_eq!(heads_of(&store), HashSet::from([op_id(2)]));
    }

    /// The marker family is read for the bootstrap and never written again.
    #[test]
    fn a_v1_server_heads_marker_is_readable_by_the_bootstrap() {
        let temp = tempfile::tempdir().unwrap();
        crate::vex_publish::write_server_heads(
            temp.path(),
            &ServerHeadsMarker::new(Some("token".to_string())),
        )
        .unwrap();
        // A v2 marker carries no heads, so it contributes nothing to a seed.
        let store = store_over(temp.path(), true);
        assert!(block_on(store.get_op_heads()).is_err());
    }
}
