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

#![expect(missing_docs)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::io::{BufReader, Seek, SeekFrom};
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::SystemTime;

use jj_backend_api::CloneBlobMode as ProtoCloneBlobMode;
use jj_backend_api::CloneViewKind as ProtoCloneViewKind;
use jj_backend_api::GetCloneManifestRequest;
use jj_backend_api::GetObjectRequest;
use jj_backend_api::GetObjectsInlineRequest;
use jj_backend_api::GetObjectsRequest;
use jj_backend_api::GetRepoRequest;
use jj_backend_api::InitRepoRequest;
use jj_backend_api::InlineObject;
use jj_backend_api::ObjectId;
use jj_backend_api::PutObjectRequest;
use jj_backend_api::PutObjectsRequest;
use jj_backend_api::ResolveOperationIdPrefixRequest;
use jj_backend_api::ResolveRefsRequest;
use jj_backend_api::VirtualRepositoryMount as ProtoVirtualRepositoryMount;
use jj_backend_api::jj_backend_client::JjBackendClient;
use jj_backend_types::{
    CloneManifest, ContentId, ObjectKind, ObjectPackEntry, SnapshotPackSet, decode_object_pack,
    decode_object_pack_reader, decode_object_pack_with_visitor,
};
use serde::Deserialize;
use serde::Serialize;
use tempfile::NamedTempFile;
use thiserror::Error;
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tracing::debug;

use crate::repo::StoreFactories;
use crate::vex_backend::VexBackend;
use crate::vex_op_heads_store::VexOpHeadsStore;
use crate::vex_op_store::VexOpStore;

pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:50051";

/// Set when the command's paged output has been closed by its reader (e.g. the
/// user quit the pager). In-flight blocking backend RPCs observe this and abort
/// promptly with a broken-pipe error so the process can exit instead of running
/// work nobody will read. Process-global because there is exactly one command
/// per process.
static OUTPUT_CLOSED: AtomicBool = AtomicBool::new(false);

/// Signal that paged output has been closed by the reader. Called by the CLI
/// pager watcher when the pager process (external) or pager thread (builtin)
/// goes away.
pub fn signal_output_closed() {
    OUTPUT_CLOSED.store(true, Ordering::SeqCst);
}

fn output_closed() -> bool {
    OUTPUT_CLOSED.load(Ordering::SeqCst)
}

/// Drive `fut` to completion, but bail out promptly with a broken-pipe error if
/// the paged output is closed while we're waiting. Polls the cancellation flag
/// on a short interval so a blocking RPC (including its retry backoff) unwinds
/// within ~100ms of the pager being quit, instead of leaving the process alive
/// until the request finishes.
async fn with_output_cancel<T, Fut>(fut: Fut) -> Result<T, VexClientError>
where
    Fut: std::future::Future<Output = Result<T, VexClientError>>,
{
    tokio::pin!(fut);
    loop {
        match tokio::time::timeout(Duration::from_millis(100), &mut fut).await {
            Ok(result) => return result,
            Err(_elapsed) => {
                if output_closed() {
                    return Err(VexClientError::Io(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "output closed before backend request completed (pager quit)",
                    )));
                }
            }
        }
    }
}

/// Sleep usable from a non-tokio (pollster-style) executor: the timer runs as
/// a task on the shared gRPC runtime and its `JoinHandle` is awaited — a
/// cooperative yield, mirroring `grpc_retry_async`. Used by the clone's
/// materializing-progress ticker, whose caller executor has no timer driver.
pub(crate) async fn shared_runtime_sleep(duration: Duration) {
    let handle = VexClient::shared_grpc_runtime().spawn(tokio::time::sleep(duration));
    drop(handle.await);
}

/// Max gRPC message size for both directions. The default tonic decode limit is
/// 4 MiB, which is smaller than legitimately large objects (e.g. a >4 MiB file
/// blob fetched inline via `GetObject` during checkout), so reads would fail
/// with "decoded message length too large". The server already allows 64 MiB
/// (`JJ_GRPC_MAX_MESSAGE_BYTES`); match it on the client for encode and decode.
const MAX_GRPC_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// A `PutObjects` batch is content-addressed and create-if-missing, so a
/// response lost during an edge reload may safely be retried verbatim. Keep the
/// retry window deliberately short: bulk import has many batches in flight and
/// should fail clearly rather than retaining all of their bodies indefinitely.
const PIPELINED_PUT_RETRY_ATTEMPTS: usize = 5;
const PIPELINED_PUT_RETRY_BASE_MS: u64 = 250;
const PIPELINED_PUT_RETRY_CAP_MS: u64 = 2_000;

/// `CommitOperation` has an explicit replay-success response for an already
/// published op head. That makes the exact maintenance rejection safe to
/// retry, but it can last much longer than a transport blip while a shadow GC
/// pass walks a large Git mirror.
const COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS: usize = 24;
const COMMIT_OPERATION_MAINTENANCE_RETRY_BASE_MS: u64 = 1_000;
const COMMIT_OPERATION_MAINTENANCE_RETRY_CAP_MS: u64 = 15_000;

/// Read a non-negative seconds value from `name`, falling back to `default` when
/// unset or unparseable. Used for env-tunable gRPC connection timeouts.
fn env_secs(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}

/// Whether `VEX_RPC_TIMING` is set — enables per-RPC wall-time logging to stderr
/// for latency attribution. Cached so the env lookup happens once.
fn rpc_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VEX_RPC_TIMING").is_ok())
}

/// RAII timer that prints a client RPC's wall time (≈ its round trip, since the
/// Vex client blocks on each call) to stderr on drop when `VEX_RPC_TIMING` is
/// set. Returns `None` (zero overhead) otherwise; the label closure only runs
/// when enabled.
struct RpcTimer {
    label: String,
    start: std::time::Instant,
}

impl RpcTimer {
    fn start(label: impl FnOnce() -> String) -> Option<Self> {
        rpc_timing_enabled().then(|| Self {
            label: label(),
            start: std::time::Instant::now(),
        })
    }
}

impl Drop for RpcTimer {
    fn drop(&mut self) {
        eprintln!(
            "[vex-rpc] {:>8.1}ms  {}",
            self.start.elapsed().as_secs_f64() * 1000.0,
            self.label
        );
    }
}

/// Process-global, always-on client transfer/cache counters. Cheap relaxed
/// atomics incremented on the hot paths (per-object reads, pack transfer,
/// hydration, checkout writes); `vex bench clone` resets them before a run and
/// snapshots them after to attribute where a clone's wall-clock went.
/// Process-global for the same reason as the other client state: one command
/// per process, many `VexClient` instances per repo.
#[derive(Debug, Default)]
pub struct VexClientStats {
    /// `GetObject` RPCs issued for blob objects.
    pub get_object_rpcs_blob: AtomicU64,
    /// `GetObject` RPCs issued for tree objects.
    pub get_object_rpcs_tree: AtomicU64,
    /// `GetObject` RPCs issued for commit objects.
    pub get_object_rpcs_commit: AtomicU64,
    /// `GetObject` RPCs issued for JJ operation objects.
    pub get_object_rpcs_op: AtomicU64,
    /// `GetObject` RPCs issued for JJ view objects.
    pub get_object_rpcs_view: AtomicU64,
    /// `GetObject` RPCs issued for all other object kinds.
    pub get_object_rpcs_other: AtomicU64,
    /// Object reads served from the local cache (or the pending-upload buffer).
    pub get_object_cache_hits: AtomicU64,
    /// Cached blob reads.
    pub get_object_cache_hits_blob: AtomicU64,
    /// Cached tree reads.
    pub get_object_cache_hits_tree: AtomicU64,
    /// Cached commit reads.
    pub get_object_cache_hits_commit: AtomicU64,
    /// Cached JJ operation reads.
    pub get_object_cache_hits_op: AtomicU64,
    /// Cached JJ view reads.
    pub get_object_cache_hits_view: AtomicU64,
    /// Cached reads for all other object kinds.
    pub get_object_cache_hits_other: AtomicU64,
    /// Objects received (and verified) via `GetObjectsInline` batch responses.
    pub objects_inline_fetched: AtomicU64,
    /// `GetObjectsInline` batch RPCs issued.
    pub inline_batches: AtomicU64,
    /// Clone packs fetched and unpacked into the local cache.
    pub packs_fetched: AtomicU64,
    /// Pack chunks fetched (presigned HTTP or gRPC fallback).
    pub pack_chunks_fetched: AtomicU64,
    /// Encoded pack bytes transferred.
    pub pack_bytes_fetched: AtomicU64,
    /// Successful direct HTTP object-store fetches (pack chunks and whole
    /// packs), bypassing the gRPC relay. Named for the common case —
    /// SigV4-presigned hint URLs — but a deployment serving hints via the
    /// unauthenticated `JJ_OBJECT_BASE_URL` route counts here too (the client
    /// cannot tell the flavors apart); with such a deployment a
    /// `JJ_PRESIGN_GET_TTL_SECS=0` rollback will NOT drive this to zero.
    /// Only fetches whose result is actually consumed are counted (see
    /// [`VexClient::http_get_async`]).
    pub presigned_fetches: AtomicU64,
    /// Bytes fetched via direct HTTP (see [`Self::presigned_fetches`] for
    /// exactly what "presigned" covers).
    pub presigned_bytes: AtomicU64,
    /// Snapshot packs fetched (roadmap/032 snapshot-pack consumption).
    pub snapshot_packs_fetched: AtomicU64,
    /// Encoded snapshot pack bytes transferred.
    pub snapshot_pack_bytes: AtomicU64,
    /// Objects unpacked from packs into the local cache.
    pub objects_unpacked: AtomicU64,
    /// Objects unpacked pack-resident — indexed into a `.packs` payload file
    /// instead of exploded into a loose cache file (roadmap/032 follow-up).
    pub objects_pack_resident: AtomicU64,
    /// Loose cache file creations avoided by the pack-resident unpack.
    pub loose_writes_avoided: AtomicU64,
    /// Objects hydrated pre-checkout via [`VexClient::get_objects_inline_batched`].
    pub hydrated_objects: AtomicU64,
    /// Bytes hydrated pre-checkout.
    pub hydrated_bytes: AtomicU64,
    /// Working-copy files written during checkout.
    pub files_written: AtomicU64,
    /// Working-copy bytes written during checkout.
    pub bytes_written: AtomicU64,
    /// Working-copy files materialized via reflink/clonefile instead of a copy.
    pub files_reflinked: AtomicU64,
    /// Pre-checkout hydration tree walks skipped because a fully-unpacked
    /// snapshot pack chain already covered the start commit (roadmap/032).
    pub snapshot_walk_skips: AtomicU64,
    /// Successful native trunk selections: the server-advertised default
    /// branch resolved through native local/remote-tracking bookmark state
    /// during `vex clone` (roadmap/066).
    pub native_trunk_resolutions: AtomicU64,
    /// Advertised native trunk bookmarks absent from native bookmark state.
    /// Native clone fails closed instead of falling back to `git/ref/*`
    /// (roadmap/066).
    pub native_trunk_missing: AtomicU64,
    /// Commits whose bytes failed native protobuf decoding and were parsed as
    /// raw Git commits. Incremented only under explicit
    /// [`VexObjectReadMode::GitCompatibility`]; any non-zero value in a native
    /// clone is a correctness regression, not a performance problem.
    pub git_compat_commit_decodes: AtomicU64,
    /// Trees parsed as raw Git trees (explicit compatibility mode only; see
    /// [`Self::git_compat_commit_decodes`]).
    pub git_compat_tree_decodes: AtomicU64,
    /// Raw Git SHA-1 names resolved through `git/object/sha1/*` mapping
    /// lookups (explicit compatibility mode only).
    pub git_mapping_names_resolved: AtomicU64,
    /// `ResolveRefs` RPCs issued for `git/object/sha1/*` mapping lookups.
    pub git_mapping_rpcs: AtomicU64,
    /// Wall-clock milliseconds spent in `git/object/sha1/*` mapping RPCs.
    pub git_mapping_elapsed_ms: AtomicU64,
}

macro_rules! for_each_vex_client_stat {
    ($macro:ident) => {
        $macro!(
            get_object_rpcs_blob,
            get_object_rpcs_tree,
            get_object_rpcs_commit,
            get_object_rpcs_op,
            get_object_rpcs_view,
            get_object_rpcs_other,
            get_object_cache_hits,
            get_object_cache_hits_blob,
            get_object_cache_hits_tree,
            get_object_cache_hits_commit,
            get_object_cache_hits_op,
            get_object_cache_hits_view,
            get_object_cache_hits_other,
            objects_inline_fetched,
            inline_batches,
            packs_fetched,
            pack_chunks_fetched,
            pack_bytes_fetched,
            presigned_fetches,
            presigned_bytes,
            snapshot_packs_fetched,
            snapshot_pack_bytes,
            objects_unpacked,
            objects_pack_resident,
            loose_writes_avoided,
            hydrated_objects,
            hydrated_bytes,
            files_written,
            bytes_written,
            files_reflinked,
            snapshot_walk_skips,
            native_trunk_resolutions,
            native_trunk_missing,
            git_compat_commit_decodes,
            git_compat_tree_decodes,
            git_mapping_names_resolved,
            git_mapping_rpcs,
            git_mapping_elapsed_ms
        )
    };
}

/// Plain-value copy of [`VexClientStats`] taken at one point in time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct VexClientStatsSnapshot {
    pub get_object_rpcs_blob: u64,
    pub get_object_rpcs_tree: u64,
    pub get_object_rpcs_commit: u64,
    pub get_object_rpcs_op: u64,
    pub get_object_rpcs_view: u64,
    pub get_object_rpcs_other: u64,
    pub get_object_cache_hits: u64,
    pub get_object_cache_hits_blob: u64,
    pub get_object_cache_hits_tree: u64,
    pub get_object_cache_hits_commit: u64,
    pub get_object_cache_hits_op: u64,
    pub get_object_cache_hits_view: u64,
    pub get_object_cache_hits_other: u64,
    pub objects_inline_fetched: u64,
    pub inline_batches: u64,
    pub packs_fetched: u64,
    pub pack_chunks_fetched: u64,
    pub pack_bytes_fetched: u64,
    pub presigned_fetches: u64,
    pub presigned_bytes: u64,
    pub snapshot_packs_fetched: u64,
    pub snapshot_pack_bytes: u64,
    pub objects_unpacked: u64,
    pub objects_pack_resident: u64,
    pub loose_writes_avoided: u64,
    pub hydrated_objects: u64,
    pub hydrated_bytes: u64,
    pub files_written: u64,
    pub bytes_written: u64,
    pub files_reflinked: u64,
    pub snapshot_walk_skips: u64,
    pub native_trunk_resolutions: u64,
    pub native_trunk_missing: u64,
    pub git_compat_commit_decodes: u64,
    pub git_compat_tree_decodes: u64,
    pub git_mapping_names_resolved: u64,
    pub git_mapping_rpcs: u64,
    pub git_mapping_elapsed_ms: u64,
}

impl VexClientStats {
    fn snapshot(&self) -> VexClientStatsSnapshot {
        macro_rules! load_fields {
            ($($field:ident),*) => {
                VexClientStatsSnapshot {
                    $($field: self.$field.load(Ordering::Relaxed),)*
                }
            };
        }
        for_each_vex_client_stat!(load_fields)
    }

    fn reset(&self) {
        macro_rules! reset_fields {
            ($($field:ident),*) => {
                { $(self.$field.store(0, Ordering::Relaxed);)* }
            };
        }
        for_each_vex_client_stat!(reset_fields)
    }

    fn record_get_object_rpc(&self, kind: ObjectKind) {
        let counter = match kind {
            ObjectKind::Blob => &self.get_object_rpcs_blob,
            ObjectKind::Tree => &self.get_object_rpcs_tree,
            ObjectKind::Commit => &self.get_object_rpcs_commit,
            ObjectKind::Op => &self.get_object_rpcs_op,
            ObjectKind::View => &self.get_object_rpcs_view,
            _ => &self.get_object_rpcs_other,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_get_object_cache_hit(&self, kind: ObjectKind) {
        self.get_object_cache_hits.fetch_add(1, Ordering::Relaxed);
        let counter = match kind {
            ObjectKind::Blob => &self.get_object_cache_hits_blob,
            ObjectKind::Tree => &self.get_object_cache_hits_tree,
            ObjectKind::Commit => &self.get_object_cache_hits_commit,
            ObjectKind::Op => &self.get_object_cache_hits_op,
            ObjectKind::View => &self.get_object_cache_hits_view,
            _ => &self.get_object_cache_hits_other,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// A server-advertised trunk bookmark resolved through native bookmark
    /// state. Called by the clone target selector in `workspace.rs`.
    pub fn record_native_trunk_resolution(&self) {
        self.native_trunk_resolutions
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A server-advertised trunk bookmark was absent from native bookmark
    /// state (native clone fails closed). Called by the clone target selector
    /// in `workspace.rs`.
    pub fn record_native_trunk_missing(&self) {
        self.native_trunk_missing.fetch_add(1, Ordering::Relaxed);
    }

    /// A commit's bytes were parsed as a raw Git commit under explicit
    /// [`VexObjectReadMode::GitCompatibility`].
    pub fn record_git_compat_commit_decode(&self) {
        self.git_compat_commit_decodes
            .fetch_add(1, Ordering::Relaxed);
    }

    /// A tree's bytes were parsed as a raw Git tree under explicit
    /// [`VexObjectReadMode::GitCompatibility`].
    pub fn record_git_compat_tree_decode(&self) {
        self.git_compat_tree_decodes.fetch_add(1, Ordering::Relaxed);
    }

    /// One `git/object/sha1/*` mapping `ResolveRefs` RPC covering
    /// `names_resolved` SHA-1 names and taking `elapsed` wall-clock time.
    pub fn record_git_mapping_rpc(&self, names_resolved: u64, elapsed: Duration) {
        self.git_mapping_rpcs.fetch_add(1, Ordering::Relaxed);
        self.git_mapping_names_resolved
            .fetch_add(names_resolved, Ordering::Relaxed);
        self.git_mapping_elapsed_ms
            .fetch_add(elapsed.as_millis() as u64, Ordering::Relaxed);
    }
}

/// The process-global [`VexClientStats`] counters.
pub fn vex_client_stats() -> &'static VexClientStats {
    static STATS: OnceLock<VexClientStats> = OnceLock::new();
    STATS.get_or_init(VexClientStats::default)
}

/// Snapshot the process-global client counters.
pub fn vex_client_stats_snapshot() -> VexClientStatsSnapshot {
    vex_client_stats().snapshot()
}

/// Reset the process-global client counters to zero (bench runs only).
pub fn vex_client_stats_reset() {
    vex_client_stats().reset();
}

/// Serializes tests (across `jj-lib` modules) that assert on the
/// process-global [`VexClientStats`] counters, so a concurrent
/// [`vex_client_stats_reset`] or counter bump in a parallel test thread
/// cannot corrupt another test's delta assertions.
#[cfg(test)]
pub(crate) fn test_stats_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(Mutex::default)
}

pub use jj_backend_types::CloneBlobMode;

/// Progress events emitted while a Vex clone runs.
///
/// These are reported through the optional [`CloneProgressFn`] passed to
/// [`crate::workspace::Workspace::clone_vex`] so a caller (e.g. the CLI) can
/// render a live progress UI. They are advisory only: the clone behaves
/// identically whether or not a sink is provided.
#[derive(Debug, Clone)]
pub enum CloneProgress {
    /// Contacting the backend and resolving repo metadata.
    Connecting,
    /// The server is building the clone manifest (cold cache); emitted
    /// repeatedly while the client polls so a slow first clone shows *why* it is
    /// waiting instead of an opaque 0%.
    ManifestBuilding {
        /// Seconds spent waiting for the manifest so far.
        waited_secs: u64,
    },
    /// A transient backend error occurred and the client is retrying. Surfaced
    /// so a stuck/slow clone shows what is going wrong.
    Retrying {
        /// The operation being retried (e.g. `"clone manifest"`).
        operation: String,
        /// A short description of the error.
        message: String,
    },
    /// The clone manifest has been fetched; totals are now known.
    ManifestReady {
        /// Number of packs to prefetch.
        packs: u64,
        /// Number of immutable objects bundled inside those packs.
        pack_objects: u64,
        /// Number of loose (non-packed) objects to prefetch.
        loose_objects: u64,
        /// Approximate total bytes to transfer for the prefetch step.
        total_bytes: u64,
        /// Objects deferred for on-demand (lazy / shallow) hydration.
        deferred_objects: u64,
    },
    /// A pack finished downloading and unpacking.
    PackFetched {
        /// Packs completed so far.
        done: u64,
        /// Total packs in the manifest.
        total: u64,
        /// Cumulative immutable objects written to the local cache.
        objects: u64,
    },
    /// A loose object finished downloading.
    LooseObjectFetched {
        /// Loose objects completed so far.
        done: u64,
        /// Total loose objects in the manifest.
        total: u64,
    },
    /// File/symlink contents for the start commit are being bulk-hydrated into
    /// the local cache before checkout (lazy clones), so materialization does
    /// not pay one RPC per file.
    Hydrating {
        /// Objects hydrated so far.
        done: u64,
        /// Total objects to hydrate.
        total: u64,
    },
    /// The local JJ repository is being opened after metadata transfer.
    LoadingRepo,
    /// The repository's operation/view graph and default commit index are being
    /// loaded or built.
    Indexing,
    /// The new workspace operation is being published before checkout.
    WorkspacePublish,
    /// Prefetch finished; the working copy is about to be materialized.
    CheckingOut,
    /// Working-copy files are being written to disk during checkout.
    Materializing {
        /// Files written so far.
        files_done: u64,
        /// Total files to write.
        files_total: u64,
    },
    /// The clone selected this existing upstream bookmark as its trunk.
    TrunkResolved {
        /// Bookmark name that should back the repo-local `trunk()` alias.
        name: String,
    },
    /// The clone is complete.
    Done,
}

/// Sink for [`CloneProgress`] events. `Send + Sync` so it can be invoked from
/// the blocking gRPC worker as well as the dedicated clone thread.
pub type CloneProgressFn = dyn Fn(CloneProgress) + Send + Sync;

#[derive(Debug, Error)]
pub enum VexConfigError {
    #[error("vex repo metadata file not found at {0}")]
    MissingMetadata(PathBuf),
    #[error("vex repo metadata path has no repo parent: {0}")]
    InvalidStorePath(PathBuf),
    #[error("vex repo metadata IO")]
    Io(#[from] std::io::Error),
    #[error("vex repo metadata JSON")]
    Json(#[from] serde_json::Error),
    #[error("invalid Vex endpoint `{endpoint}`: {message}")]
    InvalidEndpoint { endpoint: String, message: String },
    #[error("backend did not return repo information")]
    MissingRepoInfo,
}

#[derive(Debug, Error)]
pub enum VexClientError {
    #[error(transparent)]
    Config(#[from] VexConfigError),
    #[error("cache IO")]
    Io(#[from] std::io::Error),
    #[error("failed to start grpc runtime")]
    Runtime(#[source] std::io::Error),
    #[error(transparent)]
    Transport(#[from] tonic::transport::Error),
    #[error(transparent)]
    Status(#[from] tonic::Status),
    #[error("invalid grpc authorization metadata: {0}")]
    InvalidAuthorizationMetadata(String),
    #[error("invalid binary pack: {0}")]
    PackDecode(String),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error("ref update rejected: {0}")]
    RefUpdateRejected(String),
}

/// Object decode policy for the Vex backend read path (roadmap/066).
///
/// `vex clone` and every ordinary repository load are native-only: commit and
/// tree bytes must decode as native Vex protobuf objects, and a decode failure
/// is a typed read error (see `vex_backend::VexNativeObjectFormatError`) —
/// never a signal to parse raw Git bytes or resolve `git/object/sha1/*`
/// mappings. Only explicit conversion/Git-bridge callers may construct
/// [`GitCompatibility`](Self::GitCompatibility), and they do so in memory: the
/// mode is never persisted to `vex.json`, so a normal clone can never inherit
/// compatibility mode from disk.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VexObjectReadMode {
    /// Commit/tree bytes must be native Vex protobuf objects (the default).
    #[default]
    NativeOnly,
    /// Failed protobuf decodes fall back to the raw Git parsers
    /// (`parse_git_commit()` / `read_git_tree()`) and their
    /// `git/object/sha1/*` mapping lookups. Explicit opt-in only.
    GitCompatibility,
}

impl VexObjectReadMode {
    /// Whether raw Git commit/tree parsing (and its SHA-1 mapping lookups) is
    /// permitted on this read path.
    pub fn allows_git_compatibility(self) -> bool {
        matches!(self, Self::GitCompatibility)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VexRepoConfig {
    pub endpoint: String,
    pub tenant_id: String,
    pub tenant_slug: String,
    pub repo_id: String,
    pub repo_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository_scope_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub virtual_repository_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backing_repo_slug: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub virtual_root_path: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub virtual_mounts: Vec<VexVirtualRepositoryMount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub access_token: Option<String>,
    /// When true, `put_object` writes objects only to the local content-addressed
    /// cache and never issues a gRPC `PutObject` to the backend. Used by the
    /// READ_ONLY ephemeral CI runner so cloning a workspace (which creates an
    /// editable `@` working-copy commit + op-log) does not require Write access to
    /// the backend. Opt-in only; defaults to false so normal clones/commits/pushes
    /// continue to persist to the backend.
    #[serde(default)]
    pub local_writes: bool,
    /// Object decode policy for backend reads (see [`VexObjectReadMode`]).
    /// Never serialized: a normal clone's `vex.json` carries no mode field and
    /// old files without one deserialize to [`VexObjectReadMode::NativeOnly`],
    /// so compatibility mode can only be constructed explicitly in memory by a
    /// conversion/Git-bridge caller (or spelled out by hand in a test
    /// fixture), never inherited from a normal clone.
    #[serde(default, skip_serializing)]
    pub object_read_mode: VexObjectReadMode,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VexVirtualRepositoryMount {
    pub slug: String,
    pub root_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_bookmark: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_branch: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_source_commit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projected_virtual_commit_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_remote_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_provider_kind: Option<String>,
}

impl VexRepoConfig {
    pub fn metadata_path_for_repo(repo_path: &Path) -> PathBuf {
        repo_path.join("vex.json")
    }

    pub fn metadata_path_for_store(store_path: &Path) -> Result<PathBuf, VexConfigError> {
        let repo_path = store_path
            .parent()
            .ok_or_else(|| VexConfigError::InvalidStorePath(store_path.to_path_buf()))?;
        Ok(Self::metadata_path_for_repo(repo_path))
    }

    pub fn load_from_store_path(store_path: &Path) -> Result<Self, VexConfigError> {
        let path = Self::metadata_path_for_store(store_path)?;
        Self::load_from_repo_path(path.parent().unwrap())
    }

    pub fn load_from_repo_path(repo_path: &Path) -> Result<Self, VexConfigError> {
        let path = Self::metadata_path_for_repo(repo_path);
        if !path.exists() {
            return Err(VexConfigError::MissingMetadata(path));
        }
        let text = fs::read_to_string(&path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn write_to_repo_path(&self, repo_path: &Path) -> Result<(), VexConfigError> {
        let path = Self::metadata_path_for_repo(repo_path);
        let parent = path
            .parent()
            .ok_or_else(|| VexConfigError::InvalidStorePath(path.clone()))?;
        let contents = serde_json::to_vec_pretty(self)?;
        let mut temporary = NamedTempFile::new_in(parent)?;
        temporary.write_all(&contents)?;
        temporary.persist(&path).map_err(|error| error.error)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct VexClient {
    config: VexRepoConfig,
    cache_root: Option<PathBuf>,
    cache_max_bytes: Option<u64>,
    /// Mirror of `config.local_writes`. When true, `put_object` short-circuits to
    /// the local cache instead of issuing a gRPC `PutObject` (READ_ONLY CI runner).
    local_writes: bool,
    /// The cache dir was created by this clone process (and is removed with the
    /// `.jj` scaffold on failure), so unpack loose writes may skip the
    /// temp+rename atomicity dance. See [`Self::mark_fresh_clone_cache`].
    fresh_cache: bool,
    /// Test override for [`pack_resident_cache_enabled`] (`None` = env). Tests
    /// pin it directly instead of mutating the process environment (`jj-lib`
    /// forbids `unsafe`, which `set_var` now requires).
    pack_resident_override: Option<bool>,
    /// Tripped by the first presigned HTTP 403 (see
    /// [`Self::fetch_pack_chunk_with_retry`]). Object-fetch hints are minted
    /// once per prefetch run, so one expired/invalid signature means every
    /// remaining hint fails the same way; once set, direct HTTP fetches are
    /// skipped and transfers go straight to the gRPC fallback. Shared
    /// (`Arc`) across clones of this client, per-client so tests stay
    /// isolated.
    presigned_get_disabled: Arc<AtomicBool>,
}

#[derive(Debug)]
struct CacheEntry {
    path: PathBuf,
    modified: SystemTime,
    size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct PackTransferState {
    pack_content_id: String,
    chunk_count: usize,
    next_chunk_index: usize,
}

fn shared_cache_root(config: &VexRepoConfig) -> Option<PathBuf> {
    std::env::var_os("JJ_VEX_SHARED_CACHE_DIR")
        .map(PathBuf::from)
        .map(|root| root.join(&config.tenant_id).join(&config.repo_id))
}

fn cache_max_bytes() -> Option<u64> {
    std::env::var("JJ_VEX_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
}

fn proto_clone_view_kind(scope: Option<&str>) -> ProtoCloneViewKind {
    match scope {
        Some("virtual_repository") | Some("virtual") => ProtoCloneViewKind::Virtual,
        Some("composed") => ProtoCloneViewKind::Composed,
        Some("repository") | Some("physical") | None => ProtoCloneViewKind::Physical,
        Some(_) => ProtoCloneViewKind::Physical,
    }
}

fn proto_virtual_repository_mount(
    mount: &VexVirtualRepositoryMount,
) -> ProtoVirtualRepositoryMount {
    ProtoVirtualRepositoryMount {
        slug: mount.slug.clone(),
        root_path: mount.root_path.clone(),
        source_bookmark: mount.source_bookmark.clone().unwrap_or_default(),
        target_branch: mount.target_branch.clone().unwrap_or_default(),
        projection_status: mount.projection_status.clone().unwrap_or_default(),
        projected_source_commit_id: mount.projected_source_commit_id.clone().unwrap_or_default(),
        projected_virtual_commit_id: mount
            .projected_virtual_commit_id
            .clone()
            .unwrap_or_default(),
        sync_remote_url: mount.sync_remote_url.clone().unwrap_or_default(),
        sync_provider_kind: mount.sync_provider_kind.clone().unwrap_or_default(),
    }
}

/// Max buffered upload bytes before an inline flush during a snapshot. Bounds
/// peak memory for very large snapshots while still letting a normal snapshot
/// (a handful of small objects) coalesce into a single batched upload.
const PENDING_FLUSH_BYTES: usize = 32 * 1024 * 1024;
/// Companion object-count cap to [`PENDING_FLUSH_BYTES`].
const PENDING_FLUSH_OBJECTS: usize = 256;

/// Max objects per `GetObjectsInline` batch (read-side analogue of
/// [`PENDING_FLUSH_OBJECTS`]).
const INLINE_FETCH_BATCH_OBJECTS: usize = 256;
/// Estimated-bytes cap per `GetObjectsInline` batch. The *response* carries the
/// object bodies and must stay under [`MAX_GRPC_MESSAGE_BYTES`]; sizes are only
/// hints (tree entries don't record them), so leave generous headroom.
const INLINE_FETCH_BATCH_BYTES: u64 = 24 * 1024 * 1024;
/// Concurrent in-flight `GetObjectsInline` batches.
const INLINE_FETCH_CONCURRENCY: usize = 8;

/// Default number of clone packs fetched+unpacked in parallel during
/// [`VexClient::prefetch_clone_manifest`]. Overridable via
/// `VEX_CLONE_PACK_CONCURRENCY` (set `1` to restore the serial pack loop).
/// Default tuned by a 2026-07-22 production sweep (271.4 MB pinned JJ
/// fixture): pack transfer means were 24.0 s at 4×8, 7.1 s at 16×32, and only
/// 6.8 s at 32×64, so throughput flattens at 16×32 (~38 MB/s) while 64 pack
/// workers added writer-contention variance.
const PACK_FETCH_CONCURRENCY: usize = 16;

fn pack_fetch_concurrency() -> usize {
    std::env::var("VEX_CLONE_PACK_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 1)
        .unwrap_or(PACK_FETCH_CONCURRENCY)
}

/// Default number of chunk fetches in flight within one pack transfer. The
/// fetches are reorder-buffered (`.buffered(W)` yields results in input
/// order), so the single writer still appends to the `.part` file strictly in
/// chunk order. Peak buffered memory is bounded at pack workers × W × the
/// 512KiB chunk size (~256 MB worst case at the 16×32 defaults). The same
/// 2026-07-22 sweep showed 8×64 regressing ~35% from head-of-line blocking in
/// the index-ordered `.part` writer, so keep W moderate. Overridable via
/// `VEX_CLONE_CHUNK_CONCURRENCY` (set `1` to restore the serial chunk loop).
const CHUNK_FETCH_CONCURRENCY: usize = 32;

/// Effective chunk-fetch concurrency (env `VEX_CLONE_CHUNK_CONCURRENCY`,
/// default [`CHUNK_FETCH_CONCURRENCY`]). Public so `vex bench clone` can
/// record it alongside the transfer counters.
pub fn clone_chunk_concurrency() -> usize {
    std::env::var("VEX_CLONE_CHUNK_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 1)
        .unwrap_or(CHUNK_FETCH_CONCURRENCY)
}

/// Persist the pack transfer state every this many appended chunks (plus once
/// at the end and on error) instead of per chunk — the per-chunk JSON rewrite
/// was one extra file write per 512KiB on the clone critical path. A kill
/// between saves leaves the `.part` ahead of the recorded state; resume
/// truncates it back to the recorded contiguous prefix (see
/// [`VexClient::prefetch_pack_via_chunks`]), so at most this many chunks are
/// refetched.
const TRANSFER_STATE_SAVE_INTERVAL: usize = 8;

/// Whether the client consumes precomputed snapshot packs from the clone
/// manifest (roadmap/032). On by default; `VEX_CLONE_SNAPSHOT_PACKS=0` (or
/// `false`/`no`) disables consumption (rollback / bench control).
fn snapshot_packs_client_enabled() -> bool {
    !matches!(
        std::env::var("VEX_CLONE_SNAPSHOT_PACKS").ok().as_deref(),
        Some("0") | Some("false") | Some("no")
    )
}

/// Whether unpacked metadata objects (commit/tree/op/view) are kept
/// pack-resident — one decompressed payload file plus a `(offset, len)`
/// sidecar index per pack under `<cache_root>/.packs/` — instead of exploded
/// into one loose cache file each (~126k of the ~129k files a prod clone used
/// to create, ~50% of the pack phase). Blobs and symlinks always unpack
/// loose: reflink materialization and checkout streaming need real per-object
/// files. On by default; `VEX_CACHE_PACK_RESIDENT=0` (or `false`/`no`)
/// restores the all-loose unpack exactly (kill switch — pack-resident reads
/// and writes are both disabled).
fn pack_resident_cache_enabled() -> bool {
    !matches!(
        std::env::var("VEX_CACHE_PACK_RESIDENT").ok().as_deref(),
        Some("0") | Some("false") | Some("no")
    )
}

/// Bound on the decode→writer channel of the loose-object unpack writer pool:
/// how many decoded entries may sit in flight before the decode thread blocks,
/// which bounds peak memory at roughly this many object bodies per pack.
const UNPACK_WRITER_QUEUE_OBJECTS: usize = 128;

/// Loose-object writer threads per pack unpack. The measured cache-write
/// throughput stops scaling past ~4 threads (FS-bound), so cap there.
fn unpack_loose_writer_count() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 4)
}

/// Objects written this process that have not yet been uploaded, keyed by repo
/// (`endpoint` + `repo_id`) so the three Vex stores of one repo — object
/// backend, op store, op heads store — share a single buffer even though each
/// holds its own [`VexClient`].
///
/// Snapshotting the working copy writes a dependency chain — the file blob, the
/// trees above it, the working-copy commit, then the operation and view — whose
/// ids are all content hashes computed locally. Uploading them one blocking
/// `put_object` round trip at a time makes `vex status` after an edit pay the
/// backend latency several times over; buffering them here lets a single
/// pipelined `put_objects` batch publish the whole set just before the op-head
/// CAS references it (see [`VexClient::commit_op_heads`]).
///
/// Invariant: an object is written to the on-disk cache only *after* it has been
/// uploaded, so the content-addressed "cached ⟹ present on server" short circuit
/// in [`VexClient::put_object`] stays sound across processes even if this one
/// dies mid-snapshot. Reads consult this buffer before the network so a
/// within-process read of a just-written object still resolves.
static PENDING_UPLOADS: OnceLock<Mutex<HashMap<String, PendingUploads>>> = OnceLock::new();

#[derive(Default)]
struct PendingUploads {
    objects: HashMap<(ObjectKind, ContentId), Vec<u8>>,
    bytes: usize,
}

/// Process-wide pack-resident indexes, keyed by cache root. Process-global for
/// the same reason as [`PENDING_UPLOADS`]: the three Vex stores of one repo —
/// object backend, op store, op heads store — each hold their own
/// [`VexClient`], and all of them must see one coherent view of the index
/// (including self-heal drops and prune invalidation).
static PACK_INDEXES: OnceLock<Mutex<HashMap<PathBuf, Arc<PackResidentIndex>>>> = OnceLock::new();

/// In-memory overlay of the pack-resident metadata cache (roadmap/032
/// follow-up). Unpacking a clone/snapshot pack appends the metadata entries'
/// bytes to `<cache_root>/.packs/<pack_hex>.payload` and records
/// `content_id → (pack, offset, len)` both here and in an atomically-written
/// `<pack_hex>.idx` sidecar (one idx file per pack, so concurrent clones
/// sharing a cache dir never coordinate appends). The overlay is consulted by
/// [`VexClient::read_cached_object`] / [`VexClient::has_cached_object`]
/// *before* the loose files; sidecars are folded in lazily on first use.
///
/// Entries only ever describe server-served, SHA-256-verified pack contents,
/// so an index hit carries the same "cached ⟹ present on server" guarantee as
/// a loose cache file. Staleness (a payload pruned or deleted behind our
/// back) self-heals on read: see [`VexClient::read_pack_resident_object`].
#[derive(Debug)]
struct PackResidentIndex {
    /// `<cache_root>/.packs` — payload and `.idx` sidecar files live here.
    packs_dir: PathBuf,
    state: Mutex<PackIndexState>,
}

#[derive(Debug, Default)]
struct PackIndexState {
    /// Whether the on-disk `*.idx` sidecars have been folded in.
    loaded: bool,
    entries: HashMap<(ObjectKind, ContentId), PackEntryLocation>,
}

/// Where one pack-resident object's bytes live: `(payload file, offset, len)`.
#[derive(Debug, Clone)]
struct PackEntryLocation {
    /// Pack content id hex — the payload/idx file stem, shared (`Arc`) across
    /// all of the pack's entries.
    pack_hex: Arc<str>,
    offset: u64,
    len: u64,
}

/// One `.idx` sidecar line: an object's location within its pack's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PackIndexRecord {
    kind: ObjectKind,
    content_id: ContentId,
    offset: u64,
    len: u64,
}

/// First line of a pack `.idx` sidecar (format version marker). Files with a
/// different header are ignored wholesale, so the format can evolve without
/// misreading old caches.
const PACK_IDX_HEADER: &str = "vex-pack-idx-v1";

/// Serialize the `.idx` sidecar: the header line, then one
/// `<kind> <content_id> <offset> <len>` line per entry.
fn format_pack_index_file(records: &[PackIndexRecord]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(records.len() * 90 + PACK_IDX_HEADER.len() + 1);
    out.push_str(PACK_IDX_HEADER);
    out.push('\n');
    for record in records {
        writeln!(
            out,
            "{} {} {} {}",
            kind_to_str(record.kind),
            record.content_id,
            record.offset,
            record.len
        )
        .expect("writing to a String cannot fail");
    }
    out
}

/// Allocation-free decode of a 64-char hex content id.
/// `ContentId::from_hex` heap-allocates a `Vec` per call (`hex::decode`),
/// which is measurable across the ~126k sidecar records a prod-scale
/// [`PackResidentIndex::ensure_loaded`] parses on a process's first metadata
/// read. Same accepted inputs as `from_hex` (either hex case).
fn content_id_from_hex_no_alloc(s: &str) -> Option<ContentId> {
    fn nibble(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    if bytes.len() != ContentId::HEX_LEN {
        return None;
    }
    let mut out = [0_u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.chunks_exact(2)) {
        *slot = (nibble(pair[0])? << 4) | nibble(pair[1])?;
    }
    Some(ContentId::from_bytes(out))
}

/// Parse a `.idx` sidecar written by [`format_pack_index_file`]. `None` for
/// anything malformed (wrong header, junk line): the whole file is then
/// ignored and its objects simply fall back to loose/RPC reads.
///
/// This runs for *every* sidecar on a process's first metadata read
/// (`ensure_loaded`) — ~11MB of text at prod scale, on the profiled
/// `vex status` startup path — so it stays allocation lean: the only
/// per-record work is borrowed `split`s, a stack hex decode, and integer
/// parses, and the output `Vec` is pre-sized from the file length (records
/// are ~86 bytes/line).
fn parse_pack_index_file(text: &str) -> Option<Vec<PackIndexRecord>> {
    let mut lines = text.lines();
    if lines.next()? != PACK_IDX_HEADER {
        return None;
    }
    let mut records = Vec::with_capacity(text.len() / 80 + 1);
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(' ');
        let kind = kind_from_str(fields.next()?)?;
        let content_id = content_id_from_hex_no_alloc(fields.next()?)?;
        let offset = fields.next()?.parse().ok()?;
        let len = fields.next()?.parse().ok()?;
        if fields.next().is_some() {
            return None;
        }
        records.push(PackIndexRecord {
            kind,
            content_id,
            offset,
            len,
        });
    }
    Some(records)
}

impl PackResidentIndex {
    fn new(packs_dir: PathBuf) -> Self {
        Self {
            packs_dir,
            state: Mutex::new(PackIndexState::default()),
        }
    }

    fn payload_path(&self, pack_hex: &str) -> PathBuf {
        self.packs_dir.join(format!("{pack_hex}.payload"))
    }

    fn idx_path(&self, pack_hex: &str) -> PathBuf {
        self.packs_dir.join(format!("{pack_hex}.idx"))
    }

    fn lookup(&self, kind: ObjectKind, content_id: &ContentId) -> Option<PackEntryLocation> {
        let mut state = self.state.lock().unwrap();
        self.ensure_loaded(&mut state);
        state.entries.get(&(kind, *content_id)).cloned()
    }

    fn contains(&self, kind: ObjectKind, content_id: &ContentId) -> bool {
        self.lookup(kind, content_id).is_some()
    }

    /// Publish a freshly unpacked pack's entries to the overlay (its payload
    /// and `.idx` sidecar are already persisted).
    fn insert_pack(&self, pack_hex: &str, records: &[PackIndexRecord]) {
        let mut state = self.state.lock().unwrap();
        self.ensure_loaded(&mut state);
        let pack_hex: Arc<str> = Arc::from(pack_hex);
        for record in records {
            state.entries.insert(
                (record.kind, record.content_id),
                PackEntryLocation {
                    pack_hex: Arc::clone(&pack_hex),
                    offset: record.offset,
                    len: record.len,
                },
            );
        }
    }

    /// Self-heal after a payload read failure: the payload file is gone (or
    /// unreadable), so every entry pointing into it is dead. Drop them from
    /// the overlay and best-effort-remove the on-disk pair so no later
    /// process resurrects the stale entries from the sidecar.
    fn drop_pack(&self, pack_hex: &str) {
        {
            let mut state = self.state.lock().unwrap();
            state
                .entries
                .retain(|_, location| location.pack_hex.as_ref() != pack_hex);
        }
        drop(fs::remove_file(self.idx_path(pack_hex)));
        drop(fs::remove_file(self.payload_path(pack_hex)));
    }

    /// Drop every entry (prune removed the whole `.packs` dir). `loaded`
    /// stays true: the sidecars are gone with the payloads, and later unpacks
    /// re-publish through [`Self::insert_pack`].
    fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        state.entries.clear();
        state.loaded = true;
    }

    /// Fold the on-disk `*.idx` sidecars into the overlay, once. A sidecar
    /// whose payload file is missing (partially pruned/deleted cache) is
    /// dropped on the spot instead of loaded — the load-time flavor of the
    /// read-time self-heal.
    fn ensure_loaded(&self, state: &mut PackIndexState) {
        if state.loaded {
            return;
        }
        state.loaded = true;
        let Ok(dir_entries) = fs::read_dir(&self.packs_dir) else {
            return;
        };
        for dir_entry in dir_entries.flatten() {
            let path = dir_entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("idx") {
                continue;
            }
            let Some(pack_hex) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            if !self.payload_path(pack_hex).exists() {
                drop(fs::remove_file(&path));
                continue;
            }
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            let Some(records) = parse_pack_index_file(&text) else {
                continue;
            };
            let pack_hex: Arc<str> = Arc::from(pack_hex);
            // Bulk-reserve before the insert loop: at prod scale (~126k
            // records) the incremental HashMap growth is roughly half the
            // load cost.
            state.entries.reserve(records.len());
            for record in records {
                state.entries.insert(
                    (record.kind, record.content_id),
                    PackEntryLocation {
                        pack_hex: Arc::clone(&pack_hex),
                        offset: record.offset,
                        len: record.len,
                    },
                );
            }
        }
    }
}

/// Kinds served pack-resident from `.packs` payloads (when
/// [`pack_resident_cache_enabled`]). Blob and Symlink must stay loose —
/// reflink materialization (`cached_blob_path`), checkout streaming
/// (`open_cached_object`) and `read_symlink` all need real per-object files —
/// and the rarer kinds (tag/copy/manifest) conservatively stay loose with
/// them. Commits, trees, ops and views are read only through
/// `read_cached_object`/`get_object`, so they can be served straight from a
/// payload file.
fn is_pack_resident_kind(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::Commit | ObjectKind::Tree | ObjectKind::Op | ObjectKind::View
    )
}

impl VexClient {
    pub fn from_config(config: VexRepoConfig) -> Result<Self, VexConfigError> {
        Self::validate_endpoint(&config.endpoint)?;
        let local_writes = config.local_writes;
        Ok(Self {
            config,
            cache_root: None,
            cache_max_bytes: cache_max_bytes(),
            local_writes,
            fresh_cache: false,
            pack_resident_override: None,
            presigned_get_disabled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn from_store_path(store_path: &Path) -> Result<Self, VexConfigError> {
        let config = VexRepoConfig::load_from_store_path(store_path)?;
        Self::from_store_path_and_config(store_path, config)
    }

    /// Like [`Self::from_store_path`], but forces `object_read_mode` after
    /// loading `vex.json`. Needed because the mode field is never serialized
    /// (`#[serde(skip_serializing)]`), so disk-backed loads always see
    /// [`VexObjectReadMode::NativeOnly`] unless an explicit conversion/
    /// materialization caller overrides it here.
    pub fn from_store_path_with_object_read_mode(
        store_path: &Path,
        object_read_mode: VexObjectReadMode,
    ) -> Result<Self, VexConfigError> {
        let mut config = VexRepoConfig::load_from_store_path(store_path)?;
        config.object_read_mode = object_read_mode;
        Self::from_store_path_and_config(store_path, config)
    }

    fn from_store_path_and_config(
        store_path: &Path,
        config: VexRepoConfig,
    ) -> Result<Self, VexConfigError> {
        Self::validate_endpoint(&config.endpoint)?;
        let repo_path = store_path
            .parent()
            .ok_or_else(|| VexConfigError::InvalidStorePath(store_path.to_path_buf()))?;
        let cache_root = shared_cache_root(&config).unwrap_or_else(|| repo_path.join("vex-cache"));
        fs::create_dir_all(&cache_root)?;
        let local_writes = config.local_writes;
        Ok(Self {
            config,
            cache_root: Some(cache_root),
            cache_max_bytes: cache_max_bytes(),
            local_writes,
            fresh_cache: false,
            pack_resident_override: None,
            presigned_get_disabled: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn config(&self) -> &VexRepoConfig {
        &self.config
    }

    /// Whether this client is in local-write mode (READ_ONLY CI runner): writes
    /// resolve to the local cache instead of the backend. See `put_object` and
    /// [`crate::vex_op_heads_store::VexOpHeadsStore`].
    pub fn local_writes(&self) -> bool {
        self.local_writes
    }

    /// Whether this client uses the pack-resident metadata cache
    /// ([`pack_resident_cache_enabled`], overridable per client for tests).
    fn pack_resident_enabled(&self) -> bool {
        self.pack_resident_override
            .unwrap_or_else(pack_resident_cache_enabled)
    }

    /// This cache root's shared [`PackResidentIndex`], creating it on first
    /// use. `None` without a cache root or when the pack-resident cache is
    /// disabled (`VEX_CACHE_PACK_RESIDENT=0`) — every consulting call site
    /// then behaves exactly as before the pack-resident split.
    fn pack_index(&self) -> Option<Arc<PackResidentIndex>> {
        if !self.pack_resident_enabled() {
            return None;
        }
        let cache_root = self.cache_root.as_ref()?;
        let map = PACK_INDEXES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock().unwrap();
        if let Some(index) = guard.get(cache_root) {
            return Some(Arc::clone(index));
        }
        let index = Arc::new(PackResidentIndex::new(cache_root.join(".packs")));
        guard.insert(cache_root.clone(), Arc::clone(&index));
        Some(index)
    }

    /// Mark this client's cache dir as freshly created by the current clone
    /// scaffold, enabling the direct-create (no temp+rename) fast path for
    /// the unpack's loose writes. Off by default; only the repo-local
    /// `vex-cache` qualifies: it lives inside the `.jj` this clone just
    /// created (`create_jj_dir` fails if one exists) and the whole `.jj` is
    /// removed on clone failure, so a crash cannot leave a truncated cache
    /// file for `read_cached_object` (which never re-verifies hashes) to
    /// serve forever. A shared cache dir (`JJ_VEX_SHARED_CACHE_DIR`) may
    /// pre-exist and outlives a failed clone, so it keeps atomic writes.
    pub fn mark_fresh_clone_cache(&mut self) {
        self.fresh_cache = shared_cache_root(&self.config).is_none();
    }

    fn cache_path(&self, kind: ObjectKind, content_id: &ContentId) -> Option<PathBuf> {
        self.cache_root
            .as_ref()
            .map(|root| root.join(kind_to_str(kind)).join(content_id.to_string()))
    }

    fn transfer_state_root(&self) -> Option<PathBuf> {
        self.cache_root
            .as_ref()
            .map(|root| root.join(".transfer-state").join("packs"))
    }

    fn transfer_state_path(&self, pack_content_id: &ContentId) -> Option<PathBuf> {
        self.transfer_state_root()
            .map(|root| root.join(format!("{pack_content_id}.json")))
    }

    fn transfer_partial_path(&self, pack_content_id: &ContentId) -> Option<PathBuf> {
        self.transfer_state_root()
            .map(|root| root.join(format!("{pack_content_id}.part")))
    }

    /// Directory of snapshot-set markers: `<cache_root>/.snapshots/<commit_hex>`
    /// marks that the full working-tree closure of that commit has been
    /// unpacked into this cache (roadmap/032). Excluded from LRU pruning, but
    /// whenever a prune evicts any object file *all* markers are dropped (see
    /// [`Self::prune_cache_if_needed`]): the evicted objects may belong to a
    /// closure a marker vouches for, and a stale marker would otherwise
    /// permanently suppress both snapshot serving (markers are sent as haves,
    /// trimming the served chain) and the hydration walk skip — and propagate
    /// across trunk advances via deltas marked complete on top of it.
    fn snapshot_marker_root(&self) -> Option<PathBuf> {
        self.cache_root.as_ref().map(|root| root.join(".snapshots"))
    }

    /// Whether the snapshot set for `commit_hex` (64-char lowercase hex) is
    /// recorded as fully unpacked into the local cache.
    pub fn has_unpacked_snapshot(&self, commit_hex: &str) -> bool {
        if !is_snapshot_commit_hex(commit_hex) {
            return false;
        }
        self.snapshot_marker_root()
            .is_some_and(|root| root.join(commit_hex).exists())
    }

    /// Record that the snapshot set for `commit_hex` is fully unpacked.
    fn write_snapshot_marker(&self, commit_hex: &str) -> Result<(), VexClientError> {
        let Some(root) = self.snapshot_marker_root() else {
            return Ok(());
        };
        fs::create_dir_all(&root)?;
        fs::write(root.join(commit_hex), b"")?;
        Ok(())
    }

    /// Commit ids (64-char lowercase hex) of every snapshot set fully unpacked
    /// into this cache. Sent as `have_snapshot_commit_ids` so the server can
    /// trim the served snapshot chain to the delta above what we already hold.
    pub fn cached_snapshot_commit_ids(&self) -> Vec<String> {
        let Some(root) = self.snapshot_marker_root() else {
            return Vec::new();
        };
        let Ok(entries) = fs::read_dir(root) else {
            return Vec::new();
        };
        entries
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| is_snapshot_commit_hex(name))
            .collect()
    }

    fn load_pack_transfer_state(
        &self,
        pack_content_id: &ContentId,
    ) -> Result<Option<PackTransferState>, VexClientError> {
        let Some(path) = self.transfer_state_path(pack_content_id) else {
            return Ok(None);
        };
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        match serde_json::from_slice(&bytes) {
            Ok(state) => Ok(Some(state)),
            Err(err) => {
                // Corrupt/truncated state (saves are plain `fs::write`, so a
                // kill or ENOSPC mid-save can leave partial JSON behind):
                // inconsistent state means a full reset per the resume
                // contract. Drop the poisoned file so the chunk path
                // self-heals instead of erroring into the full-pack fallback
                // on every later clone sharing this cache.
                tracing::warn!(
                    error = %err,
                    path = %path.display(),
                    "corrupt pack transfer state; resetting the transfer"
                );
                drop(fs::remove_file(&path));
                Ok(None)
            }
        }
    }

    fn save_pack_transfer_state(
        &self,
        pack_content_id: &ContentId,
        state: &PackTransferState,
    ) -> Result<(), VexClientError> {
        let Some(path) = self.transfer_state_path(pack_content_id) else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let payload = serde_json::to_vec_pretty(state)
            .map_err(VexConfigError::Json)
            .map_err(VexClientError::from)?;
        fs::write(path, payload)?;
        Ok(())
    }

    /// Remove a finished (or abandoned) pack transfer's state + `.part` files.
    /// Also best-effort-removes any legacy loose `pack/<chunk_id>` cache files
    /// for `chunk_ids`: older clients' gRPC chunk fallback double-wrote every
    /// chunk into the loose cache (~41MB of dead files per prod clone). New
    /// fallback reads bypass the cache entirely (see
    /// [`Self::fetch_pack_chunk_with_retry`]); this cleans up what old clients
    /// left behind.
    fn clear_pack_transfer_state(
        &self,
        pack_content_id: &ContentId,
        chunk_ids: &[ContentId],
    ) -> Result<(), VexClientError> {
        if let Some(state_path) = self.transfer_state_path(pack_content_id) {
            drop(fs::remove_file(state_path));
        }
        if let Some(partial_path) = self.transfer_partial_path(pack_content_id) {
            drop(fs::remove_file(partial_path));
        }
        for chunk_id in chunk_ids {
            if let Some(chunk_path) = self.cache_path(ObjectKind::Pack, chunk_id) {
                drop(fs::remove_file(chunk_path));
            }
        }
        Ok(())
    }

    pub(crate) fn read_cached_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Option<Vec<u8>> {
        // Pack-resident overlay first: after a clone, metadata kinds live in
        // `.packs` payloads and never as loose files, so the in-memory lookup
        // is the common hit and skips a guaranteed failed `open()` of the
        // loose path. Anything not in the overlay — blobs, individually
        // fetched or locally written objects — falls through to the loose
        // file, which remains fully supported.
        if let Some(bytes) = self.read_pack_resident_object(kind, content_id) {
            return Some(bytes);
        }
        let path = self.cache_path(kind, content_id)?;
        let bytes = fs::read(&path).ok()?;
        debug!(kind = kind_to_str(kind), %content_id, bytes = bytes.len(), cache_path = %path.display(), "vex cache hit");
        Some(bytes)
    }

    /// Read one object out of its pack-resident payload file, if the index
    /// holds it. Self-heals a stale index: when the payload is *structurally*
    /// gone — missing (pruned or deleted behind our back) or truncated
    /// (entries point past EOF) — the whole pack's entries are dropped along
    /// with its on-disk sidecar, and the read reports a miss so the caller
    /// falls back to the loose file or the backend. Any other I/O error
    /// (EMFILE under checkout's fd pressure, EACCES from a sandbox/AV, EIO)
    /// is transient: report a miss for this one read but keep the payload,
    /// sidecar, and index entries intact so the next read retries — matching
    /// the loose path, which never deletes on a read error.
    fn read_pack_resident_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Option<Vec<u8>> {
        let index = self.pack_index()?;
        let location = index.lookup(kind, content_id)?;
        let path = index.payload_path(&location.pack_hex);
        let read = || -> std::io::Result<Vec<u8>> {
            use std::io::Read as _;
            let mut file = File::open(&path)?;
            file.seek(SeekFrom::Start(location.offset))?;
            let mut bytes = vec![0_u8; location.len as usize];
            file.read_exact(&mut bytes)?;
            Ok(bytes)
        };
        match read() {
            Ok(bytes) => {
                debug!(kind = kind_to_str(kind), %content_id, bytes = bytes.len(), pack = %location.pack_hex, "vex cache hit (pack)");
                Some(bytes)
            }
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::UnexpectedEof
                ) =>
            {
                debug!(
                    kind = kind_to_str(kind),
                    %content_id,
                    pack = %location.pack_hex,
                    error = %err,
                    "pack payload missing/truncated; dropping its index entries (self-heal)"
                );
                index.drop_pack(&location.pack_hex);
                None
            }
            Err(err) => {
                debug!(
                    kind = kind_to_str(kind),
                    %content_id,
                    pack = %location.pack_hex,
                    error = %err,
                    "pack payload unreadable (transient); treating as a cache miss"
                );
                None
            }
        }
    }

    /// Whether an object is present in the local cache, without reading it.
    ///
    /// The cache is content-addressed and only populated after a successful
    /// upload (or by clone prefetch of server-resident objects), so a hit means
    /// the object is already on the server. Callers use this to skip redundant
    /// uploads cheaply (no disk read of the blob body).
    fn has_cached_object(&self, kind: ObjectKind, content_id: &ContentId) -> bool {
        // Pack-resident entries count too: they were unpacked from
        // server-served, hash-verified packs, so "cached ⟹ present on server"
        // holds for them — without this, every push would re-upload the
        // pack-delivered metadata. The payload file is deliberately not
        // stat'ed here: even if it was pruned, the object is still on the
        // server, which is all this check vouches for.
        if self
            .pack_index()
            .is_some_and(|index| index.contains(kind, content_id))
        {
            return true;
        }
        self.cache_path(kind, content_id)
            .is_some_and(|path| path.exists())
    }

    /// Open the locally cached copy of an object for streaming, if present.
    /// Counts as a cache hit. Lets bulk readers (checkout) stream blob
    /// contents straight from disk instead of buffering whole objects in RAM.
    pub(crate) fn open_cached_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Option<fs::File> {
        let path = self.cache_path(kind, content_id)?;
        let file = fs::File::open(&path).ok()?;
        debug!(kind = kind_to_str(kind), %content_id, cache_path = %path.display(), "vex cache hit (stream)");
        vex_client_stats().record_get_object_cache_hit(kind);
        Some(file)
    }

    /// Path of the locally cached copy of an object, if present. Cache files
    /// are content-addressed and never mutated once written, so callers may
    /// clone (reflink) them — checkout's copy-on-write materialization.
    pub(crate) fn cached_object_path(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Option<PathBuf> {
        let path = self.cache_path(kind, content_id)?;
        path.exists().then_some(path)
    }

    fn write_cached_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        data: &[u8],
    ) -> Result<(), VexClientError> {
        self.write_cached_object_no_prune(kind, content_id, data)?;
        self.prune_cache_if_needed()?;
        Ok(())
    }

    /// Like [`Self::write_cached_object`] but skips the per-write prune pass.
    /// Pruning scans the whole cache directory when `JJ_VEX_CACHE_MAX_BYTES` is
    /// set — quadratic during a bulk write of N objects — so bulk writers (e.g.
    /// [`Self::get_objects_inline_batched`]) call this per object and prune
    /// once at the end.
    fn write_cached_object_no_prune(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        data: &[u8],
    ) -> Result<(), VexClientError> {
        let Some(path) = self.cache_path(kind, content_id) else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut temp = NamedTempFile::new_in(path.parent().expect("cache file has parent"))?;
        use std::io::Write as _;
        temp.write_all(data)?;
        temp.flush()?;
        temp.persist(&path).map_err(|err| err.error)?;
        debug!(kind = kind_to_str(kind), %content_id, bytes = data.len(), cache_path = %path.display(), "vex cache write");
        Ok(())
    }

    /// Persist one unpacked loose object. `direct` skips the temp+rename
    /// atomicity dance (measured 2.5x faster at clone scale) — safe ONLY for
    /// a cache dir created by this clone process (see
    /// [`Self::mark_fresh_clone_cache`]): a crash mid-write leaves a
    /// truncated file, which `read_cached_object` (never re-verifies hashes)
    /// would otherwise serve forever, but a failed clone removes the whole
    /// freshly-scaffolded `.jj` and its cache with it.
    fn write_unpacked_loose_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        data: &[u8],
        direct: bool,
    ) -> Result<(), VexClientError> {
        if !direct {
            return self.write_cached_object_no_prune(kind, content_id, data);
        }
        let Some(path) = self.cache_path(kind, content_id) else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = File::create(&path)?;
        file.write_all(data)?;
        debug!(kind = kind_to_str(kind), %content_id, bytes = data.len(), cache_path = %path.display(), "vex cache write (direct)");
        Ok(())
    }

    fn prune_cache_if_needed(&self) -> Result<(), VexClientError> {
        let (Some(cache_root), Some(limit_bytes)) = (&self.cache_root, self.cache_max_bytes) else {
            return Ok(());
        };
        let mut entries = Vec::new();
        // Skip bookkeeping dirs at the cache root (`.snapshots` set markers,
        // `.transfer-state` resumable pack state): they are tiny, and pruning
        // them would silently forfeit snapshot negotiation or break an
        // in-flight resumable transfer.
        for entry in fs::read_dir(cache_root)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                collect_cache_entries(&entry.path(), &mut entries)?;
            } else if metadata.is_file() {
                entries.push(CacheEntry {
                    path: entry.path(),
                    modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                    size_bytes: metadata.len(),
                });
            }
        }
        let mut total_bytes = entries.iter().map(|entry| entry.size_bytes).sum::<u64>();
        if total_bytes <= limit_bytes {
            return Ok(());
        }
        entries.sort_by_key(|entry| entry.modified);
        let target_bytes = limit_bytes.saturating_mul(9).saturating_div(10);
        let mut removed_files = 0_u64;
        let mut reclaimed_bytes = 0_u64;
        for entry in entries {
            if total_bytes <= target_bytes {
                break;
            }
            if fs::remove_file(&entry.path).is_ok() {
                total_bytes = total_bytes.saturating_sub(entry.size_bytes);
                removed_files += 1;
                reclaimed_bytes += entry.size_bytes;
            }
        }
        // Evicted object files may belong to closures our `.snapshots`
        // markers vouch for; a stale marker would permanently suppress both
        // snapshot serving (markers are sent as haves, so the server trims
        // the chain) and the hydration walk skip, and new deltas marked
        // complete on top of it would keep the degradation alive across trunk
        // advances. Markers are cheap to regenerate — drop them all.
        if removed_files > 0 {
            if let Some(marker_root) = self.snapshot_marker_root() {
                drop(fs::remove_dir_all(marker_root));
            }
            // The `.packs` payload/index files are excluded from the LRU scan
            // above (dot-dir), so a capped cache bounds their growth here
            // instead: any prune that evicts object files also drops the
            // pack-resident store wholesale, mirroring the marker rule. The
            // removal runs even with `VEX_CACHE_PACK_RESIDENT=0` — nothing
            // reads or writes `.packs` while the kill switch is on, so a
            // cache dir that previously ran enabled would otherwise keep its
            // whole `.packs` footprint as unreclaimable dead disk. The
            // in-memory overlay is cleared with it (`pack_index()` is `None`
            // when disabled, so the clear no-ops there); another process
            // holding stale entries self-heals on its next read (the payload
            // open fails, so the pack's entries are dropped).
            drop(fs::remove_dir_all(cache_root.join(".packs")));
            if let Some(index) = self.pack_index() {
                index.clear();
            }
        }
        debug!(
            cache_root = %cache_root.display(),
            limit_bytes,
            target_bytes,
            total_bytes,
            removed_files,
            reclaimed_bytes,
            "pruned vex cache"
        );
        Ok(())
    }

    /// Validate that `endpoint` is a well-formed URI without building a TLS
    /// connector.
    ///
    /// Each `vex` command opens three Vex stores — the object backend, the op
    /// store, and the op heads store — and every one validates the same endpoint
    /// on open. `Endpoint::from_shared` performs the same URI parsing that
    /// [`Self::endpoint`] relies on (same error surface) but attaches no TLS
    /// connector, so validation is effectively free. The one connector we
    /// actually need is built lazily, once per process, in
    /// [`Self::cached_channel`].
    fn validate_endpoint(endpoint: &str) -> Result<(), VexConfigError> {
        Endpoint::from_shared(endpoint.to_string())
            .map(|_| ())
            .map_err(|err| VexConfigError::InvalidEndpoint {
                endpoint: endpoint.to_string(),
                message: err.to_string(),
            })
    }

    /// Whether `endpoint` speaks TLS (its scheme is `https`). Plaintext `http`
    /// endpoints (e.g. a local dev backend) get no TLS connector attached.
    fn endpoint_is_https(endpoint: &str) -> bool {
        endpoint
            .split_once("://")
            .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("https"))
    }

    /// Whether to verify the server against the system trust store instead of
    /// the compiled-in webpki roots. Off by default (webpki, which needs no
    /// keychain read); set `VEX_TLS_NATIVE_ROOTS=1` when the backend is reached
    /// through a TLS-intercepting proxy that presents a private/corporate root
    /// CA the system trusts but the webpki (Mozilla) set does not.
    fn native_tls_roots_requested() -> bool {
        matches!(
            std::env::var("VEX_TLS_NATIVE_ROOTS").ok().as_deref(),
            Some("1") | Some("true") | Some("yes")
        )
    }

    fn endpoint(endpoint: &str) -> Result<Endpoint, VexConfigError> {
        let mkerr = |err: tonic::transport::Error| VexConfigError::InvalidEndpoint {
            endpoint: endpoint.to_string(),
            message: err.to_string(),
        };
        // Build with `from_shared` rather than `Endpoint::new`: `new`
        // auto-attaches, for every `https` URI, a TLS connector built from the
        // *system* root store — a ~100ms macOS keychain read + cert parse paid
        // on every short-lived `vex` command. Attach the connector ourselves
        // from the compiled-in webpki (Mozilla) roots instead — instant, no
        // keychain — falling back to the system trust store only when
        // `VEX_TLS_NATIVE_ROOTS` is set (see `native_tls_roots_requested`).
        let is_https = Self::endpoint_is_https(endpoint);
        let mut endpoint = Endpoint::from_shared(endpoint.to_string()).map_err(mkerr)?;
        if is_https {
            let tls = tonic::transport::ClientTlsConfig::new();
            let tls = if Self::native_tls_roots_requested() {
                tls.with_native_roots()
            } else {
                tls.with_webpki_roots()
            };
            endpoint = endpoint.tls_config(tls).map_err(mkerr)?;
        }
        // Bound cold-start tail latency: a `vex` process is short-lived and pays
        // a fresh TCP+TLS+HTTP/2 handshake on its first call, so cap how long a
        // hung connect/request can stall a command. HTTP/2 keepalive keeps the
        // pooled channel healthy across the calls within one command and guards
        // against an idle edge-proxy reset mid-command. Values are conservative
        // to avoid tripping server-side `too_many_pings` (ENHANCE_YOUR_CALM).
        let endpoint = endpoint
            .connect_timeout(Duration::from_secs(env_secs(
                "VEX_GRPC_CONNECT_TIMEOUT_SECS",
                10,
            )))
            .timeout(Duration::from_secs(env_secs(
                "VEX_GRPC_REQUEST_TIMEOUT_SECS",
                300,
            )))
            .tcp_keepalive(Some(Duration::from_secs(30)))
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10));
        Ok(endpoint)
    }

    fn auth_request<T>(
        message: T,
        access_token: Option<&str>,
    ) -> Result<tonic::Request<T>, tonic::Status> {
        let mut request = tonic::Request::new(message);
        if let Some(access_token) = access_token.filter(|value| !value.is_empty()) {
            let metadata = MetadataValue::try_from(format!("Bearer {access_token}"))
                .map_err(|err| tonic::Status::invalid_argument(err.to_string()))?;
            request.metadata_mut().insert("authorization", metadata);
        }
        Ok(request)
    }

    /// Shared multi-threaded runtime for all blocking gRPC calls. Reused across
    /// every call so we don't pay runtime-construction cost per request and so
    /// batches can be issued concurrently over one HTTP/2 connection.
    fn shared_grpc_runtime() -> &'static tokio::runtime::Runtime {
        static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
        RUNTIME.get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .expect("failed to build shared gRPC runtime")
        })
    }

    /// Return a cached, connected `Channel` for `endpoint_url`, establishing one
    /// on first use. tonic `Channel`s are cheap to clone and multiplex requests
    /// over a single connection, so reusing them avoids a fresh TCP+TLS+HTTP/2
    /// handshake on every object — the dominant cost when uploading thousands.
    fn cached_channel(endpoint_url: &str) -> Result<Channel, VexClientError> {
        static CHANNELS: OnceLock<Mutex<HashMap<String, Channel>>> = OnceLock::new();
        let channels = CHANNELS.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(channel) = channels.lock().unwrap().get(endpoint_url) {
            return Ok(channel.clone());
        }
        let endpoint = Self::endpoint(endpoint_url)?;
        let channel =
            Self::shared_grpc_runtime().block_on(async move { endpoint.connect().await })?;
        channels
            .lock()
            .unwrap()
            .insert(endpoint_url.to_string(), channel.clone());
        Ok(channel)
    }

    fn block_on_grpc<T, F, Fut>(endpoint: &str, f: F) -> Result<T, VexClientError>
    where
        F: FnOnce(JjBackendClient<Channel>) -> Fut,
        Fut: Future<Output = Result<T, tonic::Status>>,
    {
        let channel = Self::cached_channel(endpoint)?;
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            let client = JjBackendClient::new(channel)
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
            f(client).await.map_err(Into::into)
        }))
    }

    /// Whether a gRPC status is worth retrying. Transient transport/edge
    /// failures (a Cloudflare/Caddy 502 mid-stream surfaces as `Internal` or
    /// `Unknown`, connection resets as `Unavailable`) are retryable; semantic
    /// errors (NotFound, InvalidArgument, auth) are not.
    fn is_transient_status(status: &tonic::Status) -> bool {
        matches!(
            status.code(),
            tonic::Code::Unavailable
                | tonic::Code::Internal
                | tonic::Code::Unknown
                | tonic::Code::DeadlineExceeded
                | tonic::Code::Aborted
                | tonic::Code::ResourceExhausted
        )
    }

    /// Whether a client error is a transient blip worth riding through (network
    /// hiccup, backend restart, edge 502) rather than a hard failure.
    fn is_transient_client_error(err: &VexClientError) -> bool {
        match err {
            VexClientError::Status(status) => Self::is_transient_status(status),
            VexClientError::Transport(_) => true,
            _ => false,
        }
    }

    /// `PutObjects` is safe to replay because the server creates immutable,
    /// content-addressed objects only when missing. In addition to the normal
    /// transient statuses, Caddy can cancel an in-flight HTTP/2 stream while it
    /// reloads; only this idempotent write path treats that cancellation as
    /// retryable.
    fn is_transient_pipelined_put_error(err: &VexClientError) -> bool {
        matches!(err, VexClientError::Status(status) if status.code() == tonic::Code::Cancelled)
            || Self::is_transient_client_error(err)
    }

    fn is_commit_operation_maintenance_status(status: &tonic::Status) -> bool {
        status.code() == tonic::Code::Unavailable
            && status.message() == "repository maintenance is in progress; retry commit"
    }

    /// `CommitOperation` is the one write RPC whose exact request can safely
    /// be replayed after the acknowledgement boundary is lost: the server
    /// accepts an already-current operation head as success, while an ordinary
    /// CAS conflict remains a normal response for jj to handle. Cover both the
    /// explicit maintenance fence and a transient/cancelled transport response
    /// so a Caddy or client timeout after the server commits cannot orphan a
    /// completed conversion.
    fn is_retryable_commit_operation_status(status: &tonic::Status) -> bool {
        Self::is_commit_operation_maintenance_status(status)
            || status.code() == tonic::Code::Cancelled
            || Self::is_transient_status(status)
    }

    fn commit_operation_maintenance_retry_delay(attempt: usize) -> Duration {
        let shift = attempt.saturating_sub(1).min(6) as u32;
        let backoff_ms = COMMIT_OPERATION_MAINTENANCE_RETRY_BASE_MS
            .saturating_mul(1_u64 << shift)
            .min(COMMIT_OPERATION_MAINTENANCE_RETRY_CAP_MS);
        let jitter_ms = Self::retry_jitter_ms(backoff_ms / 4 + 1);
        Duration::from_millis(backoff_ms + jitter_ms)
    }

    /// Bounded exponential backoff for retries of one idempotent `PutObjects`
    /// batch. The small jitter keeps concurrently retried batches from
    /// reconnecting to a recovering edge at the same instant.
    fn pipelined_put_retry_delay(attempt: usize) -> Duration {
        let shift = attempt.saturating_sub(1).min(6) as u32;
        let backoff_ms = PIPELINED_PUT_RETRY_BASE_MS
            .saturating_mul(1_u64 << shift)
            .min(PIPELINED_PUT_RETRY_CAP_MS);
        let jitter_ms = Self::retry_jitter_ms(backoff_ms / 4 + 1);
        Duration::from_millis(backoff_ms + jitter_ms)
    }

    /// Retry one idempotent `PutObjects` batch. The first call is allowed to
    /// reuse the process's pooled channel; retries receive `false` so callers
    /// reconnect with a fresh channel and client rather than reusing the stream
    /// Caddy may just have drained.
    async fn retry_pipelined_put_batch<T, F, Fut>(mut send: F) -> Result<T, VexClientError>
    where
        F: FnMut(bool) -> Fut,
        Fut: Future<Output = Result<T, VexClientError>>,
    {
        for attempt in 1..=PIPELINED_PUT_RETRY_ATTEMPTS {
            match send(attempt == 1).await {
                Ok(value) => return Ok(value),
                Err(err)
                    if Self::is_transient_pipelined_put_error(&err)
                        && attempt < PIPELINED_PUT_RETRY_ATTEMPTS =>
                {
                    let delay = Self::pipelined_put_retry_delay(attempt);
                    debug!(
                        attempt,
                        retry_attempt = attempt + 1,
                        attempts = PIPELINED_PUT_RETRY_ATTEMPTS,
                        delay_ms = delay.as_millis(),
                        error = %err,
                        "transient PutObjects batch failure; reconnecting before retry"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(err) => return Err(err),
            }
        }
        unreachable!("a nonzero retry budget always returns from the loop")
    }

    /// Like [`Self::block_on_grpc`] but retries the call on transient errors
    /// with linear backoff. Used for hot read paths (e.g. the per-file
    /// `GetObject` calls a working-copy checkout makes thousands of times),
    /// where a single transient edge blip would otherwise abort the whole
    /// operation. The closure is `Fn` so it can be re-invoked per attempt.
    fn block_on_grpc_retry<T, F, Fut>(
        endpoint: &str,
        attempts: usize,
        f: F,
    ) -> Result<T, VexClientError>
    where
        F: Fn(JjBackendClient<Channel>) -> Fut,
        Fut: Future<Output = Result<T, tonic::Status>>,
    {
        let channel = Self::cached_channel(endpoint)?;
        // Retry budget. A clone's working-copy checkout makes thousands of
        // per-object `GetObject` reads; a transient edge failure (a 502
        // mid-stream) or a jj-backend restart (down for seconds to tens of
        // seconds) must be *ridden through*, not aborted. The previous policy —
        // 5 attempts with linear 200ms*attempt backoff — gave only a ~2s window,
        // so a single backend blip mid-checkout failed the whole clone
        // ("Failed to check out the initial commit"). Use exponential backoff
        // (capped) with jitter over a ~40s window, and let callers only raise
        // (never lower) the attempt count. All tunable via env for ops.
        let attempts = attempts
            .max(env_secs("VEX_GRPC_RETRY_ATTEMPTS", 10) as usize)
            .max(1);
        let base_ms = env_secs("VEX_GRPC_RETRY_BACKOFF_MS", 250).max(1);
        let cap_ms = env_secs("VEX_GRPC_RETRY_BACKOFF_CAP_MS", 8_000).max(base_ms);
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            let mut attempt = 0usize;
            loop {
                attempt += 1;
                let client = JjBackendClient::new(channel.clone())
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
                match f(client).await {
                    Ok(value) => return Ok(value),
                    Err(status) if Self::is_transient_status(&status) && attempt < attempts => {
                        // Exponential backoff capped at `cap_ms`, plus jitter, so
                        // the flood of concurrent checkout reads doesn't hammer
                        // the backend in lockstep the instant it recovers.
                        let shift = (attempt - 1).min(6) as u32;
                        let backoff_ms = base_ms.saturating_mul(1u64 << shift).min(cap_ms);
                        let jitter_ms = Self::retry_jitter_ms(backoff_ms / 2 + 1);
                        tokio::time::sleep(std::time::Duration::from_millis(
                            backoff_ms + jitter_ms,
                        ))
                        .await;
                        continue;
                    }
                    Err(status) => return Err(status.into()),
                }
            }
        }))
    }

    /// Retry a transient response for an idempotent `CommitOperation` request.
    /// This policy is deliberately not shared with other write RPCs: the same
    /// operation head is replay-safe because the server returns success if it
    /// is already current.
    fn block_on_commit_operation_maintenance_retry<T, F, Fut>(
        endpoint: &str,
        f: F,
    ) -> Result<T, VexClientError>
    where
        F: Fn(JjBackendClient<Channel>) -> Fut,
        Fut: Future<Output = Result<T, tonic::Status>>,
    {
        let channel = Self::cached_channel(endpoint)?;
        // A shadow scan of a large mirror can legitimately occupy this lane for
        // more than the default window. Operators may raise (but never reduce)
        // the bounded budget for a known maintenance window without making
        // ordinary writes retry indefinitely.
        let attempts = env_secs(
            "VEX_COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS",
            COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS as u64,
        )
        .max(COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS as u64) as usize;
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            for attempt in 1..=attempts {
                let client = JjBackendClient::new(channel.clone())
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
                match f(client).await {
                    Ok(value) => return Ok(value),
                    Err(status)
                        if Self::is_retryable_commit_operation_status(&status)
                            && attempt < attempts =>
                    {
                        let delay = Self::commit_operation_maintenance_retry_delay(attempt);
                        debug!(
                            attempt,
                            retry_attempt = attempt + 1,
                            attempts,
                            delay_ms = delay.as_millis(),
                            error = %status,
                            "retryable idempotent op-head publication failure; replaying request"
                        );
                        tokio::time::sleep(delay).await;
                    }
                    Err(status) => return Err(status.into()),
                }
            }
            unreachable!("a nonzero retry budget always returns from the loop")
        }))
    }

    /// Async sibling of [`Self::block_on_grpc_retry`] that runs the retrying gRPC
    /// call as a task on the shared multi-thread runtime and awaits its
    /// `JoinHandle`, rather than blocking the calling thread.
    ///
    /// This distinction is the whole point on the working-copy checkout hot path.
    /// `TreeState::check_out` drives thousands of per-object reads through
    /// `.buffered(store.concurrency())` on a *single-threaded* `pollster`
    /// executor. `block_on_grpc_retry` blocks that one thread until each
    /// round-trip returns, so the buffered stream can never poll more than one
    /// read at a time — the intended 32-way concurrency collapses to 1, and a
    /// full clone becomes ~one network round-trip per file. Awaiting a spawned
    /// task's handle is instead a cooperative yield point: it registers a waker
    /// and returns `Pending`, so the executor keeps polling the other buffered
    /// reads and up to `concurrency()` requests are genuinely in flight on the
    /// runtime's worker threads at once.
    async fn grpc_retry_async<T, F, Fut>(
        endpoint: &str,
        attempts: usize,
        f: F,
    ) -> Result<T, VexClientError>
    where
        F: Fn(JjBackendClient<Channel>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, tonic::Status>> + Send + 'static,
        T: Send + 'static,
    {
        let channel = Self::cached_channel(endpoint)?;
        let attempts = attempts
            .max(env_secs("VEX_GRPC_RETRY_ATTEMPTS", 10) as usize)
            .max(1);
        let base_ms = env_secs("VEX_GRPC_RETRY_BACKOFF_MS", 250).max(1);
        let cap_ms = env_secs("VEX_GRPC_RETRY_BACKOFF_CAP_MS", 8_000).max(base_ms);
        let handle = Self::shared_grpc_runtime().spawn(with_output_cancel(async move {
            let mut attempt = 0usize;
            loop {
                attempt += 1;
                let client = JjBackendClient::new(channel.clone())
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
                match f(client).await {
                    Ok(value) => return Ok(value),
                    Err(status) if Self::is_transient_status(&status) && attempt < attempts => {
                        let shift = (attempt - 1).min(6) as u32;
                        let backoff_ms = base_ms.saturating_mul(1u64 << shift).min(cap_ms);
                        let jitter_ms = Self::retry_jitter_ms(backoff_ms / 2 + 1);
                        tokio::time::sleep(std::time::Duration::from_millis(
                            backoff_ms + jitter_ms,
                        ))
                        .await;
                        continue;
                    }
                    Err(status) => return Err(status.into()),
                }
            }
        }));
        match handle.await {
            Ok(result) => result,
            Err(join_err) => Err(VexClientError::Io(std::io::Error::other(format!(
                "grpc worker task failed: {join_err}"
            )))),
        }
    }

    /// Cheap, dependency-free jitter in `[0, span)` milliseconds, seeded from the
    /// wall clock. Only used to de-correlate retry backoff across concurrent
    /// reads, so statistical quality is irrelevant.
    fn retry_jitter_ms(span: u64) -> u64 {
        if span <= 1 {
            return 0;
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(0);
        nanos % span
    }

    /// Shared pooled HTTP client for presigned-URL fetches. One per process
    /// (like [`Self::cached_channel`]) so the ~139 chunk fetches of a clone
    /// reuse pooled TLS connections to the object store instead of paying a
    /// fresh TCP+TLS handshake per request.
    ///
    /// Timeouts: `connect_timeout` bounds a tarpit connect, and `read_timeout`
    /// bounds the time between body reads — so a server that returns headers
    /// then stalls the body errors out instead of hanging a non-interactive
    /// clone forever (the only other cancellation, [`with_output_cancel`],
    /// fires solely when a pager quits). A *total* request timeout is
    /// deliberately not set: whole packs stream through this client
    /// ([`Self::block_on_http_get_to_file`]) and a large-but-progressing
    /// download must never be killed. A timed-out chunk surfaces as an error
    /// and degrades to the existing gRPC fallback
    /// ([`Self::fetch_pack_chunk_with_retry`]). Env-tunable, mirroring the
    /// gRPC endpoint's `VEX_GRPC_*_TIMEOUT_SECS` knobs.
    fn shared_http_client() -> &'static reqwest::Client {
        static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
        CLIENT.get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(env_secs(
                    "VEX_HTTP_CONNECT_TIMEOUT_SECS",
                    10,
                )))
                .read_timeout(Duration::from_secs(env_secs(
                    "VEX_HTTP_READ_TIMEOUT_SECS",
                    60,
                )))
                .build()
                .expect("static HTTP client configuration is valid")
        })
    }

    /// Spawn a presigned HTTP GET as a task on the shared runtime, with
    /// `with_output_cancel` *inside* the spawned future (the
    /// [`Self::grpc_retry_async`] pattern), and buffer the response body.
    /// Awaiting the returned `JoinHandle` is a cooperative yield, so
    /// `.buffered(W)` chunk streams genuinely overlap W requests even when
    /// driven from a plain thread's `block_on`.
    ///
    /// `expected_len` (the descriptor's `size_bytes`, when the caller knows
    /// it) caps the buffered body: a hostile or broken endpoint that streams
    /// more than the expected size errors out as soon as the cap is crossed
    /// instead of buffering an arbitrarily large body — W of these run
    /// concurrently, so the memory bound matters. An over-cap fetch hits the
    /// same retry/gRPC-fallback path as any other fetch failure.
    fn spawn_http_get(
        url: String,
        headers: std::collections::HashMap<String, String>,
        expected_len: Option<u64>,
    ) -> tokio::task::JoinHandle<Result<Vec<u8>, VexClientError>> {
        Self::shared_grpc_runtime().spawn(with_output_cancel(async move {
            let mut request = Self::shared_http_client().get(&url);
            for (name, value) in &headers {
                request = request.header(name, value);
            }
            let mut response = request.send().await?.error_for_status()?;
            // Pre-size to the expected length (capped: never trust a header
            // or descriptor for a huge up-front allocation).
            let mut bytes: Vec<u8> = Vec::with_capacity(
                usize::try_from(expected_len.unwrap_or(0))
                    .unwrap_or(usize::MAX)
                    .min(16 << 20),
            );
            while let Some(chunk) = response.chunk().await? {
                let received = (bytes.len() as u64).saturating_add(chunk.len() as u64);
                if expected_len.is_some_and(|limit| received > limit) {
                    return Err(VexClientError::PackDecode(format!(
                        "http response exceeds expected size ({} bytes)",
                        expected_len.unwrap_or(0)
                    )));
                }
                bytes.extend_from_slice(&chunk);
            }
            Ok(bytes)
        }))
    }

    /// Async presigned GET: spawns via [`Self::spawn_http_get`] and awaits the
    /// task handle (a cooperative yield point for buffered chunk streams).
    ///
    /// The `presigned_fetches`/`presigned_bytes` counters are bumped *here*,
    /// on the consumer side, not inside the spawned task: dropping this future
    /// mid-await (an error abandons a `.buffered(W)` window; the detached task
    /// still runs to completion) must not count bytes that are never consumed,
    /// or bench JSON would report `presigned_bytes > pack_bytes_fetched`.
    async fn http_get_async(
        url: String,
        headers: std::collections::HashMap<String, String>,
        expected_len: Option<u64>,
    ) -> Result<Vec<u8>, VexClientError> {
        match Self::spawn_http_get(url, headers, expected_len).await {
            Ok(result) => {
                if let Ok(bytes) = &result {
                    let stats = vex_client_stats();
                    stats.presigned_fetches.fetch_add(1, Ordering::Relaxed);
                    stats
                        .presigned_bytes
                        .fetch_add(bytes.len() as u64, Ordering::Relaxed);
                }
                result
            }
            Err(join_err) => Err(VexClientError::Io(std::io::Error::other(format!(
                "http worker task failed: {join_err}"
            )))),
        }
    }

    fn block_on_http_get(
        url: &str,
        headers: &std::collections::HashMap<String, String>,
        expected_len: Option<u64>,
    ) -> Result<Vec<u8>, VexClientError> {
        // `Runtime::block_on` (not `futures::executor::block_on`): the callers
        // are plain pack-worker threads already inside a
        // `futures::executor::block_on`, which panics on re-entry.
        Self::shared_grpc_runtime().block_on(Self::http_get_async(
            url.to_string(),
            headers.clone(),
            expected_len,
        ))
    }

    /// Stream an HTTP GET body into `out`. `max_bytes` (the pack descriptor's
    /// `size_bytes`, when known) bounds how much a hostile/broken endpoint can
    /// write to disk; crossing it fails the fetch, which degrades to the
    /// existing whole-pack gRPC fallback.
    fn block_on_http_get_to_file(
        url: &str,
        headers: &std::collections::HashMap<String, String>,
        out: &mut dyn Write,
        max_bytes: Option<u64>,
    ) -> Result<(), VexClientError> {
        let url = url.to_string();
        let headers = headers.clone();
        // The response streams from a task on the shared runtime (which owns
        // the pooled client's connections and the cancellation timer) over a
        // bounded channel to this thread, which writes it out — `out` is a
        // plain `&mut dyn Write` that cannot move into a `'static` task.
        let (mut tx, mut rx) = futures::channel::mpsc::channel::<Vec<u8>>(8);
        let handle = Self::shared_grpc_runtime().spawn(with_output_cancel(async move {
            use futures::SinkExt as _;
            let mut request = Self::shared_http_client().get(&url);
            for (name, value) in &headers {
                request = request.header(name, value);
            }
            let mut response = request.send().await?.error_for_status()?;
            let mut total_bytes = 0_u64;
            while let Some(chunk) = response.chunk().await? {
                total_bytes += chunk.len() as u64;
                if max_bytes.is_some_and(|limit| total_bytes > limit) {
                    return Err(VexClientError::PackDecode(format!(
                        "http response exceeds expected size ({} bytes)",
                        max_bytes.unwrap_or(0)
                    )));
                }
                if tx.send(chunk.to_vec()).await.is_err() {
                    // Receiver dropped: the writer failed and its error wins
                    // (checked before this task's result below).
                    return Err(VexClientError::Io(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "pack stream receiver dropped",
                    )));
                }
            }
            let stats = vex_client_stats();
            stats.presigned_fetches.fetch_add(1, Ordering::Relaxed);
            stats
                .presigned_bytes
                .fetch_add(total_bytes, Ordering::Relaxed);
            Ok(())
        }));
        let (write_result, task_result) = Self::shared_grpc_runtime().block_on(async {
            use futures::StreamExt as _;
            let mut write_result: Result<(), std::io::Error> = Ok(());
            while let Some(chunk) = rx.next().await {
                if let Err(err) = out.write_all(&chunk) {
                    write_result = Err(err);
                    break;
                }
            }
            // Closing the receiver unblocks a sender awaiting channel capacity,
            // so the task observes the drop and finishes.
            drop(rx);
            let task_result = match handle.await {
                Ok(result) => result,
                Err(join_err) => Err(VexClientError::Io(std::io::Error::other(format!(
                    "http worker task failed: {join_err}"
                )))),
            };
            (write_result, task_result)
        });
        write_result?;
        task_result?;
        out.flush()?;
        Ok(())
    }

    fn direct_fetch_pack_bytes(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
    ) -> Result<Option<Vec<u8>>, VexClientError> {
        if self.presigned_get_disabled.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let Some(hint) = hints
            .iter()
            .find(|hint| hint.object_key.ends_with(&pack.content_id.to_string()))
        else {
            return Ok(None);
        };
        if hint.url.is_empty() {
            return Ok(None);
        }
        Self::block_on_http_get(&hint.url, &hint.headers, Some(pack.size_bytes)).map(Some)
    }

    /// Fetch one pack chunk via its presigned hint URL, if any. Async (the
    /// request runs as a spawned task on the shared runtime) so the chunk
    /// stream's `.buffered(W)` genuinely overlaps W fetches. `expected_len`
    /// is the chunk descriptor's `size_bytes` (caps the buffered body).
    async fn direct_fetch_pack_blob_bytes(
        &self,
        content_id: &ContentId,
        hints: &[jj_backend_api::PresignedGet],
        expected_len: Option<u64>,
    ) -> Result<Option<Vec<u8>>, VexClientError> {
        let Some(hint) = hints
            .iter()
            .find(|hint| hint.object_key.ends_with(&content_id.to_string()))
        else {
            return Ok(None);
        };
        if hint.url.is_empty() {
            return Ok(None);
        }
        Self::http_get_async(hint.url.clone(), hint.headers.clone(), expected_len)
            .await
            .map(Some)
    }

    fn direct_fetch_pack_to_file(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
        out: &mut dyn Write,
    ) -> Result<bool, VexClientError> {
        if self.presigned_get_disabled.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let Some(hint) = hints
            .iter()
            .find(|hint| hint.object_key.ends_with(&pack.content_id.to_string()))
        else {
            return Ok(false);
        };
        if hint.url.is_empty() {
            return Ok(false);
        }
        Self::block_on_http_get_to_file(&hint.url, &hint.headers, out, Some(pack.size_bytes))?;
        Ok(true)
    }

    /// Stream a pack file's entries into the local cache via the hybrid
    /// pack-resident/loose unpack (see [`Self::unpack_pack_entries`]). Uses
    /// the no-prune cache write for the loose portion (bulk path — the
    /// prefetch prunes once at the end).
    fn prefetch_pack_entries_from_file(
        &self,
        pack_content_id: &ContentId,
        path: &Path,
        prefetched_objects: &AtomicU64,
    ) -> Result<(), VexClientError> {
        let file = File::open(path)?;
        let mut reader = Some(BufReader::new(file));
        self.unpack_pack_entries(pack_content_id, prefetched_objects, move |sink| {
            let reader = reader.take().expect("unpack drives the decode once");
            let mut write_error: Option<VexClientError> = None;
            let decode_result = decode_object_pack_with_visitor(reader, |entry| {
                sink(entry).map_err(|err| {
                    write_error = Some(err);
                    jj_backend_types::PackCodecError::Compression("cache write failed".to_string())
                })
            });
            if let Some(err) = write_error {
                return Err(err);
            }
            decode_result.map_err(|err| VexClientError::PackDecode(err.to_string()))
        })
    }

    /// Unpack a pack's entries into the local cache with the hybrid split
    /// (roadmap/032 follow-up): metadata kinds ([`is_pack_resident_kind`]) are
    /// appended once to a per-pack payload file under `<cache_root>/.packs/`
    /// and published to the [`PackResidentIndex`] overlay as
    /// `(offset, len)` records, while everything else (blobs, symlinks) is
    /// written loose as before. The payload holds exactly the indexed entries'
    /// bytes, so offsets are computed here during the streaming decode; the
    /// per-entry SHA-256 verification already ran inside the decode, and
    /// entries are published only after payload + `.idx` sidecar are persisted
    /// (both atomically, content-addressed by pack id — concurrent clones
    /// sharing a cache dir persist idempotently).
    ///
    /// The loose portion is handed over a bounded channel to a small blocking
    /// writer pool ([`unpack_loose_writer_count`]), so the decode thread is
    /// not serialized behind per-object temp+rename file creation; payload and
    /// sidecar writes stay on the decode thread.
    ///
    /// With `VEX_CACHE_PACK_RESIDENT=0` (or without a cache root) every entry
    /// unpacks loose, inline, on the decode thread — exactly the pre-split
    /// behavior.
    ///
    /// `drive` feeds the entries (from a streaming decode or an in-memory
    /// pack) into the sink it is given, in pack order.
    fn unpack_pack_entries<F>(
        &self,
        pack_content_id: &ContentId,
        prefetched_objects: &AtomicU64,
        drive: F,
    ) -> Result<(), VexClientError>
    where
        F: FnOnce(
            &mut dyn FnMut(ObjectPackEntry) -> Result<(), VexClientError>,
        ) -> Result<(), VexClientError>,
    {
        let stats = vex_client_stats();
        let index = self.pack_index();
        let (Some(cache_root), Some(index)) = (self.cache_root.as_ref(), index) else {
            // Kill switch / no cache: the all-loose inline unpack of old.
            return drive(&mut |entry| {
                self.write_cached_object_no_prune(entry.kind, &entry.content_id, &entry.data)?;
                prefetched_objects.fetch_add(1, Ordering::Relaxed);
                stats.objects_unpacked.fetch_add(1, Ordering::Relaxed);
                Ok(())
            });
        };
        let packs_dir = cache_root.join(".packs");
        let pack_hex = pack_content_id.to_string();
        let direct_create = self.fresh_cache;
        // Payload + index records accumulate on the decode thread; loose
        // entries cross the bounded channel to the writer pool. The payload
        // temp is buffered: metadata entries average a few hundred bytes, so
        // writing them straight through the `NamedTempFile` would cost one
        // `write(2)` syscall per object (~126k per prod clone) on the
        // clone-critical decode thread.
        let mut payload: Option<std::io::BufWriter<NamedTempFile>> = None;
        let mut payload_offset = 0_u64;
        let mut records: Vec<PackIndexRecord> = Vec::new();
        let write_failed = AtomicBool::new(false);
        let first_writer_error: Mutex<Option<VexClientError>> = Mutex::new(None);
        let (sender, receiver) = std::sync::mpsc::sync_channel::<(ObjectKind, ContentId, Vec<u8>)>(
            UNPACK_WRITER_QUEUE_OBJECTS,
        );
        let receiver = Mutex::new(receiver);
        let drive_result = std::thread::scope(|scope| {
            for _ in 0..unpack_loose_writer_count() {
                scope.spawn(|| {
                    loop {
                        // The receiver lock is held only while *waiting*; the
                        // write below runs unlocked, so writers overlap.
                        let message = receiver.lock().unwrap().recv();
                        let Ok((kind, content_id, data)) = message else {
                            // Channel closed: the decode is done and drained.
                            return;
                        };
                        if write_failed.load(Ordering::SeqCst) {
                            // A sibling failed; keep draining so the bounded
                            // channel never blocks the decode thread.
                            continue;
                        }
                        match self.write_unpacked_loose_object(
                            kind,
                            &content_id,
                            &data,
                            direct_create,
                        ) {
                            Ok(()) => {
                                prefetched_objects.fetch_add(1, Ordering::Relaxed);
                                stats.objects_unpacked.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(err) => {
                                write_failed.store(true, Ordering::SeqCst);
                                let mut slot = first_writer_error.lock().unwrap();
                                if slot.is_none() {
                                    *slot = Some(err);
                                }
                            }
                        }
                    }
                });
            }
            let result = drive(&mut |entry| {
                if write_failed.load(Ordering::SeqCst) {
                    return Err(VexClientError::Io(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "unpack writer failed",
                    )));
                }
                if is_pack_resident_kind(entry.kind) {
                    if payload.is_none() {
                        fs::create_dir_all(&packs_dir)?;
                        payload = Some(std::io::BufWriter::with_capacity(
                            64 * 1024,
                            NamedTempFile::new_in(&packs_dir)?,
                        ));
                    }
                    let temp = payload.as_mut().expect("payload temp just initialized");
                    temp.write_all(&entry.data)?;
                    records.push(PackIndexRecord {
                        kind: entry.kind,
                        content_id: entry.content_id,
                        offset: payload_offset,
                        len: entry.data.len() as u64,
                    });
                    payload_offset += entry.data.len() as u64;
                    prefetched_objects.fetch_add(1, Ordering::Relaxed);
                    stats.objects_unpacked.fetch_add(1, Ordering::Relaxed);
                    stats.objects_pack_resident.fetch_add(1, Ordering::Relaxed);
                    stats.loose_writes_avoided.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                } else {
                    sender
                        .send((entry.kind, entry.content_id, entry.data))
                        .map_err(|_| {
                            VexClientError::Io(std::io::Error::new(
                                std::io::ErrorKind::BrokenPipe,
                                "unpack writer pool terminated",
                            ))
                        })
                }
            });
            // Closing the channel drains the writers; the scope joins them.
            drop(sender);
            result
        });
        // A writer error is the root cause of any decode abort; it wins.
        if let Some(err) = first_writer_error.into_inner().unwrap() {
            return Err(err);
        }
        drive_result?;
        if let Some(writer) = payload {
            // `into_inner` flushes the buffer and hands back the temp file.
            let temp = writer
                .into_inner()
                .map_err(std::io::IntoInnerError::into_error)?;
            Self::persist_pack_temp(
                &packs_dir,
                temp,
                &packs_dir.join(format!("{pack_hex}.payload")),
            )?;
            // A cross-process prune may have removed the whole `.packs` dir
            // between the payload persist and here; recreate it so the idx
            // temp can be created (its persist is race-tolerant below).
            fs::create_dir_all(&packs_dir)?;
            let mut idx_temp = NamedTempFile::new_in(&packs_dir)?;
            idx_temp.write_all(format_pack_index_file(&records).as_bytes())?;
            idx_temp.flush()?;
            Self::persist_pack_temp(
                &packs_dir,
                idx_temp,
                &packs_dir.join(format!("{pack_hex}.idx")),
            )?;
            index.insert_pack(&pack_hex, &records);
            debug!(
                pack = %pack_hex,
                entries = records.len(),
                payload_bytes = payload_offset,
                "vex pack-resident unpack"
            );
        }
        Ok(())
    }

    /// Persist a `.packs` temp file to its final path, tolerating a
    /// cross-process prune having `remove_dir_all`'d the `.packs` dir (and
    /// with it the temp's source path) mid-unpack: `persist` is a `rename(2)`
    /// whose source is then gone, so it fails deterministically — but the
    /// open fd handed back by the `PersistError` still holds every byte.
    /// Recreate the dir, re-materialize a fresh temp from that fd, and retry
    /// once (a second failure propagates). Without this, a concurrent capped
    /// clone sharing the cache could turn a metadata-pack unpack — fatal to
    /// the clone — into an ENOENT.
    fn persist_pack_temp(
        packs_dir: &Path,
        temp: NamedTempFile,
        path: &Path,
    ) -> Result<(), VexClientError> {
        let mut temp = match temp.persist(path) {
            Ok(_) => return Ok(()),
            Err(err) => {
                debug!(
                    path = %path.display(),
                    error = %err.error,
                    "pack file persist failed; recreating .packs and retrying once"
                );
                err.file
            }
        };
        fs::create_dir_all(packs_dir)?;
        temp.as_file_mut().seek(SeekFrom::Start(0))?;
        let mut fresh = NamedTempFile::new_in(packs_dir)?;
        std::io::copy(temp.as_file_mut(), fresh.as_file_mut())?;
        fresh.persist(path).map_err(|err| err.error)?;
        Ok(())
    }

    /// Whether a fetch error is an HTTP 403 from a hint URL. A 403 on a
    /// signed URL is deterministic — the signature expired or is invalid —
    /// so retrying the same URL (or any sibling hint minted in the same
    /// up-front batch) is doomed.
    fn is_presigned_forbidden(err: &VexClientError) -> bool {
        matches!(
            err,
            VexClientError::Http(http) if http.status() == Some(reqwest::StatusCode::FORBIDDEN)
        )
    }

    /// Fetch one pack chunk's bytes: try the presigned hint URL (twice), then
    /// fall back to a gRPC `GetObject`. The fallback deliberately bypasses the
    /// local loose cache: the chunk bytes land in the transfer's `.part` file,
    /// so a loose `pack/<chunk_id>` copy would be pure dead weight (~41MB per
    /// prod clone before this read went cache-less).
    ///
    /// Presigned bytes are hash-verified before they are returned (a chunk's
    /// content id is the SHA-256 of exactly its bytes — the gRPC fallback
    /// verifies the same way): a size-correct but wrong-content response must
    /// never enter the `.part` file, where it would only surface much later
    /// as a decode failure of the assembled pack. On the first 403 the
    /// per-client presigned kill switch trips (see `presigned_get_disabled`):
    /// hints are minted once per prefetch, so an expired URL means every
    /// remaining hint is expired too, and the rest of the pack goes straight
    /// to gRPC instead of paying two doomed HTTPS attempts per chunk.
    async fn fetch_pack_chunk_with_retry(
        &self,
        content_id: &ContentId,
        hints: &[jj_backend_api::PresignedGet],
        expected_len: Option<u64>,
    ) -> Result<Vec<u8>, VexClientError> {
        let mut last_hint_err: Option<VexClientError> = None;
        if !self.presigned_get_disabled.load(Ordering::Relaxed) {
            for _ in 0..2 {
                match self
                    .direct_fetch_pack_blob_bytes(content_id, hints, expected_len)
                    .await
                {
                    Ok(Some(bytes)) => {
                        if ContentId::hash_bytes(&bytes) == *content_id {
                            return Ok(bytes);
                        }
                        last_hint_err = Some(VexClientError::PackDecode(format!(
                            "presigned chunk {content_id} failed hash verification"
                        )));
                    }
                    Ok(None) => break,
                    Err(err) => {
                        let forbidden = Self::is_presigned_forbidden(&err);
                        last_hint_err = Some(err);
                        if forbidden {
                            self.presigned_get_disabled.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }
            }
        }
        if let Some(err) = last_hint_err {
            // Redacted: a presigned fetch error embeds the full signed URL
            // (`X-Amz-Signature=...` query), which must never reach a log.
            debug!(%content_id, error = %redact_url_queries(&err.to_string()), "direct chunk fetch failed, falling back to grpc");
        }
        let _t = RpcTimer::start(|| "get_object/pack".to_string());
        self.fetch_object_grpc_verified(ObjectKind::Pack, content_id)
            .await
    }

    async fn prefetch_pack_via_chunks(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
        snapshot: bool,
        prefetched_objects: &AtomicU64,
    ) -> Result<bool, VexClientError> {
        self.prefetch_pack_via_chunks_with_concurrency(
            pack,
            hints,
            snapshot,
            prefetched_objects,
            clone_chunk_concurrency(),
        )
        .await
    }

    /// [`Self::prefetch_pack_via_chunks`] with an explicit chunk-fetch
    /// concurrency, so tests can pin W without mutating the process
    /// environment (`jj-lib` forbids `unsafe`, which `set_var` now requires).
    async fn prefetch_pack_via_chunks_with_concurrency(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
        snapshot: bool,
        prefetched_objects: &AtomicU64,
        concurrency: usize,
    ) -> Result<bool, VexClientError> {
        let Some(chunks) = normalized_valid_pack_chunks(pack) else {
            return Ok(false);
        };
        let Some(partial_path) = self.transfer_partial_path(&pack.content_id) else {
            return Ok(false);
        };
        if let Some(parent) = partial_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut state =
            self.load_pack_transfer_state(&pack.content_id)?
                .unwrap_or(PackTransferState {
                    pack_content_id: pack.content_id.to_string(),
                    chunk_count: chunks.len(),
                    next_chunk_index: 0,
                });
        if state.chunk_count != chunks.len() || state.next_chunk_index > chunks.len() {
            state.chunk_count = chunks.len();
            state.next_chunk_index = 0;
            drop(fs::remove_file(&partial_path));
        }
        let mut partial_file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .append(true)
            .open(&partial_path)?;
        let expected_prefix_bytes: u64 = chunks
            .iter()
            .take(state.next_chunk_index)
            .map(|chunk| chunk.size_bytes)
            .sum();
        let partial_len = partial_file.metadata()?.len();
        if partial_len > expected_prefix_bytes {
            // State saves are batched (every [`TRANSFER_STATE_SAVE_INTERVAL`]
            // chunks), so a kill between an append and the next save leaves
            // the `.part` ahead of the recorded state — possibly mid-chunk.
            // Only the recorded contiguous prefix is trustworthy; truncate
            // back to it and refetch the rest.
            debug!(
                pack = %pack.content_id,
                partial_len,
                expected_prefix_bytes,
                next_chunk_index = state.next_chunk_index,
                "pack `.part` ahead of recorded transfer state; truncating to the trusted prefix"
            );
            partial_file.set_len(expected_prefix_bytes)?;
            partial_file.seek(SeekFrom::Start(expected_prefix_bytes))?;
        } else if partial_len < expected_prefix_bytes {
            // Shorter than the recorded prefix: the state/`.part` pair is
            // inconsistent (a state file ahead of its data); restart from
            // scratch.
            partial_file.set_len(0)?;
            partial_file.seek(SeekFrom::Start(0))?;
            state.next_chunk_index = 0;
        }
        let fetch_result = self
            .fetch_chunks_into_partial(
                pack,
                &chunks,
                hints,
                snapshot,
                concurrency,
                &mut state,
                &mut partial_file,
            )
            .await;
        // Persist progress once at the end — and on error, so a resumed
        // transfer continues from the last appended chunk rather than the last
        // batched save. The fetch's own error wins over a save failure.
        let save_result = self.save_pack_transfer_state(&pack.content_id, &state);
        fetch_result?;
        save_result?;
        partial_file.flush()?;
        drop(partial_file);
        let chunk_ids: Vec<ContentId> = chunks.iter().map(|chunk| chunk.content_id).collect();
        if let Err(err) = self.prefetch_pack_entries_from_file(
            &pack.content_id,
            &partial_path,
            prefetched_objects,
        ) {
            // A fully-fetched `.part` that fails decode is poison, not
            // resumable progress: the completed state passes every resume
            // consistency check (equal length), so without clearing it every
            // future attempt would refetch nothing, re-decode the same bytes,
            // and fail forever. Only a *decode* error clears — a cache-write
            // failure (e.g. disk full, surfaced as `Io`) keeps the good
            // `.part` for a zero-refetch retry.
            if matches!(err, VexClientError::PackDecode(_)) {
                drop(self.clear_pack_transfer_state(&pack.content_id, &chunk_ids));
            }
            return Err(err);
        }
        self.clear_pack_transfer_state(&pack.content_id, &chunk_ids)?;
        Ok(true)
    }

    /// Fetch `chunks[state.next_chunk_index..]` and append them to the pack's
    /// `.part` file, advancing `state` as each chunk lands.
    ///
    /// Index-ordered fetch futures are driven `.buffered(W)`: up to W fetches
    /// run concurrently (each request is a spawned task on the shared runtime,
    /// so awaiting it is a cooperative yield and the overlap survives the pack
    /// worker's plain-thread `block_on`), while `buffered` yields results in
    /// input order — it *is* the reorder buffer. The single writer below
    /// therefore appends strictly in chunk order and the contiguous-prefix
    /// resume invariant of [`PackTransferState`] is untouched.
    #[expect(clippy::too_many_arguments)]
    async fn fetch_chunks_into_partial(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        chunks: &[jj_backend_types::PackChunkDescriptor],
        hints: &[jj_backend_api::PresignedGet],
        snapshot: bool,
        concurrency: usize,
        state: &mut PackTransferState,
        partial_file: &mut File,
    ) -> Result<(), VexClientError> {
        use futures::stream::StreamExt as _;
        let mut fetched =
            futures::stream::iter(chunks.iter().enumerate().skip(state.next_chunk_index).map(
                |(index, chunk)| async move {
                    let bytes = self
                        .fetch_pack_chunk_with_retry(
                            &chunk.content_id,
                            hints,
                            Some(chunk.size_bytes),
                        )
                        .await?;
                    Ok::<_, VexClientError>((index, bytes))
                },
            ))
            .buffered(concurrency.max(1));
        let mut chunks_since_save = 0_usize;
        while let Some(result) = fetched.next().await {
            let (index, chunk_bytes) = result?;
            let chunk = &chunks[index];
            if u64::try_from(chunk_bytes.len()).unwrap_or(u64::MAX) != chunk.size_bytes {
                // Keep the state file for debugging, but restart the next
                // attempt from scratch (the caller persists this state on its
                // way out).
                state.next_chunk_index = 0;
                return Err(VexClientError::PackDecode(format!(
                    "chunk size mismatch for pack {} chunk {}",
                    pack.content_id, index
                )));
            }
            let stats = vex_client_stats();
            stats.pack_chunks_fetched.fetch_add(1, Ordering::Relaxed);
            let bytes_counter = if snapshot {
                &stats.snapshot_pack_bytes
            } else {
                &stats.pack_bytes_fetched
            };
            bytes_counter.fetch_add(chunk_bytes.len() as u64, Ordering::Relaxed);
            partial_file.write_all(&chunk_bytes)?;
            state.next_chunk_index = index + 1;
            chunks_since_save += 1;
            if chunks_since_save >= TRANSFER_STATE_SAVE_INTERVAL {
                self.save_pack_transfer_state(&pack.content_id, state)?;
                chunks_since_save = 0;
            }
        }
        Ok(())
    }

    pub async fn init_repo(
        endpoint: &str,
        tenant_slug: &str,
        repo_slug: &str,
        access_token: Option<&str>,
    ) -> Result<VexRepoConfig, VexClientError> {
        let response = Self::block_on_grpc(endpoint, |mut client| async move {
            client
                .init_repo(Self::auth_request(
                    InitRepoRequest {
                        tenant_slug: tenant_slug.to_string(),
                        repo_slug: repo_slug.to_string(),
                    },
                    access_token,
                )?)
                .await
                .map(|response| response.into_inner())
        })?;
        let repo = response.repo.ok_or(VexConfigError::MissingRepoInfo)?;
        Ok(VexRepoConfig {
            endpoint: endpoint.to_string(),
            tenant_id: repo.tenant_id,
            tenant_slug: repo.tenant_slug,
            repo_id: repo.repo_id,
            repo_slug: repo.repo_slug,
            repository_scope_kind: Some("repository".to_string()),
            virtual_repository_id: None,
            backing_repo_slug: None,
            virtual_root_path: None,
            virtual_mounts: Vec::new(),
            access_token: access_token.map(ToOwned::to_owned),
            local_writes: false,
            object_read_mode: VexObjectReadMode::NativeOnly,
        })
    }

    pub async fn get_repo(
        endpoint: &str,
        tenant_slug: &str,
        repo_slug: &str,
        access_token: Option<&str>,
    ) -> Result<VexRepoConfig, VexClientError> {
        let response = Self::block_on_grpc_retry(endpoint, 5, |mut client| async move {
            client
                .get_repo(Self::auth_request(
                    GetRepoRequest {
                        tenant_slug: tenant_slug.to_string(),
                        repo_slug: repo_slug.to_string(),
                    },
                    access_token,
                )?)
                .await
                .map(|response| response.into_inner())
        })?;
        let repo = response.repo.ok_or(VexConfigError::MissingRepoInfo)?;
        Ok(VexRepoConfig {
            endpoint: endpoint.to_string(),
            tenant_id: repo.tenant_id,
            tenant_slug: repo.tenant_slug,
            repo_id: repo.repo_id,
            repo_slug: repo.repo_slug,
            repository_scope_kind: Some("repository".to_string()),
            virtual_repository_id: None,
            backing_repo_slug: None,
            virtual_root_path: None,
            virtual_mounts: Vec::new(),
            access_token: access_token.map(ToOwned::to_owned),
            local_writes: false,
            object_read_mode: VexObjectReadMode::NativeOnly,
        })
    }

    /// Buffer key for [`PENDING_UPLOADS`]: one entry per repo, shared by every
    /// `VexClient` (backend / op store / op heads store) pointing at it.
    fn pending_key(&self) -> String {
        format!("{}\u{0}{}", self.config.endpoint, self.config.repo_id)
    }

    /// Whether snapshot writes should be buffered and uploaded in one batch
    /// rather than one blocking round trip per object. On by default; set
    /// `VEX_BATCH_SNAPSHOT_UPLOADS=0` (or `false`/`no`) to fall back to
    /// immediate per-object PUTs. Never batches in local-write mode, where puts
    /// already stay local and there is no backend round trip to coalesce.
    fn defer_uploads_enabled(&self) -> bool {
        if self.local_writes {
            return false;
        }
        !matches!(
            std::env::var("VEX_BATCH_SNAPSHOT_UPLOADS").ok().as_deref(),
            Some("0") | Some("false") | Some("no")
        )
    }

    /// Buffer one object for later batched upload. Returns `true` when the
    /// buffer has reached its byte/object cap and the caller should flush. A
    /// no-op (returns `false`) if the object is already buffered.
    fn buffer_pending_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        data: Vec<u8>,
    ) -> bool {
        let map = PENDING_UPLOADS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = map.lock().unwrap();
        let pending = guard.entry(self.pending_key()).or_default();
        if pending.objects.contains_key(&(kind, *content_id)) {
            return false;
        }
        pending.bytes += data.len();
        pending.objects.insert((kind, *content_id), data);
        pending.bytes >= PENDING_FLUSH_BYTES || pending.objects.len() >= PENDING_FLUSH_OBJECTS
    }

    /// Whether `content_id` is already buffered for upload by this process.
    fn has_pending_object(&self, kind: ObjectKind, content_id: &ContentId) -> bool {
        PENDING_UPLOADS
            .get()
            .map(|map| {
                map.lock()
                    .unwrap()
                    .get(&self.pending_key())
                    .is_some_and(|pending| pending.objects.contains_key(&(kind, *content_id)))
            })
            .unwrap_or(false)
    }

    /// Read a buffered-but-not-yet-uploaded object — lets a within-process read
    /// of an object just written this snapshot resolve before it is flushed.
    fn read_pending_object(&self, kind: ObjectKind, content_id: &ContentId) -> Option<Vec<u8>> {
        PENDING_UPLOADS.get().and_then(|map| {
            map.lock()
                .unwrap()
                .get(&self.pending_key())
                .and_then(|pending| pending.objects.get(&(kind, *content_id)).cloned())
        })
    }

    /// Upload every buffered object for this repo in one pipelined set of
    /// `put_objects` batches, then record them in the on-disk cache. Called
    /// before the op-head CAS (and when the in-memory buffer hits its cap) so
    /// every object an operation references is durable on the server first, and
    /// so the cache only ever names objects that are already uploaded.
    ///
    /// This is a blocking call (it drives the shared gRPC runtime) intended for
    /// the same single-threaded executor that drives the other `VexClient`
    /// methods; it must not be invoked from within the shared runtime.
    pub fn flush_pending_uploads(&self) -> Result<(), VexClientError> {
        let drained: Vec<(ObjectKind, ContentId, Vec<u8>)> = {
            let Some(map) = PENDING_UPLOADS.get() else {
                return Ok(());
            };
            let mut guard = map.lock().unwrap();
            match guard.get_mut(&self.pending_key()) {
                Some(pending) if !pending.objects.is_empty() => {
                    pending.bytes = 0;
                    pending
                        .objects
                        .drain()
                        .map(|((kind, id), data)| (kind, id, data))
                        .collect()
                }
                _ => return Ok(()),
            }
        };
        let _t = RpcTimer::start(|| format!("flush_pending_uploads/{}", drained.len()));

        // Split into size/count-bounded batches and upload them concurrently
        // over the one cached connection.
        let mut batches: Vec<Vec<InlineObject>> = Vec::new();
        let mut current: Vec<InlineObject> = Vec::new();
        let mut current_bytes = 0usize;
        for (kind, id, data) in &drained {
            current_bytes += data.len();
            current.push(InlineObject {
                object: Some(ObjectId {
                    kind: kind_to_str(*kind).to_string(),
                    content_id: id.to_string(),
                }),
                data: data.clone(),
            });
            if current.len() >= PENDING_FLUSH_OBJECTS || current_bytes >= PENDING_FLUSH_BYTES {
                batches.push(std::mem::take(&mut current));
                current_bytes = 0;
            }
        }
        if !current.is_empty() {
            batches.push(current);
        }

        let channel = Self::cached_channel(&self.config.endpoint)?;
        let repo_id = self.config.repo_id.clone();
        let token = self.config.access_token.clone();
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            use futures::stream::TryStreamExt as _;
            futures::stream::iter(batches.into_iter().map(Ok::<_, VexClientError>))
                .try_for_each_concurrent(16, |objects| {
                    let channel = channel.clone();
                    let repo_id = repo_id.clone();
                    let token = token.clone();
                    async move {
                        JjBackendClient::new(channel)
                            .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                            .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                            .put_objects(Self::auth_request(
                                PutObjectsRequest { repo_id, objects },
                                token.as_deref(),
                            )?)
                            .await?;
                        Ok(())
                    }
                })
                .await
        }))?;

        // Now — and only now — is "cached ⟹ uploaded" true for these objects.
        for (kind, id, data) in &drained {
            self.write_cached_object(*kind, id, data)?;
        }
        Ok(())
    }

    pub async fn put_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        data: Vec<u8>,
    ) -> Result<(), VexClientError> {
        let _t = RpcTimer::start(|| format!("put_object/{}", kind_to_str(kind)));
        // Local-write mode (READ_ONLY CI runner): persist the object only to the
        // local content-addressed cache and never contact the backend. The clone's
        // editable `@` working-copy commit (+ its tree) and the op-log objects
        // (view/operation/op-head) are written here; reads check the cache before
        // the network (see `get_object`), so they resolve back correctly without
        // requiring Write access to the backend.
        if self.local_writes {
            self.write_cached_object(kind, content_id, &data)?;
            return Ok(());
        }
        // Content-addressed short circuit: if this object is already cached it
        // was already uploaded, so skip the round trip. This is the hot path
        // during working-copy snapshots (`vex status`), where unchanged or
        // recurring blob/tree/commit content would otherwise be re-PUT.
        if self.has_cached_object(kind, content_id) {
            return Ok(());
        }
        // Snapshot batching: buffer the object instead of uploading it inline,
        // so the blob/tree/commit/op/view chain a snapshot writes is published
        // in one pipelined `put_objects` batch at the op-head CAS rather than
        // one blocking round trip each (see [`PENDING_UPLOADS`]). Already-buffered
        // objects are deduplicated by `buffer_pending_object`.
        if self.defer_uploads_enabled() {
            if !self.has_pending_object(kind, content_id) {
                let over_cap = self.buffer_pending_object(kind, content_id, data);
                if over_cap {
                    self.flush_pending_uploads()?;
                }
            }
            return Ok(());
        }
        let cache_bytes = data.clone();
        Self::block_on_grpc(&self.config.endpoint, |mut client| async move {
            client
                .put_object(Self::auth_request(
                    PutObjectRequest {
                        repo_id: self.config.repo_id.clone(),
                        object: Some(ObjectId {
                            kind: kind_to_str(kind).to_string(),
                            content_id: content_id.to_string(),
                        }),
                        data,
                    },
                    self.config.access_token.as_deref(),
                )?)
                .await
                .map(|_| ())
        })?;
        self.write_cached_object(kind, content_id, &cache_bytes)?;
        Ok(())
    }

    /// Content id of a blob (file) object: the SHA-256 of its bytes. Matches the
    /// id [`crate::vex_backend::VexBackend`] would assign, so callers can
    /// pre-compute blob ids for bulk upload without a round trip.
    pub fn blob_content_id(data: &[u8]) -> ContentId {
        use sha2::Digest as _;
        let mut hasher = sha2::Sha256::new();
        hasher.update(data);
        let digest: [u8; 32] = hasher.finalize().into();
        ContentId::from_bytes(digest)
    }

    /// Upload many already-addressed objects in a single batched RPC. The caller
    /// is responsible for chunking so each call stays under the server's gRPC
    /// message size limit. Skips the local object cache (intended for bulk
    /// import where the objects are not needed locally afterwards).
    pub async fn put_objects(
        &self,
        objects: Vec<(ObjectKind, ContentId, Vec<u8>)>,
    ) -> Result<(), VexClientError> {
        let _t = RpcTimer::start(|| format!("put_objects[{}]", objects.len()));
        if objects.is_empty() {
            return Ok(());
        }
        let inline: Vec<InlineObject> = objects
            .into_iter()
            .map(|(kind, content_id, data)| InlineObject {
                object: Some(ObjectId {
                    kind: kind_to_str(kind).to_string(),
                    content_id: content_id.to_string(),
                }),
                data,
            })
            .collect();
        Self::block_on_grpc(&self.config.endpoint, |client| async move {
            client
                .max_encoding_message_size(64 * 1024 * 1024)
                .put_objects(Self::auth_request(
                    PutObjectsRequest {
                        repo_id: self.config.repo_id.clone(),
                        objects: inline,
                    },
                    self.config.access_token.as_deref(),
                )?)
                .await
                .map(|_| ())
        })?;
        Ok(())
    }

    /// Bulk-upload file blobs, returning their backend [`crate::backend::FileId`]s
    /// in the same order. Ids are computed locally (SHA-256), so this avoids a
    /// per-file round trip; the caller should chunk to stay under the gRPC
    /// message size limit.
    pub async fn put_file_blobs(
        &self,
        blobs: Vec<Vec<u8>>,
    ) -> Result<Vec<crate::backend::FileId>, VexClientError> {
        let mut objects = Vec::with_capacity(blobs.len());
        let mut ids = Vec::with_capacity(blobs.len());
        for data in blobs {
            let content_id = Self::blob_content_id(&data);
            ids.push(crate::backend::FileId::new(content_id.as_bytes().to_vec()));
            objects.push((ObjectKind::Blob, content_id, data));
        }
        self.put_objects(objects).await?;
        Ok(ids)
    }

    /// Upload many object batches with bounded request pipelining: up to
    /// `concurrency` `put_objects` RPCs are in flight at once over the shared
    /// cached connection, overlapping their round trips. This is the key win
    /// for bulk ingestion from a single-threaded (pollster) caller: the plain
    /// per-batch `put_objects` blocks the calling thread on the shared runtime,
    /// so successive batches cannot overlap; here all batches are driven inside
    /// one `block_on`, so the runtime can keep several requests in flight.
    pub async fn put_object_batches_pipelined(
        &self,
        batches: Vec<Vec<(ObjectKind, ContentId, Vec<u8>)>>,
        concurrency: usize,
    ) -> Result<(), VexClientError> {
        let inline_batches: Vec<Vec<InlineObject>> = batches
            .into_iter()
            .filter(|batch| !batch.is_empty())
            .map(|batch| {
                batch
                    .into_iter()
                    .map(|(kind, content_id, data)| InlineObject {
                        object: Some(ObjectId {
                            kind: kind_to_str(kind).to_string(),
                            content_id: content_id.to_string(),
                        }),
                        data,
                    })
                    .collect()
            })
            .collect();
        if inline_batches.is_empty() {
            return Ok(());
        }
        // Keep the normal fast path on the shared HTTP/2 connection. If it is
        // unavailable while being established, let each batch's retry helper
        // make a fresh connection instead of failing the entire import before
        // its first request.
        let initial_channel = match Self::cached_channel(&self.config.endpoint) {
            Ok(channel) => Some(channel),
            Err(err) if Self::is_transient_pipelined_put_error(&err) => {
                debug!(error = %err, "cached PutObjects channel unavailable; reconnecting per batch");
                None
            }
            Err(err) => return Err(err),
        };
        let endpoint = self.config.endpoint.clone();
        let repo_id = self.config.repo_id.clone();
        let token = self.config.access_token.clone();
        let concurrency = concurrency.max(1);
        Self::shared_grpc_runtime().block_on(async move {
            use futures::stream::TryStreamExt as _;
            futures::stream::iter(inline_batches.into_iter().map(Ok::<_, VexClientError>))
                .try_for_each_concurrent(concurrency, |objects| {
                    let initial_channel = initial_channel.clone();
                    let endpoint = endpoint.clone();
                    let repo_id = repo_id.clone();
                    let token = token.clone();
                    async move {
                        // Keep `objects` intact so a transient connection
                        // reset after the server accepted the body can replay
                        // the exact idempotent request.
                        Self::retry_pipelined_put_batch(move |use_initial_channel| {
                            let endpoint = endpoint.clone();
                            let initial_channel = initial_channel.clone();
                            let repo_id = repo_id.clone();
                            let token = token.clone();
                            let objects = objects.clone();
                            async move {
                                let channel = match (use_initial_channel, initial_channel) {
                                    (true, Some(channel)) => channel,
                                    _ => Self::endpoint(&endpoint)?.connect().await?,
                                };
                                JjBackendClient::new(channel)
                                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                                    .put_objects(Self::auth_request(
                                        PutObjectsRequest { repo_id, objects },
                                        token.as_deref(),
                                    )?)
                                    .await?;
                                Ok(())
                            }
                        })
                        .await
                    }
                })
                .await
        })
    }

    /// Like [`Self::put_file_blobs`], but batches by object count/byte size and
    /// uploads the batches with bounded request pipelining (see
    /// [`Self::put_object_batches_pipelined`]). Returns the destination file ids
    /// in input order. Computing ids is local, so the mapping is known without
    /// waiting for the uploads.
    pub async fn put_file_blobs_pipelined(
        &self,
        blobs: Vec<Vec<u8>>,
        max_batch_objects: usize,
        max_batch_bytes: usize,
        concurrency: usize,
    ) -> Result<Vec<crate::backend::FileId>, VexClientError> {
        let mut ids = Vec::with_capacity(blobs.len());
        let mut batches: Vec<Vec<(ObjectKind, ContentId, Vec<u8>)>> = Vec::new();
        let mut current: Vec<(ObjectKind, ContentId, Vec<u8>)> = Vec::new();
        let mut current_bytes = 0usize;
        let max_objects = max_batch_objects.max(1);
        for data in blobs {
            let content_id = Self::blob_content_id(&data);
            ids.push(crate::backend::FileId::new(content_id.as_bytes().to_vec()));
            current_bytes += data.len();
            current.push((ObjectKind::Blob, content_id, data));
            if current.len() >= max_objects || current_bytes >= max_batch_bytes {
                batches.push(std::mem::take(&mut current));
                current_bytes = 0;
            }
        }
        if !current.is_empty() {
            batches.push(current);
        }
        self.put_object_batches_pipelined(batches, concurrency)
            .await?;
        Ok(ids)
    }

    /// Bulk-upload pre-serialized tree objects (canonical bytes). Ids are
    /// derived from the bytes, matching the backend's content addressing.
    pub async fn put_tree_blobs(&self, blobs: Vec<Vec<u8>>) -> Result<(), VexClientError> {
        let objects = blobs
            .into_iter()
            .map(|data| {
                let id = Self::blob_content_id(&data);
                (ObjectKind::Tree, id, data)
            })
            .collect();
        self.put_objects(objects).await
    }

    /// Bulk-upload pre-serialized commit objects (canonical bytes).
    pub async fn put_commit_blobs(&self, blobs: Vec<Vec<u8>>) -> Result<(), VexClientError> {
        let objects = blobs
            .into_iter()
            .map(|data| {
                let id = Self::blob_content_id(&data);
                (ObjectKind::Commit, id, data)
            })
            .collect();
        self.put_objects(objects).await
    }

    pub async fn get_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Result<Vec<u8>, VexClientError> {
        let _t = RpcTimer::start(|| format!("get_object/{}", kind_to_str(kind)));
        if let Some(bytes) = self.read_cached_object(kind, content_id) {
            vex_client_stats().record_get_object_cache_hit(kind);
            return Ok(bytes);
        }
        // An object written earlier this process may still be buffered for batch
        // upload (not yet on disk or the server); serve it from the buffer.
        if let Some(bytes) = self.read_pending_object(kind, content_id) {
            vex_client_stats().record_get_object_cache_hit(kind);
            return Ok(bytes);
        }
        let bytes = self.fetch_object_grpc_verified(kind, content_id).await?;
        // A cache hit is assumed present-on-server (see `has_cached_object`)
        // and is never re-verified; the fetch above hash-verified the bytes,
        // so they may enter the cache.
        self.write_cached_object(kind, content_id, &bytes)?;
        Ok(bytes)
    }

    /// gRPC `GetObject` plus content-hash verification, *without* touching the
    /// local cache. [`Self::get_object`] layers the cache on top; pack-chunk
    /// fallback reads use this directly, since chunk bytes belong in the
    /// transfer's `.part` file, not the loose cache.
    async fn fetch_object_grpc_verified(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Result<Vec<u8>, VexClientError> {
        debug!(kind = kind_to_str(kind), %content_id, "vex cache miss");
        vex_client_stats().record_get_object_rpc(kind);
        // Own every captured value so the fetch future is `Send + 'static` and can
        // be spawned onto the shared runtime. This is what lets `check_out`'s
        // `.buffered(concurrency())` actually run reads in parallel instead of
        // serializing them behind a per-object `block_on` (see `grpc_retry_async`).
        let repo_id = self.config.repo_id.clone();
        let access_token = self.config.access_token.clone();
        let kind_str = kind_to_str(kind).to_string();
        let content_id_str = content_id.to_string();
        let bytes = Self::grpc_retry_async(&self.config.endpoint, 5, move |mut client| {
            let repo_id = repo_id.clone();
            let access_token = access_token.clone();
            let kind_str = kind_str.clone();
            let content_id_str = content_id_str.clone();
            async move {
                client
                    .get_object(Self::auth_request(
                        GetObjectRequest {
                            repo_id,
                            object: Some(ObjectId {
                                kind: kind_str,
                                content_id: content_id_str,
                            }),
                        },
                        access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner().data)
            }
        })
        .await?;
        // Verify content addressing before the bytes are used anywhere: a
        // cache hit is assumed present-on-server (see `has_cached_object`) and
        // is never re-verified, so nothing unverified may be written. This
        // also keeps `hydrate_one_batch` honest — an inline object that failed
        // its hash check is refetched through here and must not slip into the
        // cache unchecked on the second try.
        if ContentId::hash_bytes(&bytes) != *content_id {
            return Err(VexClientError::Status(tonic::Status::data_loss(format!(
                "object {}/{content_id} failed hash verification",
                kind_to_str(kind),
            ))));
        }
        Ok(bytes)
    }

    /// Bulk-fetch objects into the local cache via batched `GetObjectsInline`
    /// RPCs — the read-side analogue of [`Self::put_object_batches_pipelined`],
    /// used to pre-hydrate a lazy clone's file/symlink contents before checkout.
    ///
    /// `ids` are `(kind, content_id, estimated_size)` triples; the optional
    /// size (unknown from jj tree entries, known from manifest descriptors) only
    /// tightens batch splitting. Already-cached and duplicate ids are skipped.
    /// Batches are bounded by [`INLINE_FETCH_BATCH_OBJECTS`] /
    /// [`INLINE_FETCH_BATCH_BYTES`] and run [`INLINE_FETCH_CONCURRENCY`]-wide
    /// via the `grpc_retry_async` spawned-task pattern, so the single-threaded
    /// clone executor keeps several requests in flight. Response objects are
    /// verified (kind + SHA-256) before entering the cache; ids the response
    /// omits fall back to per-object [`Self::get_object`]. The per-write cache
    /// prune is skipped and run once at the end.
    ///
    /// Emits [`CloneProgress::Hydrating`] as batches complete and returns the
    /// number of objects fetched.
    pub async fn get_objects_inline_batched(
        &self,
        ids: Vec<(ObjectKind, ContentId, Option<u64>)>,
        progress: Option<&CloneProgressFn>,
    ) -> Result<u64, VexClientError> {
        // Dedupe and drop objects already in the local cache.
        let mut seen: HashSet<(ObjectKind, ContentId)> = HashSet::new();
        let to_fetch: Vec<(ObjectKind, ContentId, Option<u64>)> = ids
            .into_iter()
            .filter(|(kind, content_id, _)| {
                seen.insert((*kind, *content_id)) && !self.has_cached_object(*kind, content_id)
            })
            .collect();
        let total = to_fetch.len() as u64;
        if let Some(progress) = progress {
            progress(CloneProgress::Hydrating { done: 0, total });
        }
        if to_fetch.is_empty() {
            return Ok(0);
        }
        let _t = RpcTimer::start(|| format!("get_objects_inline_batched[{total}]"));
        let batches = split_inline_fetch_batches(
            to_fetch,
            INLINE_FETCH_BATCH_OBJECTS,
            INLINE_FETCH_BATCH_BYTES,
        );
        use futures::stream::StreamExt as _;
        let mut results = futures::stream::iter(
            batches
                .into_iter()
                .map(|batch| self.hydrate_one_batch(batch)),
        )
        .buffer_unordered(INLINE_FETCH_CONCURRENCY);
        let mut done = 0_u64;
        let mut first_err: Option<VexClientError> = None;
        while let Some(result) = results.next().await {
            match result {
                Ok(count) => {
                    done += count;
                    if let Some(progress) = progress {
                        progress(CloneProgress::Hydrating { done, total });
                    }
                }
                Err(err) => {
                    first_err = Some(err);
                    break;
                }
            }
        }
        drop(results);
        // The batch writes above bypass the per-write prune; settle the cache
        // size once now.
        self.prune_cache_if_needed()?;
        match first_err {
            Some(err) => Err(err),
            None => Ok(done),
        }
    }

    /// Fetch one `GetObjectsInline` batch, verify and cache its objects, and
    /// fetch whatever the response omitted (or failed verification) one object
    /// at a time. Returns the number of objects hydrated (== `batch.len()` on
    /// success).
    async fn hydrate_one_batch(
        &self,
        batch: Vec<(ObjectKind, ContentId)>,
    ) -> Result<u64, VexClientError> {
        let stats = vex_client_stats();
        let mut remaining: HashSet<(ObjectKind, ContentId)> = batch.iter().copied().collect();
        match self.fetch_inline_batch(&batch).await {
            Ok(objects) => {
                for inline in objects {
                    let Some(object) = inline.object else {
                        continue;
                    };
                    let Some(kind) = kind_from_str(&object.kind) else {
                        continue;
                    };
                    let Ok(content_id) = ContentId::from_hex(&object.content_id) else {
                        continue;
                    };
                    if !remaining.contains(&(kind, content_id)) {
                        continue;
                    }
                    // Verify content addressing before the bytes enter the
                    // cache: a cached object is assumed present on the server
                    // (see `put_object`), so nothing unverified may be written.
                    if ContentId::hash_bytes(&inline.data) != content_id {
                        debug!(kind = kind_to_str(kind), %content_id, "inline object failed hash verification; refetching individually");
                        continue;
                    }
                    self.write_cached_object_no_prune(kind, &content_id, &inline.data)?;
                    stats.objects_inline_fetched.fetch_add(1, Ordering::Relaxed);
                    stats
                        .hydrated_bytes
                        .fetch_add(inline.data.len() as u64, Ordering::Relaxed);
                    remaining.remove(&(kind, content_id));
                }
            }
            // The response overflowed a gRPC message-size cap (object sizes
            // are usually unknown at batch-split time, so the byte bound can't
            // prevent this). Bisect and retry as two inline batches instead of
            // collapsing to `batch.len()` sequential per-object RPCs.
            Err(VexClientError::Status(status))
                if status.code() == tonic::Code::OutOfRange && batch.len() > 1 =>
            {
                debug!(
                    batch_objects = batch.len(),
                    "inline batch overflowed the gRPC message cap; bisecting"
                );
                let (left, right) = batch.split_at(batch.len() / 2);
                let count = Box::pin(self.hydrate_one_batch(left.to_vec())).await?
                    + Box::pin(self.hydrate_one_batch(right.to_vec())).await?;
                return Ok(count);
            }
            Err(err) => {
                debug!(
                    error = %err,
                    batch_objects = batch.len(),
                    "inline batch fetch failed; falling back to per-object reads"
                );
            }
        }
        // The response silently omits objects the server doesn't hold (and we
        // skip any that failed verification); fetch those individually.
        for (kind, content_id) in &batch {
            if !remaining.contains(&(*kind, *content_id)) {
                continue;
            }
            let bytes = match self.get_object(*kind, content_id).await {
                Ok(bytes) => bytes,
                // Legacy repos store some symlink targets as blobs (see
                // `VexBackend::read_symlink`); mirror its fallback rather than
                // aborting the whole hydration on one NotFound.
                Err(VexClientError::Status(status))
                    if status.code() == tonic::Code::NotFound && *kind == ObjectKind::Symlink =>
                {
                    self.get_object(ObjectKind::Blob, content_id).await?
                }
                Err(err) => return Err(err),
            };
            stats
                .hydrated_bytes
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        }
        stats
            .hydrated_objects
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        Ok(batch.len() as u64)
    }

    /// Issue one `GetObjectsInline` RPC for `batch` on the shared runtime
    /// (spawned + awaited, so concurrent batches genuinely overlap). Absent
    /// objects are omitted from the response; the caller diffs and falls back.
    async fn fetch_inline_batch(
        &self,
        batch: &[(ObjectKind, ContentId)],
    ) -> Result<Vec<InlineObject>, VexClientError> {
        vex_client_stats()
            .inline_batches
            .fetch_add(1, Ordering::Relaxed);
        let repo_id = self.config.repo_id.clone();
        let access_token = self.config.access_token.clone();
        let object_ids: Vec<ObjectId> = batch
            .iter()
            .map(|(kind, content_id)| ObjectId {
                kind: kind_to_str(*kind).to_string(),
                content_id: content_id.to_string(),
            })
            .collect();
        Self::grpc_retry_async(&self.config.endpoint, 5, move |mut client| {
            let repo_id = repo_id.clone();
            let access_token = access_token.clone();
            let objects = object_ids.clone();
            async move {
                client
                    .get_objects_inline(Self::auth_request(
                        GetObjectsInlineRequest { repo_id, objects },
                        access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner().objects)
            }
        })
        .await
    }

    pub async fn get_op_heads(&self) -> Result<Vec<ContentId>, VexClientError> {
        // Always read op heads live from the server. A client-side cache is
        // unsafe here: jj records the working-copy operation locally *before* the
        // commit's server-side CAS runs, so serving a stale head lets jj build a
        // working-copy op on it; when the CAS then rejects the stale head the
        // working copy is left pinned to an orphan op, diverging from the backend
        // head (a "sibling operation" that blocks all further commands).
        let _t = RpcTimer::start(|| "get_op_heads".to_string());
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .get_op_heads(Self::auth_request(
                        jj_backend_api::GetOpHeadsRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        let ids = response
            .op_content_ids
            .into_iter()
            .map(|id| {
                ContentId::from_hex(&id).map_err(|err| {
                    tonic::Status::internal(format!("invalid op head from server: {err}"))
                })
            })
            .collect::<Result<Vec<_>, tonic::Status>>()?;
        Ok(ids)
    }

    pub async fn commit_op_heads(
        &self,
        expected: &[ContentId],
        new_head: &ContentId,
        new_view: &ContentId,
    ) -> Result<jj_backend_api::CommitOperationResponse, VexClientError> {
        let _t = RpcTimer::start(|| "commit_op_heads".to_string());
        // Publish every object buffered this process before advancing the op
        // head, so the operation the CAS installs never references an object
        // that is missing on the server. A flush failure aborts here, leaving
        // the head unchanged (the un-uploaded objects are simply unreferenced).
        self.flush_pending_uploads()?;
        let response = Self::block_on_commit_operation_maintenance_retry(
            &self.config.endpoint,
            |mut client| async move {
                client
                    .commit_operation(Self::auth_request(
                        jj_backend_api::CommitOperationRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            expected_op_head_ids: expected
                                .iter()
                                .map(ToString::to_string)
                                .collect(),
                            new_op_content_id: new_head.to_string(),
                            new_view_content_id: new_view.to_string(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            },
        )?;
        Ok(response)
    }

    pub async fn resolve_operation_id_prefix(
        &self,
        prefix: &str,
    ) -> Result<Vec<ContentId>, VexClientError> {
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .resolve_operation_id_prefix(Self::auth_request(
                        ResolveOperationIdPrefixRequest {
                            repo_id: self.config.repo_id.clone(),
                            prefix: prefix.to_string(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        response
            .matches
            .into_iter()
            .map(|id| {
                ContentId::from_hex(&id).map_err(|err| {
                    tonic::Status::internal(format!("invalid operation id from server: {err}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    pub async fn resolve_ref(&self, name: &str) -> Result<Option<String>, VexClientError> {
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .resolve_refs(Self::auth_request(
                        ResolveRefsRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            names: vec![name.to_string()],
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        Ok(response.refs.into_iter().next().map(|r| r.target_commit_id))
    }

    /// Resolve many refs by exact name in one round trip; names with no
    /// stored ref are simply absent from the result. Used for batched
    /// mapping-ref lookups (e.g. materialization git<->native identity maps)
    /// where a `resolve_ref`-per-name loop would be one network round trip
    /// per row.
    pub async fn resolve_refs(
        &self,
        names: &[String],
    ) -> Result<Vec<jj_backend_api::RefValue>, VexClientError> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .resolve_refs(Self::auth_request(
                        ResolveRefsRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            names: names.to_vec(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        Ok(response.refs)
    }

    /// List every ref whose name starts with `prefix` in one round trip. The
    /// backend returns the whole matching set unpaginated, so this is only
    /// suitable for namespaces of bounded size (e.g. a materialization
    /// identity-mapping namespace of tens of thousands of rows) — not for
    /// unbounded namespaces like `refs/heads/`.
    pub async fn list_refs(
        &self,
        prefix: &str,
    ) -> Result<Vec<jj_backend_api::RefValue>, VexClientError> {
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .list_refs(Self::auth_request(
                        jj_backend_api::ListRefsRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            prefix: prefix.to_string(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        Ok(response.refs)
    }

    /// Apply a batch of ref writes. Each update's `expected_version: None`
    /// inserts a brand-new ref (rejected if one already exists), while
    /// `Some(version)` CAS-updates an existing ref. Not retried internally:
    /// callers of batched, content-addressed writes (e.g. materialization
    /// mapping refs) should pre-filter to only the updates that are actually
    /// needed (see `materialize_mapping::plan_chunk_writes` in vex-cli) so a
    /// retry at the call site is safe to re-send verbatim. Returns an error
    /// if the backend rejects the batch (CAS conflict or validation failure);
    /// the message is the backend's `error_message`.
    pub async fn update_refs(
        &self,
        updates: Vec<jj_backend_api::RefUpdate>,
    ) -> Result<(), VexClientError> {
        if updates.is_empty() {
            return Ok(());
        }
        let response = Self::block_on_grpc(&self.config.endpoint, |mut client| async move {
            client
                .update_refs(Self::auth_request(
                    jj_backend_api::UpdateRefsRequest {
                        tenant_id: self.config.tenant_id.clone(),
                        repo_id: self.config.repo_id.clone(),
                        updates,
                        policy_lease: String::new(),
                    },
                    self.config.access_token.as_deref(),
                )?)
                .await
                .map(|response| response.into_inner())
        })?;
        if response.ok {
            Ok(())
        } else {
            Err(VexClientError::RefUpdateRejected(response.error_message))
        }
    }

    pub async fn get_clone_manifest(
        &self,
        blob_mode: CloneBlobMode,
        extra_have_snapshot_commit_ids: &[String],
        progress: Option<&CloneProgressFn>,
    ) -> Result<CloneManifest, VexClientError> {
        let clone_view_kind = proto_clone_view_kind(self.config.repository_scope_kind.as_deref());
        let virtual_mounts: Vec<jj_backend_api::VirtualRepositoryMount> = self
            .config
            .virtual_mounts
            .iter()
            .map(proto_virtual_repository_mount)
            .collect();
        // Snapshot negotiation (roadmap/032): declare every snapshot set we
        // already hold fully unpacked (shared-cache `.snapshots` markers) plus
        // any caller-supplied haves, so the server trims the served
        // `snapshot_packs` chain to the delta above them. Unknown ids are
        // ignored server-side (full chain returned), so a stale marker is
        // harmless here.
        let have_snapshot_commit_ids = assemble_snapshot_haves(
            self.cached_snapshot_commit_ids(),
            extra_have_snapshot_commit_ids,
        );
        // Building a clone manifest for a large repo can take minutes (it packs
        // tens of thousands of objects). We send `accept_pending = true` so the
        // server returns `building = true` immediately on a cache miss (and warms
        // in the background) instead of holding one RPC open past the
        // client/edge-proxy timeout. We then poll until the manifest is ready.
        // Each poll is itself transient-retryable via `block_on_grpc_retry`, so a
        // backend restart mid-wait is ridden through rather than fatal.
        let poll = std::time::Duration::from_millis(
            env_secs("VEX_CLONE_MANIFEST_POLL_MS", 3_000).max(500),
        );
        let max_wait =
            std::time::Duration::from_secs(env_secs("VEX_CLONE_MANIFEST_MAX_WAIT_SECS", 1_800));
        let started = std::time::Instant::now();
        loop {
            if started.elapsed() >= max_wait {
                return Err(tonic::Status::deadline_exceeded(format!(
                    "clone manifest not ready after {}s",
                    max_wait.as_secs()
                ))
                .into());
            }
            let virtual_mounts = virtual_mounts.clone();
            let have_snapshot_commit_ids = have_snapshot_commit_ids.clone();
            // One (non-retrying) attempt per iteration; the loop itself rides
            // both a still-`building` manifest *and* transient backend errors up
            // to `max_wait`, reporting each through `progress` so a slow/cold
            // first clone shows exactly what it is waiting on instead of a silent
            // 0%.
            let attempt = Self::block_on_grpc(&self.config.endpoint, |mut client| {
                let virtual_mounts = virtual_mounts.clone();
                let have_snapshot_commit_ids = have_snapshot_commit_ids.clone();
                async move {
                    client
                        .get_clone_manifest(Self::auth_request(
                            GetCloneManifestRequest {
                                tenant_id: self.config.tenant_id.clone(),
                                repo_id: self.config.repo_id.clone(),
                                clone_blob_mode: match blob_mode {
                                    CloneBlobMode::Eager => ProtoCloneBlobMode::Eager as i32,
                                    CloneBlobMode::Lazy => ProtoCloneBlobMode::Lazy as i32,
                                },
                                clone_view_kind: clone_view_kind as i32,
                                virtual_root_path: self
                                    .config
                                    .virtual_root_path
                                    .clone()
                                    .unwrap_or_default(),
                                virtual_mounts,
                                accept_pending: true,
                                have_snapshot_commit_ids,
                            },
                            self.config.access_token.as_deref(),
                        )?)
                        .await
                        .map(|response| response.into_inner())
                }
            });
            match attempt {
                Ok(response) if response.building => {
                    if let Some(progress) = progress {
                        progress(CloneProgress::ManifestBuilding {
                            waited_secs: started.elapsed().as_secs(),
                        });
                    }
                    // Blocking sleep matches this module's sync-over-async bridge
                    // (the RPC above already blocks the calling thread).
                    std::thread::sleep(poll);
                    continue;
                }
                Ok(response) => {
                    return serde_json::from_slice(&response.manifest_json)
                        .map_err(VexConfigError::Json)
                        .map_err(Into::into);
                }
                // Transient blip (edge 502, backend restart, deadline): surface
                // it and keep polling rather than aborting the clone.
                Err(err) if Self::is_transient_client_error(&err) => {
                    if let Some(progress) = progress {
                        progress(CloneProgress::Retrying {
                            operation: "clone manifest".to_string(),
                            message: err.to_string(),
                        });
                    }
                    std::thread::sleep(poll);
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn get_object_fetch_hints(
        &self,
        objects: &[(ObjectKind, ContentId)],
    ) -> Result<Vec<jj_backend_api::PresignedGet>, VexClientError> {
        let _t = RpcTimer::start(|| format!("get_object_fetch_hints[{}]", objects.len()));
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .get_objects(Self::auth_request(
                        GetObjectsRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            objects: objects
                                .iter()
                                .map(|(kind, content_id)| ObjectId {
                                    kind: kind_to_str(*kind).to_string(),
                                    content_id: content_id.to_string(),
                                })
                                .collect(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        Ok(response.get_instructions)
    }

    pub async fn prefetch_clone_manifest(
        &self,
        manifest: &CloneManifest,
        fetch_snapshot_packs: bool,
        progress: Option<&CloneProgressFn>,
    ) -> Result<(), VexClientError> {
        let result = self
            .prefetch_clone_manifest_impl(manifest, fetch_snapshot_packs, progress)
            .await;
        // The pack-unpack and loose-object writes in the impl all bypass the
        // per-write prune (quadratic during a bulk unpack); settle the cache
        // size once now — even when the prefetch failed partway through its
        // writes, so an aborted clone can't leave a capped cache over budget.
        // The prefetch's own error wins over a prune failure.
        result.and(self.prune_cache_if_needed())
    }

    async fn prefetch_clone_manifest_impl(
        &self,
        manifest: &CloneManifest,
        fetch_snapshot_packs: bool,
        progress: Option<&CloneProgressFn>,
    ) -> Result<(), VexClientError> {
        let prefetch_started = std::time::Instant::now();
        let prefetched_objects = AtomicU64::new(0);

        // Snapshot sets to consume (roadmap/032): the manifest's trunk
        // snapshot chain, minus sets already fully unpacked here (marker
        // present). Only meaningful with a cache to unpack into, and only when
        // the caller wants blobs at all — lazy local clones pass `true`;
        // virtual working copies and eager clones pass `false`.
        // `VEX_CLONE_SNAPSHOT_PACKS=0` is the kill switch.
        let snapshot_sets: Vec<&SnapshotPackSet> =
            if fetch_snapshot_packs && snapshot_packs_client_enabled() && self.cache_root.is_some()
            {
                manifest
                    .snapshot_packs
                    .iter()
                    .filter(|set| !self.has_unpacked_snapshot(&set.commit_id.to_string()))
                    .collect()
            } else {
                Vec::new()
            };
        // Flatten to (set index, pack) so per-set completeness is recoverable
        // after the parallel fetch. Zero-pack sets ("head alias" recorded for
        // a tree-identical trunk advance) contribute nothing here and are
        // marked purely off their base's completeness in
        // `record_snapshot_markers`.
        let snapshot_pack_refs: Vec<(usize, &jj_backend_types::PackDescriptor)> = snapshot_sets
            .iter()
            .enumerate()
            .flat_map(|(set_idx, set)| set.packs.iter().map(move |pack| (set_idx, pack)))
            .collect();

        let hinted_pack_ids = manifest
            .packs
            .iter()
            .chain(snapshot_pack_refs.iter().map(|(_, pack)| *pack))
            .flat_map(|pack| {
                std::iter::once(pack.content_id)
                    .chain(pack.chunks.iter().map(|chunk| chunk.content_id))
                    .collect::<Vec<_>>()
            })
            .collect::<HashSet<_>>();
        let pack_hints = self
            .get_object_fetch_hints(
                &hinted_pack_ids
                    .into_iter()
                    .map(|content_id| (ObjectKind::Pack, content_id))
                    .collect::<Vec<_>>(),
            )
            .await?;

        // One continuous `PackFetched` sequence across the metadata and
        // snapshot phases, so progress totals stay monotonic.
        let total_packs = (manifest.packs.len() + snapshot_pack_refs.len()) as u64;
        let packs_done = AtomicU64::new(0);

        // Metadata packs: any failure fails the clone (the jj state is
        // unusable without them).
        let metadata_packs: Vec<&jj_backend_types::PackDescriptor> =
            manifest.packs.iter().collect();
        for result in self
            .prefetch_packs_parallel(
                &metadata_packs,
                &pack_hints,
                false,
                &prefetched_objects,
                &packs_done,
                total_packs,
                progress,
            )
            .into_iter()
            .flatten()
        {
            result?;
        }

        // Snapshot packs: best-effort — checkout falls back to Stage-1
        // hydration / per-file reads for anything missing, so a failure here
        // only costs speed, never the clone.
        if !snapshot_sets.is_empty() {
            let packs: Vec<&jj_backend_types::PackDescriptor> =
                snapshot_pack_refs.iter().map(|(_, pack)| *pack).collect();
            let results = self.prefetch_packs_parallel(
                &packs,
                &pack_hints,
                true,
                &prefetched_objects,
                &packs_done,
                total_packs,
                progress,
            );
            let mut fetched_ok: HashMap<String, bool> = snapshot_sets
                .iter()
                .map(|set| (set.commit_id.to_string(), true))
                .collect();
            for ((set_idx, pack), result) in snapshot_pack_refs.iter().zip(results) {
                match result {
                    Some(Ok(())) => {}
                    Some(Err(err)) => {
                        tracing::warn!(
                            pack_content_id = %pack.content_id,
                            snapshot_commit_id = %snapshot_sets[*set_idx].commit_id,
                            // Redacted: the error may embed a signed URL.
                            error = %redact_url_queries(&err.to_string()),
                            "snapshot pack fetch failed; continuing without it"
                        );
                        fetched_ok.insert(snapshot_sets[*set_idx].commit_id.to_string(), false);
                    }
                    // Never started (an earlier pack failed): incomplete set.
                    None => {
                        fetched_ok.insert(snapshot_sets[*set_idx].commit_id.to_string(), false);
                    }
                }
            }
            self.record_snapshot_markers(&manifest.snapshot_packs, &fetched_ok);
        }

        let total_loose = manifest.objects.len() as u64;
        let mut loose_done = 0_u64;
        for object in &manifest.objects {
            loose_done += 1;
            if self
                .read_cached_object(object.kind, &object.content_id)
                .is_some()
            {
                vex_client_stats().record_get_object_cache_hit(object.kind);
                if let Some(progress) = progress {
                    progress(CloneProgress::LooseObjectFetched {
                        done: loose_done,
                        total: total_loose,
                    });
                }
                continue;
            }
            vex_client_stats().record_get_object_rpc(object.kind);
            let bytes =
                Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                    client
                        .get_object(Self::auth_request(
                            GetObjectRequest {
                                repo_id: self.config.repo_id.clone(),
                                object: Some(ObjectId {
                                    kind: kind_to_str(object.kind).to_string(),
                                    content_id: object.content_id.to_string(),
                                }),
                            },
                            self.config.access_token.as_deref(),
                        )?)
                        .await
                        .map(|response| response.into_inner().data)
                })?;
            // Bulk write: the whole prefetch prunes once at the end instead of
            // rescanning the cache per object.
            self.write_cached_object_no_prune(object.kind, &object.content_id, &bytes)?;
            prefetched_objects.fetch_add(1, Ordering::Relaxed);
            if let Some(progress) = progress {
                progress(CloneProgress::LooseObjectFetched {
                    done: loose_done,
                    total: total_loose,
                });
            }
        }
        debug!(
            repo_id = %self.config.repo_id,
            blob_mode = ?manifest.blob_mode,
            pack_count = manifest.packs.len(),
            snapshot_set_count = manifest.snapshot_packs.len(),
            snapshot_sets_fetched = snapshot_sets.len(),
            snapshot_packs_fetched = snapshot_pack_refs.len(),
            deferred_object_count = manifest.deferred_object_count,
            deferred_object_bytes = manifest.deferred_object_bytes,
            prefetched_objects = prefetched_objects.load(Ordering::Relaxed),
            elapsed_ms = prefetch_started.elapsed().as_millis(),
            "prefetched clone manifest"
        );
        Ok(())
    }

    /// Fetch and unpack `packs` with bounded parallelism (default
    /// [`PACK_FETCH_CONCURRENCY`], env `VEX_CLONE_PACK_CONCURRENCY`).
    ///
    /// Measured on the prod baseline the *sequential* pack loop was 41s of a
    /// 75s clone — ~22s of RPC round trips interleaved with serial zstd decode
    /// and per-object cache writes. Each worker thread here drives the whole
    /// existing [`Self::prefetch_one_pack`] path (chunked-resumable → streamed
    /// presigned → whole-pack gRPC, then unpack-into-cache) for one pack at a
    /// time, so per-pack transfer resumability and `with_output_cancel`
    /// responsiveness are unchanged, while up to `concurrency` packs overlap
    /// their network and decode costs. Workers are plain blocking threads (the
    /// same execution context the sequential caller had), NOT tasks on the
    /// shared runtime: the pack path mixes blocking bridges
    /// (`block_on_http_get*` blocks on tasks spawned onto the shared runtime)
    /// that must never run on the shared runtime's own workers.
    ///
    /// Emits [`CloneProgress::PackFetched`] per completed pack, with `done`
    /// accumulated in the caller-shared `packs_done` so the metadata and
    /// snapshot phases report one continuous sequence out of `total_packs`.
    ///
    /// Returns one slot per input pack, in input order: `Some(result)` for
    /// packs that ran, `None` for packs never started because an earlier pack
    /// failed (workers stop scheduling on the first failure and drain).
    #[expect(clippy::too_many_arguments)]
    fn prefetch_packs_parallel(
        &self,
        packs: &[&jj_backend_types::PackDescriptor],
        pack_hints: &[jj_backend_api::PresignedGet],
        snapshot: bool,
        prefetched_objects: &AtomicU64,
        packs_done: &AtomicU64,
        total_packs: u64,
        progress: Option<&CloneProgressFn>,
    ) -> Vec<Option<Result<(), VexClientError>>> {
        if packs.is_empty() {
            return Vec::new();
        }
        let concurrency = pack_fetch_concurrency().min(packs.len());
        let next = AtomicUsize::new(0);
        let abort = AtomicBool::new(false);
        let results: Vec<Mutex<Option<Result<(), VexClientError>>>> =
            (0..packs.len()).map(|_| Mutex::new(None)).collect();
        std::thread::scope(|scope| {
            for _ in 0..concurrency {
                scope.spawn(|| {
                    loop {
                        // Stop scheduling on the first failure (fail fast) and
                        // when the pager quit (the in-flight fetches also
                        // abort via `with_output_cancel`).
                        if abort.load(Ordering::SeqCst) || output_closed() {
                            break;
                        }
                        let index = next.fetch_add(1, Ordering::SeqCst);
                        let Some(pack) = packs.get(index) else {
                            break;
                        };
                        // Plain thread outside any runtime — the sync-over-
                        // async bridges inside `prefetch_one_pack` behave
                        // exactly as they do for the sequential caller.
                        let result = futures::executor::block_on(self.prefetch_one_pack(
                            pack,
                            pack_hints,
                            snapshot,
                            prefetched_objects,
                        ));
                        if result.is_ok() {
                            let done = packs_done.fetch_add(1, Ordering::SeqCst) + 1;
                            if let Some(progress) = progress {
                                progress(CloneProgress::PackFetched {
                                    done,
                                    total: total_packs,
                                    objects: prefetched_objects.load(Ordering::Relaxed),
                                });
                            }
                        } else {
                            abort.store(true, Ordering::SeqCst);
                        }
                        *results[index].lock().unwrap() = Some(result);
                    }
                });
            }
        });
        results
            .into_iter()
            .map(|slot| slot.into_inner().unwrap())
            .collect()
    }

    /// Write `.snapshots/<commit>` markers for every chain set whose full
    /// working-tree closure is now locally present (see
    /// [`snapshot_sets_now_complete`]). Best-effort: a marker write failure
    /// only forfeits future negotiation, never the clone.
    fn record_snapshot_markers(
        &self,
        chain: &[SnapshotPackSet],
        fetched_ok: &HashMap<String, bool>,
    ) {
        let mut already_marked: HashSet<String> = HashSet::new();
        for set in chain {
            for id in std::iter::once(&set.commit_id).chain(set.base_commit_id.as_ref()) {
                let hex = id.to_string();
                if self.has_unpacked_snapshot(&hex) {
                    already_marked.insert(hex);
                }
            }
        }
        for commit_hex in snapshot_sets_now_complete(chain, &already_marked, fetched_ok) {
            match self.write_snapshot_marker(&commit_hex) {
                Ok(()) => {
                    debug!(commit_id = %commit_hex, "recorded unpacked snapshot set marker");
                }
                Err(err) => {
                    tracing::warn!(
                        commit_id = %commit_hex,
                        error = %err,
                        "failed to write snapshot set marker"
                    );
                }
            }
        }
    }

    /// Fetch and unpack a single clone/snapshot pack into the local object
    /// cache, trying the chunked path first and falling back to streamed and
    /// then whole-pack reads. `snapshot` routes the pack/byte counters to the
    /// snapshot stats (`snapshot_packs_fetched`/`snapshot_pack_bytes`).
    /// `prefetched_objects` is incremented per object written. Cache writes
    /// skip the per-write prune; the caller prunes once after the whole
    /// prefetch.
    async fn prefetch_one_pack(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        pack_hints: &[jj_backend_api::PresignedGet],
        snapshot: bool,
        prefetched_objects: &AtomicU64,
    ) -> Result<(), VexClientError> {
        let stats = vex_client_stats();
        let packs_counter = if snapshot {
            &stats.snapshot_packs_fetched
        } else {
            &stats.packs_fetched
        };
        let bytes_counter = if snapshot {
            &stats.snapshot_pack_bytes
        } else {
            &stats.pack_bytes_fetched
        };
        match self
            .prefetch_pack_via_chunks(pack, pack_hints, snapshot, prefetched_objects)
            .await
        {
            Ok(true) => {
                packs_counter.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            Ok(false) => {}
            Err(err) => {
                debug!(
                    pack_content_id = %pack.content_id,
                    // Redacted: the error may embed a signed URL.
                    error = %redact_url_queries(&err.to_string()),
                    "chunk path failed, using full-pack fallback"
                );
            }
        }
        let mut temp_pack = NamedTempFile::new()?;
        let streamed = self
            .direct_fetch_pack_to_file(pack, pack_hints, temp_pack.as_file_mut())
            .unwrap_or(false);
        if streamed {
            self.prefetch_pack_entries_from_file(
                &pack.content_id,
                temp_pack.path(),
                prefetched_objects,
            )?;
            packs_counter.fetch_add(1, Ordering::Relaxed);
            bytes_counter.fetch_add(pack.size_bytes, Ordering::Relaxed);
            return Ok(());
        }

        let pack_bytes = match self.direct_fetch_pack_bytes(pack, pack_hints) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(_) => self.get_object(ObjectKind::Pack, &pack.content_id).await?,
        };
        bytes_counter.fetch_add(pack_bytes.len() as u64, Ordering::Relaxed);
        let object_pack = decode_object_pack(&pack_bytes)
            .or_else(|_| decode_object_pack_reader(BufReader::new(pack_bytes.as_slice())))
            .map_err(|err| VexClientError::PackDecode(err.to_string()))?;
        let mut entries = Some(object_pack.objects);
        self.unpack_pack_entries(&pack.content_id, prefetched_objects, move |sink| {
            for entry in entries.take().expect("unpack drives the entries once") {
                sink(entry)?;
            }
            Ok(())
        })?;
        packs_counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn normalize_pack_chunks(
    chunks: &[jj_backend_types::PackChunkDescriptor],
) -> Vec<jj_backend_types::PackChunkDescriptor> {
    let mut normalized = chunks.to_vec();
    normalized.sort_by_key(|chunk| (chunk.chunk_index, chunk.offset_bytes));
    normalized
}

fn normalized_valid_pack_chunks(
    pack: &jj_backend_types::PackDescriptor,
) -> Option<Vec<jj_backend_types::PackChunkDescriptor>> {
    if pack.chunks.is_empty() {
        return None;
    }
    let chunks = normalize_pack_chunks(&pack.chunks);
    let expected_count = chunks.len() as u32;
    let mut expected_offset = 0_u64;
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.chunk_count != expected_count {
            return None;
        }
        if chunk.chunk_index != index as u32 {
            return None;
        }
        if chunk.offset_bytes != expected_offset {
            return None;
        }
        expected_offset = expected_offset.saturating_add(chunk.size_bytes);
    }
    if expected_offset != pack.size_bytes {
        return None;
    }
    Some(chunks)
}

fn collect_cache_entries(root: &Path, entries: &mut Vec<CacheEntry>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            collect_cache_entries(&path, entries)?;
        } else if metadata.is_file() {
            entries.push(CacheEntry {
                path,
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                size_bytes: metadata.len(),
            });
        }
    }
    Ok(())
}

/// Whether `id` is a well-formed snapshot commit id: 64 chars of lowercase
/// hex (the wire format of `have_snapshot_commit_ids` and the `.snapshots`
/// marker file names). Also guards marker paths against traversal — anything
/// else is silently ignored.
fn is_snapshot_commit_hex(id: &str) -> bool {
    id.len() == 64 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Merge locally-marked snapshot ids with caller-supplied haves into the
/// deduplicated, validated list sent as `have_snapshot_commit_ids` on the
/// clone manifest request. Sorted so requests are deterministic.
fn assemble_snapshot_haves(marker_ids: Vec<String>, extra: &[String]) -> Vec<String> {
    let mut haves: Vec<String> = marker_ids
        .into_iter()
        .chain(extra.iter().cloned())
        .filter(|id| is_snapshot_commit_hex(id))
        .collect();
    haves.sort_unstable();
    haves.dedup();
    haves
}

/// Which sets of a snapshot chain are *newly* complete — i.e. should get
/// `.snapshots` markers — after a fetch pass, in chain (base-first) order.
///
/// A set is complete when its own packs all unpacked (`fetched_ok`, keyed by
/// commit hex; zero-pack "head alias" sets are trivially `true`) AND — for
/// delta sets — its base's closure is covered, either by an earlier element
/// of this walk or by a pre-existing marker (`already_marked`, which also
/// covers a chain the server trimmed above our declared haves). A broken
/// link therefore stops marker propagation to every delta above it, while
/// the already-cached objects remain usable for hydration fallback.
fn snapshot_sets_now_complete(
    chain: &[SnapshotPackSet],
    already_marked: &HashSet<String>,
    fetched_ok: &HashMap<String, bool>,
) -> Vec<String> {
    let mut covered: HashSet<String> = already_marked.clone();
    let mut newly_complete = Vec::new();
    for set in chain {
        let commit_hex = set.commit_id.to_string();
        if covered.contains(&commit_hex) {
            continue;
        }
        let base_covered = set
            .base_commit_id
            .as_ref()
            .is_none_or(|base| covered.contains(&base.to_string()));
        if base_covered && fetched_ok.get(&commit_hex).copied().unwrap_or(false) {
            covered.insert(commit_hex.clone());
            newly_complete.push(commit_hex);
        }
    }
    newly_complete
}

/// Split objects into `GetObjectsInline` batches bounded by object count and
/// (estimated) response bytes. Unknown sizes count as zero, so a size-less id
/// list is bounded by count alone.
fn split_inline_fetch_batches(
    ids: Vec<(ObjectKind, ContentId, Option<u64>)>,
    max_objects: usize,
    max_bytes: u64,
) -> Vec<Vec<(ObjectKind, ContentId)>> {
    let max_objects = max_objects.max(1);
    let mut batches: Vec<Vec<(ObjectKind, ContentId)>> = Vec::new();
    let mut current: Vec<(ObjectKind, ContentId)> = Vec::new();
    let mut current_bytes = 0_u64;
    for (kind, content_id, size_bytes) in ids {
        current_bytes = current_bytes.saturating_add(size_bytes.unwrap_or(0));
        current.push((kind, content_id));
        if current.len() >= max_objects || current_bytes >= max_bytes {
            batches.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Replace the query string of any URL embedded in `text` with `<redacted>`.
/// Presigned object-store URLs carry their entire authorization in the query
/// (`X-Amz-Signature=...`), and reqwest errors embed the full request URL, so
/// every log line that can carry such an error must pass through here.
fn redact_url_queries(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find('?') {
        out.push_str(&rest[..pos]);
        out.push_str("?<redacted>");
        let after = &rest[pos + 1..];
        // The query ends at the first delimiter that cannot appear in one
        // (reqwest wraps URLs in parentheses; whitespace/quotes end a token).
        let end = after
            .find([')', ' ', '\t', '\n', '"', '\''])
            .unwrap_or(after.len());
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

pub fn kind_to_str(kind: ObjectKind) -> &'static str {
    match kind {
        ObjectKind::Blob => "blob",
        ObjectKind::Tree => "tree",
        ObjectKind::Commit => "commit",
        ObjectKind::Tag => "tag",
        ObjectKind::Symlink => "symlink",
        ObjectKind::Copy => "copy",
        ObjectKind::View => "view",
        ObjectKind::Op => "op",
        ObjectKind::Pack => "pack",
        ObjectKind::Manifest => "manifest",
    }
}

/// Inverse of [`kind_to_str`]; `None` for unknown kind strings (e.g. from a
/// newer server).
fn kind_from_str(kind: &str) -> Option<ObjectKind> {
    match kind {
        "blob" => Some(ObjectKind::Blob),
        "tree" => Some(ObjectKind::Tree),
        "commit" => Some(ObjectKind::Commit),
        "tag" => Some(ObjectKind::Tag),
        "symlink" => Some(ObjectKind::Symlink),
        "copy" => Some(ObjectKind::Copy),
        "view" => Some(ObjectKind::View),
        "op" => Some(ObjectKind::Op),
        "pack" => Some(ObjectKind::Pack),
        "manifest" => Some(ObjectKind::Manifest),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jj_backend_api::PresignedGet;
    use jj_backend_types::{
        ClonePackScope, ObjectPack, PackChunkDescriptor, PackDescriptor, encode_object_pack,
    };
    use std::io::Read;
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::thread;

    /// Serializes tests that touch the process-global [`VexClientStats`]
    /// counters, so a concurrent [`vex_client_stats_reset`] cannot corrupt
    /// another test's delta assertions (tests run in parallel threads).
    /// Shared with `vex_backend`'s tests via [`crate::vex::test_stats_lock`].
    fn stats_lock() -> &'static Mutex<()> {
        test_stats_lock()
    }

    fn sample_client() -> VexClient {
        VexClient::from_config(VexRepoConfig {
            endpoint: "http://127.0.0.1:50051".to_string(),
            tenant_id: "tenant".to_string(),
            tenant_slug: "tenant".to_string(),
            repo_id: "repo".to_string(),
            repo_slug: "repo".to_string(),
            repository_scope_kind: Some("repository".to_string()),
            virtual_repository_id: None,
            backing_repo_slug: None,
            virtual_root_path: None,
            virtual_mounts: Vec::new(),
            access_token: None,
            local_writes: false,
            object_read_mode: VexObjectReadMode::NativeOnly,
        })
        .unwrap()
    }

    #[test]
    fn validate_endpoint_accepts_valid_uris_without_building_a_connector() {
        // `validate_endpoint` exists to avoid the native-root cert load that
        // `Endpoint::new` performs for https URIs; it must still accept the
        // same well-formed endpoints the real connect path uses.
        for endpoint in [
            "https://jj.vex.sc",
            "http://127.0.0.1:50051",
            "https://example.com:443/path",
        ] {
            assert!(
                VexClient::validate_endpoint(endpoint).is_ok(),
                "expected {endpoint} to validate"
            );
            // The full connect-path builder must also accept it, so validation
            // never diverges from what `cached_channel` will later parse.
            assert!(VexClient::endpoint(endpoint).is_ok());
        }
    }

    #[test]
    fn validate_endpoint_rejects_malformed_uris() {
        for endpoint in ["", "ht tp://has space", "::::"] {
            assert!(
                VexClient::validate_endpoint(endpoint).is_err(),
                "expected {endpoint:?} to be rejected"
            );
        }
    }

    #[test]
    fn endpoint_is_https_detects_scheme() {
        // Only https endpoints get a TLS connector; http (local dev) must not.
        assert!(VexClient::endpoint_is_https("https://jj.vex.sc"));
        assert!(VexClient::endpoint_is_https("HTTPS://jj.vex.sc"));
        assert!(!VexClient::endpoint_is_https("http://127.0.0.1:50051"));
        assert!(!VexClient::endpoint_is_https("127.0.0.1:50051"));
    }

    #[test]
    fn pipelined_put_treats_cancelled_and_unavailable_as_transient() {
        for status in [
            tonic::Status::cancelled("edge reloaded"),
            tonic::Status::unavailable("connection reset"),
        ] {
            assert!(VexClient::is_transient_pipelined_put_error(
                &VexClientError::Status(status)
            ));
        }
    }

    #[test]
    fn commit_operation_retries_only_the_explicit_maintenance_status() {
        assert!(VexClient::is_commit_operation_maintenance_status(
            &tonic::Status::unavailable("repository maintenance is in progress; retry commit")
        ));
        assert!(!VexClient::is_commit_operation_maintenance_status(
            &tonic::Status::unavailable("connection reset")
        ));
        assert!(!VexClient::is_commit_operation_maintenance_status(
            &tonic::Status::internal("repository maintenance is in progress; retry commit")
        ));
    }

    #[test]
    fn commit_operation_retries_only_replay_safe_transient_statuses() {
        assert!(VexClient::is_retryable_commit_operation_status(
            &tonic::Status::cancelled("Timeout expired")
        ));
        assert!(VexClient::is_retryable_commit_operation_status(
            &tonic::Status::unavailable("connection reset")
        ));
        assert!(!VexClient::is_retryable_commit_operation_status(
            &tonic::Status::permission_denied("not allowed")
        ));
        assert!(!VexClient::is_retryable_commit_operation_status(
            &tonic::Status::invalid_argument("malformed operation")
        ));
    }

    #[test]
    fn commit_operation_maintenance_retry_delay_is_bounded() {
        assert!(
            VexClient::commit_operation_maintenance_retry_delay(usize::MAX)
                <= Duration::from_millis(COMMIT_OPERATION_MAINTENANCE_RETRY_CAP_MS * 5 / 4)
        );
    }

    #[test]
    fn pipelined_put_retries_cancelled_batch_with_fresh_connection() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let attempts = Arc::new(AtomicUsize::new(0));
        let result = VexClient::shared_grpc_runtime().block_on({
            let calls = calls.clone();
            let attempts = attempts.clone();
            VexClient::retry_pipelined_put_batch(move |uses_initial_channel| {
                let calls = calls.clone();
                let attempts = attempts.clone();
                async move {
                    calls.lock().unwrap().push(uses_initial_channel);
                    if attempts.fetch_add(1, Ordering::Relaxed) == 0 {
                        Err(VexClientError::Status(tonic::Status::cancelled(
                            "edge reloaded",
                        )))
                    } else {
                        Ok(())
                    }
                }
            })
        });

        assert!(
            result.is_ok(),
            "cancelled batch should be retried: {result:?}"
        );
        assert_eq!(*calls.lock().unwrap(), vec![true, false]);
    }

    #[test]
    fn pipelined_put_retry_delay_is_bounded() {
        assert!(
            VexClient::pipelined_put_retry_delay(usize::MAX)
                <= Duration::from_millis(PIPELINED_PUT_RETRY_CAP_MS * 5 / 4)
        );
    }

    #[test]
    fn pack_transfer_state_round_trip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let pack_id = ContentId::hash_bytes(b"pack-state");
        let state = PackTransferState {
            pack_content_id: pack_id.to_string(),
            chunk_count: 4,
            next_chunk_index: 2,
        };
        client.save_pack_transfer_state(&pack_id, &state).unwrap();
        let loaded = client.load_pack_transfer_state(&pack_id).unwrap().unwrap();
        assert_eq!(loaded, state);
        // Clearing also removes legacy loose `pack/<chunk_id>` files that old
        // clients' gRPC chunk fallback double-wrote into the cache.
        let chunk_id = ContentId::hash_bytes(b"legacy-chunk");
        client
            .write_cached_object_no_prune(ObjectKind::Pack, &chunk_id, b"legacy-chunk")
            .unwrap();
        client
            .clear_pack_transfer_state(&pack_id, &[chunk_id])
            .unwrap();
        assert!(client.load_pack_transfer_state(&pack_id).unwrap().is_none());
        assert!(!client.has_cached_object(ObjectKind::Pack, &chunk_id));
    }

    #[test]
    fn normalize_pack_chunks_prefers_chunk_index_then_offset() {
        let chunks = vec![
            PackChunkDescriptor {
                content_id: ContentId::hash_bytes(b"2"),
                chunk_index: 2,
                chunk_count: 3,
                offset_bytes: 200,
                size_bytes: 10,
            },
            PackChunkDescriptor {
                content_id: ContentId::hash_bytes(b"0"),
                chunk_index: 0,
                chunk_count: 3,
                offset_bytes: 0,
                size_bytes: 10,
            },
            PackChunkDescriptor {
                content_id: ContentId::hash_bytes(b"1"),
                chunk_index: 1,
                chunk_count: 3,
                offset_bytes: 100,
                size_bytes: 10,
            },
        ];
        let normalized = normalize_pack_chunks(&chunks);
        assert_eq!(normalized[0].chunk_index, 0);
        assert_eq!(normalized[1].chunk_index, 1);
        assert_eq!(normalized[2].chunk_index, 2);
    }

    #[test]
    fn normalized_valid_pack_chunks_accepts_well_formed_chunks() {
        let pack = PackDescriptor {
            content_id: ContentId::hash_bytes(b"pack"),
            size_bytes: 30,
            scope: ClonePackScope::Full,
            chunks: vec![
                PackChunkDescriptor {
                    content_id: ContentId::hash_bytes(b"c2"),
                    chunk_index: 2,
                    chunk_count: 3,
                    offset_bytes: 20,
                    size_bytes: 10,
                },
                PackChunkDescriptor {
                    content_id: ContentId::hash_bytes(b"c0"),
                    chunk_index: 0,
                    chunk_count: 3,
                    offset_bytes: 0,
                    size_bytes: 10,
                },
                PackChunkDescriptor {
                    content_id: ContentId::hash_bytes(b"c1"),
                    chunk_index: 1,
                    chunk_count: 3,
                    offset_bytes: 10,
                    size_bytes: 10,
                },
            ],
            objects: vec![],
        };
        let chunks = normalized_valid_pack_chunks(&pack).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[2].chunk_index, 2);
    }

    #[test]
    fn normalized_valid_pack_chunks_rejects_non_contiguous_offset() {
        let pack = PackDescriptor {
            content_id: ContentId::hash_bytes(b"pack"),
            size_bytes: 30,
            scope: ClonePackScope::Full,
            chunks: vec![
                PackChunkDescriptor {
                    content_id: ContentId::hash_bytes(b"c0"),
                    chunk_index: 0,
                    chunk_count: 2,
                    offset_bytes: 0,
                    size_bytes: 10,
                },
                PackChunkDescriptor {
                    content_id: ContentId::hash_bytes(b"c1"),
                    chunk_index: 1,
                    chunk_count: 2,
                    offset_bytes: 15,
                    size_bytes: 20,
                },
            ],
            objects: vec![],
        };
        assert!(normalized_valid_pack_chunks(&pack).is_none());
    }

    #[test]
    fn split_inline_fetch_batches_bounds_by_object_count() {
        let ids: Vec<_> = (0..600_u32)
            .map(|i| {
                (
                    ObjectKind::Blob,
                    ContentId::hash_bytes(&i.to_le_bytes()),
                    None,
                )
            })
            .collect();
        let batches = split_inline_fetch_batches(ids, 256, u64::MAX);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![256, 256, 88]
        );
    }

    #[test]
    fn split_inline_fetch_batches_bounds_by_estimated_bytes() {
        // 10 MiB each with a 24 MiB cap: the third object crosses the cap, so
        // batches close at three objects apiece.
        let ids: Vec<_> = (0..7_u32)
            .map(|i| {
                (
                    ObjectKind::Blob,
                    ContentId::hash_bytes(&i.to_le_bytes()),
                    Some(10 * 1024 * 1024_u64),
                )
            })
            .collect();
        let batches = split_inline_fetch_batches(ids, 256, 24 * 1024 * 1024);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![3, 3, 1]
        );
    }

    #[test]
    fn split_inline_fetch_batches_without_sizes_ignores_byte_bound() {
        let ids: Vec<_> = (0..10_u32)
            .map(|i| {
                (
                    ObjectKind::Tree,
                    ContentId::hash_bytes(&i.to_le_bytes()),
                    None,
                )
            })
            .collect();
        // Unknown sizes count as zero bytes, so only the count bound applies.
        let batches = split_inline_fetch_batches(ids, 4, 1);
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![4, 4, 2]
        );
    }

    #[test]
    fn kind_round_trips_through_strings() {
        for kind in [
            ObjectKind::Blob,
            ObjectKind::Tree,
            ObjectKind::Commit,
            ObjectKind::Tag,
            ObjectKind::Symlink,
            ObjectKind::Copy,
            ObjectKind::View,
            ObjectKind::Op,
            ObjectKind::Pack,
            ObjectKind::Manifest,
        ] {
            assert_eq!(kind_from_str(kind_to_str(kind)), Some(kind));
        }
        assert_eq!(kind_from_str("mystery"), None);
    }

    #[test]
    fn client_stats_snapshot_and_reset() {
        // One test (not several), serialized against the other counter-bumping
        // tests via `stats_lock`, so parallel test threads never race on the
        // process-global counters.
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        vex_client_stats_reset();
        let stats = vex_client_stats();
        stats.record_get_object_rpc(ObjectKind::Blob);
        stats.record_get_object_rpc(ObjectKind::Blob);
        stats.record_get_object_rpc(ObjectKind::Tree);
        stats.record_get_object_rpc(ObjectKind::Commit);
        stats.record_get_object_rpc(ObjectKind::Op);
        stats.record_get_object_rpc(ObjectKind::View);
        stats.record_get_object_rpc(ObjectKind::Pack);
        stats.record_get_object_cache_hit(ObjectKind::Blob);
        stats.record_get_object_cache_hit(ObjectKind::Commit);
        stats.record_get_object_cache_hit(ObjectKind::Op);
        stats.record_get_object_cache_hit(ObjectKind::View);
        stats.record_get_object_cache_hit(ObjectKind::Pack);
        stats.hydrated_bytes.fetch_add(4096, Ordering::Relaxed);
        let snapshot = vex_client_stats_snapshot();
        assert_eq!(snapshot.get_object_rpcs_blob, 2);
        assert_eq!(snapshot.get_object_rpcs_tree, 1);
        assert_eq!(snapshot.get_object_rpcs_commit, 1);
        assert_eq!(snapshot.get_object_rpcs_op, 1);
        assert_eq!(snapshot.get_object_rpcs_view, 1);
        assert_eq!(snapshot.get_object_rpcs_other, 1);
        assert_eq!(snapshot.get_object_cache_hits, 5);
        assert_eq!(snapshot.get_object_cache_hits_blob, 1);
        assert_eq!(snapshot.get_object_cache_hits_commit, 1);
        assert_eq!(snapshot.get_object_cache_hits_op, 1);
        assert_eq!(snapshot.get_object_cache_hits_view, 1);
        assert_eq!(snapshot.get_object_cache_hits_other, 1);
        assert_eq!(snapshot.hydrated_bytes, 4096);
        vex_client_stats_reset();
        assert_eq!(
            vex_client_stats_snapshot(),
            VexClientStatsSnapshot::default()
        );
    }

    #[test]
    fn native_path_and_git_mapping_counters_snapshot_and_reset() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        vex_client_stats_reset();
        let stats = vex_client_stats();
        stats.record_native_trunk_resolution();
        stats.record_native_trunk_missing();
        stats.record_git_compat_commit_decode();
        stats.record_git_compat_commit_decode();
        stats.record_git_compat_tree_decode();
        stats.record_git_mapping_rpc(3, Duration::from_millis(25));
        stats.record_git_mapping_rpc(1, Duration::from_millis(5));
        let snapshot = vex_client_stats_snapshot();
        assert_eq!(snapshot.native_trunk_resolutions, 1);
        assert_eq!(snapshot.native_trunk_missing, 1);
        assert_eq!(snapshot.git_compat_commit_decodes, 2);
        assert_eq!(snapshot.git_compat_tree_decodes, 1);
        assert_eq!(snapshot.git_mapping_names_resolved, 4);
        assert_eq!(snapshot.git_mapping_rpcs, 2);
        assert_eq!(snapshot.git_mapping_elapsed_ms, 30);
        vex_client_stats_reset();
        assert_eq!(
            vex_client_stats_snapshot(),
            VexClientStatsSnapshot::default()
        );
    }

    fn hex_id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    fn snapshot_set(
        commit: ContentId,
        base: Option<ContentId>,
        pack_count: usize,
    ) -> SnapshotPackSet {
        let packs = (0..pack_count)
            .map(|index| PackDescriptor {
                content_id: ContentId::hash_bytes(&[commit.as_bytes()[0], index as u8]),
                size_bytes: 10,
                scope: ClonePackScope::Full,
                chunks: vec![],
                objects: vec![],
            })
            .collect();
        SnapshotPackSet {
            commit_id: commit,
            base_commit_id: base,
            packs,
            object_count: pack_count as u64,
            total_bytes: 10 * pack_count as u64,
        }
    }

    #[test]
    fn snapshot_marker_round_trip_and_listing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let commit_hex = hex_id(7).to_string();
        assert!(!client.has_unpacked_snapshot(&commit_hex));
        assert!(client.cached_snapshot_commit_ids().is_empty());

        client.write_snapshot_marker(&commit_hex).unwrap();
        assert!(client.has_unpacked_snapshot(&commit_hex));
        assert_eq!(
            client.cached_snapshot_commit_ids(),
            vec![commit_hex.clone()]
        );

        // Junk in the marker dir (short names, uppercase, non-hex) is ignored
        // rather than sent to the server as a bogus have.
        let marker_root = temp_dir.path().join(".snapshots");
        fs::write(marker_root.join("not-a-commit"), b"").unwrap();
        fs::write(marker_root.join(commit_hex.to_uppercase()), b"").unwrap();
        assert_eq!(client.cached_snapshot_commit_ids(), vec![commit_hex]);

        // A client without a cache root (from_config) has no markers.
        let cacheless = sample_client();
        assert!(!cacheless.has_unpacked_snapshot(&hex_id(7).to_string()));
        assert!(cacheless.cached_snapshot_commit_ids().is_empty());
    }

    /// A prune that evicts object files must drop every `.snapshots` marker:
    /// the evicted objects may belong to a closure a marker vouches for, and
    /// a stale marker would permanently suppress snapshot serving (sent as a
    /// have, trimming the served chain) and the hydration walk skip.
    #[test]
    fn prune_cache_drops_snapshot_markers_when_objects_are_evicted() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        client.cache_max_bytes = Some(8);
        let commit_hex = hex_id(3).to_string();
        client.write_snapshot_marker(&commit_hex).unwrap();

        // Under the cap: nothing evicted, markers survive.
        client
            .write_cached_object_no_prune(ObjectKind::Blob, &ContentId::hash_bytes(b"a"), b"a")
            .unwrap();
        client.prune_cache_if_needed().unwrap();
        assert!(client.has_unpacked_snapshot(&commit_hex));

        // Over the cap: the eviction invalidates every marker.
        client
            .write_cached_object_no_prune(
                ObjectKind::Blob,
                &ContentId::hash_bytes(b"big"),
                &[0_u8; 64],
            )
            .unwrap();
        client.prune_cache_if_needed().unwrap();
        assert!(!client.has_unpacked_snapshot(&commit_hex));
        assert!(client.cached_snapshot_commit_ids().is_empty());
    }

    #[test]
    fn assemble_snapshot_haves_merges_validates_and_dedupes() {
        let a = hex_id(1).to_string();
        let b = hex_id(2).to_string();
        let haves = assemble_snapshot_haves(
            vec![b.clone(), a.clone()],
            &[
                a.clone(),             // duplicate of a marker
                "not-hex".to_string(), // invalid: rejected
                a.to_uppercase(),      // invalid: uppercase rejected
                "abc123".to_string(),  // invalid: too short
            ],
        );
        let mut expected = vec![a, b];
        expected.sort_unstable();
        assert_eq!(haves, expected);
    }

    #[test]
    fn snapshot_sets_now_complete_marks_full_chain_including_zero_pack_sets() {
        // base -> delta -> zero-pack head alias (tree-identical trunk advance).
        let base = snapshot_set(hex_id(1), None, 2);
        let delta = snapshot_set(hex_id(2), Some(hex_id(1)), 1);
        let alias = snapshot_set(hex_id(3), Some(hex_id(2)), 0);
        let chain = vec![base, delta, alias];
        let fetched_ok: HashMap<String, bool> = chain
            .iter()
            .map(|set| (set.commit_id.to_string(), true))
            .collect();
        let newly = snapshot_sets_now_complete(&chain, &HashSet::new(), &fetched_ok);
        assert_eq!(
            newly,
            vec![
                hex_id(1).to_string(),
                hex_id(2).to_string(),
                hex_id(3).to_string(),
            ]
        );
    }

    #[test]
    fn snapshot_sets_now_complete_stops_at_a_broken_link() {
        let base = snapshot_set(hex_id(1), None, 1);
        let delta = snapshot_set(hex_id(2), Some(hex_id(1)), 1);
        let head = snapshot_set(hex_id(3), Some(hex_id(2)), 1);
        let chain = vec![base, delta, head];
        // The middle delta's packs failed: the head must not be marked even
        // though its own packs unpacked, because its base closure is missing.
        let fetched_ok: HashMap<String, bool> = [
            (hex_id(1).to_string(), true),
            (hex_id(2).to_string(), false),
            (hex_id(3).to_string(), true),
        ]
        .into_iter()
        .collect();
        let newly = snapshot_sets_now_complete(&chain, &HashSet::new(), &fetched_ok);
        assert_eq!(newly, vec![hex_id(1).to_string()]);
    }

    #[test]
    fn snapshot_sets_now_complete_builds_on_prior_markers() {
        // Server trimmed the chain above our declared have: the served chain
        // starts at a delta whose base is covered by an existing marker.
        let delta = snapshot_set(hex_id(2), Some(hex_id(1)), 1);
        let chain = vec![delta];
        let already_marked: HashSet<String> = [hex_id(1).to_string()].into_iter().collect();
        let fetched_ok: HashMap<String, bool> =
            [(hex_id(2).to_string(), true)].into_iter().collect();
        let newly = snapshot_sets_now_complete(&chain, &already_marked, &fetched_ok);
        assert_eq!(newly, vec![hex_id(2).to_string()]);

        // Same chain with no marker for the base: nothing gets marked.
        let newly = snapshot_sets_now_complete(&chain, &HashSet::new(), &fetched_ok);
        assert!(newly.is_empty());
    }

    #[test]
    fn is_snapshot_commit_hex_accepts_only_64_char_lowercase_hex() {
        assert!(is_snapshot_commit_hex(&hex_id(0xab).to_string()));
        assert!(!is_snapshot_commit_hex(""));
        assert!(!is_snapshot_commit_hex("abcd"));
        assert!(!is_snapshot_commit_hex(
            &hex_id(0xab).to_string().to_uppercase()
        ));
        assert!(!is_snapshot_commit_hex(&format!(
            "{}z",
            &hex_id(0xab).to_string()[..63]
        )));
    }

    #[test]
    fn direct_fetch_pack_bytes_uses_http_hint() {
        // Bumps the global presigned-fetch counters.
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = b"pack-bytes".to_vec();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0_u8; 1024];
            let _ = stream.read(&mut buf).unwrap();
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
            stream.write_all(&body).unwrap();
        });

        let content_id = ContentId::hash_bytes(b"pack");
        let pack = PackDescriptor {
            content_id,
            // The descriptor size now caps the buffered response body, so it
            // must match what the server serves.
            size_bytes: 10,
            scope: ClonePackScope::Full,
            chunks: vec![],
            objects: vec![],
        };
        let hints = vec![PresignedGet {
            object_key: format!("packs/sha256/{content_id}"),
            url: format!("http://{addr}/objects/pack/{content_id}"),
            headers: Default::default(),
        }];

        let bytes = sample_client()
            .direct_fetch_pack_bytes(&pack, &hints)
            .unwrap()
            .unwrap();

        assert_eq!(bytes, b"pack-bytes");
        server.join().unwrap();
    }

    /// Minimal HTTP server for presigned chunk fetches: serves
    /// `GET /chunks/<hex>` from an id → bytes map, one connection per request
    /// (`Connection: close`, so the shared reqwest pool opens a fresh
    /// connection each time), with a small pseudo-random delay per request so
    /// concurrent fetches complete out of order and exercise the reorder
    /// buffer. Records every requested path.
    struct ChunkServer {
        addr: SocketAddr,
        requests: Arc<Mutex<Vec<String>>>,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl ChunkServer {
        fn start(chunks: HashMap<String, Vec<u8>>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let chunks = Arc::new(chunks);
            let handle = {
                let requests = Arc::clone(&requests);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    for stream in listener.incoming() {
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        let Ok(stream) = stream else { break };
                        let requests = Arc::clone(&requests);
                        let chunks = Arc::clone(&chunks);
                        thread::spawn(move || Self::serve_one(stream, &chunks, &requests));
                    }
                })
            };
            Self {
                addr,
                requests,
                stop,
                handle: Some(handle),
            }
        }

        fn serve_one(
            mut stream: TcpStream,
            chunks: &HashMap<String, Vec<u8>>,
            requests: &Mutex<Vec<String>>,
        ) {
            // Read to the end of the request headers (requests are tiny).
            let mut buf = Vec::new();
            let mut byte = [0_u8; 1];
            while !buf.ends_with(b"\r\n\r\n") {
                match stream.read(&mut byte) {
                    Ok(1) => buf.push(byte[0]),
                    _ => return,
                }
            }
            let request = String::from_utf8_lossy(&buf);
            let path = request.split_whitespace().nth(1).unwrap_or("").to_string();
            requests.lock().unwrap().push(path.clone());
            // De-correlate completion order across concurrent fetches.
            let jitter_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| u64::from(d.subsec_nanos()) % 25)
                .unwrap_or(0);
            thread::sleep(Duration::from_millis(jitter_ms));
            match path.rsplit('/').next().and_then(|hex| chunks.get(hex)) {
                Some(body) => {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    drop(stream.write_all(header.as_bytes()));
                    drop(stream.write_all(body));
                }
                None => {
                    drop(stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    ));
                }
            }
        }

        fn requested_paths(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }

        fn url_for(&self, content_id: &ContentId) -> String {
            format!("http://{}/chunks/{content_id}", self.addr)
        }
    }

    impl Drop for ChunkServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            // Unblock the accept loop so the thread observes the stop flag.
            drop(TcpStream::connect(self.addr));
            if let Some(handle) = self.handle.take() {
                drop(handle.join());
            }
        }
    }

    /// Minimal HTTP server answering every request with `403 Forbidden` (an
    /// expired presigned URL) and counting complete requests.
    struct ForbiddenServer {
        addr: SocketAddr,
        hits: Arc<AtomicUsize>,
        stop: Arc<AtomicBool>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl ForbiddenServer {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let hits = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let handle = {
                let hits = Arc::clone(&hits);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    for stream in listener.incoming() {
                        if stop.load(Ordering::SeqCst) {
                            break;
                        }
                        let Ok(mut stream) = stream else { break };
                        let mut buf = Vec::new();
                        let mut byte = [0_u8; 1];
                        while !buf.ends_with(b"\r\n\r\n") {
                            match stream.read(&mut byte) {
                                Ok(1) => buf.push(byte[0]),
                                _ => break,
                            }
                        }
                        if !buf.ends_with(b"\r\n\r\n") {
                            // The Drop unblock connection, not a request.
                            continue;
                        }
                        hits.fetch_add(1, Ordering::SeqCst);
                        drop(stream.write_all(
                            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        ));
                    }
                })
            };
            Self {
                addr,
                hits,
                stop,
                handle: Some(handle),
            }
        }

        fn hits(&self) -> usize {
            self.hits.load(Ordering::SeqCst)
        }

        fn url_for(&self, content_id: &ContentId) -> String {
            format!("http://{}/chunks/{content_id}", self.addr)
        }
    }

    impl Drop for ForbiddenServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            drop(TcpStream::connect(self.addr));
            if let Some(handle) = self.handle.take() {
                drop(handle.join());
            }
        }
    }

    /// A valid encoded pack of pseudo-random blob objects (incompressible, so
    /// the encoded pack is large enough to split into many chunks), served
    /// chunk-by-chunk over a local [`ChunkServer`] via presigned hints.
    struct ChunkedPackFixture {
        pack: PackDescriptor,
        objects: Vec<ObjectPackEntry>,
        encoded: Vec<u8>,
        hints: Vec<PresignedGet>,
        server: ChunkServer,
    }

    fn chunked_pack_fixture(object_count: usize, chunk_size: usize) -> ChunkedPackFixture {
        // Cheap deterministic LCG data zstd cannot squash, so the encoded
        // payload really spans many chunks.
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let objects: Vec<ObjectPackEntry> = (0..object_count)
            .map(|_| {
                let data: Vec<u8> = (0..257)
                    .map(|_| {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        (seed >> 33) as u8
                    })
                    .collect();
                ObjectPackEntry {
                    kind: ObjectKind::Blob,
                    content_id: ContentId::hash_bytes(&data),
                    data,
                }
            })
            .collect();
        let encoded = encode_object_pack(&ObjectPack {
            objects: objects.clone(),
        });
        let chunk_count = encoded.len().div_ceil(chunk_size) as u32;
        assert!(chunk_count > 1, "fixture must produce a multi-chunk pack");
        let pieces: Vec<(PackChunkDescriptor, Vec<u8>)> = encoded
            .chunks(chunk_size)
            .enumerate()
            .map(|(index, piece)| {
                (
                    PackChunkDescriptor {
                        content_id: ContentId::hash_bytes(piece),
                        chunk_index: index as u32,
                        chunk_count,
                        offset_bytes: (index * chunk_size) as u64,
                        size_bytes: piece.len() as u64,
                    },
                    piece.to_vec(),
                )
            })
            .collect();
        let server = ChunkServer::start(
            pieces
                .iter()
                .map(|(descriptor, bytes)| (descriptor.content_id.to_string(), bytes.clone()))
                .collect(),
        );
        let hints = pieces
            .iter()
            .map(|(descriptor, _)| PresignedGet {
                object_key: format!("packs/chunks/sha256/{}", descriptor.content_id),
                url: server.url_for(&descriptor.content_id),
                headers: Default::default(),
            })
            .collect();
        let pack = PackDescriptor {
            content_id: ContentId::hash_bytes(&encoded),
            size_bytes: encoded.len() as u64,
            scope: ClonePackScope::Full,
            chunks: pieces
                .into_iter()
                .map(|(descriptor, _)| descriptor)
                .collect(),
            objects: vec![],
        };
        ChunkedPackFixture {
            pack,
            objects,
            encoded,
            hints,
            server,
        }
    }

    /// Run a full chunked prefetch into a fresh cache dir at the given fetch
    /// concurrency and return the unpacked object bytes, in fixture object
    /// order.
    fn run_chunked_prefetch(fixture: &ChunkedPackFixture, concurrency: usize) -> Vec<Vec<u8>> {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let counter = AtomicU64::new(0);
        let ok = futures::executor::block_on(client.prefetch_pack_via_chunks_with_concurrency(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
            concurrency,
        ))
        .unwrap();
        assert!(ok, "chunked path must handle a well-formed chunked pack");
        // Chunk fetches must not leave loose `pack/<chunk_id>` cache files.
        assert!(!temp_dir.path().join("pack").exists());
        // Transfer state is cleared after a successful unpack.
        assert!(
            client
                .load_pack_transfer_state(&fixture.pack.content_id)
                .unwrap()
                .is_none()
        );
        fixture
            .objects
            .iter()
            .map(|entry| {
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .expect("object unpacked into cache")
            })
            .collect()
    }

    /// The `.buffered(W)` reorder buffer must reassemble the pack
    /// byte-identically to the serial (W=1) loop even when chunk responses
    /// complete out of order (the test server adds random delays), and every
    /// chunk must be counted as a presigned fetch.
    #[test]
    fn chunked_prefetch_reorder_buffer_matches_serial_and_counts_presigned() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture(24, 256);
        let run_with_concurrency = |concurrency: usize| {
            let before = vex_client_stats_snapshot();
            let objects = run_chunked_prefetch(&fixture, concurrency);
            let after = vex_client_stats_snapshot();
            let chunk_count = fixture.pack.chunks.len() as u64;
            assert_eq!(
                after.presigned_fetches - before.presigned_fetches,
                chunk_count
            );
            assert_eq!(
                after.presigned_bytes - before.presigned_bytes,
                fixture.encoded.len() as u64
            );
            assert_eq!(
                after.pack_chunks_fetched - before.pack_chunks_fetched,
                chunk_count
            );
            assert_eq!(
                after.pack_bytes_fetched - before.pack_bytes_fetched,
                fixture.encoded.len() as u64
            );
            objects
        };
        let concurrent = run_with_concurrency(4);
        let serial = run_with_concurrency(1);
        assert_eq!(concurrent, serial);
        // Unpack SHA-256-verifies every entry, so matching the source objects
        // proves the reassembled pack bytes were identical to the original.
        for (entry, bytes) in fixture.objects.iter().zip(&concurrent) {
            assert_eq!(&entry.data, bytes);
        }
    }

    /// A `.part` file LONGER than the recorded state (a kill between an append
    /// and the next batched state save, possibly mid-chunk) must be truncated
    /// back to the recorded contiguous prefix — chunks before the prefix are
    /// not refetched, everything after is.
    #[test]
    fn chunked_prefetch_resume_truncates_part_longer_than_state() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture(24, 256);
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let recorded = 2_usize;
        client
            .save_pack_transfer_state(
                &fixture.pack.content_id,
                &PackTransferState {
                    pack_content_id: fixture.pack.content_id.to_string(),
                    chunk_count: fixture.pack.chunks.len(),
                    next_chunk_index: recorded,
                },
            )
            .unwrap();
        // 4.5 chunks on disk vs 2 recorded: 2.5 chunks of untrusted tail.
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        fs::write(&partial_path, &fixture.encoded[..4 * 256 + 128]).unwrap();

        let counter = AtomicU64::new(0);
        let ok = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
        ))
        .unwrap();
        assert!(ok);
        let requested = fixture.server.requested_paths();
        for chunk in &fixture.pack.chunks[..recorded] {
            let id = chunk.content_id.to_string();
            assert!(
                !requested.iter().any(|path| path.ends_with(&id)),
                "chunk {id} within the recorded prefix must not be refetched"
            );
        }
        for chunk in &fixture.pack.chunks[recorded..] {
            let id = chunk.content_id.to_string();
            assert!(
                requested.iter().any(|path| path.ends_with(&id)),
                "chunk {id} beyond the recorded prefix must be fetched"
            );
        }
        // The unpack hash-verified every entry: the truncate+resume
        // reassembled the exact original pack bytes.
        for entry in &fixture.objects {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
    }

    /// A `.part` file SHORTER than the recorded state is inconsistent (a state
    /// file ahead of its data): the transfer must fully reset and refetch
    /// every chunk.
    #[test]
    fn chunked_prefetch_resume_resets_when_part_shorter_than_state() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture(24, 256);
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        client
            .save_pack_transfer_state(
                &fixture.pack.content_id,
                &PackTransferState {
                    pack_content_id: fixture.pack.content_id.to_string(),
                    chunk_count: fixture.pack.chunks.len(),
                    next_chunk_index: 3,
                },
            )
            .unwrap();
        // Only one chunk on disk vs 3 recorded.
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        fs::write(&partial_path, &fixture.encoded[..256]).unwrap();

        let counter = AtomicU64::new(0);
        let ok = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
        ))
        .unwrap();
        assert!(ok);
        let requested = fixture.server.requested_paths();
        for chunk in &fixture.pack.chunks {
            let id = chunk.content_id.to_string();
            assert!(
                requested.iter().any(|path| path.ends_with(&id)),
                "chunk {id} must be refetched after a full reset"
            );
        }
        for entry in &fixture.objects {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
    }

    /// A size-correct but wrong-content presigned chunk response must be
    /// rejected by the per-chunk hash verification (never appended to the
    /// `.part`), and the next attempt must resume from the trusted prefix.
    #[test]
    fn chunked_prefetch_hash_verifies_presigned_chunks_and_resumes() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture(24, 256);
        let chunks = &fixture.pack.chunks;
        // A second server serving the same chunk ids, but chunk 1's bytes are
        // corrupted in place: same length (passes the old size-only check),
        // wrong content.
        let corrupt_server = ChunkServer::start(
            chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    let start = chunk.offset_bytes as usize;
                    let mut bytes =
                        fixture.encoded[start..start + chunk.size_bytes as usize].to_vec();
                    if index == 1 {
                        bytes[0] ^= 0xff;
                    }
                    (chunk.content_id.to_string(), bytes)
                })
                .collect(),
        );
        let corrupt_hints: Vec<PresignedGet> = chunks
            .iter()
            .map(|chunk| PresignedGet {
                object_key: format!("packs/chunks/sha256/{}", chunk.content_id),
                url: corrupt_server.url_for(&chunk.content_id),
                headers: Default::default(),
            })
            .collect();
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let counter = AtomicU64::new(0);
        // Chunk 1 fails hash verification twice, then the gRPC fallback fails
        // (no backend in unit tests): the transfer errors...
        futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &corrupt_hints,
            false,
            &counter,
        ))
        .unwrap_err();
        let chunk1_id = chunks[1].content_id.to_string();
        assert_eq!(
            corrupt_server
                .requested_paths()
                .iter()
                .filter(|path| path.ends_with(&chunk1_id))
                .count(),
            2,
            "the bad chunk must be retried once before the gRPC fallback"
        );
        // ...and the corrupt bytes never reached the `.part`: it holds
        // exactly the verified chunk 0, recorded as resumable progress.
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        assert_eq!(
            fs::read(&partial_path).unwrap(),
            &fixture.encoded[..chunks[0].size_bytes as usize]
        );
        assert_eq!(
            client
                .load_pack_transfer_state(&fixture.pack.content_id)
                .unwrap()
                .unwrap()
                .next_chunk_index,
            1
        );
        // A retry against a healthy server resumes past the verified prefix
        // and completes.
        let ok = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
        ))
        .unwrap();
        assert!(ok);
        let chunk0_id = chunks[0].content_id.to_string();
        assert!(
            !fixture
                .server
                .requested_paths()
                .iter()
                .any(|path| path.ends_with(&chunk0_id)),
            "the verified prefix must not be refetched"
        );
        for entry in &fixture.objects {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
    }

    /// Corrupt/truncated transfer-state JSON (a kill mid non-atomic save)
    /// must reset the transfer — not hard-error the chunk path forever — and
    /// remove the poisoned file.
    #[test]
    fn chunked_prefetch_resets_on_corrupt_transfer_state() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture(24, 256);
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let state_path = client
            .transfer_state_path(&fixture.pack.content_id)
            .unwrap();
        fs::create_dir_all(state_path.parent().unwrap()).unwrap();
        fs::write(&state_path, b"{\"pack_content_id\": trunc").unwrap();
        // The corrupt file loads as no-state and is dropped on the spot.
        assert!(
            client
                .load_pack_transfer_state(&fixture.pack.content_id)
                .unwrap()
                .is_none()
        );
        assert!(!state_path.exists());
        // And a prefetch over a re-corrupted state self-heals end to end.
        fs::write(&state_path, b"not json at all").unwrap();
        let counter = AtomicU64::new(0);
        let ok = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
        ))
        .unwrap();
        assert!(ok);
        for entry in &fixture.objects {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
    }

    /// A COMPLETED transfer whose `.part` fails decode (same-length on-disk
    /// corruption passes every resume consistency check) must clear its
    /// state + `.part` so the next attempt refetches from scratch instead of
    /// re-decoding the same poisoned bytes forever.
    #[test]
    fn chunked_prefetch_clears_poisoned_completed_transfer() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture(24, 256);
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        client
            .save_pack_transfer_state(
                &fixture.pack.content_id,
                &PackTransferState {
                    pack_content_id: fixture.pack.content_id.to_string(),
                    chunk_count: fixture.pack.chunks.len(),
                    next_chunk_index: fixture.pack.chunks.len(),
                },
            )
            .unwrap();
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        let mut poisoned = fixture.encoded.clone();
        let mid = poisoned.len() / 2;
        poisoned[mid] ^= 0xff;
        fs::write(&partial_path, &poisoned).unwrap();

        let counter = AtomicU64::new(0);
        let err = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
        ))
        .unwrap_err();
        assert!(matches!(err, VexClientError::PackDecode(_)), "{err}");
        // Nothing was fetched (the state said complete)...
        assert!(fixture.server.requested_paths().is_empty());
        // ...but the poison is gone, so the next attempt starts clean...
        assert!(
            client
                .load_pack_transfer_state(&fixture.pack.content_id)
                .unwrap()
                .is_none()
        );
        assert!(!partial_path.exists());
        // ...and succeeds by refetching every chunk.
        let ok = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            false,
            &counter,
        ))
        .unwrap();
        assert!(ok);
        for entry in &fixture.objects {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
    }

    /// Direct HTTP fetches must cap the response body at the caller's
    /// expected size (the descriptor's `size_bytes`) instead of buffering or
    /// writing unbounded bytes from a broken/hostile endpoint.
    #[test]
    fn http_get_caps_response_at_expected_size() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let content_id = ContentId::hash_bytes(b"oversize");
        let body = vec![7_u8; 10];
        let server = ChunkServer::start(
            [(content_id.to_string(), body.clone())]
                .into_iter()
                .collect(),
        );
        let url = server.url_for(&content_id);
        let headers = HashMap::new();
        assert_eq!(
            VexClient::block_on_http_get(&url, &headers, Some(10)).unwrap(),
            body
        );
        assert_eq!(
            VexClient::block_on_http_get(&url, &headers, None).unwrap(),
            body
        );
        let err = VexClient::block_on_http_get(&url, &headers, Some(4)).unwrap_err();
        assert!(err.to_string().contains("exceeds expected size"), "{err}");
        // The streaming (whole-pack) variant enforces its cap too.
        let mut out = Vec::new();
        let err =
            VexClient::block_on_http_get_to_file(&url, &headers, &mut out, Some(4)).unwrap_err();
        assert!(err.to_string().contains("exceeds expected size"), "{err}");
        let mut out = Vec::new();
        VexClient::block_on_http_get_to_file(&url, &headers, &mut out, Some(10)).unwrap();
        assert_eq!(out, body);
    }

    /// The first presigned 403 (expired/invalid signature — deterministic)
    /// must trip the per-client kill switch: no second attempt on the same
    /// chunk, and every later direct HTTP fetch is skipped outright.
    #[test]
    fn presigned_403_disables_direct_fetch_for_the_run() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let server = ForbiddenServer::start();
        let content_id = ContentId::hash_bytes(b"forbidden-chunk");
        let hints = vec![PresignedGet {
            object_key: format!("packs/chunks/sha256/{content_id}"),
            url: server.url_for(&content_id),
            headers: Default::default(),
        }];
        let client = sample_client();
        // The 403 breaks out after ONE attempt; the gRPC fallback then fails
        // (no backend in unit tests), surfacing an error.
        futures::executor::block_on(client.fetch_pack_chunk_with_retry(&content_id, &hints, None))
            .unwrap_err();
        assert_eq!(server.hits(), 1);
        assert!(client.presigned_get_disabled.load(Ordering::Relaxed));
        // Subsequent chunk fetches skip the presigned path entirely.
        drop(futures::executor::block_on(
            client.fetch_pack_chunk_with_retry(&content_id, &hints, None),
        ));
        assert_eq!(server.hits(), 1);
        // The whole-pack direct fetches are disabled too.
        let pack = PackDescriptor {
            content_id,
            size_bytes: 4,
            scope: ClonePackScope::Full,
            chunks: vec![],
            objects: vec![],
        };
        assert!(
            client
                .direct_fetch_pack_bytes(&pack, &hints)
                .unwrap()
                .is_none()
        );
        let mut out = Vec::new();
        assert!(
            !client
                .direct_fetch_pack_to_file(&pack, &hints, &mut out)
                .unwrap()
        );
        assert_eq!(server.hits(), 1);
    }

    /// A `.packs` persist racing a cross-process prune (`remove_dir_all` of
    /// the whole dir, unlinking the temp's source path) must recover from the
    /// still-open fd instead of failing the unpack — fatal for metadata
    /// packs.
    #[cfg(unix)]
    #[test]
    fn persist_pack_temp_survives_concurrent_packs_dir_removal() {
        let temp_dir = tempfile::tempdir().unwrap();
        let packs_dir = temp_dir.path().join(".packs");
        fs::create_dir_all(&packs_dir).unwrap();
        let mut temp = NamedTempFile::new_in(&packs_dir).unwrap();
        temp.write_all(b"payload-bytes").unwrap();
        temp.flush().unwrap();
        fs::remove_dir_all(&packs_dir).unwrap();
        let dest = packs_dir.join("pack.payload");
        VexClient::persist_pack_temp(&packs_dir, temp, &dest).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"payload-bytes");
    }

    /// Mixed-kind pack entries: the four metadata kinds (pack-resident) plus
    /// a blob and a symlink (always loose).
    fn hybrid_pack_entries() -> Vec<ObjectPackEntry> {
        let entry = |kind: ObjectKind, data: &[u8]| ObjectPackEntry {
            kind,
            content_id: ContentId::hash_bytes(data),
            data: data.to_vec(),
        };
        vec![
            entry(ObjectKind::Commit, b"commit-bytes"),
            entry(ObjectKind::Tree, b"tree-bytes"),
            entry(ObjectKind::Op, b"op-bytes"),
            entry(ObjectKind::View, b"view-bytes"),
            entry(ObjectKind::Blob, b"blob-bytes"),
            entry(ObjectKind::Symlink, b"symlink-target"),
        ]
    }

    fn pack_resident_client(cache_root: &Path, enabled: bool) -> VexClient {
        let mut client = sample_client();
        client.cache_root = Some(cache_root.to_path_buf());
        client.pack_resident_override = Some(enabled);
        client
    }

    /// Encode `entries` into a pack file and unpack it through the real
    /// streaming path; returns the pack content id.
    fn unpack_hybrid_pack(client: &VexClient, entries: &[ObjectPackEntry]) -> ContentId {
        let encoded = encode_object_pack(&ObjectPack {
            objects: entries.to_vec(),
        });
        let pack_id = ContentId::hash_bytes(&encoded);
        let pack_file = tempfile::NamedTempFile::new().unwrap();
        fs::write(pack_file.path(), &encoded).unwrap();
        let counter = AtomicU64::new(0);
        client
            .prefetch_pack_entries_from_file(&pack_id, pack_file.path(), &counter)
            .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), entries.len() as u64);
        pack_id
    }

    fn loose_path(cache_root: &Path, entry: &ObjectPackEntry) -> PathBuf {
        cache_root
            .join(kind_to_str(entry.kind))
            .join(entry.content_id.to_string())
    }

    /// The hybrid unpack must serve metadata reads straight from the pack
    /// payload — with NO loose file — while blobs/symlinks unpack loose, and
    /// a fresh index (a new process) must reload the sidecar identically.
    #[test]
    fn pack_resident_unpack_serves_reads_without_loose_files() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let client = pack_resident_client(temp_dir.path(), true);
        let entries = hybrid_pack_entries();
        let before = vex_client_stats_snapshot();
        let pack_id = unpack_hybrid_pack(&client, &entries);
        let after = vex_client_stats_snapshot();
        assert_eq!(after.objects_unpacked - before.objects_unpacked, 6);
        assert_eq!(
            after.objects_pack_resident - before.objects_pack_resident,
            4
        );
        assert_eq!(after.loose_writes_avoided - before.loose_writes_avoided, 4);

        let packs_dir = temp_dir.path().join(".packs");
        assert!(packs_dir.join(format!("{pack_id}.payload")).exists());
        assert!(packs_dir.join(format!("{pack_id}.idx")).exists());
        for entry in &entries {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data,
                "every unpacked object must read back byte-identically"
            );
            assert!(client.has_cached_object(entry.kind, &entry.content_id));
            let loose = loose_path(temp_dir.path(), entry);
            if is_pack_resident_kind(entry.kind) {
                assert!(
                    !loose.exists(),
                    "metadata must not explode into a loose file"
                );
            } else {
                assert!(
                    loose.exists(),
                    "blobs/symlinks stay loose for reflink/streaming"
                );
            }
        }
        // A fresh index (as a new process would build it) reloads the sidecar.
        let reloaded = PackResidentIndex::new(packs_dir);
        let commit = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Commit)
            .unwrap();
        let location = reloaded.lookup(commit.kind, &commit.content_id).unwrap();
        assert_eq!(location.pack_hex.as_ref(), pack_id.to_string());
        assert_eq!(location.len, commit.data.len() as u64);
    }

    /// `VEX_CACHE_PACK_RESIDENT=0` (here: the per-client override) must
    /// restore the pre-split behavior exactly: every entry lands as a loose
    /// file, no `.packs` dir appears, and reads are byte-identical.
    #[test]
    fn pack_resident_kill_switch_restores_all_loose_unpack() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let disabled = pack_resident_client(temp_dir.path(), false);
        let entries = hybrid_pack_entries();
        let before = vex_client_stats_snapshot();
        unpack_hybrid_pack(&disabled, &entries);
        let after = vex_client_stats_snapshot();
        assert_eq!(after.objects_unpacked - before.objects_unpacked, 6);
        assert_eq!(after.objects_pack_resident, before.objects_pack_resident);
        assert_eq!(after.loose_writes_avoided, before.loose_writes_avoided);
        assert!(!temp_dir.path().join(".packs").exists());
        for entry in &entries {
            assert!(loose_path(temp_dir.path(), entry).exists());
            assert_eq!(
                disabled
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
    }

    /// A payload deleted behind our back (prune from another process, manual
    /// cleanup) must read as a miss — the caller falls back to loose/RPC —
    /// and self-heal by dropping the whole pack's index entries + sidecar.
    #[test]
    fn pack_resident_read_self_heals_when_payload_deleted() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let client = pack_resident_client(temp_dir.path(), true);
        let entries = hybrid_pack_entries();
        let pack_id = unpack_hybrid_pack(&client, &entries);
        let packs_dir = temp_dir.path().join(".packs");
        fs::remove_file(packs_dir.join(format!("{pack_id}.payload"))).unwrap();
        let commit = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Commit)
            .unwrap();
        assert!(
            client
                .read_cached_object(commit.kind, &commit.content_id)
                .is_none()
        );
        // The whole pack self-healed out of the index, so presence checks
        // (the put_object upload skip) miss for its other entries too...
        let tree = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Tree)
            .unwrap();
        assert!(!client.has_cached_object(tree.kind, &tree.content_id));
        // ...and the stale sidecar is gone, so no later process reloads it.
        assert!(!packs_dir.join(format!("{pack_id}.idx")).exists());
        // Loose objects are untouched.
        let blob = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Blob)
            .unwrap();
        assert!(client.has_cached_object(blob.kind, &blob.content_id));
    }

    /// A *transient* payload read error (EACCES here; EMFILE/EIO in the
    /// wild) must be a plain miss for that one read — NOT trigger the
    /// self-heal, which permanently deletes the intact payload + sidecar.
    /// Only a missing or truncated payload (NotFound/UnexpectedEof) is
    /// structural and may drop the pack.
    #[cfg(unix)]
    #[test]
    fn pack_resident_transient_read_error_misses_without_dropping_pack() {
        use std::os::unix::fs::PermissionsExt as _;
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let client = pack_resident_client(temp_dir.path(), true);
        let entries = hybrid_pack_entries();
        let pack_id = unpack_hybrid_pack(&client, &entries);
        let packs_dir = temp_dir.path().join(".packs");
        let payload_path = packs_dir.join(format!("{pack_id}.payload"));
        let idx_path = packs_dir.join(format!("{pack_id}.idx"));
        let commit = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Commit)
            .unwrap();

        // Transient failure: unreadable payload => miss, but nothing deleted.
        fs::set_permissions(&payload_path, fs::Permissions::from_mode(0o000)).unwrap();
        if File::open(&payload_path).is_err() {
            // (Skipped when running as root, where mode 000 is still readable.)
            assert!(
                client
                    .read_cached_object(commit.kind, &commit.content_id)
                    .is_none()
            );
            assert!(
                payload_path.exists(),
                "transient error must not delete the payload"
            );
            assert!(
                idx_path.exists(),
                "transient error must not delete the sidecar"
            );
        }
        fs::set_permissions(&payload_path, fs::Permissions::from_mode(0o644)).unwrap();
        // The next read simply retries and succeeds.
        assert_eq!(
            client
                .read_cached_object(commit.kind, &commit.content_id)
                .unwrap(),
            commit.data
        );

        // Structural failure: truncated payload (UnexpectedEof) self-heals.
        File::options()
            .write(true)
            .open(&payload_path)
            .unwrap()
            .set_len(1)
            .unwrap();
        assert!(
            client
                .read_cached_object(commit.kind, &commit.content_id)
                .is_none()
        );
        assert!(!idx_path.exists(), "truncated payload must drop the pack");
    }

    /// The "cached ⟹ uploaded" short circuit in `put_object` must include
    /// pack-resident objects, or every push would re-upload the metadata a
    /// clone's packs delivered.
    #[test]
    fn put_object_skips_upload_for_pack_resident_objects() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), true);
        // Isolate this test's slice of the process-global pending-upload
        // buffer (keyed by endpoint + repo id).
        client.config.repo_id = "repo-pack-resident-put-skip".to_string();
        let entries = hybrid_pack_entries();
        unpack_hybrid_pack(&client, &entries);
        let commit = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Commit)
            .unwrap();
        futures::executor::block_on(client.put_object(
            commit.kind,
            &commit.content_id,
            commit.data.clone(),
        ))
        .unwrap();
        assert!(
            !client.has_pending_object(commit.kind, &commit.content_id),
            "an index hit must skip the upload entirely, not buffer it"
        );
        // A genuinely uncached object takes the normal (buffered) upload path.
        let missing_data = b"never-packed".to_vec();
        let missing_id = ContentId::hash_bytes(&missing_data);
        futures::executor::block_on(client.put_object(
            ObjectKind::Commit,
            &missing_id,
            missing_data,
        ))
        .unwrap();
        assert!(client.has_pending_object(ObjectKind::Commit, &missing_id));
    }

    /// A prune that evicts loose object files must drop the pack-resident
    /// store wholesale (its payloads are excluded from the LRU scan) and
    /// clear the in-memory overlay with it.
    #[test]
    fn prune_drops_packs_dir_and_clears_pack_index() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), true);
        client.cache_max_bytes = Some(8);
        let entries = hybrid_pack_entries();
        unpack_hybrid_pack(&client, &entries);
        assert!(temp_dir.path().join(".packs").exists());
        // The loose blob+symlink bytes alone exceed the cap: the prune evicts
        // them and must take `.packs` down too.
        client.prune_cache_if_needed().unwrap();
        assert!(!temp_dir.path().join(".packs").exists());
        let commit = entries
            .iter()
            .find(|entry| entry.kind == ObjectKind::Commit)
            .unwrap();
        assert!(!client.has_cached_object(commit.kind, &commit.content_id));
        assert!(
            client
                .read_cached_object(commit.kind, &commit.content_id)
                .is_none()
        );
    }

    /// With the pack-resident kill switch on (`VEX_CACHE_PACK_RESIDENT=0`)
    /// nothing reads or writes `.packs`, so a prune must still reclaim a
    /// `.packs` dir left behind by an earlier enabled run — otherwise the
    /// rollback orphans it as dead disk for as long as the switch is on.
    #[test]
    fn prune_drops_packs_dir_even_with_pack_resident_disabled() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let enabled = pack_resident_client(temp_dir.path(), true);
        unpack_hybrid_pack(&enabled, &hybrid_pack_entries());
        assert!(temp_dir.path().join(".packs").exists());
        let mut disabled = pack_resident_client(temp_dir.path(), false);
        disabled.cache_max_bytes = Some(8);
        // The loose blob+symlink bytes exceed the cap, so the prune evicts
        // files — and must take the orphaned `.packs` down with them.
        disabled.prune_cache_if_needed().unwrap();
        assert!(!temp_dir.path().join(".packs").exists());
    }

    /// The direct-create fast path is off by default, only enabled through
    /// `mark_fresh_clone_cache` (the clone scaffold), never for a shared
    /// cache dir — and it must produce byte-identical loose files.
    #[test]
    fn fresh_clone_cache_direct_create_is_gated_and_writes_identical_files() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), true);
        assert!(!client.fresh_cache, "direct create must default off");
        client.mark_fresh_clone_cache();
        // Fresh only when the cache is repo-local (no shared cache dir
        // configured in the environment).
        assert_eq!(
            client.fresh_cache,
            std::env::var_os("JJ_VEX_SHARED_CACHE_DIR").is_none()
        );
        client.fresh_cache = true;
        let entries = hybrid_pack_entries();
        unpack_hybrid_pack(&client, &entries);
        for entry in &entries {
            assert_eq!(
                client
                    .read_cached_object(entry.kind, &entry.content_id)
                    .unwrap(),
                entry.data
            );
        }
        for entry in entries
            .iter()
            .filter(|entry| !is_pack_resident_kind(entry.kind))
        {
            assert_eq!(
                fs::read(loose_path(temp_dir.path(), entry)).unwrap(),
                entry.data,
                "direct-create loose files must be byte-identical"
            );
        }
    }

    /// The allocation-free hex decode used by the sidecar parser must accept
    /// exactly what `ContentId::from_hex` accepts.
    #[test]
    fn content_id_hex_decode_matches_from_hex() {
        let id = ContentId::hash_bytes(b"hex-roundtrip");
        let hex = id.to_string();
        assert_eq!(content_id_from_hex_no_alloc(&hex), Some(id));
        assert_eq!(
            content_id_from_hex_no_alloc(&hex.to_uppercase()),
            Some(ContentId::from_hex(&hex.to_uppercase()).unwrap())
        );
        for junk in ["", "abc", &format!("{}z", &hex[..63]), &format!("{hex}00")] {
            assert_eq!(
                content_id_from_hex_no_alloc(junk).is_none(),
                ContentId::from_hex(junk).is_err(),
                "decoders disagree on {junk:?}"
            );
        }
    }

    #[test]
    fn pack_index_file_round_trips_and_rejects_junk() {
        let records = vec![
            PackIndexRecord {
                kind: ObjectKind::Commit,
                content_id: ContentId::hash_bytes(b"a"),
                offset: 0,
                len: 12,
            },
            PackIndexRecord {
                kind: ObjectKind::Tree,
                content_id: ContentId::hash_bytes(b"b"),
                offset: 12,
                len: 34,
            },
        ];
        let text = format_pack_index_file(&records);
        assert_eq!(parse_pack_index_file(&text).unwrap(), records);
        assert!(parse_pack_index_file("").is_none());
        assert!(parse_pack_index_file("not-the-header\n").is_none());
        assert!(
            parse_pack_index_file(&format!("{PACK_IDX_HEADER}\ncommit nothex 0 1\n")).is_none()
        );
        assert!(
            parse_pack_index_file(&format!(
                "{PACK_IDX_HEADER}\ncommit {} 0 1 extra\n",
                ContentId::hash_bytes(b"a")
            ))
            .is_none()
        );
    }

    /// A sidecar whose payload is missing (partially deleted cache) must not
    /// be loaded — and is dropped on the spot (load-time self-heal).
    #[test]
    fn pack_index_loader_drops_sidecar_without_payload() {
        let temp_dir = tempfile::tempdir().unwrap();
        let packs_dir = temp_dir.path().join(".packs");
        fs::create_dir_all(&packs_dir).unwrap();
        let content_id = ContentId::hash_bytes(b"orphan");
        let records = vec![PackIndexRecord {
            kind: ObjectKind::Commit,
            content_id,
            offset: 0,
            len: 6,
        }];
        let idx_path = packs_dir.join("deadbeef.idx");
        fs::write(&idx_path, format_pack_index_file(&records)).unwrap();
        let index = PackResidentIndex::new(packs_dir);
        assert!(index.lookup(ObjectKind::Commit, &content_id).is_none());
        assert!(!idx_path.exists());
    }

    #[test]
    fn redact_url_queries_strips_signed_query_strings() {
        // reqwest error Display wraps the URL in parentheses; the query (which
        // carries the whole SigV4 authorization) must go, the rest must stay.
        assert_eq!(
            redact_url_queries(
                "error sending request for url (https://t3.storage.dev/bucket/key?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=abc123): timed out"
            ),
            "error sending request for url (https://t3.storage.dev/bucket/key?<redacted>): timed out"
        );
        assert_eq!(redact_url_queries("no urls here"), "no urls here");
        assert_eq!(
            redact_url_queries("got https://a/b?x=1 then https://c/d?y=2"),
            "got https://a/b?<redacted> then https://c/d?<redacted>"
        );
        assert_eq!(
            redact_url_queries("trailing https://a/b?x=1"),
            "trailing https://a/b?<redacted>"
        );
    }
}

pub fn create_store_factories() -> StoreFactories {
    create_store_factories_with_object_read_mode(VexObjectReadMode::NativeOnly)
}

/// Store factories for a Vex-backed repo load, with an explicit object-read
/// mode applied after reading `vex.json`.
///
/// Ordinary clones/loads use [`create_store_factories`] ([`VexObjectReadMode::NativeOnly`]).
/// Conversion/materialization must pass [`VexObjectReadMode::GitCompatibility`]
/// because they open repos whose op-log views may still reference raw Git
/// commit bytes; the mode cannot be inherited from disk (it is never
/// serialized into `vex.json`).
pub fn create_store_factories_with_object_read_mode(
    object_read_mode: VexObjectReadMode,
) -> StoreFactories {
    let mut store_factories = StoreFactories::empty();
    store_factories.add_backend(
        VexBackend::name_static(),
        Box::new(move |_settings, store_path| {
            Ok(Box::new(VexBackend::load_with_object_read_mode(
                store_path,
                object_read_mode,
            )?))
        }),
    );
    store_factories.add_op_store(
        VexOpStore::name_static(),
        Box::new(|_settings, store_path, root_data| {
            Ok(Box::new(VexOpStore::load(store_path, root_data)?))
        }),
    );
    store_factories.add_op_heads_store(
        VexOpHeadsStore::name_static(),
        Box::new(|_settings, store_path| Ok(Box::new(VexOpHeadsStore::load(store_path)?))),
    );
    store_factories
}
