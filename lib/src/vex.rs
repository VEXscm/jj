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
use std::io::Read;
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
use jj_backend_api::FederatedHomePathChange as ProtoFederatedHomePathChange;
use jj_backend_api::FederatedHomeSubmitOperationKind as ProtoFederatedHomeSubmitOperationKind;
use jj_backend_api::GetCloneManifestRequest;
use jj_backend_api::GetFederatedHomeManifestRequest;
use jj_backend_api::GetHydrationPacksRequest;
use jj_backend_api::GetObjectRequest;
use jj_backend_api::GetObjectsInlineRequest;
use jj_backend_api::GetObjectsRequest;
use jj_backend_api::GetRepoRequest;
use jj_backend_api::InitRepoRequest;
use jj_backend_api::InlineObject;
use jj_backend_api::ObjectId;
use jj_backend_api::PlanFederatedHomeSubmitRequest;
use jj_backend_api::PutObjectRequest;
use jj_backend_api::PutObjectsRequest;
use jj_backend_api::RefNaming;
use jj_backend_api::ResolveOperationIdPrefixRequest;
use jj_backend_api::ResolveRefsRequest;
use jj_backend_api::StageFederatedHomeSubmitPartitionRequest;
use jj_backend_api::VirtualRepositoryMount as ProtoVirtualRepositoryMount;
use jj_backend_api::get_federated_home_manifest_request::Selection as FederatedHomeManifestSelection;
use jj_backend_api::jj_backend_client::JjBackendClient;
use jj_backend_types::{
    CloneManifest, ContentId, FederatedHomeManifest, HydrationPackManifest, ObjectKind,
    ObjectPackEntry, decode_object_pack, decode_object_pack_reader,
    decode_object_pack_with_visitor, decode_pack_chunk_entries, parse_pack_header,
};
pub use jj_backend_types::{
    ContentId as VexContentId, FederatedHomeComponent as VexFederatedHomeComponent,
    FederatedHomeManifest as VexFederatedHomeManifest,
    FederatedHomePathOwner as VexFederatedHomePathOwner,
    FederatedHomePathOwnerKind as VexFederatedHomePathOwnerKind,
    FederatedHomeSubmitOperationKind as VexFederatedHomeOperationKind,
    FederatedHomeSubmitPlan as VexFederatedHomeSubmitPlan, ObjectKind as VexObjectKind,
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
// Must match the backend's single-request plan-bound Home staging boundary.
// There is intentionally no transparent generic-upload fallback for a larger
// closure; the caller gets a clear error before any remote write is attempted.
const MAX_FEDERATED_HOME_STAGE_OBJECTS: usize = 512;
const MAX_FEDERATED_HOME_STAGE_BYTES: usize = 32 * 1024 * 1024;

/// Ref namespace the freshness probe (roadmap/088, D8) scopes its token to.
///
/// Bookmarks are what a user means by "am I behind": scoping to them keeps the
/// token from churning on namespaces nobody reads (tags, mapping refs, internal
/// checkpoint refs), which would report every repository as perpetually behind.
pub const REF_FRESHNESS_PREFIX: &str = "git/ref/refs/heads/";

/// A `PutObjects` batch is content-addressed and create-if-missing, so a
/// response lost during an edge reload or backend restart may safely be
/// retried verbatim (fresh connection each attempt). Bulk materialize used to
/// fail the whole import after ~6s of INTERNAL_ERROR/stream resets while
/// jj-backend recovered from memory pressure; mirror the commit-ops ladder so
/// a brief restart/pressure window (~60–90s) is survivable. Bodies are held
/// only for the in-flight pipeline window, not the whole repo.
const PIPELINED_PUT_RETRY_ATTEMPTS: usize = 12;
const PIPELINED_PUT_RETRY_BASE_MS: u64 = 1_000;
const PIPELINED_PUT_RETRY_CAP_MS: u64 = 10_000;

/// `CommitOperation` has an explicit replay-success response for an already
/// published op head. That makes the exact maintenance rejection safe to
/// retry, but it can last much longer than a transport blip while a shadow GC
/// pass walks a large Git mirror.
/// Attempts for the inline op-head publication. Deliberately small: this runs
/// on the synchronous publish path while the working-copy lock is held, so a
/// backend that accepts connections but does not answer must surface quickly
/// as a re-runnable failure rather than freeze the repository. Operators can
/// raise it (never lower it) with
/// `VEX_COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS` to ride out a long
/// maintenance window, which is also the rollback for this bound.
const COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS: usize = 2;
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

/// Per-attempt deadline for the two RPCs on the synchronous publish path
/// (`PutObjects` flushed before the CAS, and `CommitOperation` itself).
///
/// The channel-wide `VEX_GRPC_REQUEST_TIMEOUT_SECS` default of 300 s exists
/// for bulk reads; applied to a publish it means a backend that accepts the
/// connection and never answers holds the working-copy lock — and therefore
/// every other session on the machine — for minutes per attempt. These two
/// calls take the same env var with a much shorter default, so raising the
/// variable still restores the old behavior for everything at once.
fn publish_request_timeout() -> Duration {
    Duration::from_secs(env_secs("VEX_GRPC_REQUEST_TIMEOUT_SECS", 30).max(1))
}

/// The message appended when a publish exhausts its bounded budget. The
/// working copy is untouched by a failed publication — the operation simply
/// was not recorded on the server — so the actionable advice is to re-run.
const PUBLISH_TIMEOUT_HINT: &str = "your files are untouched; re-run the command to publish again";

/// Run one publish RPC attempt under `budget`, turning "never answered" into
/// `DeadlineExceeded`. The existing retry classification already treats that
/// as transient, so a wedged attempt is retried once and then surfaces as an
/// ordinary failed publication — no new error plumbing, just a bounded wait
/// and an actionable message.
async fn publish_attempt_within<T, Fut>(
    budget: Duration,
    rpc: &str,
    call: Fut,
) -> Result<T, tonic::Status>
where
    Fut: Future<Output = Result<T, tonic::Status>>,
{
    match tokio::time::timeout(budget, call).await {
        Ok(result) => result,
        Err(_elapsed) => Err(tonic::Status::deadline_exceeded(format!(
            "the backend did not answer {rpc} within {}s; {PUBLISH_TIMEOUT_HINT}",
            budget.as_secs()
        ))),
    }
}

/// Process-wide HTTP/2 `:authority` override for every Vex gRPC channel.
///
/// The Vex hosted-listener path dials a stable VIP (`http://10.88.0.2:8444`)
/// that routes by `:authority` to an internal service name
/// (e.g. `grpc.jj-backend.internal`). tonic derives `:authority` from the
/// dialed URI, so without an override the listener's catch-all 404s. Set once
/// per process (CLI flag `vex materialize --grpc-authority`, or the
/// `VEX_GRPC_AUTHORITY` env var) before the first channel is built; blank
/// values are ignored and the dialed host:port stays the authority (today's
/// behavior).
static GRPC_AUTHORITY_OVERRIDE: OnceLock<String> = OnceLock::new();

/// Set the `:authority` presented on every subsequent Vex gRPC connection in
/// this process. Returns `false` (and changes nothing) if an override was
/// already set or `authority` is blank. Channels are cached per endpoint, so
/// call this before the first Vex RPC of the process.
pub fn set_grpc_authority_override(authority: &str) -> bool {
    let authority = authority.trim();
    if authority.is_empty() {
        return false;
    }
    GRPC_AUTHORITY_OVERRIDE.set(authority.to_string()).is_ok()
}

/// The effective `:authority` override: the programmatic setter wins, then the
/// `VEX_GRPC_AUTHORITY` env var; `None` (the default) keeps tonic's derivation
/// from the dialed URI.
fn grpc_authority_override() -> Option<String> {
    if let Some(authority) = GRPC_AUTHORITY_OVERRIDE.get() {
        return Some(authority.clone());
    }
    std::env::var("VEX_GRPC_AUTHORITY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// The gRPC metadata key carrying the running client's version.
///
/// PRD 088 **D14** condition 3 ("no org is pinned to a CLI older than the
/// reference release") was literally unmeasurable while the client sent nothing
/// but an `authorization` header: the server's release registry lists *published*
/// builds, not running ones. This header is the smallest honest fix — the client
/// states what it is, once, on every request it already makes, and the server
/// records a last-seen version per tenant off the request path.
pub const CLIENT_VERSION_METADATA_KEY: &str = "x-vex-cli-version";

/// Process-wide running-client version, reported as `x-vex-cli-version`.
///
/// Set once at startup by the binary that knows it — `vex-cli` passes
/// `env!("VEX_VERSION")`, the same string `vex --version` prints. `jj_lib` is a
/// library and deliberately does not guess: an embedder that never calls
/// [`set_client_version`] and sets no `VEX_CLI_VERSION` simply sends no header,
/// and the server reports it as unknown rather than as a version it made up.
static CLIENT_VERSION: OnceLock<String> = OnceLock::new();

/// Declare the running client's version for every subsequent Vex gRPC request in
/// this process. Returns `false` (and changes nothing) if a version was already
/// set or `version` is blank.
///
/// The value is sent **verbatim**, including a dev build's `-<commit>` suffix.
/// That suffix is information — it distinguishes a source build from the clean
/// release of the same semver — and stripping it would make a locally built
/// client indistinguishable from a shipped one in the compatibility gate.
pub fn set_client_version(version: &str) -> bool {
    let version = version.trim();
    if version.is_empty() {
        return false;
    }
    CLIENT_VERSION.set(version.to_string()).is_ok()
}

/// The `x-vex-cli-version` value to send, or `None` to send no header.
///
/// Resolution: the programmatic setter wins, then the `VEX_CLI_VERSION` env var
/// (which lets a wrapper or a test declare a version without linking the CLI).
/// A value that is not printable ASCII, or is longer than the server stores, is
/// dropped rather than sent — the server would reject it anyway, and a header
/// this client cannot vouch for is worse than no header.
fn client_version_metadata() -> Option<MetadataValue<tonic::metadata::Ascii>> {
    let raw = CLIENT_VERSION.get().cloned().or_else(|| {
        std::env::var("VEX_CLI_VERSION")
            .ok()
            .map(|value| value.trim().to_string())
    })?;
    validated_client_version_metadata(&raw)
}

/// The validation half, split out so it is testable without touching the
/// process-wide [`CLIENT_VERSION`] cell.
///
/// A value that is not printable ASCII, or is longer than the server stores, is
/// dropped rather than sent: the server would reject it anyway, and a header
/// this client cannot vouch for is worse than no header — the gate reads a
/// missing header as *unknown*, which is the honest answer.
fn validated_client_version_metadata(raw: &str) -> Option<MetadataValue<tonic::metadata::Ascii>> {
    let raw = raw.trim();
    if raw.is_empty() || !raw.chars().all(|ch| ch.is_ascii_graphic()) {
        return None;
    }
    let shortened;
    let value = if raw.len() > MAX_CLIENT_VERSION_METADATA_LEN {
        // A dev build reports `X.Y.Z-<full 64-char commit>` — 70 characters,
        // over the server's 64. Dropping it silently made every source-built
        // client invisible to the compatibility gate, which is exactly
        // backwards: developers and agents run dev builds, so the population
        // most likely to be on old code was the one the telemetry could not
        // see. Observed 2026-07-29, when a 1.1.0 dev build produced no sighting
        // at all while CI runners on the clean release build did.
        //
        // Shorten the commit suffix rather than dropping the header. The
        // suffix's job is to distinguish a source build from the shipped
        // release of the same semver and to identify the commit; 12 hex
        // characters do both, and match the short-sha convention used for
        // image tags and the CLI release registry. The semver core — the part
        // the gate actually compares — is never touched.
        let (core, suffix) = raw.split_once('-')?;
        shortened = format!("{core}-{}", &suffix[..suffix.len().min(12)]);
        if shortened.len() > MAX_CLIENT_VERSION_METADATA_LEN {
            return None;
        }
        shortened.as_str()
    } else {
        raw
    };
    MetadataValue::try_from(value).ok()
}

/// Mirrors the server's `jj_client_version_sightings.cli_version VARCHAR(64)`.
/// The longest legitimate value is a dev build's `X.Y.Z-<40-char commit>`.
const MAX_CLIENT_VERSION_METADATA_LEN: usize = 64;

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
/// Guard held for the duration of one server RPC.
///
/// Every RPC constructs one, so this is the single place that knows a command
/// blocked on the server. It does two things: counts the call (always, so how
/// much a command leans on the server can be checked by a number rather than by
/// inspection), and prints a line when `rpc_timing_enabled()`.
///
/// It deliberately does NOT hold a tracing span: an `EnteredSpan` kept across an
/// await makes the enclosing future `!Send`. Span coverage of the round trip
/// comes from `#[tracing::instrument]` on the RPC methods themselves, which
/// wraps the future rather than entering a guard.
struct RpcTimer {
    label: String,
    start: std::time::Instant,
    print: bool,
}

impl RpcTimer {
    fn start(label: impl FnOnce() -> String) -> Self {
        let label = label();
        vex_client_stats()
            .blocking_rpcs
            .fetch_add(1, Ordering::Relaxed);
        Self {
            label,
            start: std::time::Instant::now(),
            print: rpc_timing_enabled(),
        }
    }
}

impl Drop for RpcTimer {
    fn drop(&mut self) {
        if !self.print {
            return;
        }
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
    /// Server RPCs issued on the calling command's critical path — every call
    /// the command blocked on, whatever its kind.
    ///
    /// This is the number that answers "did this command take the server off
    /// its critical path": a non-zero count means it did not.
    pub blocking_rpcs: AtomicU64,
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
    /// Wall-clock milliseconds spent acquiring presigned URL fetch hints
    /// (the `GetObjects` hint RPC), summed per call.
    pub pack_presign_wait_ms: AtomicU64,
    /// Wall-clock milliseconds awaiting presigned HTTP GETs (chunk fetches
    /// and streamed whole packs), summed per request across concurrent
    /// workers — with W-way concurrency the sum can exceed wall time.
    pub pack_http_wait_ms: AtomicU64,
    /// Per-pack download wall time (chunked path), summed across packs.
    /// Concurrent pack workers make the sum exceed wall time by design.
    pub pack_download_ms: AtomicU64,
    /// Per-pack decode+unpack wall time, summed across packs (chunked and
    /// whole-pack fallback paths).
    pub pack_unpack_ms: AtomicU64,
    /// Wall-clock milliseconds spent in the serial per-object `GetObject`
    /// loop that tails the clone-manifest prefetch.
    pub pack_loose_object_ms: AtomicU64,
    /// `GetOpHeads` RPCs issued, including budgeted freshness refreshes
    /// (roadmap/088). Zero on the hot path of a converged local-first repo.
    pub op_head_rpcs: AtomicU64,
    /// Op-head resolutions served from the local marker files with no RPC.
    pub op_head_local_serves: AtomicU64,
    /// Budgeted freshness refreshes attempted.
    pub refresh_attempts: AtomicU64,
    /// Freshness refreshes that exceeded their budget (or failed) and
    /// silently degraded to the local heads.
    pub refresh_timeouts: AtomicU64,
    /// Operations sitting in the pending-publish chain at the last publisher
    /// run (a level, not a total).
    pub pending_ops: AtomicU64,
    /// Op-head CAS attempts rejected because the server head had moved.
    pub publish_cas_conflicts: AtomicU64,
    /// Pending chains republished onto a newer server head after a conflict.
    pub publish_folds: AtomicU64,
    /// Local operations published as part of a coalesced chain instead of
    /// their own CAS.
    pub coalesced_ops: AtomicU64,
    /// Wall-clock milliseconds of the last publisher drain.
    pub publish_lag_ms: AtomicU64,
}

macro_rules! for_each_vex_client_stat {
    ($macro:ident) => {
        $macro!(
            blocking_rpcs,
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
            objects_unpacked,
            objects_pack_resident,
            loose_writes_avoided,
            hydrated_objects,
            hydrated_bytes,
            files_written,
            bytes_written,
            files_reflinked,
            native_trunk_resolutions,
            native_trunk_missing,
            git_compat_commit_decodes,
            git_compat_tree_decodes,
            git_mapping_names_resolved,
            git_mapping_rpcs,
            git_mapping_elapsed_ms,
            pack_presign_wait_ms,
            pack_http_wait_ms,
            pack_download_ms,
            pack_unpack_ms,
            pack_loose_object_ms,
            op_head_rpcs,
            op_head_local_serves,
            refresh_attempts,
            refresh_timeouts,
            pending_ops,
            publish_cas_conflicts,
            publish_folds,
            coalesced_ops,
            publish_lag_ms
        )
    };
}

/// Plain-value copy of [`VexClientStats`] taken at one point in time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct VexClientStatsSnapshot {
    pub blocking_rpcs: u64,
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
    pub objects_unpacked: u64,
    pub objects_pack_resident: u64,
    pub loose_writes_avoided: u64,
    pub hydrated_objects: u64,
    pub hydrated_bytes: u64,
    pub files_written: u64,
    pub bytes_written: u64,
    pub files_reflinked: u64,
    pub native_trunk_resolutions: u64,
    pub native_trunk_missing: u64,
    pub git_compat_commit_decodes: u64,
    pub git_compat_tree_decodes: u64,
    pub git_mapping_names_resolved: u64,
    pub git_mapping_rpcs: u64,
    pub git_mapping_elapsed_ms: u64,
    pub pack_presign_wait_ms: u64,
    pub pack_http_wait_ms: u64,
    pub pack_download_ms: u64,
    pub pack_unpack_ms: u64,
    pub pack_loose_object_ms: u64,
    pub op_head_rpcs: u64,
    pub op_head_local_serves: u64,
    pub refresh_attempts: u64,
    pub refresh_timeouts: u64,
    pub pending_ops: u64,
    pub publish_cas_conflicts: u64,
    pub publish_folds: u64,
    pub coalesced_ops: u64,
    pub publish_lag_ms: u64,
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
    #[error("invalid Home checkout metadata: {0}")]
    InvalidFederatedHome(String),
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
    /// A staged-upload marker names an object whose bytes are no longer in the
    /// local cache, so the work it belongs to cannot be published from this
    /// workspace. Its own variant because it is not an IO or transport failure:
    /// nothing is retryable, and the fix is a human one.
    #[error("{0}")]
    StagedObjectsMissing(String),
}

pub type VexObjectId = (VexObjectKind, VexContentId);

/// A backend-issued flat Home submit plan plus one capability for each
/// nonempty physical partition. The capabilities remain in-process only and
/// are never written to Home checkout metadata or Rails payloads.
#[derive(Debug, Clone)]
pub struct VexFederatedHomeSubmitPlanResponse {
    pub plan: VexFederatedHomeSubmitPlan,
    stage_capabilities: HashMap<String, String>,
}

impl VexFederatedHomeSubmitPlanResponse {
    pub fn stage_capability(&self, repository_id: &str) -> Option<&str> {
        self.stage_capabilities
            .get(repository_id)
            .map(String::as_str)
    }
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

pub const VEX_FEDERATED_HOME_FORMAT_VERSION: u32 = 1;
const VEX_FEDERATED_HOME_METADATA_FILE: &str = "federated-home.json";

/// Hidden local routing state for one flat Home checkout.
///
/// `manifest` is the exact canonical Rust contract returned by jj-backend.
/// Slugs, public ids, endpoints, and credentials are deliberately outside it so
/// they cannot affect its content digest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VexFederatedHomeConfig {
    pub format_version: u32,
    pub manifest_artifact_suffix: String,
    pub manifest_content_sha256: String,
    /// Exact generation of jj-backend's mutable current-manifest pointer at
    /// clone time. Submit must observe this same current pointer.
    pub manifest_generation: u64,
    pub manifest: FederatedHomeManifest,
    /// Single short-lived Home aggregate credential, bound by jj-backend to
    /// this exact manifest pointer and its component selections. It is never a
    /// reusable credential for an individual physical repository.
    pub aggregate_access_token: String,
    /// Home first, followed by components in manifest order.
    pub repositories: Vec<VexFederatedHomeRepository>,
    /// The locally synthesized flat snapshot. It is not part of the canonical
    /// ownership manifest and is filled in only after clone composition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate_base_commit_id: Option<ContentId>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VexFederatedHomeRepository {
    /// Exact `jj-backend` `RepoInfo.repo_id` resolved with the Home aggregate
    /// credential. It is explicitly not Rails `Repository#id`.
    pub repository_id: String,
    /// Rails public id and slug are request/display identity only.
    pub repository_public_id: String,
    pub repository_slug: String,
    /// Empty for Home; otherwise the Home-relative ownership root.
    pub root_path: String,
    pub endpoint: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VexFederatedHomeManifestResponse {
    pub manifest_artifact_suffix: String,
    pub manifest_content_sha256: String,
    pub manifest_generation: u64,
    pub manifest: FederatedHomeManifest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VexFederatedHomePathChange {
    pub path: String,
    pub operation: VexFederatedHomeOperationKind,
}

fn require_current_federated_home_manifest(is_current: bool) -> Result<(), VexClientError> {
    if is_current {
        Ok(())
    } else {
        Err(VexConfigError::InvalidFederatedHome(
            "the Home configuration changed since this checkout was cloned".to_string(),
        )
        .into())
    }
}

fn validate_federated_home_pointer(
    pointer: &jj_backend_api::FederatedHomeManifestPointer,
    expected_suffix: &str,
    expected_digest: &str,
    expected_generation: Option<u64>,
) -> Result<(), VexClientError> {
    if pointer.format_version != VEX_FEDERATED_HOME_FORMAT_VERSION
        || pointer.generation == 0
        || pointer.manifest_artifact_suffix != expected_suffix
        || pointer.manifest_content_sha256 != expected_digest
    {
        return Err(VexConfigError::InvalidFederatedHome(
            "backend current-manifest pointer is incoherent".to_string(),
        )
        .into());
    }
    if expected_generation.is_some_and(|generation| generation != pointer.generation) {
        return Err(VexConfigError::InvalidFederatedHome(
            "the flat Home current-manifest pointer generation changed since clone".to_string(),
        )
        .into());
    }
    Ok(())
}

impl VexFederatedHomeConfig {
    pub fn metadata_path_for_repo(repo_path: &Path) -> PathBuf {
        repo_path.join(VEX_FEDERATED_HOME_METADATA_FILE)
    }

    pub fn load_from_repo_path(repo_path: &Path) -> Result<Option<Self>, VexConfigError> {
        let path = Self::metadata_path_for_repo(repo_path);
        if !path.exists() {
            return Ok(None);
        }
        let config: Self = serde_json::from_slice(&fs::read(path)?)?;
        config.validate()?;
        Ok(Some(config))
    }

    pub fn write_to_repo_path(&self, repo_path: &Path) -> Result<(), VexConfigError> {
        self.validate()?;
        let path = Self::metadata_path_for_repo(repo_path);
        let parent = path
            .parent()
            .ok_or_else(|| VexConfigError::InvalidStorePath(path.clone()))?;
        let mut temporary = NamedTempFile::new_in(parent)?;
        temporary.write_all(&serde_json::to_vec_pretty(self)?)?;
        temporary.persist(path).map_err(|error| error.error)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<(), VexConfigError> {
        if self.format_version != VEX_FEDERATED_HOME_FORMAT_VERSION
            || self.format_version != self.manifest.format_version
        {
            return Err(VexConfigError::InvalidFederatedHome(format!(
                "unsupported format version {}",
                self.format_version
            )));
        }
        self.manifest.validate().map_err(|_| {
            VexConfigError::InvalidFederatedHome("Home manifest path layout is invalid".to_string())
        })?;
        let digest = self
            .manifest
            .content_sha256()
            .map_err(|error| VexConfigError::InvalidFederatedHome(error.to_string()))?;
        let suffix = self
            .manifest
            .artifact_suffix()
            .map_err(|error| VexConfigError::InvalidFederatedHome(error.to_string()))?;
        if digest != self.manifest_content_sha256 || suffix != self.manifest_artifact_suffix {
            return Err(VexConfigError::InvalidFederatedHome(
                "backend manifest bytes, digest, and artifact suffix disagree".to_string(),
            ));
        }
        if self.manifest_generation == 0 {
            return Err(VexConfigError::InvalidFederatedHome(
                "current manifest pointer generation must be positive".to_string(),
            ));
        }
        if !self
            .aggregate_access_token
            .strip_prefix("vexhome_")
            .is_some_and(|token| !token.is_empty())
        {
            return Err(VexConfigError::InvalidFederatedHome(
                "Home routing requires a manifest-bound aggregate credential".to_string(),
            ));
        }
        if self.repositories.len() != self.manifest.components.len() + 1 {
            return Err(VexConfigError::InvalidFederatedHome(
                "repository access routes do not exactly cover the manifest".to_string(),
            ));
        }
        let expected = std::iter::once((self.manifest.home_repository_id.as_str(), "")).chain(
            self.manifest.components.iter().map(|component| {
                (
                    component.repository_id.as_str(),
                    component.root_path.as_str(),
                )
            }),
        );
        for (repository, (repository_id, root_path)) in self.repositories.iter().zip(expected) {
            if repository.repository_id != repository_id
                || repository.root_path != root_path
                || repository.repository_public_id.trim().is_empty()
                || repository.repository_slug.trim().is_empty()
                || repository.endpoint.trim().is_empty()
            {
                return Err(VexConfigError::InvalidFederatedHome(format!(
                    "invalid or out-of-order access route for repository {}",
                    repository.repository_id
                )));
            }
        }
        Ok(())
    }

    pub fn home(&self) -> &VexFederatedHomeRepository {
        &self.repositories[0]
    }

    pub fn component_repository(&self, component_index: usize) -> &VexFederatedHomeRepository {
        &self.repositories[component_index + 1]
    }

    pub fn repository_config(
        &self,
        home_config: &VexRepoConfig,
        repository: &VexFederatedHomeRepository,
    ) -> VexRepoConfig {
        VexRepoConfig {
            endpoint: repository.endpoint.clone(),
            tenant_id: home_config.tenant_id.clone(),
            tenant_slug: home_config.tenant_slug.clone(),
            repo_id: repository.repository_id.clone(),
            repo_slug: repository.repository_slug.clone(),
            // Component object and staged-write RPCs are authorized as part of
            // the single Home facade, never as raw physical repository access.
            repository_scope_kind: Some("composed".to_string()),
            virtual_repository_id: None,
            backing_repo_slug: None,
            virtual_root_path: None,
            virtual_mounts: Vec::new(),
            access_token: Some(self.aggregate_access_token.clone()),
            local_writes: false,
            object_read_mode: VexObjectReadMode::NativeOnly,
        }
    }
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
    /// `.jj/repo` for a disk-backed client. Flat Home clients lazily read the
    /// hidden federated config here after a cache miss, including clients that
    /// were opened during clone before that config was persisted.
    repo_path: Option<PathBuf>,
    /// Immutable physical read routes resolved from hidden flat Home metadata.
    /// An absent metadata file is not cached: clone writes it only after the
    /// synthetic aggregate base is readable, and the already-open backend must
    /// discover it afterward without being rebuilt.
    federated_read_routes: Arc<OnceLock<Vec<VexRepoConfig>>>,
    /// This store is the local-only synthetic Home facade. Ordinary staged
    /// publication (notably `vex push`) must never send its objects to Home.
    federated_facade: bool,
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

/// Max bytes per staged-upload `PutObjects` batch. Bounds peak memory for a
/// push that publishes a very large staged set while still letting a normal
/// push (a handful of small objects) go out as a single batch.
const PENDING_FLUSH_BYTES: usize = 32 * 1024 * 1024;
/// Companion object-count cap to [`PENDING_FLUSH_BYTES`].
const PENDING_FLUSH_OBJECTS: usize = 256;
/// Concurrent in-flight `PutObjects` batches while publishing staged objects,
/// and — because the batches of one wave are read into memory together — the
/// number of batches held at once. Peak upload memory is therefore bounded at
/// roughly `STAGED_UPLOAD_CONCURRENCY * PENDING_FLUSH_BYTES`.
const STAGED_UPLOAD_CONCURRENCY: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagedMarkerPolicy {
    Preserve,
    Remove,
}

/// Max objects per `GetObjectsInline` batch (read-side analogue of
/// [`PENDING_FLUSH_OBJECTS`]).
const INLINE_FETCH_BATCH_OBJECTS: usize = 256;
/// Estimated-bytes cap per `GetObjectsInline` batch. The *response* carries the
/// object bodies and must stay under [`MAX_GRPC_MESSAGE_BYTES`]; sizes are only
/// hints (tree entries don't record them), so leave generous headroom.
const INLINE_FETCH_BATCH_BYTES: u64 = 24 * 1024 * 1024;
/// Default concurrent in-flight `GetObjectsInline` batches. Overridable via
/// `VEX_CLONE_HYDRATION_CONCURRENCY` for clone benchmarking and tuning.
/// A 2026-07-29 production sweep on `vex/home` measured median hydration at
/// 14.37s (8), 11.77s (12), and 12.70s (16), so 12 keeps the object store busy
/// without the contention seen at the wider setting.
const INLINE_FETCH_CONCURRENCY: usize = 12;

fn parse_inline_fetch_concurrency(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 1)
        .unwrap_or(INLINE_FETCH_CONCURRENCY)
}

fn inline_fetch_concurrency() -> usize {
    parse_inline_fetch_concurrency(
        std::env::var("VEX_CLONE_HYDRATION_CONCURRENCY")
            .ok()
            .as_deref(),
    )
}

/// Default number of clone packs fetched+unpacked in parallel during
/// [`VexClient::prefetch_clone_manifest`]. Overridable via
/// `VEX_CLONE_PACK_CONCURRENCY` (set `1` to restore the serial pack loop).
/// A 2026-07-22 production sweep (271.4 MB pinned JJ fixture) showed
/// throughput flattening around ~38 MB/s once enough requests were in
/// flight; with the server-side pack reshape (~16 MiB compressed packs,
/// 2 MiB chunks) far fewer, larger transfers carry the same bytes, so 8 pack
/// workers keep the pipe full without writer-contention variance.
const PACK_FETCH_CONCURRENCY: usize = 8;

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
/// chunk order. Chunks are now typically 2 MiB (server-side pack reshape), so
/// peak buffered memory is bounded at pack workers × W × 2 MiB — 8 × 16 ×
/// 2 MiB = 256 MB worst case, the same bound as the previous 16×32×512 KiB
/// defaults. The 2026-07-22 sweep showed very wide W regressing from
/// head-of-line blocking in the index-ordered `.part` writer, so keep W
/// moderate. Overridable via `VEX_CLONE_CHUNK_CONCURRENCY` (set `1` to
/// restore the serial chunk loop); `VEX_CLONE_PACK_CONCURRENCY` covers the
/// pack-level knob.
const CHUNK_FETCH_CONCURRENCY: usize = 16;

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

/// Process-wide pack-resident indexes, keyed by cache root. Process-global
/// because the three Vex stores of one repo — object backend, op store, op
/// heads store — each hold their own [`VexClient`], and all of them must see
/// one coherent view of the index (including self-heal drops and prune
/// invalidation).
static PACK_INDEXES: OnceLock<Mutex<HashMap<PathBuf, Arc<PackResidentIndex>>>> = OnceLock::new();

/// In-memory overlay of the pack-resident metadata cache (roadmap/032
/// follow-up). Unpacking a clone pack appends the metadata entries'
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
            repo_path: None,
            federated_read_routes: Arc::new(OnceLock::new()),
            federated_facade: false,
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

    /// Repository *initialization*: use the caller's in-memory config (nothing
    /// is on disk to load yet) but bind the object cache to `store_path`, the
    /// same place every later process will look.
    ///
    /// [`Self::from_config`] deliberately has no cache, which was harmless
    /// while every write also uploaded. Since roadmap/088 Stage 7 the cache is
    /// where a write's *only* copy lives until `vex push` publishes it, so an
    /// `init` whose backend had no cache would drop the repository's very first
    /// commit and tree on the floor: nothing local to read them back from, and
    /// nothing staged for a push to publish.
    pub fn from_config_at(
        config: VexRepoConfig,
        store_path: &Path,
    ) -> Result<Self, VexConfigError> {
        Self::from_store_path_and_config(store_path, config)
    }

    /// Bind another repository-scoped client to this checkout's single object
    /// cache. Federated Home uses this for component prefetch/publication; it
    /// does not create another store, workspace, or `.jj` directory.
    pub fn from_config_with_cache_root(
        config: VexRepoConfig,
        cache_root: PathBuf,
    ) -> Result<Self, VexConfigError> {
        Self::validate_endpoint(&config.endpoint)?;
        fs::create_dir_all(&cache_root)?;
        let local_writes = config.local_writes;
        Ok(Self {
            config,
            cache_root: Some(cache_root),
            repo_path: None,
            federated_read_routes: Arc::new(OnceLock::new()),
            federated_facade: false,
            cache_max_bytes: cache_max_bytes(),
            local_writes,
            fresh_cache: false,
            pack_resident_override: None,
            presigned_get_disabled: Arc::new(AtomicBool::new(false)),
        })
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

    /// Like [`Self::from_store_path`], but talks to `endpoint` with
    /// `access_token` instead of whatever `vex.json` recorded at clone time.
    ///
    /// `vex push` resolves auth for the command it is running (which may be a
    /// freshly minted token, or a different endpoint than the clone used) and
    /// then has to publish this workspace's staged objects with it. Everything
    /// else — repo id, tenant id, and therefore the object cache the staged
    /// markers live in — still comes from the workspace on disk, because that
    /// is the repository whose work is being published.
    pub fn from_store_path_with_auth(
        store_path: &Path,
        endpoint: &str,
        access_token: Option<&str>,
    ) -> Result<Self, VexConfigError> {
        let mut config = VexRepoConfig::load_from_store_path(store_path)?;
        config.endpoint = endpoint.to_string();
        config.access_token = access_token.map(ToOwned::to_owned);
        Self::from_store_path_and_config(store_path, config)
    }

    /// This client's repository as the catalog names it: `(tenant, repo)`
    /// slugs, so a caller can check that a workspace on disk really is the
    /// repository a command is targeting before acting on its behalf.
    pub fn repo_slugs(&self) -> (&str, &str) {
        (&self.config.tenant_slug, &self.config.repo_slug)
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
        let federated_facade = VexFederatedHomeConfig::metadata_path_for_repo(repo_path).is_file();
        Ok(Self {
            config,
            cache_root: Some(cache_root),
            repo_path: Some(repo_path.to_path_buf()),
            federated_read_routes: Arc::new(OnceLock::new()),
            federated_facade,
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

    pub fn cache_root(&self) -> Option<&Path> {
        self.cache_root.as_deref()
    }

    fn federated_read_routes(&self) -> Result<&[VexRepoConfig], VexClientError> {
        if let Some(routes) = self.federated_read_routes.get() {
            return Ok(routes);
        }
        let Some(repo_path) = self.repo_path.as_deref() else {
            return Ok(&[]);
        };
        let Some(flat_home) = VexFederatedHomeConfig::load_from_repo_path(repo_path)? else {
            // Clone persists the config only after synthesizing the flat base.
            // Do not memoize absence: this same backend remains live for the
            // initial checkout and must discover the routes if pruning causes
            // a miss after persistence.
            return Ok(&[]);
        };
        if flat_home.home().repository_id != self.config.repo_id {
            return Err(VexConfigError::InvalidFederatedHome(format!(
                "hidden Home repository {} does not match the checkout backend {}",
                flat_home.home().repository_id,
                self.config.repo_id
            ))
            .into());
        }
        let routes = flat_home
            .repositories
            .iter()
            .skip(1)
            .map(|repository| flat_home.repository_config(&self.config, repository))
            .collect::<Vec<_>>();
        for route in &routes {
            Self::validate_endpoint(&route.endpoint)?;
        }
        // Another concurrent read may have initialized the same immutable
        // routes. Either value was derived from the same validated hidden
        // config, so use the winner.
        drop(self.federated_read_routes.set(routes));
        Ok(self
            .federated_read_routes
            .get()
            .map(Vec::as_slice)
            .unwrap_or(&[]))
    }

    fn supports_federated_object_fallback(kind: ObjectKind) -> bool {
        matches!(
            kind,
            ObjectKind::Blob | ObjectKind::Symlink | ObjectKind::Tree | ObjectKind::Commit
        )
    }

    fn is_not_found(error: &VexClientError) -> bool {
        matches!(error, VexClientError::Status(status) if status.code() == tonic::Code::NotFound)
    }

    /// Fetch the exact canonical manifest selected by the backend's durable
    /// current pointer. The response digest is checked against both the bytes
    /// received and Rust's canonical serde encoding before it can be persisted.
    pub async fn get_current_federated_home_manifest(
        &self,
    ) -> Result<VexFederatedHomeManifestResponse, VexClientError> {
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .get_federated_home_manifest(Self::auth_request(
                        GetFederatedHomeManifestRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            selection: Some(FederatedHomeManifestSelection::Current(true)),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        let bytes_digest = ContentId::hash_bytes(&response.manifest_json).to_hex();
        if bytes_digest != response.manifest_content_sha256 {
            return Err(VexConfigError::InvalidFederatedHome(
                "backend manifest response digest does not match its bytes".to_string(),
            )
            .into());
        }
        let manifest: FederatedHomeManifest =
            serde_json::from_slice(&response.manifest_json).map_err(VexConfigError::Json)?;
        manifest.validate().map_err(|_| {
            VexConfigError::InvalidFederatedHome("Home manifest path layout is invalid".to_string())
        })?;
        let canonical_digest = manifest
            .content_sha256()
            .map_err(|error| VexConfigError::InvalidFederatedHome(error.to_string()))?;
        let canonical_suffix = manifest
            .artifact_suffix()
            .map_err(|error| VexConfigError::InvalidFederatedHome(error.to_string()))?;
        if canonical_digest != response.manifest_content_sha256
            || canonical_suffix != response.manifest_artifact_suffix
        {
            return Err(VexConfigError::InvalidFederatedHome(
                "backend manifest response is not canonical Rust serde JSON".to_string(),
            )
            .into());
        }
        require_current_federated_home_manifest(response.is_current)?;
        let pointer = response.current_pointer.as_ref().ok_or_else(|| {
            VexConfigError::InvalidFederatedHome(
                "backend omitted the current-manifest pointer".to_string(),
            )
        })?;
        validate_federated_home_pointer(
            pointer,
            &response.manifest_artifact_suffix,
            &response.manifest_content_sha256,
            None,
        )?;
        Ok(VexFederatedHomeManifestResponse {
            manifest_artifact_suffix: response.manifest_artifact_suffix,
            manifest_content_sha256: response.manifest_content_sha256,
            manifest_generation: pointer.generation,
            manifest,
        })
    }

    /// Ask the backend to deterministically route a flat facade diff through
    /// the exact manifest artifact cloned into this workspace. Hidden local
    /// metadata supplies credentials only; it never decides path ownership.
    pub async fn plan_federated_home_submit(
        &self,
        manifest: &FederatedHomeManifest,
        manifest_artifact_suffix: &str,
        manifest_generation: u64,
        changes: Vec<VexFederatedHomePathChange>,
    ) -> Result<VexFederatedHomeSubmitPlanResponse, VexClientError> {
        let response = Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| {
            let changes = changes.clone();
            async move {
                client
                    .plan_federated_home_submit(Self::auth_request(
                        PlanFederatedHomeSubmitRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            manifest_artifact_suffix: manifest_artifact_suffix.to_string(),
                            changes: changes
                                .into_iter()
                                .map(|change| ProtoFederatedHomePathChange {
                                    path: change.path,
                                    operation: match change.operation {
                                        VexFederatedHomeOperationKind::Delete => {
                                            ProtoFederatedHomeSubmitOperationKind::Delete as i32
                                        }
                                        VexFederatedHomeOperationKind::Upsert => {
                                            ProtoFederatedHomeSubmitOperationKind::Upsert as i32
                                        }
                                    },
                                })
                                .collect(),
                            // Cross-owner renames are intentionally already
                            // represented as source delete + destination upsert.
                            renames: Vec::new(),
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            }
        })?;
        require_current_federated_home_manifest(response.manifest_is_current)?;
        let expected_digest = manifest
            .content_sha256()
            .map_err(|error| VexConfigError::InvalidFederatedHome(error.to_string()))?;
        if response.manifest_artifact_suffix != manifest_artifact_suffix
            || response.manifest_content_sha256 != expected_digest
        {
            return Err(VexConfigError::InvalidFederatedHome(
                "backend submit plan does not identify the checkout manifest".to_string(),
            )
            .into());
        }
        let pointer = response.current_pointer.as_ref().ok_or_else(|| {
            VexConfigError::InvalidFederatedHome(
                "backend submit plan omitted the current-manifest pointer".to_string(),
            )
        })?;
        validate_federated_home_pointer(
            pointer,
            manifest_artifact_suffix,
            &expected_digest,
            Some(manifest_generation),
        )?;
        let plan: VexFederatedHomeSubmitPlan = serde_json::from_slice(&response.submit_plan_json)
            .map_err(|error| {
            VexConfigError::InvalidFederatedHome(format!(
                "cannot decode backend submit plan: {error}"
            ))
        })?;
        plan.validate_resolved_targets(manifest).map_err(|_| {
            VexConfigError::InvalidFederatedHome(
                "Home submit plan does not match this checkout or lacks target CAS state"
                    .to_string(),
            )
        })?;
        let required_capabilities = plan
            .partitions
            .iter()
            .filter(|partition| !partition.operations.is_empty())
            .map(|partition| partition.repository_id.as_str())
            .collect::<HashSet<_>>();
        let mut stage_capabilities = HashMap::with_capacity(response.partition_capabilities.len());
        for capability in response.partition_capabilities {
            if capability.repository_id.is_empty()
                || capability.capability.is_empty()
                || !required_capabilities.contains(capability.repository_id.as_str())
                || stage_capabilities
                    .insert(capability.repository_id, capability.capability)
                    .is_some()
            {
                return Err(VexConfigError::InvalidFederatedHome(
                    "backend returned invalid Home staging capabilities".to_string(),
                )
                .into());
            }
        }
        if stage_capabilities.len() != required_capabilities.len() {
            return Err(VexConfigError::InvalidFederatedHome(
                "backend omitted a Home staging capability".to_string(),
            )
            .into());
        }
        Ok(VexFederatedHomeSubmitPlanResponse {
            plan,
            stage_capabilities,
        })
    }

    /// Whether this client is in local-write mode (READ_ONLY CI runner): writes
    /// resolve to the local cache instead of the backend. See `put_object` and
    /// [`crate::vex_op_heads_store::VexOpHeadsStore`].
    pub fn local_writes(&self) -> bool {
        self.local_writes
    }

    /// Read this repository's opaque ref-freshness token (D8) under a hard
    /// wall-clock budget covering the connect handshake, with no retries.
    /// `Ok(None)` means the budget expired — the caller reports "unknown"
    /// freshness and the command carries on.
    ///
    /// The token is **opaque**: equality-compared only, never ordered, never
    /// parsed. The server derives it from per-ref state under
    /// [`REF_FRESHNESS_PREFIX`]; a different token means the refs under that
    /// scope have moved, and nothing more. It deliberately does *not* fall
    /// back to `get_op_heads`: repointing a ref question at the op log is
    /// exactly the coupling Stage 7 removes.
    ///
    /// Never fatal to a command (D8). Every caller in [`crate::vex_freshness`]
    /// turns an error into `Unknown`, and this runs after the command's output
    /// is done, so it is not on any critical path — which is why it counts
    /// against `refresh_attempts` rather than `blocking_rpcs`.
    pub fn ref_freshness_token_within(
        &self,
        budget: Duration,
    ) -> Result<Option<String>, VexClientError> {
        self.ref_freshness_token_for_within(REF_FRESHNESS_PREFIX, budget)
    }

    /// [`Self::ref_freshness_token_within`] scoped to an explicit ref-name
    /// prefix. An empty prefix means every ref outside the immutable
    /// `git/object/sha1/` mapping namespace.
    pub fn ref_freshness_token_for_within(
        &self,
        prefix: &str,
        budget: Duration,
    ) -> Result<Option<String>, VexClientError> {
        let _t = RpcTimer::start(|| "get_refs_freshness/budgeted".to_string());
        vex_client_stats()
            .refresh_attempts
            .fetch_add(1, Ordering::Relaxed);
        let endpoint = self.config.endpoint.clone();
        let request = jj_backend_api::GetRefsFreshnessRequest {
            tenant_id: self.config.tenant_id.clone(),
            repo_id: self.config.repo_id.clone(),
            prefix: prefix.to_string(),
        };
        let token = self.config.access_token.clone();
        let response = Self::shared_grpc_runtime().block_on(async move {
            let attempt = async move {
                let channel = Self::cached_channel_async(&endpoint).await?;
                JjBackendClient::new(channel)
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .get_refs_freshness(Self::auth_request(request, token.as_deref())?)
                    .await
                    .map(|response| response.into_inner())
                    .map_err(VexClientError::from)
            };
            tokio::time::timeout(budget, attempt).await
        });
        let response = match response {
            Err(_elapsed) => {
                vex_client_stats()
                    .refresh_timeouts
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            Ok(Ok(response)) => response,
            Ok(Err(err)) => {
                vex_client_stats()
                    .refresh_timeouts
                    .fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
        };
        // An empty token is the server saying "I cannot answer", not "your refs
        // are at the empty state"; reporting it as a token would let two repos
        // with no answer compare equal and look current.
        if response.refs_token.is_empty() {
            return Ok(None);
        }
        tracing::debug!(
            token = %response.refs_token,
            ref_count = response.ref_count,
            scope = %response.scope_prefix,
            "ref-freshness token"
        );
        Ok(Some(response.refs_token))
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
        // Skip bookkeeping dirs at the cache root (`.transfer-state`
        // resumable pack state): they are tiny, and pruning them would break
        // an in-flight resumable transfer.
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
        // Staged objects are the only copy of unpublished work. Federated pins
        // are the only durable copy of synthetic facade state, which is never
        // published by design. Neither class is evictable.
        let protected: HashSet<PathBuf> = self
            .staged_objects()
            .unwrap_or_default()
            .into_iter()
            .chain(self.pinned_federated_objects().unwrap_or_default())
            .filter_map(|(kind, content_id)| self.cache_path(kind, &content_id))
            .collect();
        for entry in entries {
            if total_bytes <= target_bytes {
                break;
            }
            if protected.contains(&entry.path) {
                continue;
            }
            if fs::remove_file(&entry.path).is_ok() {
                total_bytes = total_bytes.saturating_sub(entry.size_bytes);
                removed_files += 1;
                reclaimed_bytes += entry.size_bytes;
            }
        }
        if removed_files > 0 {
            // The `.packs` payload/index files are excluded from the LRU scan
            // above (dot-dir), so a capped cache bounds their growth here
            // instead: any prune that evicts object files also drops the
            // pack-resident store wholesale. The
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
        let endpoint_str = endpoint;
        let mut endpoint = Endpoint::from_shared(endpoint.to_string()).map_err(mkerr)?;
        // Listener-routed deployments dial a stable VIP but must present the
        // internal service name as `:authority` (see
        // `set_grpc_authority_override`). `Endpoint::origin` swaps the URI the
        // requests are built against (scheme + authority) while the transport
        // still connects to the dialed address.
        if let Some(authority) = grpc_authority_override() {
            let scheme = if is_https { "https" } else { "http" };
            let origin = tonic::transport::Uri::builder()
                .scheme(scheme)
                .authority(authority.as_str())
                .path_and_query("/")
                .build()
                .map_err(|err| VexConfigError::InvalidEndpoint {
                    endpoint: endpoint_str.to_string(),
                    message: format!("invalid gRPC authority override `{authority}`: {err}"),
                })?;
            endpoint = endpoint.origin(origin);
        }
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
        // Every Vex gRPC request is built here, so this is the one place the
        // client version needs to be attached. Doing it per call site would
        // guarantee that a future RPC forgets it and silently reappears as an
        // "unknown version" tenant in the compatibility gate.
        if let Some(version) = client_version_metadata() {
            request
                .metadata_mut()
                .insert(CLIENT_VERSION_METADATA_KEY, version);
        }
        Ok(request)
    }

    /// Shared multi-threaded runtime for all blocking gRPC calls. Reused across
    /// every call so we don't pay runtime-construction cost per request and so
    /// batches can be issued concurrently over one HTTP/2 connection.
    pub(crate) fn shared_grpc_runtime() -> &'static tokio::runtime::Runtime {
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
    fn channel_cache() -> &'static Mutex<HashMap<String, Channel>> {
        static CHANNELS: OnceLock<Mutex<HashMap<String, Channel>>> = OnceLock::new();
        CHANNELS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    fn cached_channel(endpoint_url: &str) -> Result<Channel, VexClientError> {
        if let Some(channel) = Self::channel_cache().lock().unwrap().get(endpoint_url) {
            return Ok(channel.clone());
        }
        let endpoint = Self::endpoint(endpoint_url)?;
        let channel =
            Self::shared_grpc_runtime().block_on(async move { endpoint.connect().await })?;
        Self::channel_cache()
            .lock()
            .unwrap()
            .insert(endpoint_url.to_string(), channel.clone());
        Ok(channel)
    }

    /// [`Self::cached_channel`] for callers already inside the shared runtime,
    /// so the connect handshake can be wrapped in a timeout instead of
    /// blocking a thread for however long the network takes.
    async fn cached_channel_async(endpoint_url: &str) -> Result<Channel, VexClientError> {
        if let Some(channel) = Self::channel_cache().lock().unwrap().get(endpoint_url) {
            return Ok(channel.clone());
        }
        let channel = Self::endpoint(endpoint_url)?.connect().await?;
        Self::channel_cache()
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
    /// batch. Sized so the full ladder (~12 attempts, 1s base, 10s cap) covers
    /// a brief backend restart/pressure window. The small jitter keeps
    /// concurrently retried batches from reconnecting at the same instant.
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
        let per_attempt = publish_request_timeout();
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            for attempt in 1..=attempts {
                let client = JjBackendClient::new(channel.clone())
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
                let result =
                    publish_attempt_within(per_attempt, "CommitOperation", f(client)).await;
                match result {
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

    /// Run one optional-capability gRPC call on the shared runtime without the
    /// general retry ladder. Feature probes such as `GetHydrationPacks` must
    /// fall back immediately when an older edge reports the unknown method as
    /// `Unknown` instead of the more specific `Unimplemented` status.
    async fn grpc_once_async<T, F, Fut>(endpoint: &str, f: F) -> Result<T, VexClientError>
    where
        F: FnOnce(JjBackendClient<Channel>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, tonic::Status>> + Send + 'static,
        T: Send + 'static,
    {
        let channel = Self::cached_channel(endpoint)?;
        let handle = Self::shared_grpc_runtime().spawn(with_output_cancel(async move {
            let client = JjBackendClient::new(channel)
                .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
            f(client).await.map_err(Into::into)
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
        let http_started = std::time::Instant::now();
        let joined = Self::spawn_http_get(url, headers, expected_len).await;
        vex_client_stats()
            .pack_http_wait_ms
            .fetch_add(http_started.elapsed().as_millis() as u64, Ordering::Relaxed);
        match joined {
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
        let http_started = std::time::Instant::now();
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
        vex_client_stats()
            .pack_http_wait_ms
            .fetch_add(http_started.elapsed().as_millis() as u64, Ordering::Relaxed);
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
        prefetched_objects: &AtomicU64,
    ) -> Result<bool, VexClientError> {
        self.prefetch_pack_via_chunks_with_concurrency(
            pack,
            hints,
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
        prefetched_objects: &AtomicU64,
        concurrency: usize,
    ) -> Result<bool, VexClientError> {
        if pack.chunk_frames {
            return self
                .prefetch_chunk_framed_pack_via_chunks(pack, hints, prefetched_objects, concurrency)
                .await;
        }
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
        let download_started = std::time::Instant::now();
        let fetch_result = self
            .fetch_chunks_into_partial(
                pack,
                &chunks,
                hints,
                concurrency,
                &mut state,
                &mut partial_file,
            )
            .await;
        vex_client_stats().pack_download_ms.fetch_add(
            download_started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        // Persist progress once at the end — and on error, so a resumed
        // transfer continues from the last appended chunk rather than the last
        // batched save. The fetch's own error wins over a save failure.
        let save_result = self.save_pack_transfer_state(&pack.content_id, &state);
        fetch_result?;
        save_result?;
        partial_file.flush()?;
        drop(partial_file);
        let chunk_ids: Vec<ContentId> = chunks.iter().map(|chunk| chunk.content_id).collect();
        let unpack_started = std::time::Instant::now();
        let unpack_result = self.prefetch_pack_entries_from_file(
            &pack.content_id,
            &partial_path,
            prefetched_objects,
        );
        vex_client_stats().pack_unpack_ms.fetch_add(
            unpack_started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        if let Err(err) = unpack_result {
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

    /// Fetch and unpack a pack whose chunks are independently decodable zstd
    /// frames. The transfer's `.part` file remains the byte-for-byte,
    /// in-order resume journal used by the legacy path; unpack consumes each
    /// verified in-memory chunk instead of rereading the growing file.
    ///
    /// A resumed prefix is decoded from `.part` before new chunks are fetched.
    /// This is deliberately idempotent: a prior process may have unpacked some
    /// of that prefix before it failed, while a crash before its unpack leaves
    /// it only in the journal. Per-object cache writes already tolerate either
    /// case, and every prefix entry is therefore made available before the
    /// transfer is declared complete.
    async fn prefetch_chunk_framed_pack_via_chunks(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
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
            partial_file.set_len(0)?;
            partial_file.seek(SeekFrom::Start(0))?;
            state.next_chunk_index = 0;
        }

        let chunk_ids: Vec<ContentId> = chunks.iter().map(|chunk| chunk.content_id).collect();
        let mut decoded_entries = 0_usize;
        let mut expected_entries = None;
        let mut prefix_file = File::open(&partial_path)?;
        for (index, chunk) in chunks.iter().take(state.next_chunk_index).enumerate() {
            let mut bytes = vec![0; chunk.size_bytes as usize];
            prefix_file.read_exact(&mut bytes)?;
            if index == 0 {
                expected_entries = Some(
                    parse_pack_header(&bytes)
                        .map_err(|err| VexClientError::PackDecode(err.to_string()))?
                        .entry_count,
                );
            }
            match self
                .unpack_chunk_frame_entries(chunk, Arc::new(bytes), prefetched_objects)
                .await
            {
                Ok(count) => decoded_entries += count,
                Err(err) => {
                    if matches!(err, VexClientError::PackDecode(_)) {
                        drop(self.clear_pack_transfer_state(&pack.content_id, &chunk_ids));
                    }
                    return Err(err);
                }
            }
        }
        drop(prefix_file);

        let download_started = std::time::Instant::now();
        let download_elapsed_ms = Arc::new(AtomicU64::new(0));
        let decode_failed = Arc::new(AtomicBool::new(false));
        let fetch_result = self
            .fetch_chunk_frames_into_partial(
                pack,
                &chunks,
                hints,
                concurrency,
                &mut state,
                &mut partial_file,
                &mut decoded_entries,
                &mut expected_entries,
                &download_elapsed_ms,
                &decode_failed,
                download_started,
                prefetched_objects,
            )
            .await;
        let downloaded_ms = download_elapsed_ms.load(Ordering::Relaxed);
        vex_client_stats().pack_download_ms.fetch_add(
            if downloaded_ms == 0 && fetch_result.is_err() {
                download_started.elapsed().as_millis() as u64
            } else {
                downloaded_ms
            },
            Ordering::Relaxed,
        );
        let save_result = self.save_pack_transfer_state(&pack.content_id, &state);
        if let Err(err) = fetch_result {
            if decode_failed.load(Ordering::Relaxed) {
                drop(self.clear_pack_transfer_state(&pack.content_id, &chunk_ids));
            }
            return Err(err);
        }
        save_result?;
        partial_file.flush()?;
        drop(partial_file);

        let Some(expected_entries) = expected_entries else {
            let err = VexClientError::PackDecode(format!(
                "chunk-framed pack {} did not contain chunk zero",
                pack.content_id
            ));
            drop(self.clear_pack_transfer_state(&pack.content_id, &chunk_ids));
            return Err(err);
        };
        if decoded_entries != expected_entries {
            let err = VexClientError::PackDecode(format!(
                "chunk-framed pack {} decoded {decoded_entries} entries; header declares {expected_entries}",
                pack.content_id
            ));
            drop(self.clear_pack_transfer_state(&pack.content_id, &chunk_ids));
            return Err(err);
        }
        self.clear_pack_transfer_state(&pack.content_id, &chunk_ids)?;
        Ok(true)
    }

    /// Decode and cache one verified chunk frame on Tokio's blocking pool.
    /// Each frame gets its own pack-resident payload id (`chunk.content_id`):
    /// the regular per-pack payload is mutable while it is being assembled, so
    /// sharing it across concurrent frame writes would lose index records.
    /// Object content IDs remain unchanged and loose writes are already
    /// idempotent, preserving skip-known-object behavior.
    async fn unpack_chunk_frame_entries(
        &self,
        chunk: &jj_backend_types::PackChunkDescriptor,
        bytes: Arc<Vec<u8>>,
        prefetched_objects: &AtomicU64,
    ) -> Result<usize, VexClientError> {
        let client = self.clone();
        let chunk_index = chunk.chunk_index;
        let chunk_content_id = chunk.content_id;
        let unpacked_objects = Arc::new(AtomicU64::new(0));
        let unpacked_objects_for_task = Arc::clone(&unpacked_objects);
        let unpack_started = std::time::Instant::now();
        let handle = Self::shared_grpc_runtime().spawn_blocking(move || {
            let entries = decode_pack_chunk_entries(&bytes, chunk_index)
                .map_err(|err| VexClientError::PackDecode(err.to_string()))?;
            let entry_count = entries.len();
            let mut entries = Some(entries);
            client.unpack_pack_entries(
                &chunk_content_id,
                unpacked_objects_for_task.as_ref(),
                move |sink| {
                    for entry in entries.take().expect("unpack drives frame entries once") {
                        sink(entry)?;
                    }
                    Ok(())
                },
            )?;
            Ok::<_, VexClientError>(entry_count)
        });
        let result = handle
            .await
            .map_err(|err| VexClientError::PackDecode(format!("chunk unpack task failed: {err}")));
        vex_client_stats().pack_unpack_ms.fetch_add(
            unpack_started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        let entry_count = result??;
        prefetched_objects.fetch_add(unpacked_objects.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(entry_count)
    }

    #[expect(clippy::too_many_arguments)]
    async fn fetch_chunk_frames_into_partial(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        chunks: &[jj_backend_types::PackChunkDescriptor],
        hints: &[jj_backend_api::PresignedGet],
        concurrency: usize,
        state: &mut PackTransferState,
        partial_file: &mut File,
        decoded_entries: &mut usize,
        expected_entries: &mut Option<usize>,
        download_elapsed_ms: &AtomicU64,
        decode_failed: &AtomicBool,
        download_started: std::time::Instant,
        prefetched_objects: &AtomicU64,
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
                    download_elapsed_ms.fetch_max(
                        download_started.elapsed().as_millis() as u64,
                        Ordering::Relaxed,
                    );
                    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != chunk.size_bytes {
                        return Err(VexClientError::PackDecode(format!(
                            "chunk size mismatch for pack {} chunk {}",
                            pack.content_id, index
                        )));
                    }
                    let header_entries = if index == 0 {
                        match parse_pack_header(&bytes) {
                            Ok(header) => Some(header.entry_count),
                            Err(err) => {
                                decode_failed.store(true, Ordering::Relaxed);
                                return Err(VexClientError::PackDecode(err.to_string()));
                            }
                        }
                    } else {
                        None
                    };
                    let bytes = Arc::new(bytes);
                    let entry_count = match self
                        .unpack_chunk_frame_entries(chunk, Arc::clone(&bytes), prefetched_objects)
                        .await
                    {
                        Ok(count) => count,
                        Err(err) => {
                            if matches!(err, VexClientError::PackDecode(_)) {
                                decode_failed.store(true, Ordering::Relaxed);
                            }
                            return Err(err);
                        }
                    };
                    Ok::<_, VexClientError>((index, bytes, entry_count, header_entries))
                },
            ))
            .buffered(concurrency.max(1));
        let mut chunks_since_save = 0_usize;
        while let Some(result) = fetched.next().await {
            let (index, chunk_bytes, entry_count, header_entries) = result?;
            if let Some(header_entries) = header_entries {
                *expected_entries = Some(header_entries);
            }
            *decoded_entries += entry_count;
            let stats = vex_client_stats();
            stats.pack_chunks_fetched.fetch_add(1, Ordering::Relaxed);
            stats
                .pack_bytes_fetched
                .fetch_add(chunk_bytes.len() as u64, Ordering::Relaxed);
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
    async fn fetch_chunks_into_partial(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        chunks: &[jj_backend_types::PackChunkDescriptor],
        hints: &[jj_backend_api::PresignedGet],
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
            stats
                .pack_bytes_fetched
                .fetch_add(chunk_bytes.len() as u64, Ordering::Relaxed);
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

    /// Whether object writes are *staged* on disk for a later `vex push`
    /// rather than uploaded inline, one blocking round trip per object.
    ///
    /// On by default. Since roadmap/088 Stage 7 `vex push` is the only
    /// publication verb, so this is the normal path: a `vex commit` must be
    /// able to finish without contacting the backend at all, and the objects it
    /// wrote must still be uploadable by a *later* process. Set
    /// `VEX_BATCH_SNAPSHOT_UPLOADS=0` (or `false`/`no`) to fall back to
    /// immediate per-object PUTs for an ordinary physical checkout. A
    /// federated Home facade always stages: its signed Plan/Stage flow is the
    /// only write boundary, and generic object PUTs are deliberately denied.
    /// Never staged in local-write mode, where puts stay local by design and
    /// are never published at all.
    fn staged_object_writes_enabled(
        local_writes: bool,
        federated_facade: bool,
        batch_override: Option<&str>,
    ) -> bool {
        if local_writes {
            return false;
        }
        federated_facade || !matches!(batch_override, Some("0") | Some("false") | Some("no"))
    }

    fn stage_writes_enabled(&self) -> bool {
        let batch_override = std::env::var("VEX_BATCH_SNAPSHOT_UPLOADS").ok();
        Self::staged_object_writes_enabled(
            self.local_writes,
            self.federated_facade,
            batch_override.as_deref(),
        )
    }

    /// Directory of staged-upload markers: `<cache_root>/.staged/<kind>/<id>`
    /// is an empty file meaning "this object's bytes are in the cache but have
    /// not been uploaded yet".
    ///
    /// This is the one thing that keeps the "cached ⟹ present on server" short
    /// circuit in [`Self::put_object`] sound now that writes no longer upload:
    /// a cached object is on the server **unless** it carries a marker. The
    /// markers are what a later `vex push` — in a later process — reads to know
    /// what to publish, which is why they must be on disk and not in a
    /// process-local buffer.
    fn staged_marker_root(&self) -> Option<PathBuf> {
        self.cache_root.as_ref().map(|root| root.join(".staged"))
    }

    fn staged_marker_path(&self, kind: ObjectKind, content_id: &ContentId) -> Option<PathBuf> {
        self.staged_marker_root()
            .map(|root| root.join(kind_to_str(kind)).join(content_id.to_string()))
    }

    /// Local-only facade objects are never published, but remain part of the
    /// editable flat working copy after aggregate acceptance. Their pins are
    /// separate from staged-upload markers so pruning preserves the bytes
    /// without making any later publication path consider them uploadable.
    fn federated_pin_root(&self) -> Option<PathBuf> {
        self.cache_root
            .as_ref()
            .map(|root| root.join(".federated-pins"))
    }

    fn federated_pin_path(&self, kind: ObjectKind, content_id: &ContentId) -> Option<PathBuf> {
        self.federated_pin_root()
            .map(|root| root.join(kind_to_str(kind)).join(content_id.to_string()))
    }

    /// Whether this object is staged for upload (cached but not yet published).
    #[cfg(test)]
    fn has_staged_object(&self, kind: ObjectKind, content_id: &ContentId) -> bool {
        self.staged_marker_path(kind, content_id)
            .is_some_and(|path| path.exists())
    }

    /// Persist one object locally and record that it still owes an upload.
    ///
    /// **The marker is written before the bytes.** A crash between the two
    /// leaves a marker for an object the cache does not hold, which the next
    /// push refuses on, loudly and by name (the marker stays, so the refusal
    /// repeats rather than decaying into silence); the reverse order would
    /// leave cached bytes that
    /// no marker names, and [`Self::put_object`]'s cache short circuit would
    /// then skip them forever — a silently dangling ref. Both writes are
    /// temp-file-plus-rename, matching the marker discipline in
    /// [`crate::vex_publish`].
    ///
    /// Deliberately skips the cache prune pass: staged objects are the only
    /// copy of unpublished work (see [`Self::prune_cache_if_needed`], which
    /// refuses to evict them for the same reason).
    fn stage_object(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        data: &[u8],
    ) -> Result<(), VexClientError> {
        if let Some(path) = self.staged_marker_path(kind, content_id) {
            let dir = path.parent().expect("staged marker has a parent");
            fs::create_dir_all(dir)?;
            let temp = NamedTempFile::new_in(dir)?;
            temp.as_file().sync_all()?;
            temp.persist(&path).map_err(|err| err.error)?;
        }
        self.write_cached_object_no_prune(kind, content_id, data)
    }

    /// Every object currently staged for upload, read off disk. Includes work
    /// staged by earlier processes, which is the whole point.
    fn staged_objects(&self) -> Result<Vec<(ObjectKind, ContentId)>, VexClientError> {
        self.object_markers(self.staged_marker_root())
    }

    fn pinned_federated_objects(&self) -> Result<Vec<(ObjectKind, ContentId)>, VexClientError> {
        self.object_markers(self.federated_pin_root())
    }

    fn object_markers(
        &self,
        root: Option<PathBuf>,
    ) -> Result<Vec<(ObjectKind, ContentId)>, VexClientError> {
        let Some(root) = root else {
            return Ok(Vec::new());
        };
        let mut objects = Vec::new();
        let kind_dirs = match fs::read_dir(&root) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err.into()),
        };
        for kind_dir in kind_dirs {
            let kind_dir = kind_dir?;
            let Some(kind) = kind_dir
                .file_name()
                .to_str()
                .and_then(|name| kind_from_str(name))
            else {
                continue;
            };
            for marker in fs::read_dir(kind_dir.path())? {
                let marker = marker?;
                let Some(content_id) = marker
                    .file_name()
                    .to_str()
                    .and_then(|name| ContentId::from_hex(name).ok())
                else {
                    continue;
                };
                objects.push((kind, content_id));
            }
        }
        // Deterministic order so a failed upload retries the same way and so
        // logs of two runs are comparable.
        objects.sort_by_key(|(kind, id)| (kind_to_str(*kind), id.to_hex()));
        Ok(objects)
    }

    /// Upload everything staged for this repo, then drop its markers.
    ///
    /// This is the publication half of `vex push`: it must run **before** any
    /// ref advances, so the ref never names a commit whose closure is missing
    /// server-side (roadmap/056's presence-before-publish invariant). Returns
    /// the number of objects uploaded.
    ///
    /// Safe to replay in every direction. `PutObjects` is content-addressed
    /// create-if-missing, so re-uploading an object a previous attempt already
    /// landed is a no-op; markers are removed only after their batch is
    /// acknowledged, so a crash mid-push re-uploads rather than forgets.
    ///
    /// This is a blocking call (it drives the shared gRPC runtime) and must not
    /// be invoked from within that runtime.
    pub fn upload_staged_objects(&self) -> Result<usize, VexClientError> {
        if self.federated_facade {
            return Err(VexClientError::StagedObjectsMissing(
                "ordinary object publication is disabled for a Home checkout; use `vex submit`"
                    .to_string(),
            ));
        }
        let staged = self.staged_objects()?;
        if staged.is_empty() {
            return Ok(0);
        }
        let _t = RpcTimer::start(|| format!("upload_staged_objects/{}", staged.len()));

        // Size/count-bounded batches, uploaded a wave at a time so peak memory
        // is bounded by the wave rather than by everything ever staged.
        let mut uploaded = 0usize;
        let mut missing = Vec::new();
        let mut wave: Vec<Vec<(ObjectKind, ContentId, Vec<u8>)>> = Vec::new();
        let mut batch: Vec<(ObjectKind, ContentId, Vec<u8>)> = Vec::new();
        let mut batch_bytes = 0usize;
        for (kind, content_id) in staged {
            let Some(data) = self.read_cached_object(kind, &content_id) else {
                // The bytes are gone (a cache wiped behind our back). Say so
                // rather than advancing a ref over a hole.
                missing.push((kind, content_id));
                continue;
            };
            batch_bytes += data.len();
            batch.push((kind, content_id, data));
            if batch.len() >= PENDING_FLUSH_OBJECTS || batch_bytes >= PENDING_FLUSH_BYTES {
                wave.push(std::mem::take(&mut batch));
                batch_bytes = 0;
            }
            if wave.len() >= STAGED_UPLOAD_CONCURRENCY {
                uploaded += self.upload_staged_wave(std::mem::take(&mut wave))?;
            }
        }
        if !batch.is_empty() {
            wave.push(batch);
        }
        if !wave.is_empty() {
            uploaded += self.upload_staged_wave(wave)?;
        }
        if !missing.is_empty() {
            return Err(VexClientError::StagedObjectsMissing(format!(
                "{} staged object(s) are recorded as unpublished but their bytes are no longer \
                 in the local cache (first: {}/{}); the local cache was cleared before these \
                 changes were pushed, so they cannot be published from this workspace",
                missing.len(),
                kind_to_str(missing[0].0),
                missing[0].1,
            )));
        }
        Ok(uploaded)
    }

    /// Publish an explicit content-addressed closure from this client's bound
    /// cache. Replays are safe because `PutObjects` is create-if-missing. This
    /// path deliberately preserves ordinary staged markers: one cached object
    /// can be required by both a component closure and the Home facade, and the
    /// marker is not repository-scoped.
    pub fn upload_cached_objects(
        &self,
        objects: impl IntoIterator<Item = VexObjectId>,
    ) -> Result<usize, VexClientError> {
        let mut seen = HashSet::new();
        let mut objects = objects
            .into_iter()
            .filter(|object| seen.insert(*object))
            .collect::<Vec<_>>();
        objects.sort_by(|left, right| {
            kind_to_str(left.0)
                .cmp(kind_to_str(right.0))
                .then_with(|| left.1.to_string().cmp(&right.1.to_string()))
        });
        if objects.is_empty() {
            return Ok(0);
        }

        let mut uploaded = 0usize;
        let mut missing = Vec::new();
        let mut wave: Vec<Vec<(ObjectKind, ContentId, Vec<u8>)>> = Vec::new();
        let mut batch: Vec<(ObjectKind, ContentId, Vec<u8>)> = Vec::new();
        let mut batch_bytes = 0usize;
        for (kind, content_id) in objects {
            let Some(data) = self.read_cached_object(kind, &content_id) else {
                missing.push((kind, content_id));
                continue;
            };
            batch_bytes += data.len();
            batch.push((kind, content_id, data));
            if batch.len() >= PENDING_FLUSH_OBJECTS || batch_bytes >= PENDING_FLUSH_BYTES {
                wave.push(std::mem::take(&mut batch));
                batch_bytes = 0;
            }
            if wave.len() >= STAGED_UPLOAD_CONCURRENCY {
                uploaded +=
                    self.upload_wave(std::mem::take(&mut wave), StagedMarkerPolicy::Preserve)?;
            }
        }
        if !batch.is_empty() {
            wave.push(batch);
        }
        if !wave.is_empty() {
            uploaded += self.upload_wave(wave, StagedMarkerPolicy::Preserve)?;
        }
        if let Some((kind, content_id)) = missing.first() {
            return Err(VexClientError::StagedObjectsMissing(format!(
                "{} object(s) in the submit closure are missing from the flat Home cache (first: {}/{}); refresh or reclone before submitting",
                missing.len(),
                kind_to_str(*kind),
                content_id
            )));
        }
        Ok(uploaded)
    }

    /// Stage one exact planned physical child closure for a flat Home submit.
    ///
    /// Unlike [`Self::upload_cached_objects`], this sends one bounded request
    /// to the dedicated capability endpoint and never falls back to generic
    /// `PutObjects`. Staged markers remain until Rails accepts the single
    /// aggregate operation and the caller retires them explicitly.
    pub fn stage_federated_home_submit_partition(
        &self,
        partition_capability: &str,
        proposed_child_commit: &ContentId,
        objects: impl IntoIterator<Item = VexObjectId>,
    ) -> Result<usize, VexClientError> {
        let mut objects = objects.into_iter().collect::<Vec<_>>();
        objects.sort_by(|left, right| {
            kind_to_str(left.0)
                .cmp(kind_to_str(right.0))
                .then_with(|| left.1.to_hex().cmp(&right.1.to_hex()))
        });
        if objects.len() > MAX_FEDERATED_HOME_STAGE_OBJECTS {
            return Err(VexConfigError::InvalidFederatedHome(format!(
                "flat Home staged-write request has {} objects (maximum {MAX_FEDERATED_HOME_STAGE_OBJECTS})",
                objects.len(),
            ))
            .into());
        }
        let mut seen = HashSet::with_capacity(objects.len());
        let mut inline = Vec::with_capacity(objects.len());
        let mut total_bytes = 0usize;
        for (kind, content_id) in objects {
            if !seen.insert((kind, content_id)) {
                return Err(VexConfigError::InvalidFederatedHome(
                    "Home child closure repeats one object".to_string(),
                )
                .into());
            }
            let data = self.read_cached_object(kind, &content_id).ok_or_else(|| {
                VexClientError::StagedObjectsMissing(format!(
                    "a planned Home child object is missing from the local cache: {}/{}",
                    kind_to_str(kind),
                    content_id
                ))
            })?;
            total_bytes = total_bytes.checked_add(data.len()).ok_or_else(|| {
                VexConfigError::InvalidFederatedHome(
                    "flat Home staged-write inline byte count overflow".to_string(),
                )
            })?;
            if total_bytes > MAX_FEDERATED_HOME_STAGE_BYTES {
                return Err(VexConfigError::InvalidFederatedHome(format!(
                    "flat Home staged-write request has {total_bytes} bytes (maximum {MAX_FEDERATED_HOME_STAGE_BYTES})",
                ))
                .into());
            }
            inline.push(InlineObject {
                object: Some(ObjectId {
                    kind: kind_to_str(kind).to_string(),
                    content_id: content_id.to_hex(),
                }),
                data,
            });
        }
        let proposed_child_commit = *proposed_child_commit;
        if !seen.contains(&(ObjectKind::Commit, proposed_child_commit)) {
            return Err(VexConfigError::InvalidFederatedHome(
                "Home child closure omits its proposed commit".to_string(),
            )
            .into());
        }
        let response = Self::block_on_grpc_retry(&self.config.endpoint, 3, |mut client| {
            let tenant_id = self.config.tenant_id.clone();
            let repo_id = self.config.repo_id.clone();
            let capability = partition_capability.to_string();
            let objects = inline.clone();
            let token = self.config.access_token.clone();
            async move {
                client
                    .stage_federated_home_submit_partition(Self::auth_request(
                        StageFederatedHomeSubmitPartitionRequest {
                            tenant_id,
                            repo_id,
                            partition_capability: capability,
                            proposed_child_commit_id: proposed_child_commit.to_hex(),
                            objects,
                        },
                        token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            }
        })?;
        if !response.ok {
            return Err(VexClientError::Status(tonic::Status::internal(
                "backend rejected a federated Home stage without a status",
            )));
        }
        Ok(response.stored_count as usize)
    }

    /// Protect exact local-only flat facade objects from cache eviction without
    /// making them eligible for upload. Call this after the aggregate API
    /// accepts a federated submit and before retiring its staged markers.
    pub fn pin_local_federated_objects(
        &self,
        objects: impl IntoIterator<Item = VexObjectId>,
    ) -> Result<usize, VexClientError> {
        let mut seen = HashSet::new();
        let mut objects = objects
            .into_iter()
            .filter(|object| seen.insert(*object))
            .collect::<Vec<_>>();
        objects.sort_by_key(|(kind, content_id)| (kind_to_str(*kind), content_id.to_hex()));
        let missing = objects
            .iter()
            .filter(|(kind, content_id)| {
                self.cache_path(*kind, content_id)
                    .is_none_or(|path| !path.is_file())
            })
            .collect::<Vec<_>>();
        if let Some((kind, content_id)) = missing.first() {
            return Err(VexClientError::StagedObjectsMissing(format!(
                "{} local Home object(s) are missing (first: {}/{}); retry or reclone Home before finalizing submit",
                missing.len(),
                kind_to_str(*kind),
                content_id
            )));
        }

        let mut pinned = 0;
        for (kind, content_id) in objects {
            let Some(path) = self.federated_pin_path(kind, &content_id) else {
                continue;
            };
            if path.is_file() {
                continue;
            }
            let parent = path.parent().expect("federated pin has a parent");
            fs::create_dir_all(parent)?;
            let temporary = NamedTempFile::new_in(parent)?;
            temporary.as_file().sync_all()?;
            temporary.persist(path).map_err(|error| error.error)?;
            pinned += 1;
        }
        Ok(pinned)
    }

    /// Retire only the exact staged markers whose federated owner uploads were
    /// acknowledged and whose single Home orchestration request succeeded.
    /// No bytes are uploaded here, and unrelated staged work is never scanned
    /// or cleared.
    pub fn retire_staged_objects(
        &self,
        objects: impl IntoIterator<Item = VexObjectId>,
    ) -> Result<usize, VexClientError> {
        let mut seen = HashSet::new();
        let mut retired = 0;
        for (kind, content_id) in objects {
            if !seen.insert((kind, content_id)) {
                continue;
            }
            let Some(path) = self.staged_marker_path(kind, &content_id) else {
                continue;
            };
            match fs::remove_file(path) {
                Ok(()) => retired += 1,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(retired)
    }

    /// Upload one wave of batches concurrently over the shared connection and
    /// drop the markers of everything it acknowledged.
    fn upload_staged_wave(
        &self,
        wave: Vec<Vec<(ObjectKind, ContentId, Vec<u8>)>>,
    ) -> Result<usize, VexClientError> {
        self.upload_wave(wave, StagedMarkerPolicy::Remove)
    }

    fn upload_wave(
        &self,
        wave: Vec<Vec<(ObjectKind, ContentId, Vec<u8>)>>,
        marker_policy: StagedMarkerPolicy,
    ) -> Result<usize, VexClientError> {
        let ids: Vec<(ObjectKind, ContentId)> = wave
            .iter()
            .flatten()
            .map(|(kind, content_id, _)| (*kind, *content_id))
            .collect();
        let batches: Vec<Vec<InlineObject>> = wave
            .into_iter()
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
        let channel = Self::cached_channel(&self.config.endpoint)?;
        let repo_id = self.config.repo_id.clone();
        let token = self.config.access_token.clone();
        let per_batch = publish_request_timeout();
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            use futures::stream::TryStreamExt as _;
            futures::stream::iter(batches.into_iter().map(Ok::<_, VexClientError>))
                .try_for_each_concurrent(STAGED_UPLOAD_CONCURRENCY, |objects| {
                    let channel = channel.clone();
                    let repo_id = repo_id.clone();
                    let token = token.clone();
                    async move {
                        let request = Self::auth_request(
                            PutObjectsRequest { repo_id, objects },
                            token.as_deref(),
                        )?;
                        let mut client = JjBackendClient::new(channel)
                            .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                            .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
                        publish_attempt_within(
                            per_batch,
                            "PutObjects",
                            client.put_objects(request),
                        )
                        .await?;
                        Ok(())
                    }
                })
                .await
        }))?;
        self.apply_uploaded_marker_policy(&ids, marker_policy);
        Ok(ids.len())
    }

    fn apply_uploaded_marker_policy(&self, ids: &[VexObjectId], marker_policy: StagedMarkerPolicy) {
        if marker_policy == StagedMarkerPolicy::Preserve {
            return;
        }
        // Now — and only now — is "cached ⟹ uploaded" true for the repository
        // whose ordinary staged publication is running, so its markers may go.
        // A failure to unlink is harmless: the next push re-uploads.
        for (kind, content_id) in ids {
            if let Some(path) = self.staged_marker_path(*kind, content_id) {
                drop(fs::remove_file(path));
            }
        }
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
        // was already uploaded *or* is already staged for upload, so there is
        // nothing new to record. This is the hot path during working-copy
        // snapshots (`vex status`), where unchanged or recurring
        // blob/tree/commit content would otherwise be re-written.
        if self.has_cached_object(kind, content_id) {
            return Ok(());
        }
        // The default path: stage the object durably and return without
        // touching the network. `vex push` is the only publication verb
        // (roadmap/088 Stage 7), and it uploads what this staged — including
        // what *earlier processes* staged, which an in-memory buffer could
        // never deliver.
        if self.stage_writes_enabled() {
            return self.stage_object(kind, content_id, &data);
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
    #[tracing::instrument(skip_all)]
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
        self.get_object_with_fetch(kind, content_id, |config, kind, content_id| async move {
            Self::fetch_object_grpc_verified_for_config(&config, kind, &content_id).await
        })
        .await
    }

    async fn get_object_with_fetch<F, Fut>(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
        mut fetch: F,
    ) -> Result<Vec<u8>, VexClientError>
    where
        F: FnMut(VexRepoConfig, ObjectKind, ContentId) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<u8>, VexClientError>>,
    {
        let _t = RpcTimer::start(|| format!("get_object/{}", kind_to_str(kind)));
        if let Some(bytes) = self.read_cached_object(kind, content_id) {
            vex_client_stats().record_get_object_cache_hit(kind);
            return Ok(bytes);
        }
        let primary_error = match fetch(self.config.clone(), kind, *content_id).await {
            Ok(bytes) => {
                self.write_cached_object(kind, content_id, &bytes)?;
                return Ok(bytes);
            }
            Err(error)
                if Self::supports_federated_object_fallback(kind) && Self::is_not_found(&error) =>
            {
                error
            }
            Err(error) => return Err(error),
        };
        let mut first_route_error = None;
        for route in self.federated_read_routes()? {
            debug!(
                kind = kind_to_str(kind),
                %content_id,
                repository_id = %route.repo_id,
                "flat Home cache miss; trying immutable physical owner fallback"
            );
            match fetch(route.clone(), kind, *content_id).await {
                Ok(bytes) => {
                    self.write_cached_object(kind, content_id, &bytes)?;
                    return Ok(bytes);
                }
                Err(error) if Self::is_not_found(&error) => {}
                Err(error) => {
                    // An earlier unrelated component may have an expired token
                    // or transient endpoint failure. Keep searching exact
                    // physical routes for the immutable content id, but report
                    // the first real route failure if no owner can serve it.
                    first_route_error.get_or_insert(error);
                }
            }
        }
        Err(first_route_error.unwrap_or(primary_error))
    }

    async fn fetch_object_grpc_verified(
        &self,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Result<Vec<u8>, VexClientError> {
        Self::fetch_object_grpc_verified_for_config(&self.config, kind, content_id).await
    }

    /// gRPC `GetObject` plus content-hash verification for one exact physical
    /// route, without touching the local cache.
    async fn fetch_object_grpc_verified_for_config(
        config: &VexRepoConfig,
        kind: ObjectKind,
        content_id: &ContentId,
    ) -> Result<Vec<u8>, VexClientError> {
        debug!(kind = kind_to_str(kind), %content_id, repository_id = %config.repo_id, "vex cache miss");
        vex_client_stats().record_get_object_rpc(kind);
        // Own every captured value so the fetch future is `Send + 'static` and can
        // be spawned onto the shared runtime. This is what lets `check_out`'s
        // `.buffered(concurrency())` actually run reads in parallel instead of
        // serializing them behind a per-object `block_on` (see `grpc_retry_async`).
        let repo_id = config.repo_id.clone();
        let access_token = config.access_token.clone();
        let kind_str = kind_to_str(kind).to_string();
        let content_id_str = content_id.to_string();
        let bytes = Self::grpc_retry_async(&config.endpoint, 5, move |mut client| {
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
    /// [`INLINE_FETCH_BATCH_BYTES`] and run [`inline_fetch_concurrency`]-wide
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
        .buffer_unordered(inline_fetch_concurrency());
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

    #[tracing::instrument(skip_all)]
    pub async fn get_op_heads(&self) -> Result<Vec<ContentId>, VexClientError> {
        // Always read op heads live from the server. A client-side cache is
        // unsafe here: jj records the working-copy operation locally *before* the
        // commit's server-side CAS runs, so serving a stale head lets jj build a
        // working-copy op on it; when the CAS then rejects the stale head the
        // working copy is left pinned to an orphan op, diverging from the backend
        // head (a "sibling operation" that blocks all further commands).
        let _t = RpcTimer::start(|| "get_op_heads".to_string());
        vex_client_stats()
            .op_head_rpcs
            .fetch_add(1, Ordering::Relaxed);
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

    /// Read the server op heads under a hard wall-clock budget covering the
    /// connect handshake as well as the request, with no retries. Returns
    /// `Ok(None)` when the budget expires — the caller keeps serving local
    /// heads. Used by the opportunistic freshness refresh (roadmap/088), which
    /// must never lengthen a read command.
    pub fn get_op_heads_within(
        &self,
        budget: Duration,
    ) -> Result<Option<Vec<ContentId>>, VexClientError> {
        let _t = RpcTimer::start(|| "get_op_heads/budgeted".to_string());
        vex_client_stats()
            .op_head_rpcs
            .fetch_add(1, Ordering::Relaxed);
        let endpoint = self.config.endpoint.clone();
        let request = jj_backend_api::GetOpHeadsRequest {
            tenant_id: self.config.tenant_id.clone(),
            repo_id: self.config.repo_id.clone(),
        };
        let token = self.config.access_token.clone();
        let response = Self::shared_grpc_runtime().block_on(async move {
            let attempt = async move {
                let channel = Self::cached_channel_async(&endpoint).await?;
                JjBackendClient::new(channel)
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .get_op_heads(Self::auth_request(request, token.as_deref())?)
                    .await
                    .map(|response| response.into_inner())
                    .map_err(VexClientError::from)
            };
            tokio::time::timeout(budget, attempt).await
        });
        let response = match response {
            Err(_elapsed) => return Ok(None),
            Ok(result) => result?,
        };
        let ids = response
            .op_content_ids
            .into_iter()
            .map(|id| {
                ContentId::from_hex(&id).map_err(|err| {
                    tonic::Status::internal(format!("invalid op head from server: {err}"))
                })
            })
            .collect::<Result<Vec<_>, tonic::Status>>()?;
        Ok(Some(ids))
    }

    /// Upload one batch of already-addressed objects under a hard wall-clock
    /// deadline covering the connect handshake, with no retries. `Ok(false)`
    /// means the deadline expired. The publisher (roadmap/088) uses this
    /// instead of [`Self::put_object_batches_pipelined`] so a wedged
    /// connection can never hold the working-copy lock indefinitely; a
    /// deadline is safe to surface because `put_objects` is content-addressed
    /// create-if-missing and the whole batch is re-sent on the next drain.
    pub fn put_objects_within(
        &self,
        objects: Vec<(ObjectKind, ContentId, Vec<u8>)>,
        budget: Duration,
    ) -> Result<bool, VexClientError> {
        if objects.is_empty() {
            return Ok(true);
        }
        let _t = RpcTimer::start(|| format!("put_objects/budgeted[{}]", objects.len()));
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
        let endpoint = self.config.endpoint.clone();
        let repo_id = self.config.repo_id.clone();
        let token = self.config.access_token.clone();
        let outcome = Self::shared_grpc_runtime().block_on(async move {
            let attempt = async move {
                let channel = Self::cached_channel_async(&endpoint).await?;
                JjBackendClient::new(channel)
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .put_objects(Self::auth_request(
                        PutObjectsRequest {
                            repo_id,
                            objects: inline,
                        },
                        token.as_deref(),
                    )?)
                    .await
                    .map(|_| ())
                    .map_err(VexClientError::from)
            };
            tokio::time::timeout(budget, attempt).await
        });
        match outcome {
            Err(_elapsed) => Ok(false),
            Ok(result) => result.map(|()| true),
        }
    }

    /// [`Self::commit_op_heads`] under a hard wall-clock deadline, with none of
    /// the maintenance-retry looping. `Ok(None)` means the deadline expired
    /// with the CAS outcome unknown; the caller must leave its queue intact and
    /// re-derive on the next drain, which the replay check on the server makes
    /// safe. Buffered objects are NOT flushed here — the publisher has already
    /// uploaded everything the new operation references.
    pub fn commit_op_heads_within(
        &self,
        expected: &[ContentId],
        new_head: &ContentId,
        new_view: &ContentId,
        budget: Duration,
    ) -> Result<Option<jj_backend_api::CommitOperationResponse>, VexClientError> {
        let _t = RpcTimer::start(|| "commit_op_heads/budgeted".to_string());
        let endpoint = self.config.endpoint.clone();
        let request = jj_backend_api::CommitOperationRequest {
            tenant_id: self.config.tenant_id.clone(),
            repo_id: self.config.repo_id.clone(),
            expected_op_head_ids: expected.iter().map(ToString::to_string).collect(),
            new_op_content_id: new_head.to_string(),
            new_view_content_id: new_view.to_string(),
            divergence_ok: crate::vex_op_head_delta::divergence_ok(),
            max_op_heads: crate::vex_op_head_delta::max_op_heads(),
        };
        let token = self.config.access_token.clone();
        let outcome = Self::shared_grpc_runtime().block_on(async move {
            let attempt = async move {
                let channel = Self::cached_channel_async(&endpoint).await?;
                JjBackendClient::new(channel)
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .commit_operation(Self::auth_request(request, token.as_deref())?)
                    .await
                    .map(|response| response.into_inner())
                    .map_err(VexClientError::from)
            };
            tokio::time::timeout(budget, attempt).await
        });
        match outcome {
            Err(_elapsed) => Ok(None),
            Ok(result) => result.map(Some),
        }
    }

    #[tracing::instrument(skip_all)]
    pub async fn commit_op_heads(
        &self,
        expected: &[ContentId],
        new_head: &ContentId,
        new_view: &ContentId,
    ) -> Result<jj_backend_api::CommitOperationResponse, VexClientError> {
        let _t = RpcTimer::start(|| "commit_op_heads".to_string());
        // Publish every staged object before advancing the op head, so the
        // operation the CAS installs never references an object that is missing
        // on the server. An upload failure aborts here, leaving the head
        // unchanged (the un-uploaded objects are simply unreferenced).
        self.upload_staged_objects()?;
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
                            divergence_ok: crate::vex_op_head_delta::divergence_ok(),
                            max_op_heads: crate::vex_op_head_delta::max_op_heads(),
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
                            naming: RefNaming::Legacy as i32,
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
                            naming: RefNaming::Legacy as i32,
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
                            resolve_git_target_class: false,
                            naming: RefNaming::Legacy as i32,
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
        progress: Option<&CloneProgressFn>,
    ) -> Result<CloneManifest, VexClientError> {
        let clone_view_kind = proto_clone_view_kind(self.config.repository_scope_kind.as_deref());
        let virtual_mounts: Vec<jj_backend_api::VirtualRepositoryMount> = self
            .config
            .virtual_mounts
            .iter()
            .map(proto_virtual_repository_mount)
            .collect();
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
        // Transient errors (edge 502/503, backend restart) retry on a short
        // exponential backoff instead of the build-poll interval: a warm
        // manifest serves in well under a second, so waiting a flat 3 s after
        // one edge blip quantized the whole manifest phase to ~3+ s
        // (measured 2.7-5.7 s outliers vs a ~0.8 s healthy floor).
        let transient_backoff_floor =
            std::time::Duration::from_millis(env_secs("VEX_CLONE_MANIFEST_RETRY_MS", 250).max(50));
        let mut transient_backoff = transient_backoff_floor;
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
            // One (non-retrying) attempt per iteration; the loop itself rides
            // both a still-`building` manifest *and* transient backend errors up
            // to `max_wait`, reporting each through `progress` so a slow/cold
            // first clone shows exactly what it is waiting on instead of a silent
            // 0%.
            let attempt = Self::block_on_grpc(&self.config.endpoint, |mut client| {
                let virtual_mounts = virtual_mounts.clone();
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
                                // Roadmap 032's snapshot packs are retired; the
                                // proto field survives for wire compatibility.
                                have_snapshot_commit_ids: Vec::new(),
                            },
                            self.config.access_token.as_deref(),
                        )?)
                        .await
                        .map(|response| response.into_inner())
                }
            });
            match attempt {
                Ok(response) if response.building => {
                    transient_backoff = transient_backoff_floor;
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
                    std::thread::sleep(transient_backoff);
                    transient_backoff = (transient_backoff * 2).min(poll);
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
        let presign_started = std::time::Instant::now();
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
        vex_client_stats().pack_presign_wait_ms.fetch_add(
            presign_started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        Ok(response.get_instructions)
    }

    /// Ask the backend to pack the exact blob-like objects needed by the
    /// selected clone checkout. This is intentionally separate from the
    /// reusable clone manifest: the checkout target is resolved from live refs
    /// after that immutable metadata seed has been consumed.
    pub async fn get_hydration_packs(
        &self,
        objects: &[(ObjectKind, ContentId)],
    ) -> Result<HydrationPackManifest, VexClientError> {
        let object_ids: Vec<ObjectId> = objects
            .iter()
            .map(|(kind, content_id)| ObjectId {
                kind: kind_to_str(*kind).to_string(),
                content_id: content_id.to_string(),
            })
            .collect();
        let repo_id = self.config.repo_id.clone();
        let access_token = self.config.access_token.clone();
        let response = Self::grpc_once_async(&self.config.endpoint, move |mut client| async move {
            client
                .get_hydration_packs(Self::auth_request(
                    GetHydrationPacksRequest {
                        repo_id,
                        objects: object_ids,
                    },
                    access_token.as_deref(),
                )?)
                .await
                .map(|response| response.into_inner())
        })
        .await?;
        serde_json::from_slice(&response.manifest_json)
            .map_err(VexConfigError::Json)
            .map_err(Into::into)
    }

    /// Fetch and unpack a hydration-pack response into the clone's local
    /// cache. Blob and symlink entries remain loose, as required by checkout.
    pub async fn prefetch_hydration_packs(
        &self,
        manifest: &HydrationPackManifest,
        progress: Option<&CloneProgressFn>,
    ) -> Result<(), VexClientError> {
        let prefetched_objects = AtomicU64::new(0);
        let hinted_pack_ids = manifest
            .packs
            .iter()
            .flat_map(|pack| {
                std::iter::once(pack.content_id)
                    .chain(pack.chunks.iter().map(|chunk| chunk.content_id))
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
        let total_packs = manifest.packs.len() as u64;
        let packs_done = AtomicU64::new(0);
        let packs = manifest.packs.iter().collect::<Vec<_>>();
        let result = self
            .prefetch_packs_parallel(
                &packs,
                &pack_hints,
                &prefetched_objects,
                &packs_done,
                total_packs,
                progress,
            )
            .into_iter()
            .flatten()
            .collect::<Result<Vec<_>, _>>()
            .map(|_| ());
        result.and(self.prune_cache_if_needed())?;
        debug!(
            repo_id = %self.config.repo_id,
            pack_count = manifest.packs.len(),
            object_count = manifest.object_count,
            total_bytes = manifest.total_bytes,
            prefetched_objects = prefetched_objects.load(Ordering::Relaxed),
            "prefetched clone hydration packs"
        );
        Ok(())
    }

    pub async fn prefetch_clone_manifest(
        &self,
        manifest: &CloneManifest,
        progress: Option<&CloneProgressFn>,
    ) -> Result<(), VexClientError> {
        let result = self.prefetch_clone_manifest_impl(manifest, progress).await;
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
        progress: Option<&CloneProgressFn>,
    ) -> Result<(), VexClientError> {
        let prefetch_started = std::time::Instant::now();
        let prefetched_objects = AtomicU64::new(0);

        let hinted_pack_ids = manifest
            .packs
            .iter()
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

        let total_packs = manifest.packs.len() as u64;
        let packs_done = AtomicU64::new(0);

        // Metadata packs: any failure fails the clone (the jj state is
        // unusable without them).
        let metadata_packs: Vec<&jj_backend_types::PackDescriptor> =
            manifest.packs.iter().collect();
        for result in self
            .prefetch_packs_parallel(
                &metadata_packs,
                &pack_hints,
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

        let total_loose = manifest.objects.len() as u64;
        let mut loose_done = 0_u64;
        let loose_started = std::time::Instant::now();
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
        vex_client_stats().pack_loose_object_ms.fetch_add(
            loose_started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
        debug!(
            repo_id = %self.config.repo_id,
            blob_mode = ?manifest.blob_mode,
            pack_count = manifest.packs.len(),
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
    /// accumulated in the caller-shared `packs_done`.
    ///
    /// Returns one slot per input pack, in input order: `Some(result)` for
    /// packs that ran, `None` for packs never started because an earlier pack
    /// failed (workers stop scheduling on the first failure and drain).
    fn prefetch_packs_parallel(
        &self,
        packs: &[&jj_backend_types::PackDescriptor],
        pack_hints: &[jj_backend_api::PresignedGet],
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

    /// Fetch and unpack a single clone pack into the local object cache,
    /// trying the chunked path first and falling back to streamed and then
    /// whole-pack reads. `prefetched_objects` is incremented per object
    /// written. Cache writes skip the per-write prune; the caller prunes once
    /// after the whole prefetch.
    async fn prefetch_one_pack(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        pack_hints: &[jj_backend_api::PresignedGet],
        prefetched_objects: &AtomicU64,
    ) -> Result<(), VexClientError> {
        let stats = vex_client_stats();
        let packs_counter = &stats.packs_fetched;
        let bytes_counter = &stats.pack_bytes_fetched;
        match self
            .prefetch_pack_via_chunks(pack, pack_hints, prefetched_objects)
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
            let unpack_started = std::time::Instant::now();
            self.prefetch_pack_entries_from_file(
                &pack.content_id,
                temp_pack.path(),
                prefetched_objects,
            )?;
            stats.pack_unpack_ms.fetch_add(
                unpack_started.elapsed().as_millis() as u64,
                Ordering::Relaxed,
            );
            packs_counter.fetch_add(1, Ordering::Relaxed);
            bytes_counter.fetch_add(pack.size_bytes, Ordering::Relaxed);
            return Ok(());
        }

        let pack_bytes = match self.direct_fetch_pack_bytes(pack, pack_hints) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(_) => self.get_object(ObjectKind::Pack, &pack.content_id).await?,
        };
        bytes_counter.fetch_add(pack_bytes.len() as u64, Ordering::Relaxed);
        let unpack_started = std::time::Instant::now();
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
        stats.pack_unpack_ms.fetch_add(
            unpack_started.elapsed().as_millis() as u64,
            Ordering::Relaxed,
        );
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
pub(crate) fn kind_from_str(kind: &str) -> Option<ObjectKind> {
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
        encode_object_pack_chunked,
    };
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::thread;

    /// Serializes tests that touch the process-global [`VexClientStats`]
    /// counters, so a concurrent [`vex_client_stats_reset`] cannot corrupt
    /// another test's delta assertions (tests run in parallel threads).
    /// Shared with `vex_backend`'s tests via [`crate::vex::test_stats_lock`].
    fn stats_lock() -> &'static Mutex<()> {
        test_stats_lock()
    }

    #[test]
    fn federated_facade_ignores_the_generic_upload_opt_out() {
        for batch_override in [Some("0"), Some("false"), Some("no")] {
            assert!(VexClient::staged_object_writes_enabled(
                false,
                true,
                batch_override
            ));
            assert!(!VexClient::staged_object_writes_enabled(
                false,
                false,
                batch_override
            ));
        }
        assert!(!VexClient::staged_object_writes_enabled(true, true, None));
    }

    #[test]
    fn a_client_version_is_sent_verbatim_including_a_dev_build_suffix() {
        for version in [
            "1.0.0",
            "0.12.0-3ce9e572",
            &format!("1.2.3-{}", "a".repeat(40)),
        ] {
            assert_eq!(
                validated_client_version_metadata(version)
                    .as_ref()
                    .and_then(|value| value.to_str().ok()),
                Some(version),
                "{version} must reach the server unchanged; the commit suffix is information"
            );
        }
        assert_eq!(
            validated_client_version_metadata("  1.0.0  ")
                .as_ref()
                .and_then(|value| value.to_str().ok()),
            Some("1.0.0"),
        );
    }

    #[test]
    fn an_unsendable_client_version_yields_no_header_at_all() {
        // No header is the honest signal: the gate reads it as *unknown*. A
        // mangled or truncated one would be read as a real version.
        for bad in [
            "",
            "   ",
            "1.0.0-café",
            "1.0.0\r\nx-injected: 1",
            "1.0.0 dev",
            &"9".repeat(MAX_CLIENT_VERSION_METADATA_LEN + 1),
        ] {
            assert!(
                validated_client_version_metadata(bad).is_none(),
                "must send no header for {bad:?}"
            );
        }
    }

    /// A dev build reports `X.Y.Z-<full 64-char commit>` = 70 characters, over
    /// the server's 64-character column. That used to drop the header entirely,
    /// making every source-built client invisible to the D14 gate — the exact
    /// population most likely to be running old code. The suffix is shortened
    /// instead, so the build is still reported and still distinguishable from
    /// the clean release of the same semver.
    #[test]
    fn a_dev_build_version_is_shortened_rather_than_dropped() {
        let sha = "09d948a1d1322130c26c281fa2c2e709bde791532550d088cf0908db3d8d0702";
        assert_eq!(sha.len(), 64, "a real commit id");
        let raw = format!("1.1.0-{sha}");
        assert!(
            raw.len() > MAX_CLIENT_VERSION_METADATA_LEN,
            "the case only exists because the dev version overflows: {}",
            raw.len()
        );

        let sent = validated_client_version_metadata(&raw)
            .expect("a dev build must still report a version");
        let sent = sent.to_str().expect("ascii");
        assert_eq!(sent, "1.1.0-09d948a1d132");
        assert!(sent.len() <= MAX_CLIENT_VERSION_METADATA_LEN);

        // The semver core is what the gate compares, so it must survive intact,
        // and the result must remain distinguishable from the clean release.
        assert!(sent.starts_with("1.1.0-"));
        assert_ne!(sent, "1.1.0");

        // A clean release version is still sent verbatim, untouched.
        let release = validated_client_version_metadata("1.1.0").expect("release version");
        assert_eq!(release.to_str().expect("ascii"), "1.1.0");
    }

    #[test]
    fn set_client_version_ignores_a_blank_declaration() {
        assert!(!set_client_version(""));
        assert!(!set_client_version("   "));
    }

    #[test]
    fn every_request_carries_the_client_version_alongside_the_auth_token() {
        // `auth_request` is the single construction point for every Vex gRPC
        // request, which is what stops a future RPC from silently omitting the
        // header and reappearing as an "unknown version" tenant in the gate.
        set_client_version("1.4.2-deadbeef");
        let request = VexClient::auth_request((), Some("token-abc")).expect("build request");

        assert_eq!(
            request
                .metadata()
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer token-abc"),
        );
        assert_eq!(
            request
                .metadata()
                .get(CLIENT_VERSION_METADATA_KEY)
                .and_then(|value| value.to_str().ok()),
            client_version_metadata()
                .as_ref()
                .and_then(|value| value.to_str().ok()),
            "the header must be whatever this process resolved, on every request"
        );
        assert!(
            client_version_metadata().is_some(),
            "a version was declared, so one must be sent"
        );
    }

    #[test]
    fn an_unauthenticated_request_still_declares_its_version() {
        set_client_version("1.4.2-deadbeef");
        let request = VexClient::auth_request((), None).expect("build request");

        assert!(request.metadata().get("authorization").is_none());
        assert!(
            request
                .metadata()
                .get(CLIENT_VERSION_METADATA_KEY)
                .is_some()
        );
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
    fn flat_home_stage_client_matches_the_server_512_object_and_32_mib_bounds() {
        let client = sample_client();
        let proposed_commit = ContentId::from_bytes([1; 32]);
        let error = client
            .stage_federated_home_submit_partition(
                "test-capability",
                &proposed_commit,
                std::iter::repeat((ObjectKind::Blob, ContentId::from_bytes([2; 32])))
                    .take(MAX_FEDERATED_HOME_STAGE_OBJECTS + 1),
            )
            .expect_err("the client must not send a 513-object Home stage request");
        assert!(matches!(
            error,
            VexClientError::Config(VexConfigError::InvalidFederatedHome(message))
                if message.contains("512")
        ));

        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = sample_client();
        client.cache_root = Some(temp_dir.path().to_path_buf());
        let data = vec![0; MAX_FEDERATED_HOME_STAGE_BYTES + 1];
        let blob_id = ContentId::hash_bytes(&data);
        client
            .write_cached_object_no_prune(ObjectKind::Blob, &blob_id, &data)
            .unwrap();
        let error = client
            .stage_federated_home_submit_partition(
                "test-capability",
                &proposed_commit,
                [(ObjectKind::Blob, blob_id)],
            )
            .expect_err("the client must not send a Home closure above 32 MiB");
        assert!(matches!(
            error,
            VexClientError::Config(VexConfigError::InvalidFederatedHome(message))
                if message.contains(&MAX_FEDERATED_HOME_STAGE_BYTES.to_string())
        ));
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
    fn pipelined_put_retry_ladder_covers_backend_restart_window() {
        // Sum of uncapped backoff (no jitter) across the inter-attempt delays
        // should land in the ~60–90s band so bulk import survives a short
        // jj-backend OOM/restart rather than failing after a few seconds.
        let mut total_ms = 0u64;
        for attempt in 1..PIPELINED_PUT_RETRY_ATTEMPTS {
            let shift = attempt.saturating_sub(1).min(6) as u32;
            total_ms += PIPELINED_PUT_RETRY_BASE_MS
                .saturating_mul(1_u64 << shift)
                .min(PIPELINED_PUT_RETRY_CAP_MS);
        }
        assert!(
            (60_000..=90_000).contains(&total_ms),
            "expected ~60–90s retry window, got {total_ms}ms \
             (attempts={PIPELINED_PUT_RETRY_ATTEMPTS} base={PIPELINED_PUT_RETRY_BASE_MS} \
             cap={PIPELINED_PUT_RETRY_CAP_MS})"
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
            chunk_frames: false,
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
            chunk_frames: false,
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
    fn inline_fetch_concurrency_parser_uses_default_for_missing_or_invalid_values() {
        assert_eq!(
            parse_inline_fetch_concurrency(None),
            INLINE_FETCH_CONCURRENCY
        );
        assert_eq!(
            parse_inline_fetch_concurrency(Some("not-a-number")),
            INLINE_FETCH_CONCURRENCY
        );
        assert_eq!(
            parse_inline_fetch_concurrency(Some("0")),
            INLINE_FETCH_CONCURRENCY
        );
    }

    #[test]
    fn inline_fetch_concurrency_parser_accepts_positive_values() {
        assert_eq!(parse_inline_fetch_concurrency(Some("1")), 1);
        assert_eq!(parse_inline_fetch_concurrency(Some("16")), 16);
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
            chunk_frames: false,
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
        chunked_pack_fixture_with_frames(object_count, chunk_size, false)
    }

    fn chunked_pack_fixture_with_frames(
        object_count: usize,
        chunk_size: usize,
        chunk_frames: bool,
    ) -> ChunkedPackFixture {
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
        let object_pack = ObjectPack {
            objects: objects.clone(),
        };
        let (encoded, boundaries) = if chunk_frames {
            encode_object_pack_chunked(&object_pack, 3, chunk_size)
        } else {
            (encode_object_pack(&object_pack), Vec::new())
        };
        let chunk_count = if chunk_frames {
            boundaries.len() as u32
        } else {
            encoded.len().div_ceil(chunk_size) as u32
        };
        assert!(chunk_count > 1, "fixture must produce a multi-chunk pack");
        let legacy_boundaries: Vec<usize> = (0..encoded.len()).step_by(chunk_size).collect();
        let chunk_boundaries = if chunk_frames {
            boundaries
        } else {
            legacy_boundaries
        };
        let pieces: Vec<(PackChunkDescriptor, Vec<u8>)> = chunk_boundaries
            .iter()
            .enumerate()
            .map(|(index, start)| {
                let end = chunk_boundaries
                    .get(index + 1)
                    .copied()
                    .unwrap_or(encoded.len());
                let piece = &encoded[*start..end];
                (
                    PackChunkDescriptor {
                        content_id: ContentId::hash_bytes(piece),
                        chunk_index: index as u32,
                        chunk_count,
                        offset_bytes: *start as u64,
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
            chunk_frames,
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

    #[test]
    fn chunk_framed_prefetch_streams_independent_frames_and_counts_presigned() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture_with_frames(24, 256, true);
        let before = vex_client_stats_snapshot();
        let unpacked = run_chunked_prefetch(&fixture, 4);
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
        for (entry, bytes) in fixture.objects.iter().zip(unpacked) {
            assert_eq!(entry.data, bytes);
        }
    }

    #[test]
    fn chunk_framed_prefetch_decodes_recorded_resume_prefix() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture_with_frames(24, 256, true);
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
        let prefix_bytes: usize = fixture.pack.chunks[..recorded]
            .iter()
            .map(|chunk| chunk.size_bytes as usize)
            .sum();
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        fs::write(&partial_path, &fixture.encoded[..prefix_bytes]).unwrap();

        let counter = AtomicU64::new(0);
        assert!(
            futures::executor::block_on(client.prefetch_pack_via_chunks(
                &fixture.pack,
                &fixture.hints,
                &counter,
            ))
            .unwrap()
        );
        let requested = fixture.server.requested_paths();
        for chunk in &fixture.pack.chunks[..recorded] {
            assert!(
                !requested
                    .iter()
                    .any(|path| path.ends_with(&chunk.content_id.to_string())),
                "recorded frame must be decoded from .part, not fetched again"
            );
        }
        assert_eq!(
            counter.load(Ordering::Relaxed),
            fixture.objects.len() as u64
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

    #[test]
    fn chunk_framed_prefetch_hash_failure_keeps_verified_prefix_for_resume() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture_with_frames(24, 256, true);
        let corrupt_server = ChunkServer::start(
            fixture
                .pack
                .chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    let start = chunk.offset_bytes as usize;
                    let end = start + chunk.size_bytes as usize;
                    let mut bytes = fixture.encoded[start..end].to_vec();
                    if index == 1 {
                        bytes[0] ^= 0xff;
                    }
                    (chunk.content_id.to_string(), bytes)
                })
                .collect(),
        );
        let corrupt_hints: Vec<PresignedGet> = fixture
            .pack
            .chunks
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
        futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &corrupt_hints,
            &counter,
        ))
        .unwrap_err();
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        let first_chunk_len = fixture.pack.chunks[0].size_bytes as usize;
        assert_eq!(
            fs::read(&partial_path).unwrap(),
            fixture.encoded[..first_chunk_len]
        );
        assert_eq!(
            client
                .load_pack_transfer_state(&fixture.pack.content_id)
                .unwrap()
                .unwrap()
                .next_chunk_index,
            1
        );

        assert!(
            futures::executor::block_on(client.prefetch_pack_via_chunks(
                &fixture.pack,
                &fixture.hints,
                &counter,
            ))
            .unwrap()
        );
        assert!(
            !fixture
                .server
                .requested_paths()
                .iter()
                .any(|path| { path.ends_with(&fixture.pack.chunks[0].content_id.to_string()) }),
            "the verified frame prefix must not be refetched"
        );
    }

    #[test]
    fn chunk_framed_prefetch_rejects_header_entry_count_mismatch() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let fixture = chunked_pack_fixture_with_frames(24, 256, true);
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
        let mut incomplete_header = fixture.encoded.clone();
        incomplete_header[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let partial_path = client
            .transfer_partial_path(&fixture.pack.content_id)
            .unwrap();
        fs::write(&partial_path, incomplete_header).unwrap();

        let counter = AtomicU64::new(0);
        let err = futures::executor::block_on(client.prefetch_pack_via_chunks(
            &fixture.pack,
            &fixture.hints,
            &counter,
        ))
        .unwrap_err();
        assert!(matches!(err, VexClientError::PackDecode(_)), "{err}");
        assert!(
            client
                .load_pack_transfer_state(&fixture.pack.content_id)
                .unwrap()
                .is_none()
        );
        assert!(!partial_path.exists());
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
            chunk_frames: false,
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

    /// A backend that accepts the call and never answers must not hold the
    /// synchronous publish (and the working-copy lock with it) open: the
    /// attempt fails within its budget, as a status the retry classification
    /// already knows, carrying advice the user can act on.
    #[test]
    fn an_unanswered_publish_attempt_fails_within_its_budget() {
        let budget = Duration::from_millis(50);
        let started = std::time::Instant::now();
        let status = VexClient::shared_grpc_runtime()
            .block_on(publish_attempt_within::<(), _>(
                budget,
                "CommitOperation",
                std::future::pending(),
            ))
            .unwrap_err();

        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
        assert!(status.message().contains(PUBLISH_TIMEOUT_HINT), "{status}");
        assert!(
            VexClient::is_retryable_commit_operation_status(&status),
            "an unanswered attempt must stay retryable inside the budget"
        );
    }

    /// The synchronous publish must not inherit the 300 s bulk-read default.
    #[test]
    fn the_publish_request_timeout_is_short_by_default() {
        assert_eq!(publish_request_timeout(), Duration::from_secs(30));
        assert_eq!(COMMIT_OPERATION_MAINTENANCE_RETRY_ATTEMPTS, 2);
    }

    /// The "cached ⟹ uploaded" short circuit in `put_object` must include
    /// pack-resident objects, or every push would re-upload the metadata a
    /// clone's packs delivered.
    #[test]
    fn put_object_skips_upload_for_pack_resident_objects() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), true);
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
            !client.has_staged_object(commit.kind, &commit.content_id),
            "an index hit must skip the write entirely, not stage it for upload"
        );
        // A genuinely uncached object takes the normal (staged) path.
        let missing_data = b"never-packed".to_vec();
        let missing_id = ContentId::hash_bytes(&missing_data);
        futures::executor::block_on(client.put_object(
            ObjectKind::Commit,
            &missing_id,
            missing_data.clone(),
        ))
        .unwrap();
        assert!(client.has_staged_object(ObjectKind::Commit, &missing_id));
        // ... and staging is durable: the bytes are on disk, so a *later*
        // process can publish them.
        assert_eq!(
            client.read_cached_object(ObjectKind::Commit, &missing_id),
            Some(missing_data)
        );
    }

    /// An orphaned marker — recorded as unpublished, but its bytes are gone —
    /// must stop the push with a diagnosis, not be skipped. Advancing a ref
    /// over the hole is the one outcome that silently corrupts the repository.
    ///
    /// No backend is needed: the only staged object here has no bytes, so
    /// there is nothing to upload and the check is reached before any RPC.
    #[test]
    fn upload_refuses_when_a_staged_object_lost_its_bytes() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), false);
        client.config.repo_id = "repo-staged-orphan".to_string();
        let data = b"bytes that will be deleted".to_vec();
        let content_id = ContentId::hash_bytes(&data);
        futures::executor::block_on(client.put_object(ObjectKind::Commit, &content_id, data))
            .unwrap();
        // Simulate a cache cleared behind our back: the marker survives, the
        // object does not.
        std::fs::remove_file(client.cache_path(ObjectKind::Commit, &content_id).unwrap()).unwrap();

        let error = client.upload_staged_objects().unwrap_err();
        assert!(
            matches!(error, VexClientError::StagedObjectsMissing(_)),
            "expected a staged-objects-missing refusal, got: {error}"
        );
        assert!(
            error.to_string().contains(&content_id.to_hex()),
            "the refusal must name the object that cannot be published: {error}"
        );
        // The marker is deliberately left in place: dropping it would turn a
        // loud failure into a silent one on the next push.
        assert!(client.has_staged_object(ObjectKind::Commit, &content_id));
    }

    #[test]
    fn component_closure_acknowledgement_preserves_a_shared_home_marker() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut home = pack_resident_client(temp_dir.path(), false);
        home.config.repo_id = "41".to_string();
        let data = b"object required by Home and component".to_vec();
        let content_id = ContentId::hash_bytes(&data);
        futures::executor::block_on(home.put_object(ObjectKind::Blob, &content_id, data)).unwrap();
        let closure = [(ObjectKind::Blob, content_id)];
        assert!(home.has_staged_object(ObjectKind::Blob, &content_id));

        let mut component = home.clone();
        component.config.repo_id = "57".to_string();
        component.apply_uploaded_marker_policy(&closure, StagedMarkerPolicy::Preserve);
        assert!(
            home.has_staged_object(ObjectKind::Blob, &content_id),
            "component publication must not hide the object from Home staged publication"
        );

        home.apply_uploaded_marker_policy(&closure, StagedMarkerPolicy::Remove);
        assert!(
            !home.has_staged_object(ObjectKind::Blob, &content_id),
            "the ordinary Home publication owns marker removal after acknowledgement"
        );
    }

    #[test]
    fn exact_marker_retirement_leaves_unrelated_staged_work() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), false);
        client.config.repo_id = "41".to_string();
        let accepted_data = b"accepted federated child".to_vec();
        let accepted_id = ContentId::hash_bytes(&accepted_data);
        let unrelated_data = b"unrelated later work".to_vec();
        let unrelated_id = ContentId::hash_bytes(&unrelated_data);
        futures::executor::block_on(client.put_object(
            ObjectKind::Commit,
            &accepted_id,
            accepted_data,
        ))
        .unwrap();
        futures::executor::block_on(client.put_object(
            ObjectKind::Blob,
            &unrelated_id,
            unrelated_data,
        ))
        .unwrap();

        let retired = client
            .retire_staged_objects([(ObjectKind::Commit, accepted_id)])
            .unwrap();

        assert_eq!(retired, 1);
        assert!(!client.has_staged_object(ObjectKind::Commit, &accepted_id));
        assert!(
            client.has_staged_object(ObjectKind::Blob, &unrelated_id),
            "aggregate success must not clear an unrelated global marker"
        );
    }

    #[test]
    fn accepted_facade_pin_survives_marker_retirement_and_cache_pruning() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), false);
        client.config.repo_id = "41".to_string();
        client.cache_max_bytes = Some(1);
        let facade_data = b"local synthetic aggregate commit".to_vec();
        let facade_id = ContentId::hash_bytes(&facade_data);
        futures::executor::block_on(client.put_object(
            ObjectKind::Commit,
            &facade_id,
            facade_data.clone(),
        ))
        .unwrap();

        assert_eq!(
            client
                .pin_local_federated_objects([(ObjectKind::Commit, facade_id)])
                .unwrap(),
            1
        );
        assert_eq!(
            client
                .retire_staged_objects([(ObjectKind::Commit, facade_id)])
                .unwrap(),
            1
        );
        assert!(!client.has_staged_object(ObjectKind::Commit, &facade_id));

        let evictable_data = b"ordinary server-backed cache entry";
        let evictable_id = ContentId::hash_bytes(evictable_data);
        client
            .write_cached_object(ObjectKind::Blob, &evictable_id, evictable_data)
            .unwrap();

        assert_eq!(
            client.read_cached_object(ObjectKind::Commit, &facade_id),
            Some(facade_data),
            "a local facade object stays readable after its upload marker is retired"
        );
    }

    fn write_flat_home_read_fallback_fixture(
        workspace_root: &Path,
    ) -> (PathBuf, VexFederatedHomeConfig) {
        let repo_path = workspace_root.join(".jj/repo");
        let store_path = repo_path.join("store");
        fs::create_dir_all(&store_path).unwrap();
        let mut home_config = sample_client().config;
        home_config.repo_id = "9001".to_string();
        home_config.repo_slug = "home".to_string();
        home_config.repository_scope_kind = Some("composed".to_string());
        home_config.access_token = Some("vexhome_aggregate".to_string());
        home_config.write_to_repo_path(&repo_path).unwrap();
        let mut manifest = FederatedHomeManifest {
            format_version: VEX_FEDERATED_HOME_FORMAT_VERSION,
            home_repository_id: "9001".to_string(),
            home_bookmark: "main".to_string(),
            home_revision: ContentId::from_bytes([1; 32]),
            components: vec![jj_backend_types::FederatedHomeComponent {
                repository_id: "9002".to_string(),
                root_path: "apps/web".to_string(),
                selected_bookmark: "main".to_string(),
                selected_revision: ContentId::from_bytes([2; 32]),
            }],
            path_owners: Vec::new(),
        };
        manifest.path_owners = manifest.canonical_path_owners();
        let flat_home = VexFederatedHomeConfig {
            format_version: VEX_FEDERATED_HOME_FORMAT_VERSION,
            manifest_artifact_suffix: manifest.artifact_suffix().unwrap(),
            manifest_content_sha256: manifest.content_sha256().unwrap(),
            manifest_generation: 3,
            manifest,
            aggregate_access_token: "vexhome_aggregate".to_string(),
            repositories: vec![
                VexFederatedHomeRepository {
                    repository_id: "9001".to_string(),
                    repository_public_id: "repository_home".to_string(),
                    repository_slug: "home".to_string(),
                    root_path: String::new(),
                    endpoint: "http://127.0.0.1:1".to_string(),
                },
                VexFederatedHomeRepository {
                    repository_id: "9002".to_string(),
                    repository_public_id: "repository_web".to_string(),
                    repository_slug: "web".to_string(),
                    root_path: "apps/web".to_string(),
                    endpoint: "http://127.0.0.1:2".to_string(),
                },
            ],
            aggregate_base_commit_id: Some(ContentId::from_bytes([3; 32])),
        };
        flat_home.write_to_repo_path(&repo_path).unwrap();
        (store_path, flat_home)
    }

    #[test]
    fn evicted_component_object_is_read_through_its_hidden_physical_route() {
        let temp_dir = tempfile::tempdir().unwrap();
        let workspace_root = temp_dir.path().join("home");
        let (store_path, _flat_home) = write_flat_home_read_fallback_fixture(&workspace_root);
        let client = VexClient::from_store_path(&store_path).unwrap();
        let data = b"component blob restored after eviction".to_vec();
        let content_id = ContentId::hash_bytes(&data);
        client
            .write_cached_object_no_prune(ObjectKind::Blob, &content_id, &data)
            .unwrap();
        let cache_path = client.cache_path(ObjectKind::Blob, &content_id).unwrap();
        fs::remove_file(&cache_path).unwrap();

        let attempts = Arc::new(Mutex::new(Vec::new()));
        let restored = futures::executor::block_on(client.get_object_with_fetch(
            ObjectKind::Blob,
            &content_id,
            {
                let attempts = Arc::clone(&attempts);
                let data = data.clone();
                move |config, _kind, _content_id| {
                    let attempts = Arc::clone(&attempts);
                    let data = data.clone();
                    async move {
                        attempts.lock().unwrap().push((
                            config.repo_id.clone(),
                            config.access_token.clone().unwrap_or_default(),
                        ));
                        if config.repo_id == "9002" {
                            Ok(data)
                        } else {
                            Err(VexClientError::Status(tonic::Status::not_found(
                                "object is not in physical Home",
                            )))
                        }
                    }
                }
            },
        ))
        .unwrap();

        assert_eq!(restored, data);
        assert_eq!(
            *attempts.lock().unwrap(),
            vec![
                ("9001".to_string(), "vexhome_aggregate".to_string()),
                ("9002".to_string(), "vexhome_aggregate".to_string()),
            ]
        );
        assert_eq!(fs::read(cache_path).unwrap(), data);
        assert!(!workspace_root.join("apps/web/.jj").exists());
        assert!(!workspace_root.join(".gitmodules").exists());
    }

    #[test]
    fn ordinary_push_refuses_a_flat_facade_without_consuming_markers() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path().join("repo");
        let store_path = repo_path.join("store");
        fs::create_dir_all(&store_path).unwrap();
        let config = sample_client().config;
        config.write_to_repo_path(&repo_path).unwrap();
        fs::write(
            VexFederatedHomeConfig::metadata_path_for_repo(&repo_path),
            b"{}",
        )
        .unwrap();
        let client = VexClient::from_store_path(&store_path).unwrap();
        let data = b"local facade commit".to_vec();
        let content_id = ContentId::hash_bytes(&data);
        futures::executor::block_on(client.put_object(ObjectKind::Commit, &content_id, data))
            .unwrap();

        let error = client.upload_staged_objects().unwrap_err();

        assert!(error.to_string().contains("disabled for a Home checkout"));
        assert!(
            client.has_staged_object(ObjectKind::Commit, &content_id),
            "the refusal must preserve the facade marker for federated retry"
        );
    }

    #[test]
    fn federated_pointer_requires_exact_positive_clone_generation() {
        let pointer = jj_backend_api::FederatedHomeManifestPointer {
            format_version: VEX_FEDERATED_HOME_FORMAT_VERSION,
            generation: 11,
            manifest_artifact_suffix: "federated-home/v1/digest.json".to_string(),
            manifest_content_sha256: "digest".to_string(),
        };

        validate_federated_home_pointer(
            &pointer,
            "federated-home/v1/digest.json",
            "digest",
            Some(11),
        )
        .unwrap();
        let stale = validate_federated_home_pointer(
            &pointer,
            "federated-home/v1/digest.json",
            "digest",
            Some(10),
        )
        .unwrap_err();
        assert!(stale.to_string().contains("generation changed since clone"));

        let mut incoherent = pointer;
        incoherent.generation = 0;
        let error = validate_federated_home_pointer(
            &incoherent,
            "federated-home/v1/digest.json",
            "digest",
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("pointer is incoherent"));
    }

    #[test]
    fn backend_non_current_manifest_is_stale() {
        let error = require_current_federated_home_manifest(false).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("changed since this checkout was cloned")
        );
    }

    /// Whoever publishes a repository's staged objects has to bind to the same
    /// store path the backend that staged them did — the markers live in the
    /// object cache derived from it, and a client built with
    /// [`VexClient::from_config`] has no cache at all, so it would report
    /// "nothing staged" and let a ref advance over objects the backend never
    /// received.
    ///
    /// `vex materialize` (`convert_native::publish_staged_objects`) depends on
    /// exactly this: it publishes through [`VexClient::from_config_at`] at the
    /// throwaway backing repo's store path, while the objects were staged by
    /// the `VexBackend` that [`VexClient::from_store_path`] opened there
    /// (VEX-308).
    #[test]
    fn a_client_opened_at_a_store_path_shares_its_staging_area() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path().join("repo");
        let store_path = repo_path.join("store");
        fs::create_dir_all(&store_path).unwrap();
        let mut config = sample_client().config;
        config.repo_id = "repo-staged-shared".to_string();
        config.write_to_repo_path(&repo_path).unwrap();

        // What the repo's backend does: open from the store path on disk and
        // write an object, which stages it.
        let staging = VexClient::from_store_path(&store_path).unwrap();
        let data = b"a merge commit only this run has".to_vec();
        let content_id = ContentId::hash_bytes(&data);
        futures::executor::block_on(staging.put_object(ObjectKind::Commit, &content_id, data))
            .unwrap();

        // What the publisher does: build from an in-memory config (carrying a
        // freshly rotated token) bound to the same store path.
        let publisher = VexClient::from_config_at(config, &store_path).unwrap();

        assert!(
            publisher.has_staged_object(ObjectKind::Commit, &content_id),
            "the publisher must see what the repository's own backend staged"
        );
    }

    /// Staged objects are the only copy of unpushed work, so a prune must not
    /// evict them however far over its cap the cache is.
    #[test]
    fn prune_never_evicts_staged_objects() {
        let _guard = stats_lock().lock().unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let mut client = pack_resident_client(temp_dir.path(), false);
        client.config.repo_id = "repo-staged-prune".to_string();
        client.cache_max_bytes = Some(1);
        let data = b"unpushed-commit-object".to_vec();
        let content_id = ContentId::hash_bytes(&data);
        futures::executor::block_on(client.put_object(
            ObjectKind::Commit,
            &content_id,
            data.clone(),
        ))
        .unwrap();
        client.prune_cache_if_needed().unwrap();
        assert!(client.has_staged_object(ObjectKind::Commit, &content_id));
        assert_eq!(
            client.read_cached_object(ObjectKind::Commit, &content_id),
            Some(data)
        );
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
