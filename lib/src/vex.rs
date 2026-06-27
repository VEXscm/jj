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
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::SystemTime;

use jj_backend_api::CloneBlobMode as ProtoCloneBlobMode;
use jj_backend_api::GetCloneManifestRequest;
use jj_backend_api::GetObjectRequest;
use jj_backend_api::GetObjectsRequest;
use jj_backend_api::GetRepoRequest;
use jj_backend_api::InitRepoRequest;
use jj_backend_api::InlineObject;
use jj_backend_api::ObjectId;
use jj_backend_api::PutObjectRequest;
use jj_backend_api::PutObjectsRequest;
use jj_backend_api::ResolveOperationIdPrefixRequest;
use jj_backend_api::jj_backend_client::JjBackendClient;
use jj_backend_types::{
    CloneManifest, ContentId, ObjectKind, decode_object_pack, decode_object_pack_reader,
    decode_object_pack_with_visitor,
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

/// Max gRPC message size for both directions. The default tonic decode limit is
/// 4 MiB, which is smaller than legitimately large objects (e.g. a >4 MiB file
/// blob fetched inline via `GetObject` during checkout), so reads would fail
/// with "decoded message length too large". The server already allows 64 MiB
/// (`JJ_GRPC_MAX_MESSAGE_BYTES`); match it on the client for encode and decode.
const MAX_GRPC_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

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
    /// Prefetch finished; the working copy is about to be materialized.
    CheckingOut,
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
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VexRepoConfig {
    pub endpoint: String,
    pub tenant_id: String,
    pub tenant_slug: String,
    pub repo_id: String,
    pub repo_slug: String,
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
        fs::write(path, serde_json::to_vec_pretty(self)?)?;
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

/// Max buffered upload bytes before an inline flush during a snapshot. Bounds
/// peak memory for very large snapshots while still letting a normal snapshot
/// (a handful of small objects) coalesce into a single batched upload.
const PENDING_FLUSH_BYTES: usize = 32 * 1024 * 1024;
/// Companion object-count cap to [`PENDING_FLUSH_BYTES`].
const PENDING_FLUSH_OBJECTS: usize = 256;

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

impl VexClient {
    pub fn from_config(config: VexRepoConfig) -> Result<Self, VexConfigError> {
        Self::validate_endpoint(&config.endpoint)?;
        let local_writes = config.local_writes;
        Ok(Self {
            config,
            cache_root: None,
            cache_max_bytes: cache_max_bytes(),
            local_writes,
        })
    }

    pub fn from_store_path(store_path: &Path) -> Result<Self, VexConfigError> {
        let config = VexRepoConfig::load_from_store_path(store_path)?;
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
        let bytes = fs::read(path)?;
        Ok(Some(
            serde_json::from_slice(&bytes)
                .map_err(VexConfigError::Json)
                .map_err(VexClientError::from)?,
        ))
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

    fn clear_pack_transfer_state(&self, pack_content_id: &ContentId) -> Result<(), VexClientError> {
        if let Some(state_path) = self.transfer_state_path(pack_content_id) {
            drop(fs::remove_file(state_path));
        }
        if let Some(partial_path) = self.transfer_partial_path(pack_content_id) {
            drop(fs::remove_file(partial_path));
        }
        Ok(())
    }

    fn read_cached_object(&self, kind: ObjectKind, content_id: &ContentId) -> Option<Vec<u8>> {
        let path = self.cache_path(kind, content_id)?;
        let bytes = fs::read(&path).ok()?;
        debug!(kind = kind_to_str(kind), %content_id, bytes = bytes.len(), cache_path = %path.display(), "vex cache hit");
        Some(bytes)
    }

    /// Whether an object is present in the local cache, without reading it.
    ///
    /// The cache is content-addressed and only populated after a successful
    /// upload (or by clone prefetch of server-resident objects), so a hit means
    /// the object is already on the server. Callers use this to skip redundant
    /// uploads cheaply (no disk read of the blob body).
    fn has_cached_object(&self, kind: ObjectKind, content_id: &ContentId) -> bool {
        self.cache_path(kind, content_id)
            .is_some_and(|path| path.exists())
    }

    fn write_cached_object(
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
        self.prune_cache_if_needed()?;
        Ok(())
    }

    fn prune_cache_if_needed(&self) -> Result<(), VexClientError> {
        let (Some(cache_root), Some(limit_bytes)) = (&self.cache_root, self.cache_max_bytes) else {
            return Ok(());
        };
        let mut entries = Vec::new();
        collect_cache_entries(cache_root, &mut entries)?;
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
            .connect_timeout(Duration::from_secs(env_secs("VEX_GRPC_CONNECT_TIMEOUT_SECS", 10)))
            .timeout(Duration::from_secs(env_secs("VEX_GRPC_REQUEST_TIMEOUT_SECS", 120)))
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
        let attempts = attempts.max(1);
        Self::shared_grpc_runtime().block_on(with_output_cancel(async move {
            let mut attempt = 0;
            loop {
                attempt += 1;
                let client = JjBackendClient::new(channel.clone())
                    .max_decoding_message_size(MAX_GRPC_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_GRPC_MESSAGE_BYTES);
                match f(client).await {
                    Ok(value) => return Ok(value),
                    Err(status) if Self::is_transient_status(&status) && attempt < attempts => {
                        tokio::time::sleep(std::time::Duration::from_millis(200 * attempt as u64))
                            .await;
                        continue;
                    }
                    Err(status) => return Err(status.into()),
                }
            }
        }))
    }

    fn block_on_http_get(
        url: &str,
        headers: &std::collections::HashMap<String, String>,
    ) -> Result<Vec<u8>, VexClientError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(VexClientError::Runtime)?;
        runtime.block_on(with_output_cancel(async move {
            let client = reqwest::Client::new();
            let mut request = client.get(url);
            for (name, value) in headers {
                request = request.header(name, value);
            }
            let response = request.send().await?.error_for_status()?;
            let bytes = response.bytes().await?;
            Ok(bytes.to_vec())
        }))
    }

    fn block_on_http_get_to_file(
        url: &str,
        headers: &std::collections::HashMap<String, String>,
        out: &mut dyn Write,
    ) -> Result<(), VexClientError> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(VexClientError::Runtime)?;
        runtime.block_on(with_output_cancel(async move {
            let client = reqwest::Client::new();
            let mut request = client.get(url);
            for (name, value) in headers {
                request = request.header(name, value);
            }
            let mut response = request.send().await?.error_for_status()?;
            while let Some(chunk) = response.chunk().await? {
                out.write_all(&chunk)?;
            }
            out.flush()?;
            Ok(())
        }))
    }

    fn direct_fetch_pack_bytes(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
    ) -> Result<Option<Vec<u8>>, VexClientError> {
        let Some(hint) = hints
            .iter()
            .find(|hint| hint.object_key.ends_with(&pack.content_id.to_string()))
        else {
            return Ok(None);
        };
        if hint.url.is_empty() {
            return Ok(None);
        }
        Self::block_on_http_get(&hint.url, &hint.headers).map(Some)
    }

    fn direct_fetch_pack_blob_bytes(
        &self,
        content_id: &ContentId,
        hints: &[jj_backend_api::PresignedGet],
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
        Self::block_on_http_get(&hint.url, &hint.headers).map(Some)
    }

    fn direct_fetch_pack_to_file(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
        out: &mut dyn Write,
    ) -> Result<bool, VexClientError> {
        let Some(hint) = hints
            .iter()
            .find(|hint| hint.object_key.ends_with(&pack.content_id.to_string()))
        else {
            return Ok(false);
        };
        if hint.url.is_empty() {
            return Ok(false);
        }
        Self::block_on_http_get_to_file(&hint.url, &hint.headers, out)?;
        Ok(true)
    }

    fn prefetch_pack_entries_from_file(
        &self,
        path: &Path,
        prefetched_objects: &mut u64,
    ) -> Result<(), VexClientError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut write_error: Option<VexClientError> = None;
        let decode_result = decode_object_pack_with_visitor(reader, |entry| {
            match self.write_cached_object(entry.kind, &entry.content_id, &entry.data) {
                Ok(()) => {
                    *prefetched_objects += 1;
                    Ok(())
                }
                Err(err) => {
                    write_error = Some(err);
                    Err(jj_backend_types::PackCodecError::Compression(
                        "cache write failed".to_string(),
                    ))
                }
            }
        });
        if let Some(err) = write_error {
            return Err(err);
        }
        decode_result.map_err(|err| VexClientError::PackDecode(err.to_string()))
    }

    async fn fetch_pack_blob_with_retry(
        &self,
        content_id: &ContentId,
        hints: &[jj_backend_api::PresignedGet],
    ) -> Result<Vec<u8>, VexClientError> {
        let mut last_hint_err: Option<VexClientError> = None;
        for _ in 0..2 {
            match self.direct_fetch_pack_blob_bytes(content_id, hints) {
                Ok(Some(bytes)) => return Ok(bytes),
                Ok(None) => break,
                Err(err) => last_hint_err = Some(err),
            }
        }
        if let Some(err) = last_hint_err {
            debug!(%content_id, error = %err, "direct chunk fetch failed, falling back to grpc");
        }
        self.get_object(ObjectKind::Pack, content_id).await
    }

    async fn prefetch_pack_via_chunks(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        hints: &[jj_backend_api::PresignedGet],
        prefetched_objects: &mut u64,
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
        if partial_file.metadata()?.len() != expected_prefix_bytes {
            partial_file.set_len(0)?;
            partial_file.seek(SeekFrom::Start(0))?;
            state.next_chunk_index = 0;
        }
        for (idx, chunk) in chunks.iter().enumerate().skip(state.next_chunk_index) {
            let chunk_bytes = self
                .fetch_pack_blob_with_retry(&chunk.content_id, hints)
                .await?;
            if u64::try_from(chunk_bytes.len()).unwrap_or(u64::MAX) != chunk.size_bytes {
                // Keep state file for debugging, but restart next attempt from scratch.
                state.next_chunk_index = 0;
                self.save_pack_transfer_state(&pack.content_id, &state)?;
                return Err(VexClientError::PackDecode(format!(
                    "chunk size mismatch for pack {} chunk {}",
                    pack.content_id, idx
                )));
            }
            partial_file.write_all(&chunk_bytes)?;
            state.next_chunk_index = idx + 1;
            self.save_pack_transfer_state(&pack.content_id, &state)?;
        }
        partial_file.flush()?;
        drop(partial_file);
        self.prefetch_pack_entries_from_file(&partial_path, prefetched_objects)?;
        self.clear_pack_transfer_state(&pack.content_id)?;
        Ok(true)
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
            access_token: access_token.map(ToOwned::to_owned),
            local_writes: false,
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
            access_token: access_token.map(ToOwned::to_owned),
            local_writes: false,
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
    fn buffer_pending_object(&self, kind: ObjectKind, content_id: &ContentId, data: Vec<u8>) -> bool {
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
        let channel = Self::cached_channel(&self.config.endpoint)?;
        let repo_id = self.config.repo_id.clone();
        let token = self.config.access_token.clone();
        let concurrency = concurrency.max(1);
        Self::shared_grpc_runtime().block_on(async move {
            use futures::stream::TryStreamExt as _;
            futures::stream::iter(inline_batches.into_iter().map(Ok::<_, VexClientError>))
                .try_for_each_concurrent(concurrency, |objects| {
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
            return Ok(bytes);
        }
        // An object written earlier this process may still be buffered for batch
        // upload (not yet on disk or the server); serve it from the buffer.
        if let Some(bytes) = self.read_pending_object(kind, content_id) {
            return Ok(bytes);
        }
        debug!(kind = kind_to_str(kind), %content_id, "vex cache miss");
        let bytes = Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
            client
                .get_object(Self::auth_request(
                    GetObjectRequest {
                        repo_id: self.config.repo_id.clone(),
                        object: Some(ObjectId {
                            kind: kind_to_str(kind).to_string(),
                            content_id: content_id.to_string(),
                        }),
                    },
                    self.config.access_token.as_deref(),
                )?)
                .await
                .map(|response| response.into_inner().data)
        })?;
        self.write_cached_object(kind, content_id, &bytes)?;
        Ok(bytes)
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
        let response = Self::block_on_grpc(&self.config.endpoint, |mut client| async move {
            client
                .commit_operation(Self::auth_request(
                    jj_backend_api::CommitOperationRequest {
                        tenant_id: self.config.tenant_id.clone(),
                        repo_id: self.config.repo_id.clone(),
                        expected_op_head_ids: expected.iter().map(ToString::to_string).collect(),
                        new_op_content_id: new_head.to_string(),
                        new_view_content_id: new_view.to_string(),
                    },
                    self.config.access_token.as_deref(),
                )?)
                .await
                .map(|response| response.into_inner())
        })?;
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

    pub async fn get_clone_manifest(
        &self,
        blob_mode: CloneBlobMode,
    ) -> Result<CloneManifest, VexClientError> {
        let response =
            Self::block_on_grpc_retry(&self.config.endpoint, 5, |mut client| async move {
                client
                    .get_clone_manifest(Self::auth_request(
                        GetCloneManifestRequest {
                            tenant_id: self.config.tenant_id.clone(),
                            repo_id: self.config.repo_id.clone(),
                            clone_blob_mode: match blob_mode {
                                CloneBlobMode::Eager => ProtoCloneBlobMode::Eager as i32,
                                CloneBlobMode::Lazy => ProtoCloneBlobMode::Lazy as i32,
                            },
                        },
                        self.config.access_token.as_deref(),
                    )?)
                    .await
                    .map(|response| response.into_inner())
            })?;
        serde_json::from_slice(&response.manifest_json)
            .map_err(VexConfigError::Json)
            .map_err(Into::into)
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
        progress: Option<&CloneProgressFn>,
    ) -> Result<(), VexClientError> {
        let prefetch_started = std::time::Instant::now();
        let mut prefetched_objects = 0_u64;
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
        for (index, pack) in manifest.packs.iter().enumerate() {
            self.prefetch_one_pack(pack, &pack_hints, &mut prefetched_objects)
                .await?;
            if let Some(progress) = progress {
                progress(CloneProgress::PackFetched {
                    done: index as u64 + 1,
                    total: total_packs,
                    objects: prefetched_objects,
                });
            }
        }
        let total_loose = manifest.objects.len() as u64;
        let mut loose_done = 0_u64;
        for object in &manifest.objects {
            loose_done += 1;
            if self
                .read_cached_object(object.kind, &object.content_id)
                .is_some()
            {
                if let Some(progress) = progress {
                    progress(CloneProgress::LooseObjectFetched {
                        done: loose_done,
                        total: total_loose,
                    });
                }
                continue;
            }
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
            self.write_cached_object(object.kind, &object.content_id, &bytes)?;
            prefetched_objects += 1;
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
            deferred_object_count = manifest.deferred_object_count,
            deferred_object_bytes = manifest.deferred_object_bytes,
            prefetched_objects,
            elapsed_ms = prefetch_started.elapsed().as_millis(),
            "prefetched clone manifest"
        );
        Ok(())
    }

    /// Fetch and unpack a single clone pack into the local object cache,
    /// trying the chunked path first and falling back to streamed and then
    /// whole-pack reads. `prefetched_objects` is incremented per object written.
    async fn prefetch_one_pack(
        &self,
        pack: &jj_backend_types::PackDescriptor,
        pack_hints: &[jj_backend_api::PresignedGet],
        prefetched_objects: &mut u64,
    ) -> Result<(), VexClientError> {
        match self
            .prefetch_pack_via_chunks(pack, pack_hints, prefetched_objects)
            .await
        {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(err) => {
                debug!(
                    pack_content_id = %pack.content_id,
                    error = %err,
                    "chunk path failed, using full-pack fallback"
                );
            }
        }
        let mut temp_pack = NamedTempFile::new()?;
        let streamed = self
            .direct_fetch_pack_to_file(pack, pack_hints, temp_pack.as_file_mut())
            .unwrap_or(false);
        if streamed {
            self.prefetch_pack_entries_from_file(temp_pack.path(), prefetched_objects)?;
            return Ok(());
        }

        let pack_bytes = match self.direct_fetch_pack_bytes(pack, pack_hints) {
            Ok(Some(bytes)) => bytes,
            Ok(None) | Err(_) => self.get_object(ObjectKind::Pack, &pack.content_id).await?,
        };
        let object_pack = decode_object_pack(&pack_bytes)
            .or_else(|_| decode_object_pack_reader(BufReader::new(pack_bytes.as_slice())))
            .map_err(|err| VexClientError::PackDecode(err.to_string()))?;
        for entry in object_pack.objects {
            self.write_cached_object(entry.kind, &entry.content_id, &entry.data)?;
            *prefetched_objects += 1;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use jj_backend_api::PresignedGet;
    use jj_backend_types::{ClonePackScope, PackChunkDescriptor, PackDescriptor};
    use std::io::Read;
    use std::net::TcpListener;
    use std::thread;

    fn sample_client() -> VexClient {
        VexClient::from_config(VexRepoConfig {
            endpoint: "http://127.0.0.1:50051".to_string(),
            tenant_id: "tenant".to_string(),
            tenant_slug: "tenant".to_string(),
            repo_id: "repo".to_string(),
            repo_slug: "repo".to_string(),
            access_token: None,
            local_writes: false,
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
        client.clear_pack_transfer_state(&pack_id).unwrap();
        assert!(client.load_pack_transfer_state(&pack_id).unwrap().is_none());
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
    fn direct_fetch_pack_bytes_uses_http_hint() {
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
            size_bytes: 4,
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
}

pub fn create_store_factories() -> StoreFactories {
    let mut store_factories = StoreFactories::empty();
    store_factories.add_backend(
        VexBackend::name_static(),
        Box::new(|_settings, store_path| Ok(Box::new(VexBackend::load(store_path)?))),
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
