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

//! Local-first op log: durability modes, the pending-publish queue, and the
//! publisher that drains it (roadmap/088).
//!
//! In [`VexDurability::Sync`] — the default — none of this runs and every
//! operation still CASes the backend inline. In the deferred modes an
//! operation is durable once its objects and marker files are on local disk;
//! the publisher moves it to the server afterwards, either at the end of the
//! command or at the next sync barrier.
//!
//! Marker files live beside the existing `vex-local-heads` /
//! `vex-pending-registration` markers in `.jj/repo/op_heads/`, share their
//! write-then-rename discipline, and carry a `v` field. An unreadable or
//! future-versioned marker is never guessed at: the caller falls back to
//! synchronous publication, which is always correct.

#![expect(missing_docs)]

use std::collections::BTreeSet;
use std::fmt;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use async_trait::async_trait;
use jj_backend_types::ContentId;
use jj_backend_types::ObjectKind;
use prost::Message as _;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::Digest as _;
use sha2::Sha256;
use tempfile::NamedTempFile;

use crate::object_id::ObjectId as _;
use crate::op_store::Operation;
use crate::op_store::OperationId;
use crate::op_store::ViewId;
use crate::simple_op_store::operation_from_proto;
use crate::simple_op_store::operation_to_proto;
use crate::vex::VexClient;
use crate::vex::VexClientError;
use crate::vex::kind_from_str;
use crate::vex::kind_to_str;
use crate::vex::vex_client_stats;

/// Name of the file (inside the op_heads store dir) where the local-write CI
/// runner and every deferred-publish repo record their op head(s). One hex
/// content id per line.
pub const LOCAL_HEADS_FILE: &str = "vex-local-heads";
/// Ordered chain of locally committed but unpublished operations.
pub const PENDING_PUBLISH_FILE: &str = "vex-pending-publish";
/// Last confirmed server op-head set, and which local operation it stands for.
pub const SERVER_HEADS_FILE: &str = "vex-server-heads";

/// Marker schema version. Bump only for an incompatible change: an older
/// client reading a newer marker falls back to synchronous publication rather
/// than misreading it.
pub const MARKER_VERSION: u32 = 1;

/// Objects per `put_objects` call while draining a queue. Sized so a batch
/// completes well inside the public edge's request timeout even on a slow link;
/// a whole backlog in one request does not.
const UPLOAD_BATCH_OBJECTS: usize = 64;

/// How many times the publisher re-derives its CAS base before giving up and
/// leaving the chain queued.
const MAX_PUBLISH_ATTEMPTS: usize = 3;

/// Whether a burst publishes as one rewritten operation instead of one CAS per
/// operation.
///
/// Off by default; set `VEX_PUBLISH_COALESCE=1` to turn it on.
///
/// The reason to want it: a burst costs one op-head CAS per operation, so with
/// several agents writing to one repository the queue can grow faster than it
/// drains — every command adds an operation while the drain retires them one at
/// a time.
///
/// The reason it is still not the default: the rewrite used to publish an
/// operation that was a *sibling* of the local tip, which permanently diverged
/// the repository. That specific defect is fixed — the rewrite now carries the
/// tip as an extra parent (see [`VexPublisher::head_to_publish`]), leaving the
/// working copy stale rather than divergent — but the convergence properties
/// the sequential drain is tested for (an interrupted drain resuming, a moved
/// server head converging, a stale recorded base costing no rejected CAS) are
/// not yet proven for the coalesced path. Prove those, and measure the win,
/// before flipping this.
fn coalescing_enabled() -> bool {
    matches!(
        std::env::var("VEX_PUBLISH_COALESCE").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

const ID_LENGTH: usize = 32;

/// When the op-head CAS runs relative to the command that produced the
/// operation.
///
/// The modes weaken *when* the server must be consistent, never whether
/// publication is correct: every mode drains through the same publisher and
/// the same CAS. [`crate::vex::VexRepoConfig::local_writes`] is a separate,
/// stronger opt-out that never publishes at all.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VexDurability {
    /// Objects flush and the op head CASes before the command returns. No
    /// longer the default: it is the mode to choose when a command's return
    /// must mean "the server has it", independent of any later barrier.
    Sync,
    /// Operations are recorded locally and drained before the process exits.
    FlushOnExit,
    /// Operations are recorded locally; the publisher drains best-effort at
    /// the end of the command and, failing that, at the next command or sync
    /// barrier. The default for interactive use: a command returns once the
    /// operation is durable on local disk, and `push`/`pull`/`submit`/`land`
    /// still flush and confirm before their own server work.
    #[default]
    LocalFirst,
}

impl VexDurability {
    pub fn is_sync(&self) -> bool {
        matches!(self, Self::Sync)
    }

    /// Whether this is the serialized default. `vex.json` omits the field at
    /// the default so a clone written by a newer CLI stays readable by an
    /// older one; any non-default mode is written explicitly.
    pub fn is_serialized_default(&self) -> bool {
        matches!(self, Self::LocalFirst)
    }

    /// Whether operations are recorded locally and published out of band.
    pub fn defers_publish(self) -> bool {
        !self.is_sync()
    }

    /// Whether the process must not exit with a non-empty queue.
    pub fn blocks_on_exit(self) -> bool {
        matches!(self, Self::FlushOnExit)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::FlushOnExit => "flush-on-exit",
            Self::LocalFirst => "local-first",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "sync" => Some(Self::Sync),
            "flush-on-exit" => Some(Self::FlushOnExit),
            "local-first" => Some(Self::LocalFirst),
            _ => None,
        }
    }

    /// `VEX_DURABILITY` overrides the repo's configured mode. An unrecognized
    /// value keeps the configured mode and warns once — a typo must not
    /// silently change durability.
    /// Whether this process looks like a disposable CI machine. Vex's own
    /// runners export `CI=true`, as does every mainstream CI system. Such a
    /// machine can be destroyed the moment its job ends, so a queue that has
    /// not drained is lost work rather than merely deferred work.
    fn ephemeral_environment() -> bool {
        Self::is_ephemeral_marker(std::env::var("CI").ok().as_deref())
    }

    /// Pure form of the `CI` check, so the policy is testable without mutating
    /// process environment.
    fn is_ephemeral_marker(value: Option<&str>) -> bool {
        let Some(value) = value else {
            return false;
        };
        let value = value.trim().to_ascii_lowercase();
        !value.is_empty() && value != "0" && value != "false"
    }

    /// Pure form of the CI adjustment applied inside [`Self::resolve`].
    fn for_environment(configured: Self, ephemeral: bool) -> Self {
        if configured == Self::LocalFirst && ephemeral {
            Self::FlushOnExit
        } else {
            configured
        }
    }

    pub fn resolve(configured: Self) -> Self {
        // An explicit `VEX_DURABILITY` always wins, including over the CI
        // adjustment below, so a runner can still opt into any mode.
        // Same deferred fast path on CI, but the process may not exit until
        // the queue is on the server.
        let configured = Self::for_environment(configured, Self::ephemeral_environment());
        let Ok(raw) = std::env::var("VEX_DURABILITY") else {
            return configured;
        };
        if raw.trim().is_empty() {
            return configured;
        }
        match Self::parse(&raw) {
            Some(mode) => mode,
            None => {
                static WARNED: OnceLock<()> = OnceLock::new();
                WARNED.get_or_init(|| {
                    tracing::warn!(
                        value = raw,
                        configured = configured.as_str(),
                        "unrecognized VEX_DURABILITY; keeping the configured durability mode"
                    );
                });
                configured
            }
        }
    }
}

/// A marker file could not be understood. Callers treat this as "this repo has
/// no usable deferred-publish state" and fall back to synchronous publication.
#[derive(Debug)]
pub enum MarkerError {
    Io(std::io::Error),
    Corrupt { file: &'static str, message: String },
    UnsupportedVersion { file: &'static str, version: u32 },
}

impl fmt::Display for MarkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "marker io error: {err}"),
            Self::Corrupt { file, message } => write!(f, "marker {file} is unreadable: {message}"),
            Self::UnsupportedVersion { file, version } => {
                write!(f, "marker {file} has unsupported version {version}")
            }
        }
    }
}

impl std::error::Error for MarkerError {}

impl From<std::io::Error> for MarkerError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

trait VersionedMarker: DeserializeOwned + Serialize {
    const FILE: &'static str;
    fn version(&self) -> u32;
}

/// Last confirmed server op-head set.
///
/// `published_local_head` records which *local* operation those server heads
/// stand for. It is `None` while the two agree byte for byte (a directly
/// published operation), and `Some` after a coalesced publish, where the
/// server holds a rewritten operation carrying the local tip's view. Without
/// it a converged repo would look permanently diverged and every command would
/// re-merge.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ServerHeadsMarker {
    pub v: u32,
    #[serde(default)]
    pub heads: Vec<String>,
    #[serde(default)]
    pub published_local_head: Option<String>,
    #[serde(default)]
    pub updated_unix: i64,
}

impl VersionedMarker for ServerHeadsMarker {
    const FILE: &'static str = SERVER_HEADS_FILE;

    fn version(&self) -> u32 {
        self.v
    }
}

impl ServerHeadsMarker {
    pub fn new(heads: Vec<ContentId>, published_local_head: Option<ContentId>) -> Self {
        Self {
            v: MARKER_VERSION,
            heads: heads.iter().map(ToString::to_string).collect(),
            published_local_head: published_local_head.map(|id| id.to_string()),
            updated_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|elapsed| elapsed.as_secs() as i64)
                .unwrap_or_default(),
        }
    }

    pub fn head_ids(&self) -> Result<Vec<ContentId>, MarkerError> {
        parse_ids(SERVER_HEADS_FILE, &self.heads)
    }

    /// Whether `local` is the local operation these server heads stand for.
    pub fn stands_for(&self, local: &[ContentId]) -> bool {
        match &self.published_local_head {
            Some(id) => local.len() == 1 && local[0].to_string() == *id,
            None => id_set(local) == self.heads.iter().cloned().collect::<BTreeSet<_>>(),
        }
    }
}

/// One locally committed, unpublished operation and the objects it introduced.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PendingOpEntry {
    pub op: String,
    #[serde(default)]
    pub objects: Vec<PendingObject>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PendingObject {
    pub kind: String,
    pub id: String,
}

/// The ordered chain of unpublished operations, plus the server head set the
/// chain was built on.
///
/// `base_heads` is bookkeeping for the read path — it is what
/// [`crate::vex_freshness::known_divergence`] compares the recorded server
/// heads against — not an input to publication. The sequential drain CASes
/// each operation against its own recorded parents, and the backend requires a
/// published operation's parents to *cover* the CAS `expected` set, so a stale
/// base costs nothing.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PendingPublishMarker {
    pub v: u32,
    #[serde(default)]
    pub base_heads: Vec<String>,
    #[serde(default)]
    pub ops: Vec<PendingOpEntry>,
}

impl VersionedMarker for PendingPublishMarker {
    const FILE: &'static str = PENDING_PUBLISH_FILE;

    fn version(&self) -> u32 {
        self.v
    }
}

impl PendingPublishMarker {
    pub fn new(base_heads: &[ContentId]) -> Self {
        Self {
            v: MARKER_VERSION,
            base_heads: base_heads.iter().map(ToString::to_string).collect(),
            ops: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn base_ids(&self) -> Result<Vec<ContentId>, MarkerError> {
        parse_ids(PENDING_PUBLISH_FILE, &self.base_heads)
    }

    pub fn contains(&self, op: &ContentId) -> bool {
        let hex = op.to_string();
        self.ops.iter().any(|entry| entry.op == hex)
    }

    pub fn push(&mut self, op: &ContentId, objects: &[(ObjectKind, ContentId)]) {
        self.ops.push(PendingOpEntry {
            op: op.to_string(),
            objects: objects
                .iter()
                .map(|(kind, id)| PendingObject {
                    kind: kind_to_str(*kind).to_string(),
                    id: id.to_string(),
                })
                .collect(),
        });
    }
}

fn parse_ids(file: &'static str, values: &[String]) -> Result<Vec<ContentId>, MarkerError> {
    values
        .iter()
        .map(|value| {
            ContentId::from_hex(value).map_err(|err| MarkerError::Corrupt {
                file,
                message: err.to_string(),
            })
        })
        .collect()
}

fn id_set(ids: &[ContentId]) -> BTreeSet<String> {
    ids.iter().map(ToString::to_string).collect()
}

fn read_marker<T: VersionedMarker>(dir: &Path) -> Result<Option<T>, MarkerError> {
    let text = match std::fs::read_to_string(dir.join(T::FILE)) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(MarkerError::Io(err)),
    };
    let marker: T = serde_json::from_str(&text).map_err(|err| MarkerError::Corrupt {
        file: T::FILE,
        message: err.to_string(),
    })?;
    if marker.version() != MARKER_VERSION {
        return Err(MarkerError::UnsupportedVersion {
            file: T::FILE,
            version: marker.version(),
        });
    }
    Ok(Some(marker))
}

fn write_marker<T: VersionedMarker>(dir: &Path, marker: &T) -> Result<(), MarkerError> {
    let data = serde_json::to_vec_pretty(marker).map_err(|err| MarkerError::Corrupt {
        file: T::FILE,
        message: err.to_string(),
    })?;
    std::fs::create_dir_all(dir)?;
    let mut temporary = NamedTempFile::new_in(dir)?;
    temporary.write_all(&data)?;
    temporary.flush()?;
    temporary
        .persist(dir.join(T::FILE))
        .map_err(|err| MarkerError::Io(err.error))?;
    Ok(())
}

fn remove_marker(dir: &Path, file: &str) -> Result<(), MarkerError> {
    match std::fs::remove_file(dir.join(file)) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(MarkerError::Io(err)),
    }
}

pub fn read_pending_publish(dir: &Path) -> Result<Option<PendingPublishMarker>, MarkerError> {
    read_marker(dir)
}

pub fn write_pending_publish(dir: &Path, marker: &PendingPublishMarker) -> Result<(), MarkerError> {
    write_marker(dir, marker)
}

pub fn read_server_heads(dir: &Path) -> Result<Option<ServerHeadsMarker>, MarkerError> {
    read_marker(dir)
}

pub fn write_server_heads(dir: &Path, marker: &ServerHeadsMarker) -> Result<(), MarkerError> {
    write_marker(dir, marker)
}

/// Read op heads previously recorded locally. `None` when nothing has been
/// recorded, so callers fall back to the backend.
pub fn read_local_heads(dir: &Path) -> Result<Option<Vec<OperationId>>, MarkerError> {
    let text = match std::fs::read_to_string(dir.join(LOCAL_HEADS_FILE)) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(MarkerError::Io(err)),
    };
    let ids = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            ContentId::from_hex(line)
                .map(|id| OperationId::new(id.as_bytes().to_vec()))
                .map_err(|err| MarkerError::Corrupt {
                    file: LOCAL_HEADS_FILE,
                    message: err.to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((!ids.is_empty()).then_some(ids))
}

/// Record op heads locally, replacing any previous set.
pub fn write_local_heads(dir: &Path, heads: &[ContentId]) -> Result<(), MarkerError> {
    std::fs::create_dir_all(dir)?;
    let mut body = String::new();
    for head in heads {
        body.push_str(&head.to_string());
        body.push('\n');
    }
    let mut temporary = NamedTempFile::new_in(dir)?;
    temporary.write_all(body.as_bytes())?;
    temporary.flush()?;
    temporary
        .persist(dir.join(LOCAL_HEADS_FILE))
        .map_err(|err| MarkerError::Io(err.error))?;
    Ok(())
}

pub fn content_id_from_op_id(id: &OperationId) -> Option<ContentId> {
    let bytes = id.to_bytes();
    let bytes: [u8; ID_LENGTH] = bytes.try_into().ok()?;
    Some(ContentId::from_bytes(bytes))
}

pub fn op_id_from_content_id(id: &ContentId) -> OperationId {
    OperationId::new(id.as_bytes().to_vec())
}

fn content_id_from_view_id(id: &ViewId) -> Option<ContentId> {
    let bytes: [u8; ID_LENGTH] = id.to_bytes().try_into().ok()?;
    Some(ContentId::from_bytes(bytes))
}

fn sha256_content_id(data: &[u8]) -> ContentId {
    let mut hasher = Sha256::new();
    hasher.update(data);
    ContentId::from_bytes(hasher.finalize().into())
}

fn is_root_content_id(id: &ContentId) -> bool {
    id.as_bytes().iter().all(|byte| *byte == 0)
}

/// Result of one `commit_op_heads` CAS.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CasOutcome {
    pub ok: bool,
    pub current_heads: Vec<ContentId>,
    pub error_message: String,
}

/// The backend operations the publisher needs. A trait so the publisher's
/// conflict, coalescing and replay paths can be exercised against a fake
/// without a live backend.
#[async_trait]
pub trait PublishTransport: Send + Sync {
    /// Object bytes from the local cache; `None` when absent.
    fn read_object(&self, kind: ObjectKind, id: &ContentId) -> Option<Vec<u8>>;

    async fn put_objects(
        &self,
        objects: Vec<(ObjectKind, ContentId, Vec<u8>)>,
    ) -> Result<(), PublishError>;

    async fn commit_op_heads(
        &self,
        expected: &[ContentId],
        new_head: &ContentId,
        new_view: &ContentId,
    ) -> Result<CasOutcome, PublishError>;

    async fn get_op_heads(&self) -> Result<Vec<ContentId>, PublishError>;
}

#[derive(Debug)]
pub enum PublishError {
    Marker(MarkerError),
    Transport(String),
    MissingObject {
        kind: String,
        id: String,
    },
    Corrupt(String),
    /// A backend call did not answer within its deadline. Distinct from
    /// [`Self::Transport`] because the CAS outcome is *unknown*, not failed:
    /// the queue stays intact and the next drain re-derives it, which the
    /// server's replay check makes safe.
    Deadline {
        rpc: &'static str,
        budget: Duration,
    },
}

impl fmt::Display for PublishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Marker(err) => write!(f, "{err}"),
            Self::Transport(message) => write!(f, "{message}"),
            Self::MissingObject { kind, id } => write!(
                f,
                "object {kind}/{id} referenced by an unpublished operation is missing from the \
                 local cache"
            ),
            Self::Corrupt(message) => write!(f, "{message}"),
            Self::Deadline { rpc, budget } => write!(
                f,
                "backend call {rpc} did not answer within {}s; the operations stay queued and \
                 publish on the next attempt",
                budget.as_secs_f32()
            ),
        }
    }
}

impl std::error::Error for PublishError {}

impl From<MarkerError> for PublishError {
    fn from(err: MarkerError) -> Self {
        Self::Marker(err)
    }
}

impl From<VexClientError> for PublishError {
    fn from(err: VexClientError) -> Self {
        Self::Transport(err.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublishOutcome {
    /// Nothing was queued.
    Idle,
    /// The chain reached the server.
    Published {
        /// Local operations the CAS covered.
        ops: usize,
        /// Whether the published operation is a rewrite carrying the chain's
        /// tip view rather than the tip operation itself.
        coalesced: bool,
        head: ContentId,
        elapsed_ms: u64,
    },
    /// The server head moved to state this repo has not merged yet. The chain
    /// stays queued and the recorded server heads are updated, so the next
    /// repo load merges them locally and the following drain succeeds.
    ServerHeadMoved { server_heads: Vec<ContentId> },
}

impl PublishOutcome {
    pub fn published_ops(&self) -> usize {
        match self {
            Self::Published { ops, .. } => *ops,
            _ => 0,
        }
    }
}

/// Drains the pending-publish chain to the backend.
pub struct VexPublisher<'a> {
    dir: &'a Path,
    transport: &'a dyn PublishTransport,
    coalesce: bool,
}

impl<'a> VexPublisher<'a> {
    pub fn new(dir: &'a Path, transport: &'a dyn PublishTransport) -> Self {
        Self {
            dir,
            transport,
            coalesce: coalescing_enabled(),
        }
    }

    /// Choose the drain strategy explicitly instead of reading the
    /// environment. Only the rewrite-coalescing hazard documented on
    /// [`Self::drain_coalesced`] makes this interesting.
    pub fn with_coalescing(mut self, coalesce: bool) -> Self {
        self.coalesce = coalesce;
        self
    }

    /// Publish every queued operation: upload the objects they introduced,
    /// then advance the server head. The default strategy is one CAS per
    /// queued operation under its own id ([`Self::drain_sequentially`]); the
    /// opt-in coalescing strategy rewrites the burst into a single operation
    /// parented on the chain's base and is documented — with its hazard — on
    /// [`Self::drain_coalesced`].
    pub async fn drain(&self) -> Result<PublishOutcome, PublishError> {
        let start = Instant::now();
        let Some(chain) = read_pending_publish(self.dir)? else {
            return Ok(PublishOutcome::Idle);
        };
        if chain.is_empty() {
            remove_marker(self.dir, PENDING_PUBLISH_FILE)?;
            return Ok(PublishOutcome::Idle);
        }
        let stats = vex_client_stats();
        stats
            .pending_ops
            .store(chain.ops.len() as u64, Ordering::Relaxed);

        self.upload_chain_objects(&chain).await?;

        if self.coalesce {
            return self.drain_coalesced(chain, start).await;
        }
        self.drain_sequentially(chain, start).await
    }

    /// Publish each queued operation under its own id, in order, with
    /// `expected` set to that operation's real parents.
    ///
    /// This is the only strategy that cannot diverge. Operation ids are
    /// preserved end to end, so the server head after a drain is literally the
    /// operation the working copy is pinned to; every other client sees the
    /// same history this repo has. It costs one CAS per operation instead of
    /// one per burst, but all of them are off the interactive path.
    async fn drain_sequentially(
        &self,
        mut chain: PendingPublishMarker,
        start: Instant,
    ) -> Result<PublishOutcome, PublishError> {
        let stats = vex_client_stats();
        let mut published = 0;
        let mut head = None;
        // Remembered before the drain consumes entries: a folded clone
        // registration is only ever the chain's first operation.
        let first_queued = chain.ops.first().map(|entry| entry.op.clone());
        while let Some(entry) = chain.ops.first() {
            let id = ContentId::from_hex(&entry.op)
                .map_err(|err| PublishError::Corrupt(format!("invalid pending op id: {err}")))?;
            let operation = self.read_operation(&id)?;
            let view = content_id_from_view_id(&operation.view_id)
                .ok_or_else(|| PublishError::Corrupt("invalid view id length".to_string()))?;
            let expected = operation_parents(&operation);

            let outcome = self
                .transport
                .commit_op_heads(&expected, &id, &view)
                .await?;
            if outcome.ok || outcome.current_heads == [id] {
                chain.ops.remove(0);
                self.commit_progress(&chain, &id, first_queued.as_deref())?;
                published += 1;
                head = Some(id);
                continue;
            }

            stats.publish_cas_conflicts.fetch_add(1, Ordering::Relaxed);
            let server_heads = self.transport.get_op_heads().await?;
            // A previous drain may have published further down the chain than
            // its marker update recorded. Resume from wherever the server got
            // to rather than republishing.
            if let Some(index) = chain.ops.iter().position(|queued| {
                server_heads.len() == 1 && queued.op == server_heads[0].to_string()
            }) {
                chain.ops.drain(..=index);
                self.commit_progress(&chain, &server_heads[0], first_queued.as_deref())?;
                published += index + 1;
                head = Some(server_heads[0]);
                stats.publish_folds.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            // The server moved to work this repo has *already merged*: jj
            // resolved the divergent op heads into a merge operation, which is
            // queued like any other. That merge is the one operation in the
            // chain that can be CASed against the moved head, so publish it
            // directly instead of dead-ending.
            if let Some((index, merge)) = self.merge_forward_target(&chain, &server_heads)? {
                let outcome = self
                    .transport
                    .commit_op_heads(&server_heads, &merge.id, &merge.view)
                    .await?;
                if outcome.ok || outcome.current_heads == [merge.id] {
                    chain.ops.drain(..=index);
                    self.commit_progress(&chain, &merge.id, first_queued.as_deref())?;
                    published += index + 1;
                    head = Some(merge.id);
                    stats.publish_folds.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                tracing::debug!(
                    merge = %merge.id,
                    error = outcome.error_message,
                    "publishing the merge of the moved server head lost its own CAS race"
                );
            }
            self.record_server_heads_only(&server_heads)?;
            tracing::warn!(
                queued = chain.ops.len(),
                server_heads = ?server_heads.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "server op head moved to unmerged state; keeping the pending chain queued"
            );
            return Ok(PublishOutcome::ServerHeadMoved { server_heads });
        }

        let elapsed_ms = start.elapsed().as_millis() as u64;
        stats.pending_ops.store(0, Ordering::Relaxed);
        stats.publish_lag_ms.store(elapsed_ms, Ordering::Relaxed);
        match head {
            Some(head) => Ok(PublishOutcome::Published {
                ops: published,
                coalesced: false,
                head,
                elapsed_ms,
            }),
            None => Ok(PublishOutcome::Idle),
        }
    }

    /// Persist the drain's progress: the server is now at `head`, and only the
    /// operations still in `chain` remain queued. Written after every CAS so a
    /// crash mid-drain resumes instead of replaying.
    fn commit_progress(
        &self,
        chain: &PendingPublishMarker,
        head: &ContentId,
        first_queued: Option<&str>,
    ) -> Result<(), PublishError> {
        write_server_heads(self.dir, &ServerHeadsMarker::new(vec![*head], None))?;
        if chain.is_empty() {
            remove_marker(self.dir, PENDING_PUBLISH_FILE)?;
            self.clear_folded_registration(first_queued)?;
        } else {
            let mut remaining = chain.clone();
            remaining.base_heads = vec![head.to_string()];
            write_pending_publish(self.dir, &remaining)?;
        }
        Ok(())
    }

    /// Publish a whole burst as one rewritten operation carrying the tip's
    /// view.
    ///
    /// DISABLED BY DEFAULT — see [`coalescing_enabled`]. The rewrite's parents
    /// are the chain's base, so whenever the burst holds more than one
    /// operation the published operation is a genuine *sibling* of the local
    /// tip rather than a descendant. The working copy stays pinned to the
    /// local operation, so any reader that resolves heads from the server (a
    /// `sync`-mode command, another clone) loads a sibling of the working
    /// copy's operation and jj refuses to proceed. Publishing a merge of the
    /// two is now permitted by the backend's parent rule, but it does not fix
    /// this: the rewrite's id is not the local tip's, so the working copy is
    /// still pinned to an operation no reader resolves to. Before this can be
    /// turned on, the publisher must also move the working copy's operation
    /// pointer to the rewrite (or record it as a successor so jj can integrate
    /// it).
    async fn drain_coalesced(
        &self,
        mut chain: PendingPublishMarker,
        start: Instant,
    ) -> Result<PublishOutcome, PublishError> {
        let stats = vex_client_stats();
        let mut base = chain.base_ids()?;
        for attempt in 0..MAX_PUBLISH_ATTEMPTS {
            let tip = self.chain_tip(&chain)?;
            let (head, view, coalesced) = self.head_to_publish(&tip, &base).await?;
            let outcome = self.transport.commit_op_heads(&base, &head, &view).await?;
            if outcome.ok || outcome.current_heads == [head] {
                return self.record_published(&chain, &tip.id, &head, coalesced, start);
            }
            stats.publish_cas_conflicts.fetch_add(1, Ordering::Relaxed);
            let server_heads = self.transport.get_op_heads().await?;
            if server_heads == [head] {
                return self.record_published(&chain, &tip.id, &head, coalesced, start);
            }
            match self.refold(&mut chain, &server_heads, &tip)? {
                Some(new_base) if attempt + 1 < MAX_PUBLISH_ATTEMPTS => {
                    stats.publish_folds.fetch_add(1, Ordering::Relaxed);
                    base = new_base;
                }
                _ => {
                    self.record_server_heads_only(&server_heads)?;
                    tracing::warn!(
                        queued = chain.ops.len(),
                        server_heads = ?server_heads.iter().map(ToString::to_string).collect::<Vec<_>>(),
                        "server op head moved to unmerged state; keeping the pending chain queued"
                    );
                    return Ok(PublishOutcome::ServerHeadMoved { server_heads });
                }
            }
        }
        let server_heads = self.transport.get_op_heads().await?;
        self.record_server_heads_only(&server_heads)?;
        Ok(PublishOutcome::ServerHeadMoved { server_heads })
    }

    /// Every object the queued operations introduced, re-uploaded
    /// unconditionally. `put_objects` is content-addressed create-if-missing,
    /// so replay after a crash — or after server-side GC collected an object
    /// that was uploaded but never referenced by a published head — costs
    /// bandwidth and nothing else.
    async fn upload_chain_objects(&self, chain: &PendingPublishMarker) -> Result<(), PublishError> {
        let mut objects = Vec::new();
        for entry in &chain.ops {
            for object in &entry.objects {
                let kind = kind_from_str(&object.kind).ok_or_else(|| {
                    PublishError::Corrupt(format!("unknown object kind {}", object.kind))
                })?;
                let id = ContentId::from_hex(&object.id).map_err(|err| {
                    PublishError::Corrupt(format!("invalid pending object id: {err}"))
                })?;
                let Some(data) = self.transport.read_object(kind, &id) else {
                    return Err(PublishError::MissingObject {
                        kind: object.kind.clone(),
                        id: object.id.clone(),
                    });
                };
                objects.push((kind, id, data));
            }
        }
        // Upload in bounded batches rather than one call.
        //
        // A backlog is exactly when this matters: the whole queue's objects in a
        // single request outgrows the edge's request timeout, the call is killed,
        // and no progress is recorded — so a queue that grew large enough could
        // never drain again, which is the terminal state of a queue that grows
        // faster than it drains. Each batch is separately idempotent
        // (content-addressed create-if-missing), so a failure part-way through
        // still leaves everything it already uploaded on the server and the next
        // attempt resumes from there.
        for batch in objects.chunks(UPLOAD_BATCH_OBJECTS) {
            self.transport.put_objects(batch.to_vec()).await?;
        }
        Ok(())
    }

    fn chain_tip(&self, chain: &PendingPublishMarker) -> Result<ChainTip, PublishError> {
        let entry = chain
            .ops
            .last()
            .ok_or_else(|| PublishError::Corrupt("pending chain is empty".to_string()))?;
        let id = ContentId::from_hex(&entry.op)
            .map_err(|err| PublishError::Corrupt(format!("invalid pending op id: {err}")))?;
        Ok(ChainTip { id })
    }

    /// The operation to CAS: the chain tip itself when its parents already
    /// equal the base (the single-operation case, which keeps local and server
    /// operation ids identical), otherwise a rewrite of the tip parented on
    /// the base.
    async fn head_to_publish(
        &self,
        tip: &ChainTip,
        base: &[ContentId],
    ) -> Result<(ContentId, ContentId, bool), PublishError> {
        let operation = self.read_operation(&tip.id)?;
        let view = content_id_from_view_id(&operation.view_id)
            .ok_or_else(|| PublishError::Corrupt("invalid view id length".to_string()))?;
        if id_set(&operation_parents(&operation)) == id_set(base) {
            return Ok((tip.id, view, false));
        }
        // Carry the local tip as an extra parent, not just the CAS base.
        //
        // Rewriting onto the base alone is what made coalescing unsafe: the
        // published operation was then a *sibling* of the operation the working
        // copy is pinned to, and the next command died with an unrecoverable
        // sibling-operation error. Keeping the tip as a parent makes the working
        // copy merely stale, which jj recovers by itself, and the backend
        // accepts it because a published operation's parents need only *cover*
        // the CAS expectation. The view is still the tip's, so the whole burst
        // publishes as one operation.
        let mut parents: Vec<OperationId> = base.iter().map(op_id_from_content_id).collect();
        if !base.contains(&tip.id) {
            parents.push(op_id_from_content_id(&tip.id));
        }
        let rewritten = Operation {
            view_id: operation.view_id.clone(),
            parents,
            metadata: operation.metadata.clone(),
            commit_predecessors: operation.commit_predecessors.clone(),
        };
        let data = operation_to_proto(&rewritten).encode_to_vec();
        let id = sha256_content_id(&data);
        self.transport
            .put_objects(vec![(ObjectKind::Op, id, data)])
            .await?;
        Ok((id, view, true))
    }

    /// The earliest queued operation whose parents *cover* `server_heads`, and
    /// which can therefore be CASed straight onto the moved server head.
    ///
    /// This is how a divergent repository converges. When the server head
    /// moves under a queued chain the read path serves both heads (see
    /// [`crate::vex_freshness::known_divergence`]), jj's own op-head
    /// resolution merges their views, and the resulting merge operation —
    /// whose parents are exactly the two heads — is queued like any other
    /// local operation. Publishing it advances the server from its current
    /// head to a descendant of *both* sides.
    ///
    /// Nothing queued ahead of the merge is dropped: those operations are the
    /// merge's own local parent and that parent's ancestors, so publishing the
    /// merge makes every one of them an ancestor of the new server head. Their
    /// objects have already been uploaded by
    /// [`Self::upload_chain_objects`], and the backend's presence proof
    /// re-checks every parent operation on each publish, so a merge whose
    /// ancestry did not reach the server is rejected rather than accepted with
    /// a hole.
    ///
    /// Operations *after* the merge stay queued and publish normally on the
    /// following iterations, because their parent is then the server head.
    fn merge_forward_target(
        &self,
        chain: &PendingPublishMarker,
        server_heads: &[ContentId],
    ) -> Result<Option<(usize, MergeTarget)>, PublishError> {
        // An empty expectation is the initial-publication contract, not a
        // merge: the backend deliberately keeps it strict, so there is nothing
        // to fold onto here.
        if server_heads.is_empty() {
            return Ok(None);
        }
        let wanted = id_set(server_heads);
        for (index, entry) in chain.ops.iter().enumerate() {
            let id = ContentId::from_hex(&entry.op)
                .map_err(|err| PublishError::Corrupt(format!("invalid pending op id: {err}")))?;
            let operation = self.read_operation(&id)?;
            let parents = id_set(&operation_parents(&operation));
            if !wanted.is_subset(&parents) {
                continue;
            }
            let view = content_id_from_view_id(&operation.view_id)
                .ok_or_else(|| PublishError::Corrupt("invalid view id length".to_string()))?;
            return Ok(Some((index, MergeTarget { id, view })));
        }
        Ok(None)
    }

    /// A CAS conflict this repo can resolve without merging: the server head
    /// is one of our own queued operations (a previous drain got further than
    /// its marker update), or it is exactly what the chain is parented on and
    /// only the recorded base was stale. Anything else means the server holds
    /// work this repo has not seen, which only a jj op merge can reconcile.
    ///
    /// Only the (default-off) coalescing drain uses this; the sequential drain
    /// recovers the same two shapes inline, plus the merge convergence in
    /// [`Self::merge_forward_target`], which deliberately does not apply to a
    /// rewrite-coalescing publish: a coalesced head is a sibling of the local
    /// tip, so folding a merge onto it would compound the divergence rather
    /// than resolve it.
    fn refold(
        &self,
        chain: &mut PendingPublishMarker,
        server_heads: &[ContentId],
        tip: &ChainTip,
    ) -> Result<Option<Vec<ContentId>>, PublishError> {
        let [head] = server_heads else {
            return Ok(None);
        };
        if let Some(index) = chain
            .ops
            .iter()
            .position(|entry| entry.op == head.to_string())
        {
            chain.ops.drain(..=index);
            if chain.is_empty() {
                return Ok(None);
            }
            chain.base_heads = vec![head.to_string()];
            write_pending_publish(self.dir, chain)?;
            return Ok(Some(vec![*head]));
        }
        let parents = operation_parents(&self.read_operation(&tip.id)?);
        if parents == vec![*head] {
            chain.base_heads = vec![head.to_string()];
            write_pending_publish(self.dir, chain)?;
            return Ok(Some(vec![*head]));
        }
        Ok(None)
    }

    fn read_operation(&self, id: &ContentId) -> Result<Operation, PublishError> {
        let data = self
            .transport
            .read_object(ObjectKind::Op, id)
            .ok_or_else(|| PublishError::MissingObject {
                kind: kind_to_str(ObjectKind::Op).to_string(),
                id: id.to_string(),
            })?;
        let proto = crate::protos::simple_op_store::Operation::decode(&*data)
            .map_err(|err| PublishError::Corrupt(format!("undecodable operation {id}: {err}")))?;
        operation_from_proto(proto)
            .map_err(|err| PublishError::Corrupt(format!("invalid operation {id}: {err}")))
    }

    fn record_published(
        &self,
        chain: &PendingPublishMarker,
        local_tip: &ContentId,
        head: &ContentId,
        coalesced: bool,
        start: Instant,
    ) -> Result<PublishOutcome, PublishError> {
        let published_local_head = (head != local_tip).then_some(*local_tip);
        write_server_heads(
            self.dir,
            &ServerHeadsMarker::new(vec![*head], published_local_head),
        )?;
        remove_marker(self.dir, PENDING_PUBLISH_FILE)?;
        self.clear_folded_registration(chain.ops.first().map(|entry| entry.op.as_str()))?;
        let elapsed_ms = start.elapsed().as_millis() as u64;
        let stats = vex_client_stats();
        stats.pending_ops.store(0, Ordering::Relaxed);
        stats.publish_lag_ms.store(elapsed_ms, Ordering::Relaxed);
        if coalesced {
            stats
                .coalesced_ops
                .fetch_add(chain.ops.len().saturating_sub(1) as u64, Ordering::Relaxed);
        }
        tracing::debug!(
            ops = chain.ops.len(),
            coalesced,
            head = %head,
            elapsed_ms,
            "published pending operation chain"
        );
        Ok(PublishOutcome::Published {
            ops: chain.ops.len(),
            coalesced,
            head: *head,
            elapsed_ms,
        })
    }

    /// A clone's deferred registration (roadmap/076) enters the queue as its
    /// first entry; once that entry is published the marker must not outlive
    /// it, or the read path would keep serving heads on its behalf. A marker
    /// naming any other operation belongs to a clone still in flight and is
    /// left alone.
    fn clear_folded_registration(&self, first_queued: Option<&str>) -> Result<(), PublishError> {
        let path = self
            .dir
            .join(crate::vex_op_heads_store::PENDING_REGISTRATION_FILE);
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(());
        };
        let folded = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|marker| marker.get("pending_op_id")?.as_str().map(str::to_owned))
            .is_some_and(|pending| first_queued == Some(pending.as_str()));
        if folded {
            remove_marker(
                self.dir,
                crate::vex_op_heads_store::PENDING_REGISTRATION_FILE,
            )?;
        }
        Ok(())
    }

    /// Record where the server actually is without claiming anything about the
    /// local chain, so the next repo load serves both heads and lets jj merge.
    fn record_server_heads_only(&self, server_heads: &[ContentId]) -> Result<(), PublishError> {
        write_server_heads(
            self.dir,
            &ServerHeadsMarker::new(server_heads.to_vec(), None),
        )?;
        Ok(())
    }
}

struct ChainTip {
    id: ContentId,
}

/// A queued operation that can be published straight onto a moved server head.
struct MergeTarget {
    id: ContentId,
    view: ContentId,
}

fn operation_parents(operation: &Operation) -> Vec<ContentId> {
    operation
        .parents
        .iter()
        .filter_map(content_id_from_op_id)
        .filter(|id| !is_root_content_id(id))
        .collect()
}

/// [`PublishTransport`] backed by a live [`VexClient`].
///
/// Every call carries a hard deadline. The publisher runs while the
/// working-copy lock is held, and a backend that accepts a connection but
/// never answers would otherwise freeze every other session on the machine —
/// observed in production against the synchronous publish path. Missing the
/// deadline is always recoverable: objects are content-addressed
/// create-if-missing, and an unanswered CAS leaves the queue intact for the
/// next drain, which the server's replay check resolves.
pub struct VexClientTransport<'a> {
    client: &'a VexClient,
    deadline: Duration,
}

impl<'a> VexClientTransport<'a> {
    pub fn new(client: &'a VexClient) -> Self {
        Self {
            client,
            deadline: publish_rpc_deadline(),
        }
    }

    /// Override the per-RPC deadline (tests, and callers with their own
    /// budget).
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = deadline;
        self
    }
}

/// Per-RPC deadline for the publisher, from `VEX_PUBLISH_RPC_TIMEOUT_MS`.
/// Generous by design — a large first publish legitimately moves real bytes —
/// but finite.
pub fn publish_rpc_deadline() -> Duration {
    let millis = std::env::var("VEX_PUBLISH_RPC_TIMEOUT_MS")
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .unwrap_or(DEFAULT_PUBLISH_RPC_TIMEOUT_MS);
    Duration::from_millis(millis)
}

/// Default per-RPC publisher deadline.
pub const DEFAULT_PUBLISH_RPC_TIMEOUT_MS: u64 = 30_000;

/// Objects per `put_objects` batch. Matches the snapshot flush caps in
/// `vex.rs`; each batch is one deadline-bounded request.
const PUBLISH_BATCH_OBJECTS: usize = 256;
const PUBLISH_BATCH_BYTES: usize = 32 * 1024 * 1024;

#[async_trait]
impl PublishTransport for VexClientTransport<'_> {
    fn read_object(&self, kind: ObjectKind, id: &ContentId) -> Option<Vec<u8>> {
        self.client.read_cached_object(kind, id)
    }

    async fn put_objects(
        &self,
        objects: Vec<(ObjectKind, ContentId, Vec<u8>)>,
    ) -> Result<(), PublishError> {
        if objects.is_empty() {
            return Ok(());
        }
        let mut batch = Vec::new();
        let mut batch_bytes = 0;
        for object in objects {
            batch_bytes += object.2.len();
            batch.push(object);
            if batch.len() >= PUBLISH_BATCH_OBJECTS || batch_bytes >= PUBLISH_BATCH_BYTES {
                self.put_batch(std::mem::take(&mut batch))?;
                batch_bytes = 0;
            }
        }
        self.put_batch(batch)
    }

    async fn commit_op_heads(
        &self,
        expected: &[ContentId],
        new_head: &ContentId,
        new_view: &ContentId,
    ) -> Result<CasOutcome, PublishError> {
        let response = self
            .client
            .commit_op_heads_within(expected, new_head, new_view, self.deadline)?
            .ok_or(PublishError::Deadline {
                rpc: "CommitOperation",
                budget: self.deadline,
            })?;
        let current_heads = response
            .current_op_head_ids
            .iter()
            .map(|id| {
                ContentId::from_hex(id)
                    .map_err(|err| PublishError::Corrupt(format!("invalid op head: {err}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CasOutcome {
            ok: response.ok,
            current_heads,
            error_message: response.error_message,
        })
    }

    async fn get_op_heads(&self) -> Result<Vec<ContentId>, PublishError> {
        self.client
            .get_op_heads_within(self.deadline)?
            .ok_or(PublishError::Deadline {
                rpc: "GetOpHeads",
                budget: self.deadline,
            })
    }
}

impl VexClientTransport<'_> {
    fn put_batch(&self, batch: Vec<(ObjectKind, ContentId, Vec<u8>)>) -> Result<(), PublishError> {
        if self.client.put_objects_within(batch, self.deadline)? {
            Ok(())
        } else {
            Err(PublishError::Deadline {
                rpc: "PutObjects",
                budget: self.deadline,
            })
        }
    }
}

/// Drain the queue at `dir` using a client built from the repo it belongs to.
/// Returns [`PublishOutcome::Idle`] without constructing a client when nothing
/// is queued, so this is cheap to call on every command.
pub fn ensure_published_at(dir: &Path) -> Result<PublishOutcome, PublishError> {
    match read_pending_publish(dir) {
        Ok(None) => return Ok(PublishOutcome::Idle),
        Ok(Some(chain)) if chain.is_empty() => return Ok(PublishOutcome::Idle),
        Ok(Some(_)) => {}
        Err(err) => return Err(err.into()),
    }
    let client =
        VexClient::from_store_path(dir).map_err(|err| PublishError::Transport(err.to_string()))?;
    let transport = VexClientTransport::new(&client);
    pollster::block_on(VexPublisher::new(dir, &transport).drain())
}

/// The `op_heads` store directory of the repo containing `cwd`, following the
/// `.jj/repo` file indirection used by secondary workspaces.
pub fn find_op_heads_dir(cwd: &Path) -> Option<PathBuf> {
    for directory in cwd.ancestors() {
        let jj_dir = directory.join(".jj");
        let repo_path = jj_dir.join("repo");
        let op_heads = repo_path.join("op_heads");
        if op_heads.is_dir() {
            return Some(op_heads);
        }
        if let Ok(reference) = std::fs::read_to_string(&repo_path) {
            let shared = jj_dir.join(reference.trim()).join("op_heads");
            if shared.is_dir() {
                return Some(shared);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use futures::executor::block_on;

    use super::*;
    use crate::op_store::OperationMetadata;

    fn id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; ID_LENGTH])
    }

    fn operation(view: u8, parents: &[ContentId]) -> Operation {
        let timestamp = crate::backend::Timestamp {
            timestamp: crate::backend::MillisSinceEpoch(0),
            tz_offset: 0,
        };
        Operation {
            view_id: ViewId::new(id(view).as_bytes().to_vec()),
            parents: parents.iter().map(op_id_from_content_id).collect(),
            metadata: OperationMetadata {
                time: crate::op_store::TimestampRange {
                    start: timestamp,
                    end: timestamp,
                },
                description: "test".to_string(),
                hostname: String::new(),
                username: String::new(),
                is_snapshot: false,
                workspace_name: None,
                attributes: std::collections::BTreeMap::new(),
            },
            commit_predecessors: Some(std::collections::BTreeMap::new()),
        }
    }

    #[derive(Default)]
    struct FakeState {
        objects: HashMap<(ObjectKind, ContentId), Vec<u8>>,
        heads: Vec<ContentId>,
        put_calls: usize,
        cas_calls: usize,
        get_head_calls: usize,
        /// Head sets the next CAS attempts observe instead of `heads`, popped
        /// in order — used to simulate a head that moved under us.
        cas_conflicts: Vec<Vec<ContentId>>,
        /// 1-based CAS call number that simulates a backend accepting the
        /// call and never answering within its deadline.
        deadline_on_call: Option<usize>,
    }

    #[derive(Default)]
    struct FakeTransport {
        state: Mutex<FakeState>,
    }

    impl FakeTransport {
        /// Publish an operation object the way a local snapshot would: bytes in
        /// the local cache only.
        fn stage_operation(&self, operation: &Operation) -> ContentId {
            let data = operation_to_proto(operation).encode_to_vec();
            let id = sha256_content_id(&data);
            self.state
                .lock()
                .unwrap()
                .objects
                .insert((ObjectKind::Op, id), data);
            id
        }

        fn set_heads(&self, heads: Vec<ContentId>) {
            self.state.lock().unwrap().heads = heads;
        }

        fn heads(&self) -> Vec<ContentId> {
            self.state.lock().unwrap().heads.clone()
        }

        fn cas_calls(&self) -> usize {
            self.state.lock().unwrap().cas_calls
        }

        fn uploaded(&self) -> usize {
            self.state.lock().unwrap().put_calls
        }
    }

    #[async_trait]
    impl PublishTransport for FakeTransport {
        fn read_object(&self, kind: ObjectKind, id: &ContentId) -> Option<Vec<u8>> {
            self.state
                .lock()
                .unwrap()
                .objects
                .get(&(kind, *id))
                .cloned()
        }

        async fn put_objects(
            &self,
            objects: Vec<(ObjectKind, ContentId, Vec<u8>)>,
        ) -> Result<(), PublishError> {
            let mut state = self.state.lock().unwrap();
            state.put_calls += objects.len();
            for (kind, id, data) in objects {
                state.objects.insert((kind, id), data);
            }
            Ok(())
        }

        async fn commit_op_heads(
            &self,
            expected: &[ContentId],
            new_head: &ContentId,
            _new_view: &ContentId,
        ) -> Result<CasOutcome, PublishError> {
            let mut state = self.state.lock().unwrap();
            state.cas_calls += 1;
            if state.deadline_on_call == Some(state.cas_calls) {
                return Err(PublishError::Deadline {
                    rpc: "CommitOperation",
                    budget: Duration::from_secs(30),
                });
            }
            // The backend's parent contract, mirrored so the client tests are
            // held to the same rule the service enforces
            // (`ensure_operation_matches_expected_heads`): the new
            // operation's parents must *cover* the CAS expectation. Extra
            // parents — which is what a merge operation has — are allowed;
            // dropping an expected head is not. A violation is a validation
            // error, not a CAS conflict, exactly as in `commit_operation`.
            if let Some(bytes) = state.objects.get(&(ObjectKind::Op, *new_head)).cloned()
                && let Ok(proto) = crate::protos::simple_op_store::Operation::decode(&*bytes)
                && let Ok(operation) = operation_from_proto(proto)
            {
                let parents = id_set(&operation_parents(&operation));
                let expected_set = id_set(expected);
                let covered = if expected_set.is_empty() {
                    parents.is_empty()
                } else {
                    expected_set.is_subset(&parents)
                };
                if !covered {
                    return Err(PublishError::Transport(format!(
                        "operation parent set {parents:?} does not cover expected operation heads \
                         {expected_set:?}"
                    )));
                }
            }
            if let Some(conflict) = state.cas_conflicts.pop() {
                state.heads = conflict.clone();
                return Ok(CasOutcome {
                    ok: false,
                    current_heads: conflict,
                    error_message: "cas conflict".to_string(),
                });
            }
            if id_set(&state.heads) != id_set(expected) {
                let current = state.heads.clone();
                return Ok(CasOutcome {
                    ok: false,
                    current_heads: current,
                    error_message: "cas conflict".to_string(),
                });
            }
            state.heads = vec![*new_head];
            Ok(CasOutcome {
                ok: true,
                current_heads: vec![*new_head],
                error_message: String::new(),
            })
        }

        async fn get_op_heads(&self) -> Result<Vec<ContentId>, PublishError> {
            let mut state = self.state.lock().unwrap();
            state.get_head_calls += 1;
            Ok(state.heads.clone())
        }
    }

    /// Builds a chain of `count` operations on `base` in the fake's object
    /// store and writes the matching marker. Returns the operation ids.
    fn queue_chain(
        dir: &Path,
        fake: &FakeTransport,
        base: ContentId,
        count: usize,
    ) -> Vec<ContentId> {
        let mut marker = PendingPublishMarker::new(&[base]);
        let mut parent = base;
        let mut ids = Vec::new();
        for index in 0..count {
            let op = operation(100 + index as u8, &[parent]);
            let op_id = fake.stage_operation(&op);
            marker.push(&op_id, &[(ObjectKind::Op, op_id)]);
            ids.push(op_id);
            parent = op_id;
        }
        write_pending_publish(dir, &marker).unwrap();
        ids
    }

    #[test]
    fn durability_parses_modes_and_env_override() {
        assert_eq!(VexDurability::parse("sync"), Some(VexDurability::Sync));
        assert_eq!(
            VexDurability::parse("Local_First"),
            Some(VexDurability::LocalFirst)
        );
        assert_eq!(
            VexDurability::parse("flush-on-exit"),
            Some(VexDurability::FlushOnExit)
        );
        assert_eq!(VexDurability::parse("nonsense"), None);
        // Local-first is the default: a command returns once the operation is
        // durable locally, and the sync barriers still confirm the server.
        assert_eq!(VexDurability::default(), VexDurability::LocalFirst);
        assert!(VexDurability::default().is_serialized_default());
        assert!(!VexDurability::Sync.defers_publish());
        assert!(VexDurability::LocalFirst.defers_publish());
        assert!(VexDurability::FlushOnExit.blocks_on_exit());
        assert!(!VexDurability::LocalFirst.blocks_on_exit());
    }

    /// A disposable CI machine can be destroyed the instant its job ends, so
    /// the default deferred mode must not let it exit with an undrained queue.
    #[test]
    fn ci_environments_default_to_flushing_before_exit() {
        assert!(!VexDurability::is_ephemeral_marker(None));
        assert!(!VexDurability::is_ephemeral_marker(Some("")));
        assert!(!VexDurability::is_ephemeral_marker(Some("0")));
        assert!(!VexDurability::is_ephemeral_marker(Some("false")));
        assert!(!VexDurability::is_ephemeral_marker(Some(" FALSE ")));
        assert!(VexDurability::is_ephemeral_marker(Some("true")));
        assert!(VexDurability::is_ephemeral_marker(Some("1")));

        // A developer machine keeps the non-blocking default; an ephemeral
        // runner must drain before the process exits.
        assert_eq!(
            VexDurability::for_environment(VexDurability::LocalFirst, false),
            VexDurability::LocalFirst
        );
        assert_eq!(
            VexDurability::for_environment(VexDurability::LocalFirst, true),
            VexDurability::FlushOnExit
        );
        // Explicit modes are left exactly as configured.
        assert_eq!(
            VexDurability::for_environment(VexDurability::Sync, true),
            VexDurability::Sync
        );
        assert_eq!(
            VexDurability::for_environment(VexDurability::FlushOnExit, false),
            VexDurability::FlushOnExit
        );
    }

    #[test]
    fn markers_round_trip_and_reject_unknown_versions() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();

        let server = ServerHeadsMarker::new(vec![id(1)], Some(id(2)));
        write_server_heads(dir, &server).unwrap();
        assert_eq!(read_server_heads(dir).unwrap().unwrap(), server);
        assert!(server.stands_for(&[id(2)]));
        assert!(!server.stands_for(&[id(1)]));

        let mut chain = PendingPublishMarker::new(&[id(1)]);
        chain.push(&id(3), &[(ObjectKind::Op, id(3))]);
        write_pending_publish(dir, &chain).unwrap();
        assert_eq!(read_pending_publish(dir).unwrap().unwrap(), chain);
        assert!(chain.contains(&id(3)));

        std::fs::write(dir.join(PENDING_PUBLISH_FILE), r#"{"v":99,"ops":[]}"#).unwrap();
        assert!(matches!(
            read_pending_publish(dir),
            Err(MarkerError::UnsupportedVersion { .. })
        ));
        std::fs::write(dir.join(PENDING_PUBLISH_FILE), "not json").unwrap();
        assert!(matches!(
            read_pending_publish(dir),
            Err(MarkerError::Corrupt { .. })
        ));
    }

    #[test]
    fn local_heads_round_trip() {
        let temp = tempfile::tempdir().unwrap();
        assert!(read_local_heads(temp.path()).unwrap().is_none());
        write_local_heads(temp.path(), &[id(7), id(8)]).unwrap();
        assert_eq!(
            read_local_heads(temp.path()).unwrap().unwrap(),
            vec![op_id_from_content_id(&id(7)), op_id_from_content_id(&id(8))]
        );
    }

    #[test]
    fn drain_without_a_queue_is_idle() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let publisher = VexPublisher::new(temp.path(), &fake);
        assert_eq!(block_on(publisher.drain()).unwrap(), PublishOutcome::Idle);
        assert_eq!(fake.cas_calls(), 0);
    }

    #[test]
    fn single_pending_operation_publishes_under_its_own_id() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        let ids = queue_chain(temp.path(), &fake, id(1), 1);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let outcome = block_on(publisher.drain()).unwrap();

        assert_eq!(
            outcome,
            PublishOutcome::Published {
                ops: 1,
                coalesced: false,
                head: ids[0],
                elapsed_ms: outcome.elapsed_ms(),
            }
        );
        assert_eq!(fake.heads(), vec![ids[0]]);
        assert_eq!(fake.cas_calls(), 1);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
        let server = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(server.heads, vec![ids[0].to_string()]);
        assert_eq!(server.published_local_head, None);
    }

    #[test]
    fn ten_local_operations_publish_under_their_own_ids() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        let ids = queue_chain(temp.path(), &fake, id(1), 10);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let outcome = block_on(publisher.drain()).unwrap();

        assert_eq!(outcome.published_ops(), 10);
        assert!(matches!(
            outcome,
            PublishOutcome::Published {
                coalesced: false,
                ..
            }
        ));
        assert_eq!(
            fake.cas_calls(),
            10,
            "one CAS per operation is the price of never rewriting an id"
        );
        assert_eq!(
            fake.heads(),
            vec![ids[9]],
            "the server head must be the local tip itself"
        );
        let server = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(server.published_local_head, None);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    /// The reproducer from the 088 AFTER benchmark
    /// (`/tmp/rm088-bench/after/RESULTS.md`): a burst of local operations, a
    /// drain that dies partway through against a slow backend, then a later
    /// drain. The published head must end up being the working copy's own
    /// operation — the coalescing rewrite made it a *sibling* instead, which
    /// jj cannot load and cannot repair.
    #[test]
    fn a_drain_interrupted_midway_still_leaves_the_server_on_the_local_tip() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        let ids = queue_chain(temp.path(), &fake, id(1), 3);
        // The second CAS never answers, exactly as the degraded backend did.
        fake.state.lock().unwrap().deadline_on_call = Some(2);

        let publisher = VexPublisher::new(temp.path(), &fake);
        assert!(matches!(
            block_on(publisher.drain()),
            Err(PublishError::Deadline { .. })
        ));

        // Progress is durable and the rest stays queued.
        assert_eq!(fake.heads(), vec![ids[0]]);
        let queued = read_pending_publish(temp.path()).unwrap().unwrap();
        assert_eq!(queued.ops.len(), 2);
        assert_eq!(
            read_server_heads(temp.path())
                .unwrap()
                .unwrap()
                .published_local_head,
            None,
            "no rewrite may be recorded, at any point in the drain"
        );

        // The backend recovers; the rest of the burst publishes.
        fake.state.lock().unwrap().deadline_on_call = None;
        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();
        assert_eq!(outcome.published_ops(), 2);

        // The invariant the sibling bug violated: the server head IS the
        // operation the working copy is pinned to, so loading the repo at the
        // server head can never produce a sibling of the working copy's
        // operation.
        let local_tip = ids[2];
        assert_eq!(fake.heads(), vec![local_tip]);
        let server = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(server.heads, vec![local_tip.to_string()]);
        assert_eq!(server.published_local_head, None);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    /// Coalescing is off by default precisely because it publishes an
    /// operation that is a sibling of the local tip. Pinned here so the hazard
    /// cannot be re-enabled by accident.
    #[test]
    fn coalescing_publishes_one_cas_and_keeps_the_tip_in_its_lineage() {
        assert!(
            !coalescing_enabled(),
            "coalescing stays opt-in until the sequential drain's convergence \
             properties are proven for it"
        );

        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        let ids = queue_chain(temp.path(), &fake, id(1), 3);

        let publisher = VexPublisher::new(temp.path(), &fake).with_coalescing(true);
        let outcome = block_on(publisher.drain()).unwrap();

        // The point of coalescing: a burst of three operations costs one CAS.
        assert_eq!(outcome.published_ops(), 3);
        assert_eq!(fake.cas_calls(), 1);

        let published = fake.heads();
        assert_ne!(published[0], ids[2], "the coalesced head is a rewrite");
        let rewritten = fake.read_object(ObjectKind::Op, &published[0]).unwrap();
        let rewritten = operation_from_proto(
            crate::protos::simple_op_store::Operation::decode(&*rewritten).unwrap(),
        )
        .unwrap();
        let parents = operation_parents(&rewritten);
        // The CAS base has to be covered, or the backend rejects the publish.
        assert!(
            parents.contains(&id(1)),
            "the rewrite must cover the CAS base"
        );
        // And the local tip has to remain an ancestor. Without this the
        // published operation is a *sibling* of the operation the working copy
        // is pinned to, and the next command fails with an unrecoverable
        // sibling-operation error. With it the working copy is merely stale,
        // which jj recovers by itself.
        assert!(
            parents.contains(&ids[2]),
            "the rewrite must keep the local tip as a parent so the working copy \
             is left stale rather than divergent"
        );
    }

    #[test]
    fn a_stale_recorded_base_no_longer_costs_a_rejected_cas() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        // The chain is parented on id(2) but its marker recorded id(1).
        let mut marker = PendingPublishMarker::new(&[id(1)]);
        let op = operation(50, &[id(2)]);
        let op_id = fake.stage_operation(&op);
        marker.push(&op_id, &[(ObjectKind::Op, op_id)]);
        write_pending_publish(temp.path(), &marker).unwrap();
        fake.set_heads(vec![id(2)]);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let outcome = block_on(publisher.drain()).unwrap();

        assert_eq!(outcome.published_ops(), 1);
        assert_eq!(fake.heads(), vec![op_id]);
        assert_eq!(
            fake.cas_calls(),
            1,
            "sequential publication CASes against the operation's real parents, so a stale \
             recorded base is simply unused"
        );
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    #[test]
    fn partially_published_chain_resumes_from_the_server_head() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 2);
        // A previous drain published the first operation but died before
        // clearing the marker.
        fake.set_heads(vec![ids[0]]);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let outcome = block_on(publisher.drain()).unwrap();

        // Both entries leave the queue: the first was recognized as already
        // published, the second was CASed now.
        assert_eq!(outcome.published_ops(), 2);
        assert_eq!(fake.heads(), vec![ids[1]]);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    #[test]
    fn unmergeable_server_head_keeps_the_chain_queued() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 3);
        // Another client published unrelated work.
        fake.set_heads(vec![id(9)]);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let outcome = block_on(publisher.drain()).unwrap();

        assert_eq!(
            outcome,
            PublishOutcome::ServerHeadMoved {
                server_heads: vec![id(9)]
            }
        );
        let chain = read_pending_publish(temp.path()).unwrap().unwrap();
        assert_eq!(chain.ops.len(), 3, "no queued operation may be dropped");
        assert_eq!(chain.ops.last().unwrap().op, ids[2].to_string());
        assert_eq!(
            read_server_heads(temp.path()).unwrap().unwrap().heads,
            vec![id(9).to_string()],
            "the moved server head is recorded so the next load merges it"
        );
        // Every object still reached the server: only the head pointer is
        // unresolved.
        assert!(fake.uploaded() >= 3);
    }

    /// Queue the merge jj produces when the read path serves both the local
    /// and the moved server head, exactly as
    /// `VexOpHeadsStore::record_pending_operation` would: the chain is rebased
    /// onto the recorded server heads and the merge is appended.
    fn queue_merge(
        dir: &Path,
        fake: &FakeTransport,
        server_heads: &[ContentId],
        local_tip: ContentId,
    ) -> ContentId {
        let mut chain = read_pending_publish(dir).unwrap().unwrap();
        let mut parents = vec![local_tip];
        parents.extend_from_slice(server_heads);
        let merge = operation(200, &parents);
        let merge_id = fake.stage_operation(&merge);
        chain.base_heads = server_heads.iter().map(ToString::to_string).collect();
        chain.push(&merge_id, &[(ObjectKind::Op, merge_id)]);
        write_pending_publish(dir, &chain).unwrap();
        merge_id
    }

    fn published_operation(fake: &FakeTransport, id: &ContentId) -> Operation {
        let bytes = fake.read_object(ObjectKind::Op, id).unwrap();
        operation_from_proto(crate::protos::simple_op_store::Operation::decode(&*bytes).unwrap())
            .unwrap()
    }

    /// The roadmap/088 divergence dead end: another session moved the server
    /// head while a chain was queued. The first drain cannot advance, but it
    /// records where the server is; jj then merges the two heads and the merge
    /// enters the queue; the next drain publishes it. Nothing is dropped and
    /// the final server head descends from both sides.
    #[test]
    fn a_moved_server_head_converges_once_jj_merges_it() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 3);
        let remote = fake.stage_operation(&operation(90, &[id(1)]));
        fake.set_heads(vec![remote]);

        // Phase one: no queued operation can be CASed onto the moved head yet.
        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();
        assert_eq!(
            outcome,
            PublishOutcome::ServerHeadMoved {
                server_heads: vec![remote]
            }
        );
        assert_eq!(
            read_pending_publish(temp.path()).unwrap().unwrap().ops.len(),
            3,
            "no queued operation may be dropped while divergence is unresolved"
        );
        assert_eq!(
            read_server_heads(temp.path()).unwrap().unwrap().heads,
            vec![remote.to_string()],
            "the moved head is recorded so the next repo load serves both heads"
        );

        // Phase two: jj merged the two heads; the merge is queued like any
        // other local operation.
        let merge = queue_merge(temp.path(), &fake, &[remote], ids[2]);
        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();

        assert_eq!(outcome.published_ops(), 4, "the whole queue is drained");
        assert!(matches!(
            outcome,
            PublishOutcome::Published {
                coalesced: false,
                ..
            }
        ));
        assert_eq!(
            fake.heads(),
            vec![merge],
            "the server head is the merge itself, under its own id"
        );
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
        let server = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(server.heads, vec![merge.to_string()]);
        assert_eq!(
            server.published_local_head, None,
            "no rewrite: the published head is the local operation itself"
        );

        // The new head descends from both sides, so nothing either session did
        // was lost — the local tip and the remote head are both its parents.
        let parents = operation_parents(&published_operation(&fake, &merge));
        assert!(parents.contains(&ids[2]), "the local tip must be a parent");
        assert!(parents.contains(&remote), "the remote head must be a parent");
        // Every queued operation reached the server as an object, so the whole
        // local op log is readable from the new head.
        for op in &ids {
            assert!(fake.read_object(ObjectKind::Op, op).is_some());
        }
    }

    /// The state a stuck workspace was actually found in: the chain's recorded
    /// base is neither the server head nor an ancestor of it, and the tip's
    /// parents are not the server head either. Before the merge was
    /// publishable this returned `ServerHeadMoved` forever and only deleting
    /// the marker files could recover it.
    #[test]
    fn a_chain_whose_base_is_unrelated_to_the_server_head_still_converges() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 2);
        let remote = fake.stage_operation(&operation(90, &[id(1)]));
        fake.set_heads(vec![remote]);
        let merge = queue_merge(temp.path(), &fake, &[remote], ids[1]);

        // Corrupt the recorded base into something unrelated to both sides:
        // the drain must not depend on it at all.
        let mut chain = read_pending_publish(temp.path()).unwrap().unwrap();
        chain.base_heads = vec![id(200).to_string()];
        write_pending_publish(temp.path(), &chain).unwrap();

        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();

        assert_eq!(outcome.published_ops(), 3);
        assert_eq!(fake.heads(), vec![merge]);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    /// Operations recorded *after* the merge are still published one by one
    /// under their own ids, so the final server head is the working copy's own
    /// operation rather than the merge.
    #[test]
    fn operations_queued_after_the_merge_publish_under_their_own_ids() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 2);
        let remote = fake.stage_operation(&operation(90, &[id(1)]));
        fake.set_heads(vec![remote]);
        let merge = queue_merge(temp.path(), &fake, &[remote], ids[1]);

        let mut chain = read_pending_publish(temp.path()).unwrap().unwrap();
        let follow_up = operation(210, &[merge]);
        let follow_up_id = fake.stage_operation(&follow_up);
        chain.push(&follow_up_id, &[(ObjectKind::Op, follow_up_id)]);
        write_pending_publish(temp.path(), &chain).unwrap();

        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();

        assert_eq!(outcome.published_ops(), 4);
        assert_eq!(
            fake.heads(),
            vec![follow_up_id],
            "the local tip, not the merge, is where the server ends up"
        );
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    /// A drain that dies after publishing the merge resumes from the merge
    /// rather than replaying it or dead-ending again.
    #[test]
    fn a_drain_interrupted_after_the_merge_resumes_correctly() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 2);
        let remote = fake.stage_operation(&operation(90, &[id(1)]));
        fake.set_heads(vec![remote]);
        let merge = queue_merge(temp.path(), &fake, &[remote], ids[1]);
        let mut chain = read_pending_publish(temp.path()).unwrap().unwrap();
        let follow_up_id = fake.stage_operation(&operation(210, &[merge]));
        chain.push(&follow_up_id, &[(ObjectKind::Op, follow_up_id)]);
        write_pending_publish(temp.path(), &chain).unwrap();

        // Call 1 is the doomed CAS of the chain's head, call 2 is the merge;
        // the follow-up's CAS never answers.
        fake.state.lock().unwrap().deadline_on_call = Some(3);
        assert!(matches!(
            block_on(VexPublisher::new(temp.path(), &fake).drain()),
            Err(PublishError::Deadline { .. })
        ));
        assert_eq!(fake.heads(), vec![merge], "the merge is durably published");
        let queued = read_pending_publish(temp.path()).unwrap().unwrap();
        assert_eq!(queued.ops.len(), 1);
        assert_eq!(queued.ops[0].op, follow_up_id.to_string());
        assert_eq!(queued.base_heads, vec![merge.to_string()]);

        fake.state.lock().unwrap().deadline_on_call = None;
        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();
        assert_eq!(outcome.published_ops(), 1);
        assert_eq!(fake.heads(), vec![follow_up_id]);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    /// An operation that merely *claims* to be publishable is not enough: the
    /// backend rejects a parent set that drops the current head, and the
    /// publisher leaves the queue intact rather than losing it.
    #[test]
    fn a_queued_operation_that_drops_the_server_head_is_not_folded_onto_it() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        queue_chain(temp.path(), &fake, id(1), 2);
        let remote = fake.stage_operation(&operation(90, &[id(1)]));
        fake.set_heads(vec![remote]);

        let outcome = block_on(VexPublisher::new(temp.path(), &fake).drain()).unwrap();

        assert_eq!(
            outcome,
            PublishOutcome::ServerHeadMoved {
                server_heads: vec![remote]
            },
            "no queued operation covers the server head, so none may be CASed onto it"
        );
        assert_eq!(fake.heads(), vec![remote], "the server head is untouched");
        assert_eq!(
            read_pending_publish(temp.path()).unwrap().unwrap().ops.len(),
            2
        );
    }

    #[test]
    fn replayed_publish_after_an_ambiguous_cas_is_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let ids = queue_chain(temp.path(), &fake, id(1), 1);
        // The CAS landed server-side but the response never arrived, so the
        // marker survived: the server already holds our operation.
        fake.set_heads(vec![ids[0]]);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let outcome = block_on(publisher.drain()).unwrap();

        assert_eq!(outcome.published_ops(), 1);
        assert_eq!(fake.heads(), vec![ids[0]]);
        assert!(read_pending_publish(temp.path()).unwrap().is_none());
    }

    #[test]
    fn a_cas_that_misses_its_deadline_keeps_the_chain_queued() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        let ids = queue_chain(temp.path(), &fake, id(1), 2);
        fake.state.lock().unwrap().deadline_on_call = Some(1);

        let publisher = VexPublisher::new(temp.path(), &fake);
        let err = block_on(publisher.drain()).unwrap_err();

        assert!(matches!(err, PublishError::Deadline { .. }));
        assert!(
            err.to_string().contains("stay queued"),
            "the deadline error must say the work is not lost: {err}"
        );
        let chain = read_pending_publish(temp.path()).unwrap().unwrap();
        assert_eq!(chain.ops.len(), 2);
        assert_eq!(chain.ops.last().unwrap().op, ids[1].to_string());
        assert_eq!(fake.heads(), vec![id(1)], "the server head is untouched");
    }

    #[test]
    fn the_publisher_deadline_is_finite_and_env_tunable() {
        assert_eq!(
            publish_rpc_deadline(),
            Duration::from_millis(DEFAULT_PUBLISH_RPC_TIMEOUT_MS)
        );
    }

    #[test]
    fn missing_staged_object_fails_loudly() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        let mut marker = PendingPublishMarker::new(&[id(1)]);
        marker.push(&id(5), &[(ObjectKind::Blob, id(6))]);
        write_pending_publish(temp.path(), &marker).unwrap();

        let publisher = VexPublisher::new(temp.path(), &fake);
        assert!(matches!(
            block_on(publisher.drain()),
            Err(PublishError::MissingObject { .. })
        ));
        assert!(
            read_pending_publish(temp.path()).unwrap().is_some(),
            "a failed drain must leave the queue intact"
        );
    }

    #[test]
    fn op_heads_dir_resolves_through_the_repo_indirection() {
        let temp = tempfile::tempdir().unwrap();
        let workspace = temp.path().join("work");
        let op_heads = workspace.join(".jj/repo/op_heads");
        std::fs::create_dir_all(&op_heads).unwrap();
        assert_eq!(find_op_heads_dir(&workspace).unwrap(), op_heads);
        assert_eq!(
            find_op_heads_dir(&workspace.join("src/deep")).unwrap(),
            op_heads
        );

        let secondary = temp.path().join("secondary");
        std::fs::create_dir_all(secondary.join(".jj/shared/op_heads")).unwrap();
        std::fs::write(secondary.join(".jj/repo"), "shared\n").unwrap();
        assert_eq!(
            find_op_heads_dir(&secondary).unwrap(),
            secondary.join(".jj/shared/op_heads")
        );

        assert!(find_op_heads_dir(temp.path()).is_none());
    }

    #[test]
    fn publishing_a_folded_clone_registration_clears_its_marker() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        let registration = operation(60, &[id(1)]);
        let registration_id = fake.stage_operation(&registration);
        let mut marker = PendingPublishMarker::new(&[id(1)]);
        marker.push(&registration_id, &[]);
        let follow_up = operation(61, &[registration_id]);
        let follow_up_id = fake.stage_operation(&follow_up);
        marker.push(&follow_up_id, &[(ObjectKind::Op, follow_up_id)]);
        write_pending_publish(temp.path(), &marker).unwrap();
        let registration_path = temp
            .path()
            .join(crate::vex_op_heads_store::PENDING_REGISTRATION_FILE);
        std::fs::write(
            &registration_path,
            format!(
                r#"{{"workspace_name":"vex-clone-1","pending_op_id":"{registration_id}","server_head_ids":["{}"]}}"#,
                id(1)
            ),
        )
        .unwrap();
        fake.set_heads(vec![id(1)]);

        let publisher = VexPublisher::new(temp.path(), &fake);
        assert_eq!(block_on(publisher.drain()).unwrap().published_ops(), 2);
        assert!(!registration_path.exists());
    }

    #[test]
    fn an_unrelated_registration_marker_survives_a_publish() {
        let temp = tempfile::tempdir().unwrap();
        let fake = FakeTransport::default();
        fake.set_heads(vec![id(1)]);
        queue_chain(temp.path(), &fake, id(1), 1);
        let registration_path = temp
            .path()
            .join(crate::vex_op_heads_store::PENDING_REGISTRATION_FILE);
        std::fs::write(
            &registration_path,
            r#"{"workspace_name":"vex-clone-2","pending_op_id":null,"server_head_ids":[]}"#,
        )
        .unwrap();

        let publisher = VexPublisher::new(temp.path(), &fake);
        assert_eq!(block_on(publisher.drain()).unwrap().published_ops(), 1);
        assert!(registration_path.exists());
    }

    #[test]
    fn ensure_published_is_idle_without_a_queue() {
        let temp = tempfile::tempdir().unwrap();
        assert_eq!(
            ensure_published_at(temp.path()).unwrap(),
            PublishOutcome::Idle
        );
    }

    impl PublishOutcome {
        fn elapsed_ms(&self) -> u64 {
            match self {
                Self::Published { elapsed_ms, .. } => *elapsed_ms,
                _ => 0,
            }
        }
    }
}
