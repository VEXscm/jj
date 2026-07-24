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

use crate::op_store::Operation;
use crate::op_store::OperationId;
use crate::op_store::ViewId;
use crate::object_id::ObjectId as _;
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

/// How many times the publisher re-derives its CAS base before giving up and
/// leaving the chain queued.
const MAX_PUBLISH_ATTEMPTS: usize = 3;

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
    /// Today's behavior: objects flush and the op head CASes before the
    /// command returns.
    #[default]
    Sync,
    /// Operations are recorded locally and drained before the process exits.
    FlushOnExit,
    /// Operations are recorded locally; the publisher drains best-effort at
    /// the end of the command and, failing that, at the next command or sync
    /// barrier.
    LocalFirst,
}

impl VexDurability {
    pub fn is_sync(&self) -> bool {
        matches!(self, Self::Sync)
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
    pub fn resolve(configured: Self) -> Self {
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
    Corrupt {
        file: &'static str,
        message: String,
    },
    UnsupportedVersion {
        file: &'static str,
        version: u32,
    },
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
/// chain was built on. `base_heads` is the only `expected` the chain can ever
/// be published with: the backend validates that a committed operation's
/// parents equal the CAS `expected` set exactly.
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
    MissingObject { kind: String, id: String },
    Corrupt(String),
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
}

impl<'a> VexPublisher<'a> {
    pub fn new(dir: &'a Path, transport: &'a dyn PublishTransport) -> Self {
        Self { dir, transport }
    }

    /// Publish every queued operation: upload the objects they introduced,
    /// then run a single op-head CAS that advances the server from the chain's
    /// base to its tip. Intermediate operations are coalesced into that one
    /// CAS — the backend requires a published operation's parents to equal the
    /// CAS `expected` set exactly, so a chain of more than one operation can
    /// only be published as a rewrite carrying the tip's view.
    pub async fn drain(&self) -> Result<PublishOutcome, PublishError> {
        let start = Instant::now();
        let Some(mut chain) = read_pending_publish(self.dir)? else {
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

        let mut base = chain.base_ids()?;
        for attempt in 0..MAX_PUBLISH_ATTEMPTS {
            let tip = self.chain_tip(&chain)?;
            let (head, view, coalesced) = self.head_to_publish(&tip, &base).await?;
            let outcome = self
                .transport
                .commit_op_heads(&base, &head, &view)
                .await?;
            if outcome.ok || outcome.current_heads == [head] {
                return self.record_published(&chain, &tip.id, &head, coalesced, start);
            }
            stats
                .publish_cas_conflicts
                .fetch_add(1, Ordering::Relaxed);
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
        self.transport.put_objects(objects).await
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
        let rewritten = Operation {
            view_id: operation.view_id.clone(),
            parents: base.iter().map(op_id_from_content_id).collect(),
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

    /// A CAS conflict this repo can resolve without merging: the server head
    /// is one of our own queued operations (a previous drain got further than
    /// its marker update), or it is exactly what the chain is parented on and
    /// only the recorded base was stale. Anything else means the server holds
    /// work this repo has not seen, which only a jj op merge can reconcile.
    fn refold(
        &self,
        chain: &mut PendingPublishMarker,
        server_heads: &[ContentId],
        tip: &ChainTip,
    ) -> Result<Option<Vec<ContentId>>, PublishError> {
        let [head] = server_heads else {
            return Ok(None);
        };
        if let Some(index) = chain.ops.iter().position(|entry| entry.op == head.to_string()) {
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
        let data = self.transport.read_object(ObjectKind::Op, id).ok_or_else(|| {
            PublishError::MissingObject {
                kind: kind_to_str(ObjectKind::Op).to_string(),
                id: id.to_string(),
            }
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
        // The clone's deferred registration, if any, is the chain's first
        // entry; it is published now and its marker must not outlive it.
        remove_marker(self.dir, crate::vex_op_heads_store::PENDING_REGISTRATION_FILE)?;
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

    /// Record where the server actually is without claiming anything about the
    /// local chain, so the next repo load serves both heads and lets jj merge.
    fn record_server_heads_only(&self, server_heads: &[ContentId]) -> Result<(), PublishError> {
        write_server_heads(self.dir, &ServerHeadsMarker::new(server_heads.to_vec(), None))?;
        Ok(())
    }
}

struct ChainTip {
    id: ContentId,
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
pub struct VexClientTransport<'a> {
    client: &'a VexClient,
}

impl<'a> VexClientTransport<'a> {
    pub fn new(client: &'a VexClient) -> Self {
        Self { client }
    }
}

/// Objects per `put_objects` batch. Matches the snapshot flush caps in
/// `vex.rs`; the batches pipeline over the one cached connection.
const PUBLISH_BATCH_OBJECTS: usize = 256;
const PUBLISH_BATCH_BYTES: usize = 32 * 1024 * 1024;
const PUBLISH_BATCH_CONCURRENCY: usize = 16;

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
        let mut batches = Vec::new();
        let mut current = Vec::new();
        let mut current_bytes = 0;
        for object in objects {
            current_bytes += object.2.len();
            current.push(object);
            if current.len() >= PUBLISH_BATCH_OBJECTS || current_bytes >= PUBLISH_BATCH_BYTES {
                batches.push(std::mem::take(&mut current));
                current_bytes = 0;
            }
        }
        if !current.is_empty() {
            batches.push(current);
        }
        self.client
            .put_object_batches_pipelined(batches, PUBLISH_BATCH_CONCURRENCY)
            .await?;
        Ok(())
    }

    async fn commit_op_heads(
        &self,
        expected: &[ContentId],
        new_head: &ContentId,
        new_view: &ContentId,
    ) -> Result<CasOutcome, PublishError> {
        let response = self
            .client
            .commit_op_heads(expected, new_head, new_view)
            .await?;
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
        Ok(self.client.get_op_heads().await?)
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
    let client = VexClient::from_store_path(dir)
        .map_err(|err| PublishError::Transport(err.to_string()))?;
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
            self.state.lock().unwrap().objects.get(&(kind, *id)).cloned()
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
    fn queue_chain(dir: &Path, fake: &FakeTransport, base: ContentId, count: usize) -> Vec<ContentId> {
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
        assert!(VexDurability::default().is_sync());
        assert!(!VexDurability::Sync.defers_publish());
        assert!(VexDurability::LocalFirst.defers_publish());
        assert!(VexDurability::FlushOnExit.blocks_on_exit());
        assert!(!VexDurability::LocalFirst.blocks_on_exit());
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
    fn ten_local_operations_coalesce_into_one_cas() {
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
                coalesced: true,
                ..
            }
        ));
        assert_eq!(fake.cas_calls(), 1);
        let server = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(
            server.published_local_head,
            Some(ids[9].to_string()),
            "a coalesced publish must record which local head the server head stands for"
        );
        // The published operation carries the tip's view and the base's parents.
        let published = fake.heads();
        assert_eq!(published.len(), 1);
        assert_ne!(published[0], ids[9]);
    }

    #[test]
    fn stale_base_refolds_onto_the_real_server_head_and_retries() {
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
        assert_eq!(fake.cas_calls(), 2, "one rejected CAS, then the refolded one");
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

        assert_eq!(outcome.published_ops(), 1);
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
