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

//! Local-first op log: the on-disk bookkeeping markers that live beside the
//! operation heads (roadmap/088).
//!
//! Since Stage 7 the operation log is *local*. Op heads are stored in
//! `.jj/repo/op_heads/heads/` by [`crate::vex_op_heads_store::VexOpHeadsStore`]
//! (delegating to [`crate::simple_op_heads_store::SimpleOpHeadsStore`]), and
//! operation/view objects are written to `.jj/repo/op_store/` by
//! [`crate::vex_op_store::VexOpStore`]. Nothing in this module publishes
//! anything any more: the deferred-publish queue and its publisher are gone
//! (D10).
//!
//! What survives here is bookkeeping:
//!
//! - [`ServerHeadsMarker`] — the ref-freshness probe record read and written by
//!   [`crate::vex_freshness`].
//! - [`read_local_heads`] and [`read_pending_publish`] — **read-only**, kept for
//!   one release as the one-time bootstrap sources that seed `heads/` for a
//!   repository cloned before Stage 7. `TODO(088 Stage 9)`: delete with the
//!   marker family.
//! - [`find_op_heads_dir`] — locating the repo from a working directory.
//!
//! Markers share a write-then-rename discipline and carry a `v` field. An
//! unreadable or future-versioned marker is never guessed at.

#![expect(missing_docs)]

use std::fmt;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use jj_backend_types::ContentId;
use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tempfile::NamedTempFile;

use crate::object_id::ObjectId as _;
use crate::op_store::OperationId;

/// Name of the file (inside the op_heads store dir) where pre-Stage-7 clients
/// recorded their op head(s). One hex content id per line. Read-only now: it is
/// a bootstrap source for `heads/`, never head storage.
pub const LOCAL_HEADS_FILE: &str = "vex-local-heads";
/// Ordered chain of locally committed but unpublished operations, as written by
/// a pre-Stage-7 client. Read-only now, for the same reason.
pub const PENDING_PUBLISH_FILE: &str = "vex-pending-publish";
/// Ref-freshness probe record. Since v2 it no longer carries op heads.
pub const SERVER_HEADS_FILE: &str = "vex-server-heads";

/// Marker schema versions written by this client. Each file carries its own:
/// they are separate schemas that happen to share a write discipline, and a
/// shared constant would version-bump a file whose contents never changed.
pub const PENDING_PUBLISH_MARKER_VERSION: u32 = 2;

/// Oldest [`PendingPublishMarker`] version this client still reads. Reading is
/// all it does: the queue is never written again.
pub const MIN_PENDING_PUBLISH_MARKER_VERSION: u32 = 1;

/// [`ServerHeadsMarker`] moved to v2 when it stopped recording op heads and
/// started recording the ref-freshness probe. v1 is still read (as "no token
/// yet"), so rolling the CLI back is not a one-way door.
pub const SERVER_HEADS_MARKER_VERSION: u32 = 2;

/// Oldest [`ServerHeadsMarker`] version this client still reads.
pub const MIN_SERVER_HEADS_MARKER_VERSION: u32 = 1;

const ID_LENGTH: usize = 32;

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or_default()
}

/// A marker file could not be understood. Callers treat this as "this repo has
/// no usable marker state".
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
    /// The version this client writes.
    const VERSION: u32;
    /// The oldest version it still reads.
    const MIN_VERSION: u32;
    fn version(&self) -> u32;
}

/// Record of this repository's ref-freshness probe (D8).
///
/// v2 carries an opaque, repo-scoped `ref_token` plus the outcome of the last
/// probe, success or not, so `vex doctor` can report a failing probe across
/// processes. It deliberately holds **no operation ids**: op heads are local,
/// and a marker that named them would be a second, competing head store.
///
/// A v1 marker (which carried `heads` / `published_local_head`) reads as "no
/// token yet". Its `heads` field is still parsed, because the one-time
/// bootstrap in [`crate::vex_op_heads_store`] seeds `heads/` from it for
/// repositories cloned before Stage 7. `TODO(088 Stage 9)`: drop `heads` with
/// the rest of the marker family.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ServerHeadsMarker {
    pub v: u32,
    /// Opaque ref-freshness token as of the last time this repository's refs
    /// were known to *match* the server's. Never overwritten by a probe that
    /// finds the server has moved — that would erase the evidence of being
    /// behind one probe after acquiring it. `None` until the first successful
    /// probe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_token: Option<String>,
    /// When [`Self::ref_token`] was last confirmed.
    #[serde(default)]
    pub updated_unix: i64,
    /// The server's token as of the last probe, recorded only when it differs
    /// from [`Self::ref_token`] — i.e. this repository is behind. `None` means
    /// the last probe found the server unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_token: Option<String>,
    /// When this repository was first observed to be behind, kept across
    /// probes so "behind since" does not reset every time the probe re-runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behind_since_unix: Option<i64>,
    /// When a probe was last attempted, whether or not it succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_probe_unix: Option<i64>,
    /// Why the last probe failed, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_probe_error: Option<String>,
    /// **v1 only, read-only.** Op heads a pre-Stage-7 client recorded here.
    /// Never written again; read solely by the one-time `heads/` bootstrap.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub heads: Vec<String>,
}

impl VersionedMarker for ServerHeadsMarker {
    const FILE: &'static str = SERVER_HEADS_FILE;
    const VERSION: u32 = SERVER_HEADS_MARKER_VERSION;
    const MIN_VERSION: u32 = MIN_SERVER_HEADS_MARKER_VERSION;

    fn version(&self) -> u32 {
        self.v
    }
}

impl ServerHeadsMarker {
    /// A fresh v2 marker carrying `ref_token` as of now.
    pub fn new(ref_token: Option<String>) -> Self {
        let now = now_unix();
        Self {
            v: SERVER_HEADS_MARKER_VERSION,
            ref_token,
            updated_unix: now,
            pending_token: None,
            behind_since_unix: None,
            last_probe_unix: Some(now),
            last_probe_error: None,
            heads: Vec::new(),
        }
    }

    /// Op heads recorded by a pre-Stage-7 client, for the one-time bootstrap.
    pub fn head_ids(&self) -> Result<Vec<ContentId>, MarkerError> {
        parse_ids(SERVER_HEADS_FILE, &self.heads)
    }

    /// Record a probe that found this repository in sync with the server, or a
    /// point at which it was brought back into sync (a clone, pull, or push).
    /// Clears any recorded "behind". Upgrades a v1 marker in place.
    pub fn record_success(&mut self, ref_token: Option<String>) {
        let now = now_unix();
        self.v = SERVER_HEADS_MARKER_VERSION;
        self.ref_token = ref_token;
        self.updated_unix = now;
        self.pending_token = None;
        self.behind_since_unix = None;
        self.last_probe_unix = Some(now);
        self.last_probe_error = None;
    }

    /// Record a probe that found the server's token different from the
    /// confirmed one: this repository is behind.
    ///
    /// [`Self::ref_token`] and [`Self::updated_unix`] are deliberately left
    /// alone. They answer "when were we last actually in sync", which is the
    /// number `vex status` reports and the one that would be destroyed by
    /// adopting the server's new token here — after which the very next read
    /// would call a stale repository current.
    pub fn record_behind(&mut self, server_token: String) {
        let now = now_unix();
        self.v = SERVER_HEADS_MARKER_VERSION;
        self.behind_since_unix = Some(self.behind_since_unix.unwrap_or(now));
        self.pending_token = Some(server_token);
        self.last_probe_unix = Some(now);
        self.last_probe_error = None;
    }

    /// Record a failed probe. The last known token and the time it was
    /// confirmed are deliberately kept: "we last saw token T at time U, and the
    /// probe has been failing since V" is the report `vex doctor` needs, and
    /// clearing the token would make a transient outage look like a repo that
    /// has never been probed.
    pub fn record_failure(&mut self, error: String) {
        self.v = SERVER_HEADS_MARKER_VERSION;
        self.last_probe_unix = Some(now_unix());
        self.last_probe_error = Some(error);
    }
}

/// One locally committed, unpublished operation, as a pre-Stage-7 client wrote
/// it. Read-only; retained for the one-time `heads/` bootstrap.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PendingOpEntry {
    pub op: String,
    #[serde(default)]
    pub removes: Vec<String>,
    #[serde(default)]
    pub objects: Vec<PendingObject>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PendingObject {
    pub kind: String,
    pub id: String,
}

/// A pre-Stage-7 publish queue. Read-only; retained for the one-time `heads/`
/// bootstrap. `TODO(088 Stage 9)`: delete with the marker family.
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
    const VERSION: u32 = PENDING_PUBLISH_MARKER_VERSION;
    const MIN_VERSION: u32 = MIN_PENDING_PUBLISH_MARKER_VERSION;

    fn version(&self) -> u32 {
        self.v
    }
}

impl PendingPublishMarker {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// The queue's last entry: the operation a pre-Stage-7 session had adopted
    /// locally, and therefore the head to seed `heads/` with.
    pub fn tip_ids(&self) -> Result<Vec<ContentId>, MarkerError> {
        let Some(tip) = self.ops.last() else {
            return Ok(Vec::new());
        };
        parse_ids(PENDING_PUBLISH_FILE, std::slice::from_ref(&tip.op))
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
    if !(T::MIN_VERSION..=T::VERSION).contains(&marker.version()) {
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

/// Delete a marker file, tolerating its absence.
///
/// `TODO(088 Stage 9)`: this is the mechanism that retires the whole
/// `vex-*` marker family once the compatibility window closes; nothing calls it
/// in the meantime, because Stage 7 deliberately leaves the legacy markers on
/// disk so a rolled-back CLI still finds them.
#[expect(dead_code)]
fn remove_marker(dir: &Path, file: &str) -> Result<(), MarkerError> {
    match std::fs::remove_file(dir.join(file)) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(MarkerError::Io(err)),
    }
}

/// Read a pre-Stage-7 publish queue. Bootstrap source only.
pub fn read_pending_publish(dir: &Path) -> Result<Option<PendingPublishMarker>, MarkerError> {
    read_marker(dir)
}

pub fn read_server_heads(dir: &Path) -> Result<Option<ServerHeadsMarker>, MarkerError> {
    read_marker(dir)
}

pub fn write_server_heads(dir: &Path, marker: &ServerHeadsMarker) -> Result<(), MarkerError> {
    write_marker(dir, marker)
}

/// Read op heads recorded locally by a pre-Stage-7 client. `None` when nothing
/// was recorded. Bootstrap source only — head storage is `heads/`.
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

pub fn content_id_from_op_id(id: &OperationId) -> Option<ContentId> {
    let bytes = id.to_bytes();
    let bytes: [u8; ID_LENGTH] = bytes.try_into().ok()?;
    Some(ContentId::from_bytes(bytes))
}

pub fn op_id_from_content_id(id: &ContentId) -> OperationId {
    OperationId::new(id.as_bytes().to_vec())
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
    use super::*;

    fn id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; ID_LENGTH])
    }

    #[test]
    fn a_v1_server_heads_marker_reads_as_no_token_yet() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join(SERVER_HEADS_FILE),
            format!(
                r#"{{"v":1,"heads":["{}"],"published_local_head":"{}","updated_unix":7}}"#,
                id(1),
                id(5)
            ),
        )
        .unwrap();

        let marker = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(marker.ref_token, None);
        assert_eq!(marker.last_probe_unix, None);
        assert_eq!(marker.last_probe_error, None);
        // The v1 head list survives solely so the one-time bootstrap can seed
        // `heads/` from it.
        assert_eq!(marker.head_ids().unwrap(), vec![id(1)]);
    }

    #[test]
    fn ref_token_marker_round_trips() {
        let temp = tempfile::tempdir().unwrap();
        let marker = ServerHeadsMarker::new(Some("token-a".to_string()));
        write_server_heads(temp.path(), &marker).unwrap();

        let read = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(read.v, SERVER_HEADS_MARKER_VERSION);
        assert_eq!(read.ref_token.as_deref(), Some("token-a"));
        assert_eq!(read.last_probe_error, None);
        assert!(read.last_probe_unix.is_some());
        // No operation ids are recorded any more.
        assert!(read.heads.is_empty());
    }

    #[test]
    fn recording_a_probe_failure_keeps_the_last_token() {
        let temp = tempfile::tempdir().unwrap();
        let mut marker = ServerHeadsMarker::new(Some("token-a".to_string()));
        let confirmed = marker.updated_unix;
        marker.record_failure("connect timed out".to_string());
        write_server_heads(temp.path(), &marker).unwrap();

        let read = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(read.ref_token.as_deref(), Some("token-a"));
        assert_eq!(read.updated_unix, confirmed);
        assert_eq!(read.last_probe_error.as_deref(), Some("connect timed out"));
    }

    #[test]
    fn markers_reject_unknown_versions() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join(SERVER_HEADS_FILE), r#"{"v":99}"#).unwrap();
        assert!(matches!(
            read_server_heads(temp.path()),
            Err(MarkerError::UnsupportedVersion { .. })
        ));
        std::fs::write(temp.path().join(PENDING_PUBLISH_FILE), r#"{"v":99}"#).unwrap();
        assert!(matches!(
            read_pending_publish(temp.path()),
            Err(MarkerError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn local_heads_and_a_legacy_queue_read_back_for_the_bootstrap() {
        let temp = tempfile::tempdir().unwrap();
        assert!(read_local_heads(temp.path()).unwrap().is_none());
        std::fs::write(
            temp.path().join(LOCAL_HEADS_FILE),
            format!("{}\n{}\n", id(1), id(2)),
        )
        .unwrap();
        assert_eq!(
            read_local_heads(temp.path()).unwrap().unwrap(),
            vec![op_id_from_content_id(&id(1)), op_id_from_content_id(&id(2))]
        );

        std::fs::write(
            temp.path().join(PENDING_PUBLISH_FILE),
            format!(
                r#"{{"v":2,"base_heads":["{}"],"ops":[{{"op":"{}"}},{{"op":"{}"}}]}}"#,
                id(1),
                id(3),
                id(4)
            ),
        )
        .unwrap();
        let chain = read_pending_publish(temp.path()).unwrap().unwrap();
        assert!(!chain.is_empty());
        assert_eq!(chain.tip_ids().unwrap(), vec![id(4)]);
    }

    #[test]
    fn op_heads_dir_resolves_through_the_repo_indirection() {
        let temp = tempfile::tempdir().unwrap();
        let repo = temp.path().join("repo");
        let op_heads = repo.join(".jj/repo/op_heads");
        std::fs::create_dir_all(&op_heads).unwrap();
        let nested = repo.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_op_heads_dir(&nested).unwrap(), op_heads);

        // A secondary workspace points at the shared repo through a file.
        let workspace = temp.path().join("ws");
        std::fs::create_dir_all(workspace.join(".jj")).unwrap();
        let shared = workspace.join(".jj/shared/op_heads");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::write(workspace.join(".jj/repo"), "shared").unwrap();
        assert_eq!(find_op_heads_dir(&workspace).unwrap(), shared);

        assert_eq!(find_op_heads_dir(temp.path()), None);
    }
}
