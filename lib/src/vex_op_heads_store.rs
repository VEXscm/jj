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

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use async_trait::async_trait;
use prost::Message as _;
use sha2::Digest as _;
use sha2::Sha256;

use crate::backend::BackendInitError;
use crate::object_id::ObjectId as _;
use crate::op_heads_store::OpHeadsStore;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_heads_store::OpHeadsStoreLock;
use crate::op_store::Operation;
use crate::op_store::OperationId;
use crate::op_store::View;
use crate::op_store::ViewId;
use crate::ref_name::WorkspaceNameBuf;
use crate::simple_op_store::operation_from_proto;
use crate::simple_op_store::operation_to_proto;
use crate::simple_op_store::view_from_proto;
use crate::simple_op_store::view_to_proto;
use crate::vex::VexClient;
use crate::vex::VexRepoConfig;

const ID_LENGTH: usize = 32;

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

fn sha256_content_id(data: &[u8]) -> jj_backend_types::ContentId {
    let mut hasher = Sha256::new();
    hasher.update(data);
    jj_backend_types::ContentId::from_bytes(hasher.finalize().into())
}

#[derive(Debug)]
struct VexNoopLock;

impl OpHeadsStoreLock for VexNoopLock {}

/// Name of the file (inside the op_heads store dir) where the local-write CI
/// runner records its op head(s). One hex content id per line. Deferred-
/// registration clones reuse the same file to serve their locally committed
/// workspace operation as the repo's op head until the registration is
/// published.
const LOCAL_HEADS_FILE: &str = "vex-local-heads";

/// Name of the marker file (inside the op_heads store dir) recording a
/// deferred workspace registration. See [`PendingRegistration`].
const PENDING_REGISTRATION_FILE: &str = "vex-pending-registration";

/// Persistent record of a clone whose workspace registration was deferred: the
/// workspace operation was committed locally only, and no op-head CAS reached
/// the server. The next `update_op_heads` that would CAS the server publishes
/// ("folds") the registration first.
///
/// Two states, distinguished by `pending_op_id`:
///
/// - **Armed** (`pending_op_id == None`): written by `Workspace::clone_vex`
///   before it commits the workspace operation. Tells `update_op_heads` to
///   record that operation locally instead of CASing the server.
/// - **Pending** (`pending_op_id == Some`): written by `update_op_heads` once
///   the workspace operation committed. `get_op_heads` now serves the local
///   heads file, and the next server-bound `update_op_heads` folds the
///   registration.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
struct PendingRegistration {
    /// Name of the workspace the deferred operation registers. Needed to
    /// rebuild the registration on a newer server head after a CAS conflict.
    workspace_name: String,
    /// Hex content id of the locally committed workspace operation.
    #[serde(default)]
    pending_op_id: Option<String>,
    /// Hex content ids of the server op heads the pending operation was built
    /// on (its non-root parents). The server validates that a committed
    /// operation's parents equal the CAS `expected` set exactly, so this is
    /// the only `expected` the pending operation can ever be published with.
    #[serde(default)]
    server_head_ids: Vec<String>,
}

/// Builds the view for a deferred workspace registration rebuilt onto a newer
/// server head. The new server view wins wholesale; only the deferred
/// workspace's wc-commit entry is carried over (and its wc commit becomes a
/// visible head). The clone-time trunk-bookmark write is intentionally not
/// replayed: it only ever (re)creates the bookmark at the clone's start
/// commit, and the newer server view's bookmark state is authoritative.
fn registration_view_on_head(
    mut head_view: View,
    pending_view: &View,
    workspace_name: &str,
) -> Result<View, String> {
    let name = WorkspaceNameBuf::from(workspace_name.to_owned());
    let Some(wc_commit_id) = pending_view.wc_commit_ids.get(&name) else {
        return Err(format!(
            "pending registration operation has no working-copy commit for workspace \
             {workspace_name}"
        ));
    };
    head_view.head_ids.insert(wc_commit_id.clone());
    head_view.wc_commit_ids.insert(name, wc_commit_id.clone());
    Ok(head_view)
}

#[derive(Debug, Clone)]
pub struct VexOpHeadsStore {
    client: VexClient,
    /// The `op_heads` store directory: home of the local heads file and the
    /// deferred-registration marker.
    store_dir: PathBuf,
    /// When true, local-write mode is active (READ_ONLY CI runner): op heads
    /// are recorded to the local heads file instead of the backend, and read
    /// back from it. Local heads stay authoritative forever and are never
    /// folded to the server.
    local_writes: bool,
}

impl VexOpHeadsStore {
    pub fn name_static() -> &'static str {
        "vex_op_heads_store"
    }

    pub fn init(config: VexRepoConfig, store_path: &Path) -> Result<Self, BackendInitError> {
        let local_writes = config.local_writes;
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        Ok(Self {
            client,
            store_dir: store_path.to_path_buf(),
            local_writes,
        })
    }

    pub fn load(store_path: &Path) -> Result<Self, crate::backend::BackendLoadError> {
        let client = VexClient::from_store_path(store_path)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))?;
        let local_writes = client.local_writes();
        Ok(Self {
            client,
            store_dir: store_path.to_path_buf(),
            local_writes,
        })
    }

    /// Arms deferred workspace registration for a clone: the next
    /// `update_op_heads` (the clone's workspace operation) is recorded locally
    /// instead of CASed to the server, and is published transparently by the
    /// first mutating operation that reaches the backend. Must be called before
    /// the workspace operation commits. Not for `local_writes` repos, whose
    /// local heads are already authoritative and never published.
    pub fn arm_deferred_registration(
        store_path: &Path,
        workspace_name: &str,
    ) -> std::io::Result<()> {
        let marker = PendingRegistration {
            workspace_name: workspace_name.to_owned(),
            pending_op_id: None,
            server_head_ids: Vec::new(),
        };
        let data = serde_json::to_vec_pretty(&marker).map_err(std::io::Error::other)?;
        fs::create_dir_all(store_path)?;
        fs::write(store_path.join(PENDING_REGISTRATION_FILE), data)
    }

    fn local_heads_path(&self) -> PathBuf {
        self.store_dir.join(LOCAL_HEADS_FILE)
    }

    fn pending_registration_path(&self) -> PathBuf {
        self.store_dir.join(PENDING_REGISTRATION_FILE)
    }

    /// Read op heads previously recorded locally (local-write mode or a
    /// deferred registration). Returns `None` when no head has been recorded,
    /// so callers fall back to the backend.
    fn read_local_heads(&self) -> Result<Option<Vec<OperationId>>, OpHeadsStoreError> {
        let path = self.local_heads_path();
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(OpHeadsStoreError::Read(Box::new(err)));
            }
        };
        let ids = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| {
                jj_backend_types::ContentId::from_hex(line)
                    .map(|id| OperationId::new(id.as_bytes().to_vec()))
                    .map_err(|err| OpHeadsStoreError::Read(Box::new(std::io::Error::other(err))))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ids.is_empty() {
            Ok(None)
        } else {
            Ok(Some(ids))
        }
    }

    /// Record op heads locally, replacing any previous set.
    fn write_local_heads(&self, new_id: &OperationId) -> Result<(), OpHeadsStoreError> {
        let path = self.local_heads_path();
        let content_id = to_content_id(new_id)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        }
        fs::write(path, format!("{content_id}\n")).map_err(|err| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(err),
        })
    }

    fn read_pending_registration(&self) -> Result<Option<PendingRegistration>, OpHeadsStoreError> {
        let path = self.pending_registration_path();
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(OpHeadsStoreError::Read(Box::new(err))),
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|err| OpHeadsStoreError::Read(Box::new(err)))
    }

    fn write_pending_registration(
        &self,
        marker: &PendingRegistration,
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let data = serde_json::to_vec_pretty(marker).map_err(|err| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(err),
        })?;
        fs::write(self.pending_registration_path(), data).map_err(|err| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(err),
        })
    }

    /// Deletes the deferred-registration state. The local heads file goes
    /// first: if marker deletion then fails, `get_op_heads` already falls back
    /// to the server and a later fold self-heals via the server's replay check.
    fn clear_deferred_registration(&self, new_id: &OperationId) -> Result<(), OpHeadsStoreError> {
        for path in [self.local_heads_path(), self.pending_registration_path()] {
            if let Err(err) = fs::remove_file(&path)
                && err.kind() != std::io::ErrorKind::NotFound
            {
                return Err(OpHeadsStoreError::Write {
                    new_op_id: new_id.clone(),
                    source: Box::new(err),
                });
            }
        }
        Ok(())
    }

    /// Records the clone's workspace operation locally instead of CASing the
    /// server. The operation's objects normally upload as a batch right before
    /// the op-head CAS; with no CAS here, flush them explicitly so the local
    /// cache holds them durably and a later fold from another process finds
    /// them on the server.
    fn record_deferred_registration(
        &self,
        mut marker: PendingRegistration,
        expected: &[jj_backend_types::ContentId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        self.client
            .flush_pending_uploads()
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        self.write_local_heads(new_id)?;
        marker.pending_op_id = Some(to_content_id(new_id)?.to_string());
        marker.server_head_ids = expected.iter().map(ToString::to_string).collect();
        self.write_pending_registration(&marker, new_id)
    }

    async fn read_operation_object(
        &self,
        id: &jj_backend_types::ContentId,
        new_id: &OperationId,
    ) -> Result<Operation, OpHeadsStoreError> {
        let data = self
            .client
            .get_object(jj_backend_types::ObjectKind::Op, id)
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        let proto = crate::protos::simple_op_store::Operation::decode(&*data).map_err(|err| {
            OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            }
        })?;
        operation_from_proto(proto).map_err(|err| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(err),
        })
    }

    async fn read_view_object(
        &self,
        id: &ViewId,
        new_id: &OperationId,
    ) -> Result<View, OpHeadsStoreError> {
        let bytes = id.to_bytes();
        if bytes.len() != ID_LENGTH {
            return Err(OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(std::io::Error::other(format!(
                    "invalid view id length: expected {ID_LENGTH}, got {}",
                    bytes.len()
                ))),
            });
        }
        let mut content_bytes = [0; ID_LENGTH];
        content_bytes.copy_from_slice(&bytes);
        let content_id = jj_backend_types::ContentId::from_bytes(content_bytes);
        let data = self
            .client
            .get_object(jj_backend_types::ObjectKind::View, &content_id)
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        let proto = crate::protos::simple_op_store::View::decode(&*data).map_err(|err| {
            OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            }
        })?;
        view_from_proto(proto).map_err(|err| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(err),
        })
    }

    /// The server head moved between clone and first publish, so the pending
    /// operation can never be accepted. Rebuild the registration as a fresh
    /// operation parented on the current server head.
    async fn rebuild_pending_registration(
        &self,
        marker: &PendingRegistration,
        pending_op: &jj_backend_types::ContentId,
        current_heads: &[jj_backend_types::ContentId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let write_err = |message: String| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(std::io::Error::other(message)),
        };
        let [head] = current_heads else {
            return Err(write_err(format!(
                "cannot republish deferred workspace registration: expected exactly one server \
                 op head, got {}",
                current_heads.len()
            )));
        };
        let head_operation = self.read_operation_object(head, new_id).await?;
        let head_view = self
            .read_view_object(&head_operation.view_id, new_id)
            .await?;
        let pending_operation = self.read_operation_object(pending_op, new_id).await?;
        let pending_view = self
            .read_view_object(&pending_operation.view_id, new_id)
            .await?;
        let new_view = registration_view_on_head(head_view, &pending_view, &marker.workspace_name)
            .map_err(write_err)?;

        let view_data = view_to_proto(&new_view).encode_to_vec();
        let view_content_id = sha256_content_id(&view_data);
        self.client
            .put_object(
                jj_backend_types::ObjectKind::View,
                &view_content_id,
                view_data,
            )
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;

        let new_operation = Operation {
            view_id: ViewId::new(view_content_id.as_bytes().to_vec()),
            parents: vec![OperationId::new(head.as_bytes().to_vec())],
            metadata: pending_operation.metadata.clone(),
            commit_predecessors: pending_operation.commit_predecessors.clone(),
        };
        let op_data = operation_to_proto(&new_operation).encode_to_vec();
        let op_content_id = sha256_content_id(&op_data);
        self.client
            .put_object(jj_backend_types::ObjectKind::Op, &op_content_id, op_data)
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;

        let response = self
            .client
            .commit_op_heads(std::slice::from_ref(head), &op_content_id, &op_content_id)
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        if response.ok {
            tracing::info!(
                workspace = marker.workspace_name,
                pending_op = %pending_op,
                republished_op = %op_content_id,
                "republished deferred workspace registration on the current server head"
            );
            Ok(())
        } else {
            Err(write_err(
                "CAS conflict on op heads: server head moved while republishing the deferred \
                 workspace registration"
                    .to_string(),
            ))
        }
    }

    /// Publishes a pending deferred workspace registration before the current
    /// operation's own CAS. A moved server head requires rebuilding the
    /// registration on that head, then reporting a CAS conflict so the caller
    /// reloads and rebuilds its user mutation.
    async fn fold_pending_registration(
        &self,
        marker: &PendingRegistration,
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let write_err = |message: String| OpHeadsStoreError::Write {
            new_op_id: new_id.clone(),
            source: Box::new(std::io::Error::other(message)),
        };
        let pending_hex = marker
            .pending_op_id
            .as_deref()
            .expect("fold requires a recorded pending op");
        let pending_op = jj_backend_types::ContentId::from_hex(pending_hex)
            .map_err(|err| write_err(format!("invalid pending registration op id: {err}")))?;
        let expected = marker
            .server_head_ids
            .iter()
            .map(|id| jj_backend_types::ContentId::from_hex(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| write_err(format!("invalid pending registration parent id: {err}")))?;
        let response = self
            .client
            .commit_op_heads(&expected, &pending_op, &pending_op)
            .await
            .map_err(|err| OpHeadsStoreError::Write {
                new_op_id: new_id.clone(),
                source: Box::new(err),
            })?;
        if response.ok {
            tracing::info!(
                workspace = marker.workspace_name,
                pending_op = %pending_op,
                "published deferred workspace registration"
            );
            return self.clear_deferred_registration(new_id);
        }
        let current_heads = response
            .current_op_head_ids
            .iter()
            .map(|id| jj_backend_types::ContentId::from_hex(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| write_err(format!("invalid op head from server: {err}")))?;
        if current_heads == [pending_op] {
            return self.clear_deferred_registration(new_id);
        }
        self.rebuild_pending_registration(marker, &pending_op, &current_heads, new_id)
            .await?;
        self.clear_deferred_registration(new_id)?;
        Err(write_err(
            "CAS conflict on op heads: the deferred workspace registration was republished on \
             the current server head; reload and retry"
                .to_string(),
        ))
    }

    async fn commit_new_head(
        &self,
        expected: &[jj_backend_types::ContentId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let new_content_id = to_content_id(new_id)?;
        let response = self
            .client
            .commit_op_heads(expected, &new_content_id, &new_content_id)
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
                source: Box::new(std::io::Error::other(response.error_message)),
            })
        }
    }
}

#[async_trait]
impl OpHeadsStore for VexOpHeadsStore {
    fn name(&self) -> &str {
        Self::name_static()
    }

    async fn update_op_heads(
        &self,
        old_ids: &[OperationId],
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
        let expected = old_ids
            .iter()
            .filter(|id| !is_root_operation_id(id))
            .map(to_content_id)
            .collect::<Result<Vec<_>, _>>()?;
        // The root operation is synthetic and never written to the remote object store.
        // JJ bootstraps new repos by pointing op heads at that all-zero id first, and the
        // first real operation CASes against that synthetic parent.
        if expected.is_empty() && is_root_operation_id(new_id) {
            return Ok(());
        }
        // Local-write mode (READ_ONLY CI runner): the op-head pointer update is a
        // backend write (`commit_op_heads` -> gRPC `CommitOperation`) that the
        // READ_ONLY token rejects ("repository access token lacks required
        // permission"). Record the new head locally instead so the clone's
        // working-copy operation is recorded without contacting the backend; the
        // referenced operation/view objects are already in the local cache (see
        // `VexClient::put_object`). `get_op_heads` reads this file back. Local
        // heads stay authoritative forever — they are never folded to the server.
        if self.local_writes {
            return self.write_local_heads(new_id);
        }
        if let Some(marker) = self.read_pending_registration()? {
            match &marker.pending_op_id {
                None => return self.record_deferred_registration(marker, &expected, new_id),
                Some(_) => self.fold_pending_registration(&marker, new_id).await?,
            }
        }
        self.commit_new_head(&expected, new_id).await
    }

    async fn get_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        // Local-write mode: once the runner has recorded an op head locally, it is
        // authoritative for this ephemeral workspace (we never advance the backend
        // head), so serve it without a backend round trip. Before the first local
        // write (e.g. resolving the clone's starting head) fall through to the
        // backend read, which the READ_ONLY token is allowed to perform.
        if self.local_writes {
            if let Some(local) = self.read_local_heads()? {
                return Ok(local);
            }
        } else if let Some(marker) = self.read_pending_registration()? {
            if marker.pending_op_id.is_some()
                && let Some(local) = self.read_local_heads()?
            {
                return Ok(local);
            }
        }
        let ids = self
            .client
            .get_op_heads()
            .await
            .map_err(|err| OpHeadsStoreError::Read(Box::new(err)))?;
        Ok(ids
            .into_iter()
            .map(|id| OperationId::new(id.as_bytes().to_vec()))
            .collect())
    }

    async fn lock(&self) -> Result<Box<dyn OpHeadsStoreLock + '_>, OpHeadsStoreError> {
        Ok(Box::new(VexNoopLock))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::collections::HashSet;

    use futures::executor::block_on;

    use super::*;
    use crate::backend::CommitId;
    use crate::op_store::RefTarget;
    use crate::vex::VexObjectReadMode;

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
            object_read_mode: VexObjectReadMode::NativeOnly,
        }
    }

    fn store_at(dir: &Path, local_writes: bool) -> VexOpHeadsStore {
        VexOpHeadsStore::init(test_config(local_writes), dir).unwrap()
    }

    fn op_id(byte: u8) -> OperationId {
        OperationId::new(vec![byte; ID_LENGTH])
    }

    #[test]
    fn armed_registration_records_local_head_and_marker() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        VexOpHeadsStore::arm_deferred_registration(temp.path(), "vex-clone-1").unwrap();

        assert_eq!(store.read_pending_registration().unwrap().unwrap(), {
            PendingRegistration {
                workspace_name: "vex-clone-1".to_string(),
                pending_op_id: None,
                server_head_ids: Vec::new(),
            }
        });
        assert!(store.read_local_heads().unwrap().is_none());

        let parent = op_id(1);
        let workspace_op = op_id(2);
        block_on(store.update_op_heads(std::slice::from_ref(&parent), &workspace_op)).unwrap();

        assert_eq!(
            store.read_local_heads().unwrap(),
            Some(vec![workspace_op.clone()])
        );
        let marker = store.read_pending_registration().unwrap().unwrap();
        assert_eq!(marker.workspace_name, "vex-clone-1");
        assert_eq!(
            marker.pending_op_id.as_deref(),
            Some(to_content_id(&workspace_op).unwrap().to_string().as_str())
        );
        assert_eq!(
            marker.server_head_ids,
            vec![to_content_id(&parent).unwrap().to_string()]
        );
        assert_eq!(
            block_on(store.get_op_heads()).unwrap(),
            vec![workspace_op.clone()]
        );

        store.clear_deferred_registration(&workspace_op).unwrap();
        assert!(store.read_local_heads().unwrap().is_none());
        assert!(store.read_pending_registration().unwrap().is_none());
        store.clear_deferred_registration(&workspace_op).unwrap();
    }

    #[test]
    fn armed_registration_with_root_parent_records_empty_server_heads() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        VexOpHeadsStore::arm_deferred_registration(temp.path(), "vex-clone-1").unwrap();

        let workspace_op = op_id(2);
        block_on(store.update_op_heads(&[OperationId::new(vec![0; ID_LENGTH])], &workspace_op))
            .unwrap();
        let marker = store.read_pending_registration().unwrap().unwrap();
        assert!(marker.server_head_ids.is_empty());
    }

    #[test]
    fn local_writes_mode_keeps_local_heads_authoritative_and_ignores_marker() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), true);
        VexOpHeadsStore::arm_deferred_registration(temp.path(), "vex-clone-1").unwrap();

        let first = op_id(3);
        block_on(store.update_op_heads(&[op_id(1)], &first)).unwrap();
        assert_eq!(block_on(store.get_op_heads()).unwrap(), vec![first.clone()]);

        let second = op_id(4);
        block_on(store.update_op_heads(std::slice::from_ref(&first), &second)).unwrap();
        assert_eq!(
            block_on(store.get_op_heads()).unwrap(),
            vec![second.clone()]
        );
        let marker = store.read_pending_registration().unwrap().unwrap();
        assert_eq!(marker.pending_op_id, None);
    }

    #[test]
    fn stale_local_heads_without_marker_are_not_served_in_normal_mode() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_at(temp.path(), false);
        store.write_local_heads(&op_id(5)).unwrap();
        assert!(block_on(store.get_op_heads()).is_err());
    }

    #[test]
    fn pending_registration_marker_round_trips() {
        let armed = PendingRegistration {
            workspace_name: "ws".to_string(),
            pending_op_id: None,
            server_head_ids: Vec::new(),
        };
        let json = serde_json::to_string(&armed).unwrap();
        assert_eq!(
            serde_json::from_str::<PendingRegistration>(&json).unwrap(),
            armed
        );

        let pending = PendingRegistration {
            workspace_name: "ws".to_string(),
            pending_op_id: Some("aa".repeat(32)),
            server_head_ids: vec!["bb".repeat(32)],
        };
        let json = serde_json::to_string(&pending).unwrap();
        assert_eq!(
            serde_json::from_str::<PendingRegistration>(&json).unwrap(),
            pending
        );
    }

    #[test]
    fn registration_view_on_head_carries_workspace_entry_only() {
        let commit = |byte: u8| CommitId::new(vec![byte; ID_LENGTH]);
        let mut pending_view = View {
            head_ids: HashSet::from([commit(1), commit(2)]),
            local_bookmarks: BTreeMap::new(),
            local_tags: BTreeMap::new(),
            remote_views: BTreeMap::new(),
            git_refs: BTreeMap::new(),
            git_head: RefTarget::absent(),
            wc_commit_ids: BTreeMap::new(),
        };
        pending_view
            .wc_commit_ids
            .insert(WorkspaceNameBuf::from("vex-clone-1".to_string()), commit(2));
        pending_view
            .local_bookmarks
            .insert("master".into(), RefTarget::normal(commit(1)));

        let head_view = View {
            head_ids: HashSet::from([commit(3)]),
            local_bookmarks: BTreeMap::new(),
            local_tags: BTreeMap::new(),
            remote_views: BTreeMap::new(),
            git_refs: BTreeMap::new(),
            git_head: RefTarget::absent(),
            wc_commit_ids: BTreeMap::from([(
                WorkspaceNameBuf::from("other".to_string()),
                commit(3),
            )]),
        };

        let rebuilt = registration_view_on_head(head_view, &pending_view, "vex-clone-1").unwrap();
        assert_eq!(rebuilt.head_ids, HashSet::from([commit(2), commit(3)]));
        assert_eq!(
            rebuilt.wc_commit_ids,
            BTreeMap::from([
                (WorkspaceNameBuf::from("other".to_string()), commit(3)),
                (WorkspaceNameBuf::from("vex-clone-1".to_string()), commit(2)),
            ])
        );
        assert!(rebuilt.local_bookmarks.is_empty());

        let head_view = View {
            head_ids: HashSet::new(),
            local_bookmarks: BTreeMap::new(),
            local_tags: BTreeMap::new(),
            remote_views: BTreeMap::new(),
            git_refs: BTreeMap::new(),
            git_head: RefTarget::absent(),
            wc_commit_ids: BTreeMap::new(),
        };
        assert!(registration_view_on_head(head_view, &pending_view, "unknown").is_err());
    }
}
