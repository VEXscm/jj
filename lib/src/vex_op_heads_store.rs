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

use crate::backend::BackendInitError;
use crate::object_id::ObjectId as _;
use crate::op_heads_store::OpHeadsStore;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_heads_store::OpHeadsStoreLock;
use crate::op_store::OperationId;
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

#[derive(Debug)]
struct VexNoopLock;

impl OpHeadsStoreLock for VexNoopLock {}

/// Name of the file (inside the op_heads store dir) where the local-write CI
/// runner records its op head(s). One hex content id per line.
const LOCAL_HEADS_FILE: &str = "vex-local-heads";

#[derive(Debug, Clone)]
pub struct VexOpHeadsStore {
    client: VexClient,
    /// When `Some`, local-write mode is active (READ_ONLY CI runner): op heads
    /// are recorded to this file instead of the backend, and read back from it.
    /// Path is the `op_heads` store directory.
    local_heads_dir: Option<PathBuf>,
}

impl VexOpHeadsStore {
    pub fn name_static() -> &'static str {
        "vex_op_heads_store"
    }

    pub fn init(config: VexRepoConfig, store_path: &Path) -> Result<Self, BackendInitError> {
        let local = config.local_writes.then(|| store_path.to_path_buf());
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        Ok(Self {
            client,
            local_heads_dir: local,
        })
    }

    pub fn load(store_path: &Path) -> Result<Self, crate::backend::BackendLoadError> {
        let client = VexClient::from_store_path(store_path)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))?;
        let local_heads_dir = client.local_writes().then(|| store_path.to_path_buf());
        Ok(Self {
            client,
            local_heads_dir,
        })
    }

    fn local_heads_path(&self) -> Option<PathBuf> {
        self.local_heads_dir
            .as_ref()
            .map(|dir| dir.join(LOCAL_HEADS_FILE))
    }

    /// Read op heads previously recorded locally (local-write mode). Returns
    /// `None` when local-write mode is off or no head has been recorded yet, so
    /// callers fall back to the backend.
    fn read_local_heads(&self) -> Result<Option<Vec<OperationId>>, OpHeadsStoreError> {
        let Some(path) = self.local_heads_path() else {
            return Ok(None);
        };
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

    /// Record op heads locally (local-write mode), replacing any previous set.
    fn write_local_heads(
        &self,
        path: &Path,
        new_id: &OperationId,
    ) -> Result<(), OpHeadsStoreError> {
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
        // `VexClient::put_object`). `get_op_heads` reads this file back.
        if let Some(path) = self.local_heads_path() {
            return self.write_local_heads(&path, new_id);
        }
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
                source: Box::new(std::io::Error::other(response.error_message)),
            })
        }
    }

    async fn get_op_heads(&self) -> Result<Vec<OperationId>, OpHeadsStoreError> {
        // Local-write mode: once the runner has recorded an op head locally, it is
        // authoritative for this ephemeral workspace (we never advance the backend
        // head), so serve it without a backend round trip. Before the first local
        // write (e.g. resolving the clone's starting head) fall through to the
        // backend read, which the READ_ONLY token is allowed to perform.
        if let Some(local) = self.read_local_heads()? {
            return Ok(local);
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
