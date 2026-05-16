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

use std::path::Path;

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

#[derive(Debug, Clone)]
pub struct VexOpHeadsStore {
    client: VexClient,
}

impl VexOpHeadsStore {
    pub fn name_static() -> &'static str {
        "vex_op_heads_store"
    }

    pub fn init(config: VexRepoConfig) -> Result<Self, BackendInitError> {
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        Ok(Self { client })
    }

    pub fn load(store_path: &Path) -> Result<Self, crate::backend::BackendLoadError> {
        let client = VexClient::from_store_path(store_path)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))?;
        Ok(Self { client })
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
