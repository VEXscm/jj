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
use std::time::SystemTime;

use async_trait::async_trait;
use prost::Message as _;
use sha2::Digest as _;
use sha2::Sha256;

use crate::backend::BackendInitError;
use crate::object_id::HexPrefix;
use crate::object_id::ObjectId as _;
use crate::object_id::PrefixResolution;
use crate::op_store::OpStore;
use crate::op_store::OpStoreError;
use crate::op_store::OpStoreResult;
use crate::op_store::Operation;
use crate::op_store::OperationId;
use crate::op_store::RootOperationData;
use crate::op_store::View;
use crate::op_store::ViewId;
use crate::simple_op_store::operation_from_proto;
use crate::simple_op_store::operation_to_proto;
use crate::simple_op_store::view_from_proto;
use crate::simple_op_store::view_to_proto;
use crate::vex::VexClient;
use crate::vex::VexRepoConfig;

const ID_LENGTH: usize = 32;

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn to_content_id(id: &[u8], object_type: &str) -> OpStoreResult<jj_backend_types::ContentId> {
    if id.len() != ID_LENGTH {
        return Err(OpStoreError::Other(
            std::io::Error::other(format!(
                "invalid {object_type} hash length: expected {ID_LENGTH}, got {}",
                id.len()
            ))
            .into(),
        ));
    }
    let mut bytes = [0; ID_LENGTH];
    bytes.copy_from_slice(id);
    Ok(jj_backend_types::ContentId::from_bytes(bytes))
}

fn map_status_error(
    err: crate::vex::VexClientError,
    object_type: &'static str,
    hash: String,
) -> OpStoreError {
    match err {
        crate::vex::VexClientError::Status(status) if status.code() == tonic::Code::NotFound => {
            OpStoreError::ObjectNotFound {
                object_type: object_type.to_string(),
                hash,
                source: status.into(),
            }
        }
        other => OpStoreError::ReadObject {
            object_type: object_type.to_string(),
            hash,
            source: Box::new(other),
        },
    }
}

#[derive(Debug, Clone)]
pub struct VexOpStore {
    client: VexClient,
    root_data: RootOperationData,
    root_operation_id: OperationId,
    root_view_id: ViewId,
}

impl VexOpStore {
    pub fn name_static() -> &'static str {
        "vex_op_store"
    }

    pub fn init(
        config: VexRepoConfig,
        root_data: RootOperationData,
    ) -> Result<Self, BackendInitError> {
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        Ok(Self::new(client, root_data))
    }

    pub fn load(
        store_path: &Path,
        root_data: RootOperationData,
    ) -> Result<Self, crate::backend::BackendLoadError> {
        let client = VexClient::from_store_path(store_path)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))?;
        Ok(Self::new(client, root_data))
    }

    fn new(client: VexClient, root_data: RootOperationData) -> Self {
        Self {
            client,
            root_data,
            root_operation_id: OperationId::from_bytes(&[0; ID_LENGTH]),
            root_view_id: ViewId::from_bytes(&[0; ID_LENGTH]),
        }
    }
}

#[async_trait]
impl OpStore for VexOpStore {
    fn name(&self) -> &str {
        Self::name_static()
    }

    fn root_operation_id(&self) -> &OperationId {
        &self.root_operation_id
    }

    async fn read_view(&self, id: &ViewId) -> OpStoreResult<View> {
        if *id == self.root_view_id {
            return Ok(View::make_root(self.root_data.root_commit_id.clone()));
        }

        let content_id = to_content_id(&id.to_bytes(), "view")?;
        let data = self
            .client
            .get_object(jj_backend_types::ObjectKind::View, &content_id)
            .await
            .map_err(|err| map_status_error(err, "view", id.hex()))?;
        let proto = crate::protos::simple_op_store::View::decode(&*data).map_err(|err| {
            OpStoreError::ReadObject {
                object_type: "view".to_string(),
                hash: id.hex(),
                source: err.into(),
            }
        })?;
        view_from_proto(proto).map_err(|err| OpStoreError::ReadObject {
            object_type: "view".to_string(),
            hash: id.hex(),
            source: err.into(),
        })
    }

    async fn write_view(&self, contents: &View) -> OpStoreResult<ViewId> {
        let data = view_to_proto(contents).encode_to_vec();
        let content_id = jj_backend_types::ContentId::from_bytes(sha256_bytes(&data));
        self.client
            .put_object(jj_backend_types::ObjectKind::View, &content_id, data)
            .await
            .map_err(|err| OpStoreError::WriteObject {
                object_type: "view",
                source: Box::new(err),
            })?;
        Ok(ViewId::new(content_id.as_bytes().to_vec()))
    }

    async fn read_operation(&self, id: &OperationId) -> OpStoreResult<Operation> {
        if *id == self.root_operation_id {
            return Ok(Operation::make_root(self.root_view_id.clone()));
        }

        let content_id = to_content_id(&id.to_bytes(), "operation")?;
        let data = self
            .client
            .get_object(jj_backend_types::ObjectKind::Op, &content_id)
            .await
            .map_err(|err| map_status_error(err, "operation", id.hex()))?;
        let proto = crate::protos::simple_op_store::Operation::decode(&*data).map_err(|err| {
            OpStoreError::ReadObject {
                object_type: "operation".to_string(),
                hash: id.hex(),
                source: err.into(),
            }
        })?;
        operation_from_proto(proto).map_err(|err| OpStoreError::ReadObject {
            object_type: "operation".to_string(),
            hash: id.hex(),
            source: err.into(),
        })
    }

    async fn write_operation(&self, contents: &Operation) -> OpStoreResult<OperationId> {
        let data = operation_to_proto(contents).encode_to_vec();
        let content_id = jj_backend_types::ContentId::from_bytes(sha256_bytes(&data));
        self.client
            .put_object(jj_backend_types::ObjectKind::Op, &content_id, data)
            .await
            .map_err(|err| OpStoreError::WriteObject {
                object_type: "operation",
                source: Box::new(err),
            })?;
        Ok(OperationId::new(content_id.as_bytes().to_vec()))
    }

    async fn resolve_operation_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> OpStoreResult<PrefixResolution<OperationId>> {
        let mut matches = Vec::new();
        if prefix.matches(&self.root_operation_id) {
            matches.push(self.root_operation_id.clone());
        }

        let remote_matches = self
            .client
            .resolve_operation_id_prefix(&prefix.hex())
            .await
            .map_err(|err| OpStoreError::Other(err.into()))?;
        matches.extend(
            remote_matches
                .into_iter()
                .map(|id| OperationId::new(id.as_bytes().to_vec())),
        );

        matches.sort();
        matches.dedup();
        Ok(match matches.len() {
            0 => PrefixResolution::NoMatch,
            1 => PrefixResolution::SingleMatch(matches.pop().unwrap()),
            _ => PrefixResolution::AmbiguousMatch,
        })
    }

    async fn gc(&self, _head_ids: &[OperationId], _keep_newer: SystemTime) -> OpStoreResult<()> {
        Ok(())
    }
}
