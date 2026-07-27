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

//! Vex operation/view object storage (roadmap/088 Stage 7).
//!
//! Operations and views are **local**. They are written to
//! `<repo>/op_store/{operations,views}/<hex>` and never uploaded: the operation
//! log is the client's own history, and publishing it is what Stage 7 removes.
//!
//! This deliberately does *not* delegate to
//! [`crate::simple_op_store::SimpleOpStore`], which hashes with blake2b-512 and
//! mints 64-byte ids. Vex ids are 32-byte sha256 of the same protobuf encoding,
//! and they are already recorded — in operation parent lists, in server op
//! heads, in the legacy `vex-local-heads` marker — so changing the hash would
//! break continuity with every id a repository already holds. The encoding and
//! the id are exactly what they were; only the destination changed.
//!
//! Reads fall back to the backend once. A repository cloned before this change
//! has its history only on the server, so a miss is fetched and then written
//! into the local file, which makes the fallback self-healing rather than a
//! permanent dependency. `local_writes` repos never fall back — they are
//! forbidden from contacting the backend.

#![expect(missing_docs)]

use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;

use async_trait::async_trait;
use prost::Message as _;
use sha2::Digest as _;
use sha2::Sha256;
use tempfile::NamedTempFile;

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

const OPERATIONS_DIR: &str = "operations";
const VIEWS_DIR: &str = "views";

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

fn not_found(object_type: &'static str, hash: String) -> OpStoreError {
    OpStoreError::ObjectNotFound {
        object_type: object_type.to_string(),
        hash,
        source: Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no such object in the local operation store",
        )),
    }
}

#[derive(Debug, Clone)]
pub struct VexOpStore {
    client: VexClient,
    /// `<repo>/op_store`, holding `operations/` and `views/`.
    dir: PathBuf,
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
        store_path: &Path,
        root_data: RootOperationData,
    ) -> Result<Self, BackendInitError> {
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        Self::new(client, store_path, root_data).map_err(|err| BackendInitError(err.into()))
    }

    pub fn load(
        store_path: &Path,
        root_data: RootOperationData,
    ) -> Result<Self, crate::backend::BackendLoadError> {
        let client = VexClient::from_store_path(store_path)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))?;
        Self::new(client, store_path, root_data)
            .map_err(|err| crate::backend::BackendLoadError(err.into()))
    }

    /// `create_dir_all` on both subdirectories rather than `create_dir`: the
    /// `op_store` directory itself already exists by the time either
    /// constructor runs, and a repository cloned before Stage 7 has the
    /// directory but neither subdirectory.
    fn new(
        client: VexClient,
        store_path: &Path,
        root_data: RootOperationData,
    ) -> std::io::Result<Self> {
        let dir = store_path.to_path_buf();
        std::fs::create_dir_all(dir.join(OPERATIONS_DIR))?;
        std::fs::create_dir_all(dir.join(VIEWS_DIR))?;
        Ok(Self {
            client,
            dir,
            root_data,
            root_operation_id: OperationId::from_bytes(&[0; ID_LENGTH]),
            root_view_id: ViewId::from_bytes(&[0; ID_LENGTH]),
        })
    }

    fn object_path(&self, subdir: &str, hex: &str) -> PathBuf {
        self.dir.join(subdir).join(hex)
    }

    /// Write-tmp-then-rename, so a reader never observes a partial object and a
    /// crashed write leaves nothing to mistake for one.
    fn write_object(&self, subdir: &str, hex: &str, data: &[u8]) -> std::io::Result<()> {
        let dir = self.dir.join(subdir);
        let mut temporary = NamedTempFile::new_in(&dir)?;
        temporary.write_all(data)?;
        temporary.flush()?;
        temporary
            .persist(dir.join(hex))
            .map_err(|err| err.error)
            .map(|_| ())
    }

    fn read_object(&self, subdir: &str, hex: &str) -> std::io::Result<Option<Vec<u8>>> {
        match std::fs::read(self.object_path(subdir, hex)) {
            Ok(data) => Ok(Some(data)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Fetch a pre-Stage-7 object from the backend and cache it locally, so the
    /// fallback heals itself instead of being paid again on every read.
    async fn fetch_and_cache(
        &self,
        kind: jj_backend_types::ObjectKind,
        subdir: &str,
        object_type: &'static str,
        hex: &str,
        content_id: &jj_backend_types::ContentId,
    ) -> OpStoreResult<Vec<u8>> {
        if self.client.local_writes() {
            return Err(not_found(object_type, hex.to_string()));
        }
        let data = self
            .client
            .get_object(kind, content_id)
            .await
            .map_err(|err| map_status_error(err, object_type, hex.to_string()))?;
        if let Err(err) = self.write_object(subdir, hex, &data) {
            tracing::debug!(error = %err, object_type, hex, "could not cache a fetched object");
        }
        Ok(data)
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

        let hex = id.hex();
        let data = match self
            .read_object(VIEWS_DIR, &hex)
            .map_err(|err| OpStoreError::ReadObject {
                object_type: "view".to_string(),
                hash: hex.clone(),
                source: err.into(),
            })? {
            Some(data) => data,
            None => {
                let content_id = to_content_id(&id.to_bytes(), "view")?;
                self.fetch_and_cache(
                    jj_backend_types::ObjectKind::View,
                    VIEWS_DIR,
                    "view",
                    &hex,
                    &content_id,
                )
                .await?
            }
        };
        let proto = crate::protos::simple_op_store::View::decode(&*data).map_err(|err| {
            OpStoreError::ReadObject {
                object_type: "view".to_string(),
                hash: hex.clone(),
                source: err.into(),
            }
        })?;
        view_from_proto(proto).map_err(|err| OpStoreError::ReadObject {
            object_type: "view".to_string(),
            hash: hex,
            source: err.into(),
        })
    }

    async fn write_view(&self, contents: &View) -> OpStoreResult<ViewId> {
        let data = view_to_proto(contents).encode_to_vec();
        let hash = sha256_bytes(&data);
        let id = ViewId::from_bytes(&hash);
        self.write_object(VIEWS_DIR, &id.hex(), &data)
            .map_err(|err| OpStoreError::WriteObject {
                object_type: "view",
                source: Box::new(err),
            })?;
        Ok(id)
    }

    async fn read_operation(&self, id: &OperationId) -> OpStoreResult<Operation> {
        if *id == self.root_operation_id {
            return Ok(Operation::make_root(self.root_view_id.clone()));
        }

        let hex = id.hex();
        let data = match self.read_object(OPERATIONS_DIR, &hex).map_err(|err| {
            OpStoreError::ReadObject {
                object_type: "operation".to_string(),
                hash: hex.clone(),
                source: err.into(),
            }
        })? {
            Some(data) => data,
            None => {
                let content_id = to_content_id(&id.to_bytes(), "operation")?;
                self.fetch_and_cache(
                    jj_backend_types::ObjectKind::Op,
                    OPERATIONS_DIR,
                    "operation",
                    &hex,
                    &content_id,
                )
                .await?
            }
        };
        let proto = crate::protos::simple_op_store::Operation::decode(&*data).map_err(|err| {
            OpStoreError::ReadObject {
                object_type: "operation".to_string(),
                hash: hex.clone(),
                source: err.into(),
            }
        })?;
        operation_from_proto(proto).map_err(|err| OpStoreError::ReadObject {
            object_type: "operation".to_string(),
            hash: hex,
            source: err.into(),
        })
    }

    async fn write_operation(&self, contents: &Operation) -> OpStoreResult<OperationId> {
        let data = operation_to_proto(contents).encode_to_vec();
        let hash = sha256_bytes(&data);
        let id = OperationId::from_bytes(&hash);
        self.write_object(OPERATIONS_DIR, &id.hex(), &data)
            .map_err(|err| OpStoreError::WriteObject {
                object_type: "operation",
                source: Box::new(err),
            })?;
        Ok(id)
    }

    async fn resolve_operation_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> OpStoreResult<PrefixResolution<OperationId>> {
        let mut matches = Vec::new();
        if prefix.matches(&self.root_operation_id) {
            matches.push(self.root_operation_id.clone());
        }

        for entry in std::fs::read_dir(self.dir.join(OPERATIONS_DIR))
            .map_err(|err| OpStoreError::Other(err.into()))?
        {
            let name = entry
                .map_err(|err| OpStoreError::Other(err.into()))?
                .file_name();
            let Some(name) = name.to_str() else { continue };
            if let Some(bytes) = crate::hex_util::decode_hex(name) {
                let id = OperationId::new(bytes);
                if prefix.matches(&id) {
                    matches.push(id);
                }
            }
        }

        // A pre-Stage-7 clone's history is only on the server, so an id the
        // user names may not be local yet. Same reason as the read fallback,
        // and it goes away with it.
        if !self.client.local_writes() {
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
        }

        matches.sort();
        matches.dedup();
        Ok(match matches.len() {
            0 => PrefixResolution::NoMatch,
            1 => PrefixResolution::SingleMatch(matches.pop().unwrap()),
            _ => PrefixResolution::AmbiguousMatch,
        })
    }

    async fn gc(&self, _head_ids: &[OperationId], _keep_newer: SystemTime) -> OpStoreResult<()> {
        // TODO(088 Stage 9): prune unreachable local operation/view files. Op
        // and View objects are the only ones this PRD reclaims (D13); commit
        // object retention is unchanged.
        Ok(())
    }
}
