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

use std::fmt::Debug;
use std::path::Path;
use std::pin::Pin;
use std::time::SystemTime;

use async_trait::async_trait;
use futures::AsyncRead;
use futures::AsyncReadExt as _;
use futures::StreamExt as _;
use futures::io::Cursor;
use futures::stream;
use futures::stream::BoxStream;
use prost::Message as _;
use sha2::Digest as _;
use sha2::Sha256;

use crate::backend::Backend;
use crate::backend::BackendError;
use crate::backend::BackendInitError;
use crate::backend::BackendLoadError;
use crate::backend::BackendResult;
use crate::backend::ChangeId;
use crate::backend::Commit;
use crate::backend::CommitId;
use crate::backend::CopyHistory;
use crate::backend::CopyId;
use crate::backend::CopyRecord;
use crate::backend::FileId;
use crate::backend::RelatedCopy;
use crate::backend::SecureSig;
use crate::backend::SigningFn;
use crate::backend::SymlinkId;
use crate::backend::Tree;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::backend::make_root_commit;
use crate::index::Index;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::simple_backend::commit_from_proto;
use crate::simple_backend::commit_to_proto;
use crate::simple_backend::tree_from_proto;
use crate::simple_backend::tree_to_proto;
use crate::vex::VexClient;
use crate::vex::VexRepoConfig;

const ID_LENGTH: usize = 32;
const CHANGE_ID_LENGTH: usize = 16;

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Serialize a tree exactly as [`VexBackend::write_tree`] would, returning its
/// content-addressed [`TreeId`] and canonical bytes. Lets callers pre-address
/// trees so they can be uploaded in bulk instead of one round trip each.
pub fn serialize_tree(tree: &crate::backend::Tree) -> (crate::backend::TreeId, Vec<u8>) {
    let data = crate::simple_backend::tree_to_proto(tree).encode_to_vec();
    let id = crate::backend::TreeId::new(sha256_bytes(&data).to_vec());
    (id, data)
}

/// Serialize a commit exactly as [`VexBackend::write_commit`] would (without
/// signing), returning its content-addressed [`CommitId`] and canonical bytes.
pub fn serialize_commit(commit: &Commit) -> (CommitId, Vec<u8>) {
    let data = crate::simple_backend::commit_to_proto(commit).encode_to_vec();
    let id = CommitId::new(sha256_bytes(&data).to_vec());
    (id, data)
}

fn to_content_id(id: &[u8], object_type: &str) -> BackendResult<jj_backend_types::ContentId> {
    if id.len() != ID_LENGTH {
        return Err(BackendError::InvalidHashLength {
            expected: ID_LENGTH,
            actual: id.len(),
            object_type: object_type.to_string(),
            hash: hex_bytes(id),
        });
    }
    let mut bytes = [0; ID_LENGTH];
    bytes.copy_from_slice(id);
    Ok(jj_backend_types::ContentId::from_bytes(bytes))
}

fn map_status_error(
    err: crate::vex::VexClientError,
    object_type: &str,
    hash: String,
) -> BackendError {
    match err {
        crate::vex::VexClientError::Status(status) if status.code() == tonic::Code::NotFound => {
            BackendError::ObjectNotFound {
                object_type: object_type.to_string(),
                hash,
                source: status.into(),
            }
        }
        other => BackendError::ReadObject {
            object_type: object_type.to_string(),
            hash,
            source: Box::new(other),
        },
    }
}

#[derive(Debug, Clone)]
pub struct VexBackend {
    client: VexClient,
    virtual_root_path: Option<RepoPathBuf>,
    root_commit_id: CommitId,
    root_change_id: ChangeId,
    empty_tree_id: TreeId,
}

impl VexBackend {
    pub fn name_static() -> &'static str {
        "vex"
    }

    pub fn init(config: VexRepoConfig) -> Result<Self, BackendInitError> {
        let client = VexClient::from_config(config).map_err(|err| BackendInitError(err.into()))?;
        Ok(Self::new(client))
    }

    pub fn load(store_path: &Path) -> Result<Self, BackendLoadError> {
        let client =
            VexClient::from_store_path(store_path).map_err(|err| BackendLoadError(err.into()))?;
        Ok(Self::new(client))
    }

    fn new(client: VexClient) -> Self {
        let empty_tree_proto = tree_to_proto(&Tree::default()).encode_to_vec();
        let empty_tree_id = TreeId::new(sha256_bytes(&empty_tree_proto).to_vec());
        let virtual_root_path = client
            .config()
            .virtual_root_path
            .as_deref()
            .filter(|path| !path.is_empty() && *path != ".")
            .and_then(|path| RepoPathBuf::from_internal_string(path.to_string()).ok());
        Self {
            client,
            virtual_root_path,
            root_commit_id: CommitId::from_bytes(&[0; ID_LENGTH]),
            root_change_id: ChangeId::from_bytes(&[0; CHANGE_ID_LENGTH]),
            empty_tree_id,
        }
    }

    async fn read_object_bytes(
        &self,
        kind: jj_backend_types::ObjectKind,
        id: &impl crate::object_id::ObjectId,
    ) -> BackendResult<Vec<u8>> {
        let content_id = to_content_id(&id.to_bytes(), &id.object_type())?;
        self.client
            .get_object(kind, &content_id)
            .await
            .map_err(|err| map_status_error(err, &id.object_type(), id.hex()))
    }

    async fn write_object_bytes(
        &self,
        kind: jj_backend_types::ObjectKind,
        object_type: &'static str,
        data: Vec<u8>,
    ) -> BackendResult<jj_backend_types::ContentId> {
        let content_id = jj_backend_types::ContentId::from_bytes(sha256_bytes(&data));
        self.client
            .put_object(kind, &content_id, data)
            .await
            .map_err(|err| BackendError::WriteObject {
                object_type,
                source: Box::new(err),
            })?;
        Ok(content_id)
    }

    async fn read_physical_tree(&self, id: &TreeId) -> BackendResult<Tree> {
        let data = self
            .read_object_bytes(jj_backend_types::ObjectKind::Tree, id)
            .await?;
        let proto = crate::protos::simple_store::Tree::decode(&*data).map_err(|err| {
            BackendError::ReadObject {
                object_type: "tree".to_string(),
                hash: id.hex(),
                source: err.into(),
            }
        })?;
        Ok(tree_from_proto(proto))
    }

    async fn project_tree_to_virtual_root(&self, root_tree: Tree) -> BackendResult<Tree> {
        let Some(virtual_root_path) = &self.virtual_root_path else {
            return Ok(root_tree);
        };

        let mut tree = root_tree;
        for component in virtual_root_path.components() {
            let Some(TreeValue::Tree(child_tree_id)) = tree.value(component).cloned() else {
                return Ok(Tree::default());
            };
            tree = self.read_physical_tree(&child_tree_id).await?;
        }
        Ok(tree)
    }
}

#[async_trait]
impl Backend for VexBackend {
    fn name(&self) -> &str {
        Self::name_static()
    }

    fn commit_id_length(&self) -> usize {
        ID_LENGTH
    }

    fn change_id_length(&self) -> usize {
        CHANGE_ID_LENGTH
    }

    fn root_commit_id(&self) -> &CommitId {
        &self.root_commit_id
    }

    fn root_change_id(&self) -> &ChangeId {
        &self.root_change_id
    }

    fn empty_tree_id(&self) -> &TreeId {
        &self.empty_tree_id
    }

    fn concurrency(&self) -> usize {
        32
    }

    async fn read_file(
        &self,
        path: &RepoPath,
        id: &FileId,
    ) -> BackendResult<Pin<Box<dyn AsyncRead + Send>>> {
        let data = self
            .read_object_bytes(jj_backend_types::ObjectKind::Blob, id)
            .await
            .map_err(|err| match err {
                BackendError::ReadObject { source, .. }
                | BackendError::ObjectNotFound { source, .. } => BackendError::ReadFile {
                    path: path.to_owned(),
                    id: id.clone(),
                    source,
                },
                other => other,
            })?;
        Ok(Box::pin(Cursor::new(data)))
    }

    async fn write_file(
        &self,
        _path: &RepoPath,
        contents: &mut (dyn AsyncRead + Send + Unpin),
    ) -> BackendResult<FileId> {
        let mut data = Vec::new();
        contents
            .read_to_end(&mut data)
            .await
            .map_err(|err| BackendError::WriteObject {
                object_type: "file",
                source: err.into(),
            })?;
        let content_id = self
            .write_object_bytes(jj_backend_types::ObjectKind::Blob, "file", data)
            .await?;
        Ok(FileId::new(content_id.as_bytes().to_vec()))
    }

    async fn read_symlink(&self, _path: &RepoPath, id: &SymlinkId) -> BackendResult<String> {
        let data = self
            .read_object_bytes(jj_backend_types::ObjectKind::Symlink, id)
            .await?;
        String::from_utf8(data).map_err(|err| BackendError::InvalidUtf8 {
            object_type: "symlink".to_string(),
            hash: id.hex(),
            source: err.utf8_error(),
        })
    }

    async fn write_symlink(&self, _path: &RepoPath, target: &str) -> BackendResult<SymlinkId> {
        let content_id = self
            .write_object_bytes(
                jj_backend_types::ObjectKind::Symlink,
                "symlink",
                target.as_bytes().to_vec(),
            )
            .await?;
        Ok(SymlinkId::new(content_id.as_bytes().to_vec()))
    }

    async fn read_copy(&self, _id: &CopyId) -> BackendResult<CopyHistory> {
        Err(BackendError::Unsupported(
            "The Vex backend doesn't support copy history yet".to_string(),
        ))
    }

    async fn write_copy(&self, _copy: &CopyHistory) -> BackendResult<CopyId> {
        Err(BackendError::Unsupported(
            "The Vex backend doesn't support copy history yet".to_string(),
        ))
    }

    async fn get_related_copies(&self, _copy_id: &CopyId) -> BackendResult<Vec<RelatedCopy>> {
        Err(BackendError::Unsupported(
            "The Vex backend doesn't support copy history yet".to_string(),
        ))
    }

    async fn read_tree(&self, path: &RepoPath, id: &TreeId) -> BackendResult<Tree> {
        if *id == self.empty_tree_id {
            return Ok(Tree::default());
        }
        let tree = self.read_physical_tree(id).await?;
        if path.is_root() {
            self.project_tree_to_virtual_root(tree).await
        } else {
            Ok(tree)
        }
    }

    async fn write_tree(&self, _path: &RepoPath, tree: &Tree) -> BackendResult<TreeId> {
        let data = tree_to_proto(tree).encode_to_vec();
        let content_id = self
            .write_object_bytes(jj_backend_types::ObjectKind::Tree, "tree", data)
            .await?;
        Ok(TreeId::new(content_id.as_bytes().to_vec()))
    }

    async fn read_commit(&self, id: &CommitId) -> BackendResult<Commit> {
        if *id == self.root_commit_id {
            return Ok(make_root_commit(
                self.root_change_id.clone(),
                self.empty_tree_id.clone(),
            ));
        }

        let data = self
            .read_object_bytes(jj_backend_types::ObjectKind::Commit, id)
            .await?;
        let proto = crate::protos::simple_store::Commit::decode(&*data).map_err(|err| {
            BackendError::ReadObject {
                object_type: "commit".to_string(),
                hash: id.hex(),
                source: err.into(),
            }
        })?;
        Ok(commit_from_proto(proto))
    }

    async fn write_commit(
        &self,
        mut commit: Commit,
        sign_with: Option<&mut SigningFn>,
    ) -> BackendResult<(CommitId, Commit)> {
        if commit.parents.is_empty() {
            return Err(BackendError::Other(
                std::io::Error::other("Cannot write a commit with no parents").into(),
            ));
        }

        let mut proto = commit_to_proto(&commit);
        if let Some(sign) = sign_with {
            let data = proto.encode_to_vec();
            let sig = sign(&data).map_err(|err| BackendError::Other(Box::new(err)))?;
            proto.secure_sig = Some(sig.clone());
            commit.secure_sig = Some(SecureSig { data, sig });
        }
        let data = proto.encode_to_vec();
        let content_id = self
            .write_object_bytes(jj_backend_types::ObjectKind::Commit, "commit", data)
            .await?;
        Ok((CommitId::new(content_id.as_bytes().to_vec()), commit))
    }

    fn get_copy_records(
        &self,
        _paths: Option<&[RepoPathBuf]>,
        _root: &CommitId,
        _head: &CommitId,
    ) -> BackendResult<BoxStream<'_, BackendResult<CopyRecord>>> {
        Ok(stream::empty().boxed())
    }

    fn gc(&self, _index: &dyn Index, _keep_newer: SystemTime) -> BackendResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::VexBackend;
    use crate::backend::Backend as _;
    use crate::repo_path::RepoPath;
    use crate::vex::VexRepoConfig;
    use pollster::FutureExt as _;

    #[test]
    fn empty_tree_is_served_locally() {
        let backend = VexBackend::init(VexRepoConfig {
            endpoint: "http://127.0.0.1:1".to_string(),
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
        })
        .unwrap();

        let tree = backend
            .read_tree(RepoPath::root(), backend.empty_tree_id())
            .block_on()
            .unwrap();

        assert_eq!(tree, crate::backend::Tree::default());
    }
}
