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
use std::path::PathBuf;
use std::pin::Pin;
use std::time::SystemTime;

use async_trait::async_trait;
use futures::AsyncRead;
use futures::AsyncReadExt as _;
use futures::StreamExt as _;
use futures::io::AllowStdIo;
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
use crate::backend::MillisSinceEpoch;
use crate::backend::RelatedCopy;
use crate::backend::SecureSig;
use crate::backend::Signature;
use crate::backend::SigningFn;
use crate::backend::SymlinkId;
use crate::backend::Timestamp;
use crate::backend::Tree;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::backend::make_root_commit;
use crate::index::Index;
use crate::merge::Merge;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;
use crate::repo_path::RepoPathBuf;
use crate::repo_path::RepoPathComponentBuf;
use crate::simple_backend::commit_from_proto;
use crate::simple_backend::commit_to_proto;
use crate::simple_backend::tree_from_proto;
use crate::simple_backend::tree_to_proto;
use crate::vex::VexClient;
use crate::vex::VexObjectReadMode;
use crate::vex::VexRepoConfig;
use crate::vex::vex_client_stats;

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

/// Typed error for a native-only read path that fetched commit/tree bytes
/// which are not a native Vex protobuf object (roadmap/066).
///
/// Returned (as the source of a [`BackendError::ReadObject`]) when a backend
/// in [`VexObjectReadMode::NativeOnly`] — every normal clone/load — fails to
/// decode an object natively. It names the object kind and the native
/// repository contract, and deliberately carries no object contents,
/// credentials, or signed URLs. It is never a signal to retry through the raw
/// Git parsers: Git-format data belongs to `vex git clone`, Git smart HTTP,
/// or an explicit conversion that opted into
/// [`VexObjectReadMode::GitCompatibility`].
#[derive(Debug, thiserror::Error)]
#[error(
    "{object_kind} object is not a native Vex protobuf {object_kind}; this repository is read \
     under the native-only object contract (native Vex clones never traverse raw Git objects — \
     complete native conversion, or use `vex git clone`/an explicit Git compatibility operation \
     for Git-format data)"
)]
pub struct VexNativeObjectFormatError {
    /// The object kind that failed native decoding (`"commit"` or `"tree"`).
    pub object_kind: &'static str,
    /// The underlying protobuf decode failure (no object contents).
    #[source]
    pub source: prost::DecodeError,
}

fn native_only_read_error(
    object_kind: &'static str,
    hash: String,
    source: prost::DecodeError,
) -> BackendError {
    BackendError::ReadObject {
        object_type: object_kind.to_string(),
        hash,
        source: Box::new(VexNativeObjectFormatError {
            object_kind,
            source,
        }),
    }
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
    /// Object decode policy (roadmap/066). [`VexObjectReadMode::NativeOnly`]
    /// for every normal clone/load; only explicit conversion/Git-bridge
    /// callers construct a config carrying
    /// [`VexObjectReadMode::GitCompatibility`].
    object_read_mode: VexObjectReadMode,
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
        let object_read_mode = client.config().object_read_mode;
        Self {
            client,
            virtual_root_path,
            root_commit_id: CommitId::from_bytes(&[0; ID_LENGTH]),
            root_change_id: ChangeId::from_bytes(&[0; CHANGE_ID_LENGTH]),
            empty_tree_id,
            object_read_mode,
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

    /// Resolve one raw Git SHA-1 to its native content ID through the
    /// `git/object/sha1/*` compatibility namespace. Reachable only from the
    /// raw Git parsers ([`Self::read_git_tree`] / [`Self::parse_git_commit`]),
    /// which are themselves gated on explicit
    /// [`VexObjectReadMode::GitCompatibility`] — a native-only read never
    /// issues these RPCs.
    async fn git_mapping(&self, kind: &str, oid_hex: &str) -> BackendResult<Vec<u8>> {
        debug_assert!(
            self.object_read_mode.allows_git_compatibility(),
            "git_mapping() reached on a NativeOnly read path"
        );
        let ref_name = format!("git/object/sha1/{kind}/{oid_hex}");
        let started = std::time::Instant::now();
        let resolved = self.client.resolve_ref(&ref_name).await;
        vex_client_stats().record_git_mapping_rpc(1, started.elapsed());
        let content_id = resolved
            .map_err(|err| BackendError::ReadObject {
                object_type: "git-mapping".to_string(),
                hash: oid_hex.to_string(),
                source: Box::new(err),
            })?
            .ok_or_else(|| BackendError::ObjectNotFound {
                object_type: "git-mapping".to_string(),
                hash: oid_hex.to_string(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("missing {ref_name}"),
                )),
            })?;
        Ok(jj_backend_types::ContentId::from_hex(&content_id)
            .map_err(|err| BackendError::ReadObject {
                object_type: "git-mapping".to_string(),
                hash: oid_hex.to_string(),
                source: Box::new(err),
            })?
            .as_bytes()
            .to_vec())
    }

    async fn read_git_tree(&self, data: &[u8]) -> BackendResult<Tree> {
        let mut entries = Vec::new();
        let mut index = 0;
        while index < data.len() {
            let mode_end = data[index..]
                .iter()
                .position(|byte| *byte == b' ')
                .ok_or_else(|| BackendError::ReadObject {
                    object_type: "git-tree".to_string(),
                    hash: "<tree>".to_string(),
                    source: "invalid git tree".into(),
                })?
                + index;
            let mode = std::str::from_utf8(&data[index..mode_end]).map_err(|err| {
                BackendError::ReadObject {
                    object_type: "git-tree".to_string(),
                    hash: "<tree>".to_string(),
                    source: err.into(),
                }
            })?;
            index = mode_end + 1;
            let name_end = data[index..]
                .iter()
                .position(|byte| *byte == 0)
                .ok_or_else(|| BackendError::ReadObject {
                    object_type: "git-tree".to_string(),
                    hash: "<tree>".to_string(),
                    source: "invalid git tree".into(),
                })?
                + index;
            let name = std::str::from_utf8(&data[index..name_end]).map_err(|err| {
                BackendError::ReadObject {
                    object_type: "git-tree".to_string(),
                    hash: "<tree>".to_string(),
                    source: err.into(),
                }
            })?;
            index = name_end + 1;
            let oid_end = index + 20;
            if oid_end > data.len() {
                return Err(BackendError::ReadObject {
                    object_type: "git-tree".to_string(),
                    hash: "<tree>".to_string(),
                    source: "truncated git tree".into(),
                });
            }
            let oid_hex = hex_bytes(&data[index..oid_end]);
            index = oid_end;
            let value = match mode {
                "40000" | "040000" => {
                    let id = self.git_mapping("tree", &oid_hex).await?;
                    TreeValue::Tree(TreeId::new(id))
                }
                "120000" => {
                    let id = self.git_mapping("blob", &oid_hex).await?;
                    TreeValue::Symlink(SymlinkId::new(id))
                }
                "160000" => {
                    TreeValue::GitSubmodule(CommitId::from_bytes(&data[index - 20..oid_end]))
                }
                "100755" => {
                    let id = self.git_mapping("blob", &oid_hex).await?;
                    TreeValue::File {
                        id: FileId::new(id),
                        executable: true,
                        copy_id: CopyId::placeholder(),
                    }
                }
                _ => {
                    let id = self.git_mapping("blob", &oid_hex).await?;
                    TreeValue::File {
                        id: FileId::new(id),
                        executable: false,
                        copy_id: CopyId::placeholder(),
                    }
                }
            };
            entries.push((RepoPathComponentBuf::new(name.to_string()).unwrap(), value));
        }
        entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        Ok(Tree::from_sorted_entries(entries))
    }

    async fn parse_git_commit(&self, data: &[u8], id: &CommitId) -> BackendResult<Commit> {
        let text = String::from_utf8_lossy(data);
        let (headers, message) = text.split_once("\n\n").unwrap_or((text.as_ref(), ""));
        let mut root_tree = None;
        for line in headers.lines() {
            if let Some(rest) = line.strip_prefix("tree ") {
                root_tree = Some(TreeId::new(self.git_mapping("tree", rest).await?));
            }
        }
        let root_tree = root_tree.ok_or_else(|| BackendError::ReadObject {
            object_type: "commit".to_string(),
            hash: id.hex(),
            source: "git commit missing tree".into(),
        })?;
        let signature = Signature {
            name: String::new(),
            email: String::new(),
            timestamp: Timestamp {
                timestamp: MillisSinceEpoch(0),
                tz_offset: 0,
            },
        };
        Ok(Commit {
            parents: vec![self.root_commit_id.clone()],
            predecessors: vec![id.clone()],
            root_tree: Merge::resolved(root_tree),
            conflict_labels: Merge::resolved(String::new()),
            change_id: ChangeId::from_bytes(&sha256_bytes(data)[0..CHANGE_ID_LENGTH]),
            description: message.to_string(),
            author: signature.clone(),
            committer: signature,
            secure_sig: None,
        })
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
        // Cache hit: stream from the on-disk cache file instead of buffering
        // the whole blob — checkout holds up to `concurrency()` readers at
        // once, so whole-blob buffers multiply peak RSS.
        let content_id = to_content_id(&id.to_bytes(), &id.object_type())?;
        if let Some(file) = self
            .client
            .open_cached_object(jj_backend_types::ObjectKind::Blob, &content_id)
        {
            return Ok(Box::pin(AllowStdIo::new(file)));
        }
        // Miss: fetch the whole blob (which also writes it to the cache) and
        // serve it from memory.
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

    fn cached_blob_path(&self, id: &FileId) -> Option<PathBuf> {
        let content_id = to_content_id(&id.to_bytes(), &id.object_type()).ok()?;
        self.client
            .cached_object_path(jj_backend_types::ObjectKind::Blob, &content_id)
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
        // Legacy repos (and snapshot packs built from them) may hold the
        // target bytes under either kind — old clients wrote symlink targets
        // as blobs. Cover both kinds in the local cache up front, matching
        // the RPC fallback order below; otherwise a target unpacked from a
        // snapshot pack under the other kind costs a NotFound round-trip
        // during checkout.
        let content_id = to_content_id(&id.to_bytes(), &id.object_type())?;
        let cached = [
            jj_backend_types::ObjectKind::Symlink,
            jj_backend_types::ObjectKind::Blob,
        ]
        .into_iter()
        .find_map(|kind| self.client.read_cached_object(kind, &content_id));
        if cached.is_some() {
            crate::vex::vex_client_stats()
                .record_get_object_cache_hit(jj_backend_types::ObjectKind::Symlink);
        }
        let data = match cached {
            Some(data) => data,
            None => match self
                .read_object_bytes(jj_backend_types::ObjectKind::Symlink, id)
                .await
            {
                Ok(data) => data,
                Err(_) => {
                    let file_id = FileId::new(id.to_bytes());
                    self.read_object_bytes(jj_backend_types::ObjectKind::Blob, &file_id)
                        .await?
                }
            },
        };
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
        let data = self
            .read_object_bytes(jj_backend_types::ObjectKind::Tree, id)
            .await?;
        let tree = match crate::protos::simple_store::Tree::decode(&*data) {
            Ok(proto) => tree_from_proto(proto),
            // A decode failure is NOT a mode selector: under the native-only
            // contract it is a typed read error, and only a backend whose
            // config explicitly opted into Git compatibility (conversion /
            // Git-bridge callers) may parse the bytes as a raw Git tree.
            Err(err) => match self.object_read_mode {
                VexObjectReadMode::NativeOnly => {
                    return Err(native_only_read_error("tree", id.hex(), err));
                }
                VexObjectReadMode::GitCompatibility => {
                    vex_client_stats().record_git_compat_tree_decode();
                    self.read_git_tree(&data).await?
                }
            },
        };
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
        match crate::protos::simple_store::Commit::decode(&*data) {
            Ok(proto) => Ok(commit_from_proto(proto)),
            // See `read_tree`: decode failure is a typed native error, never
            // an implicit switch into the raw Git compatibility parser.
            Err(err) => match self.object_read_mode {
                VexObjectReadMode::NativeOnly => {
                    Err(native_only_read_error("commit", id.hex(), err))
                }
                VexObjectReadMode::GitCompatibility => {
                    vex_client_stats().record_git_compat_commit_decode();
                    self.parse_git_commit(&data, id).await
                }
            },
        }
    }

    async fn prefetch_commits(&self, ids: &[CommitId]) -> BackendResult<()> {
        let ids = ids
            .iter()
            .filter(|id| *id != &self.root_commit_id)
            .map(|id| {
                to_content_id(&id.to_bytes(), &id.object_type())
                    .map(|content_id| (jj_backend_types::ObjectKind::Commit, content_id, None))
            })
            .collect::<BackendResult<Vec<_>>>()?;
        self.client
            .get_objects_inline_batched(ids, None)
            .await
            .map_err(|error| BackendError::ReadObject {
                object_type: "commit batch".to_string(),
                hash: "multiple commits".to_string(),
                source: Box::new(error),
            })?;
        Ok(())
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
    use std::path::Path;

    use super::VexBackend;
    use super::VexNativeObjectFormatError;
    use super::sha256_bytes;
    use crate::backend::Backend as _;
    use crate::backend::BackendError;
    use crate::backend::ChangeId;
    use crate::backend::Commit;
    use crate::backend::CommitId;
    use crate::backend::FileId;
    use crate::backend::MillisSinceEpoch;
    use crate::backend::Signature;
    use crate::backend::Timestamp;
    use crate::backend::TreeValue;
    use crate::merge::Merge;
    use crate::object_id::ObjectId as _;
    use crate::repo_path::RepoPath;
    use crate::repo_path::RepoPathComponentBuf;
    use crate::vex::VexObjectReadMode;
    use crate::vex::VexRepoConfig;
    use crate::vex::test_stats_lock;
    use crate::vex::vex_client_stats_snapshot;
    use futures::AsyncReadExt as _;
    use pollster::FutureExt as _;

    fn sample_config() -> VexRepoConfig {
        VexRepoConfig {
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
            object_read_mode: VexObjectReadMode::NativeOnly,
        }
    }

    /// Scaffold a loadable repo dir whose `vex.json` carries the given object
    /// read mode, and return a backend loaded from it. The endpoint is
    /// unreachable (`127.0.0.1:1`), so any attempted RPC — e.g. a
    /// `git/object/sha1/*` mapping lookup — fails instead of silently
    /// succeeding; objects are served from the local `vex-cache` only.
    fn load_backend_with_mode(repo_dir: &Path, mode: VexObjectReadMode) -> VexBackend {
        let store_dir = repo_dir.join("store");
        std::fs::create_dir_all(&store_dir).unwrap();
        // `object_read_mode` is deliberately never serialized (a normal clone
        // must not persist compatibility mode), so spell the field out by
        // hand for the compatibility fixture.
        let mut value = serde_json::to_value(sample_config()).unwrap();
        if mode.allows_git_compatibility() {
            value["object_read_mode"] = serde_json::Value::String("git_compatibility".to_string());
        }
        std::fs::write(
            repo_dir.join("vex.json"),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
        VexBackend::load(&store_dir).unwrap()
    }

    /// Write `data` into the local object cache under its content address so
    /// backend reads resolve without any network.
    fn put_cached_object(repo_dir: &Path, kind_dir: &str, data: &[u8]) -> Vec<u8> {
        let id = sha256_bytes(data).to_vec();
        let path = repo_dir
            .join("vex-cache")
            .join(kind_dir)
            .join(super::hex_bytes(&id));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, data).unwrap();
        id
    }

    fn native_format_source(err: &BackendError) -> Option<&VexNativeObjectFormatError> {
        match err {
            BackendError::ReadObject { source, .. } => {
                source.downcast_ref::<VexNativeObjectFormatError>()
            }
            _ => None,
        }
    }

    #[test]
    fn empty_tree_is_served_locally() {
        let backend = VexBackend::init(sample_config()).unwrap();

        let tree = backend
            .read_tree(RepoPath::root(), backend.empty_tree_id())
            .block_on()
            .unwrap();

        assert_eq!(tree, crate::backend::Tree::default());
    }

    /// Backend loaded from a store path with the blob already in the local
    /// cache: `read_file` must stream from the cache file (the endpoint above
    /// is unreachable, so any RPC attempt would fail) and `cached_blob_path`
    /// must expose the cache file for reflink materialization.
    #[test]
    fn read_file_streams_from_cache_and_reports_cached_blob_path() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_dir = temp_dir.path().join("repo");
        let store_dir = repo_dir.join("store");
        std::fs::create_dir_all(&store_dir).unwrap();
        sample_config().write_to_repo_path(&repo_dir).unwrap();
        let backend = VexBackend::load(&store_dir).unwrap();

        let data = b"cached blob contents";
        let id = FileId::new(sha256_bytes(data).to_vec());
        let uncached_id = FileId::new(sha256_bytes(b"never cached").to_vec());
        let cache_file = repo_dir
            .join("vex-cache")
            .join("blob")
            .join(super::hex_bytes(&id.to_bytes()));
        std::fs::create_dir_all(cache_file.parent().unwrap()).unwrap();
        std::fs::write(&cache_file, data).unwrap();

        assert_eq!(backend.cached_blob_path(&id), Some(cache_file));
        assert_eq!(backend.cached_blob_path(&uncached_id), None);

        let mut reader = backend.read_file(RepoPath::root(), &id).block_on().unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).block_on().unwrap();
        assert_eq!(buf, data);
    }

    /// Native protobuf commits and trees decode successfully under the
    /// default `NativeOnly` mode, entirely from the local cache.
    #[test]
    fn native_only_mode_reads_native_protobuf_commit_and_tree() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_dir = temp_dir.path().join("repo");
        let backend = load_backend_with_mode(&repo_dir, VexObjectReadMode::NativeOnly);

        let tree = crate::backend::Tree::from_sorted_entries(vec![(
            RepoPathComponentBuf::new("file".to_string()).unwrap(),
            TreeValue::File {
                id: FileId::new(sha256_bytes(b"contents").to_vec()),
                executable: false,
                copy_id: crate::backend::CopyId::placeholder(),
            },
        )]);
        let (tree_id, tree_bytes) = super::serialize_tree(&tree);
        put_cached_object(&repo_dir, "tree", &tree_bytes);

        let signature = Signature {
            name: "author".to_string(),
            email: "author@example.test".to_string(),
            timestamp: Timestamp {
                timestamp: MillisSinceEpoch(0),
                tz_offset: 0,
            },
        };
        let commit = Commit {
            parents: vec![CommitId::from_bytes(&[0; super::ID_LENGTH])],
            predecessors: vec![],
            root_tree: Merge::resolved(tree_id.clone()),
            conflict_labels: Merge::resolved(String::new()),
            change_id: ChangeId::from_bytes(&[7; super::CHANGE_ID_LENGTH]),
            description: "native protobuf commit".to_string(),
            author: signature.clone(),
            committer: signature,
            secure_sig: None,
        };
        let (commit_id, commit_bytes) = super::serialize_commit(&commit);
        put_cached_object(&repo_dir, "commit", &commit_bytes);

        let read_tree = backend
            .read_tree(RepoPath::root(), &tree_id)
            .block_on()
            .unwrap();
        assert_eq!(read_tree, tree);

        let read_commit = backend.read_commit(&commit_id).block_on().unwrap();
        assert_eq!(read_commit.description, "native protobuf commit");
        assert_eq!(read_commit.root_tree, Merge::resolved(tree_id));
    }

    /// Raw Git commit bytes are a typed native object-format error under
    /// `NativeOnly` — the read never enters `parse_git_commit()` and the
    /// error names the contract without echoing the object contents.
    #[test]
    fn native_only_mode_rejects_raw_git_commit_bytes() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_dir = temp_dir.path().join("repo");
        let backend = load_backend_with_mode(&repo_dir, VexObjectReadMode::NativeOnly);
        let data = b"tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904\n\nsecret commit message";
        let id = CommitId::new(put_cached_object(&repo_dir, "commit", data));

        let err = backend.read_commit(&id).block_on().unwrap_err();

        let typed = native_format_source(&err).expect("expected VexNativeObjectFormatError");
        assert_eq!(typed.object_kind, "commit");
        let message = typed.to_string();
        assert!(message.contains("native-only"), "message: {message}");
        // The typed error names the contract, never the object contents.
        assert!(!message.contains("secret commit message"));
        assert!(!message.contains("4b825dc6"));
    }

    /// Raw Git tree bytes are rejected under `NativeOnly` without a single
    /// `git/object/sha1/*` mapping lookup. A mapping attempt would both fail
    /// differently (the endpoint is unreachable) and bump the mapping
    /// counters, which must stay untouched.
    #[test]
    fn native_only_mode_rejects_raw_git_tree_bytes_without_mapping_lookup() {
        let _guard = test_stats_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_dir = temp_dir.path().join("repo");
        let backend = load_backend_with_mode(&repo_dir, VexObjectReadMode::NativeOnly);
        let mut data = b"40000 dir\0".to_vec();
        data.extend_from_slice(&[0xaa; 20]);
        let id = crate::backend::TreeId::new(put_cached_object(&repo_dir, "tree", &data));

        let before = vex_client_stats_snapshot();
        let err = backend
            .read_tree(RepoPath::root(), &id)
            .block_on()
            .unwrap_err();
        let after = vex_client_stats_snapshot();

        let typed = native_format_source(&err).expect("expected VexNativeObjectFormatError");
        assert_eq!(typed.object_kind, "tree");
        assert_eq!(after.git_mapping_rpcs, before.git_mapping_rpcs);
        assert_eq!(
            after.git_mapping_names_resolved,
            before.git_mapping_names_resolved
        );
        assert_eq!(
            after.git_compat_tree_decodes,
            before.git_compat_tree_decodes
        );
    }

    /// Explicit `GitCompatibility` preserves the raw Git tree parser: a
    /// gitlink-only tree (which needs no SHA-1 mapping lookup) still decodes,
    /// and the compatibility decode counter records it.
    #[test]
    fn git_compatibility_mode_decodes_raw_git_tree() {
        let _guard = test_stats_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_dir = temp_dir.path().join("repo");
        let backend = load_backend_with_mode(&repo_dir, VexObjectReadMode::GitCompatibility);
        let submodule_oid = [0x04; 20];
        let mut data = b"160000 sub\0".to_vec();
        data.extend_from_slice(&submodule_oid);
        let id = crate::backend::TreeId::new(put_cached_object(&repo_dir, "tree", &data));

        let before = vex_client_stats_snapshot();
        let tree = backend.read_tree(RepoPath::root(), &id).block_on().unwrap();
        let after = vex_client_stats_snapshot();

        let component = RepoPathComponentBuf::new("sub".to_string()).unwrap();
        assert_eq!(
            tree.value(&component),
            Some(&TreeValue::GitSubmodule(CommitId::from_bytes(
                &submodule_oid
            )))
        );
        assert_eq!(
            after.git_compat_tree_decodes,
            before.git_compat_tree_decodes + 1
        );
        // A gitlink entry carries its id inline; no mapping RPC is needed.
        assert_eq!(after.git_mapping_rpcs, before.git_mapping_rpcs);
    }

    /// Explicit `GitCompatibility` still routes non-protobuf commit bytes
    /// into `parse_git_commit()` (here failing on a commit with no tree
    /// header — a parser error, not the native-only typed error) and records
    /// the compatibility decode.
    #[test]
    fn git_compatibility_mode_still_enters_raw_git_commit_parser() {
        let _guard = test_stats_lock()
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_dir = temp_dir.path().join("repo");
        let backend = load_backend_with_mode(&repo_dir, VexObjectReadMode::GitCompatibility);
        let data = b"parent deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n\nno tree header";
        let id = CommitId::new(put_cached_object(&repo_dir, "commit", data));

        let before = vex_client_stats_snapshot();
        let err = backend.read_commit(&id).block_on().unwrap_err();
        let after = vex_client_stats_snapshot();

        // The compatibility parser ran (and failed on its own terms); the
        // native-only typed rejection was NOT raised.
        assert!(native_format_source(&err).is_none(), "unexpected: {err:?}");
        assert!(matches!(err, BackendError::ReadObject { .. }));
        assert_eq!(
            after.git_compat_commit_decodes,
            before.git_compat_commit_decodes + 1
        );
    }
}
