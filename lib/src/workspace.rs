// Copyright 2021 The Jujutsu Authors
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

use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use thiserror::Error;

use crate::backend::BackendInitError;
use crate::commit::Commit;
use crate::file_util;
use crate::file_util::BadPathEncoding;
use crate::file_util::IoResultExt as _;
use crate::file_util::PathError;
use crate::local_working_copy::LocalWorkingCopy;
use crate::local_working_copy::LocalWorkingCopyFactory;
use crate::merged_tree::MergedTree;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_store::OperationId;
use crate::ref_name::WorkspaceName;
use crate::ref_name::WorkspaceNameBuf;
use crate::repo::BackendInitializer;
use crate::repo::CheckOutCommitError;
use crate::repo::IndexStoreInitializer;
use crate::repo::OpHeadsStoreInitializer;
use crate::repo::OpStoreInitializer;
use crate::repo::ReadonlyRepo;
use crate::repo::Repo as _;
use crate::repo::RepoInitError;
use crate::repo::RepoLoader;
use crate::repo::StoreFactories;
use crate::repo::StoreLoadError;
use crate::repo::SubmoduleStoreInitializer;
use crate::repo::read_store_type;
use crate::rewrite::merge_commit_trees;
use crate::settings::UserSettings;
use crate::signing::SignInitError;
use crate::signing::Signer;
use crate::simple_backend::SimpleBackend;
use crate::transaction::TransactionCommitError;
use crate::vex::CloneBlobMode;
use crate::vex::VexRepoConfig;
use crate::vex::create_store_factories;
use crate::vex_backend::VexBackend;
use crate::vex_op_heads_store::VexOpHeadsStore;
use crate::vex_op_store::VexOpStore;
use crate::virtual_working_copy::VirtualWorkingCopy;
use crate::virtual_working_copy::VirtualWorkingCopyFactory;
use crate::working_copy::CheckoutError;
use crate::working_copy::CheckoutStats;
use crate::working_copy::LockedWorkingCopy;
use crate::working_copy::WorkingCopy;
use crate::working_copy::WorkingCopyFactory;
use crate::working_copy::WorkingCopyStateError;
use crate::workspace_store::SimpleWorkspaceStore;
use crate::workspace_store::WorkspaceStore as _;
use crate::workspace_store::WorkspaceStoreError;

#[derive(Error, Debug)]
pub enum WorkspaceInitError {
    #[error("The destination repo ({0}) already exists")]
    DestinationExists(PathBuf),
    #[error("Repo path could not be encoded")]
    EncodeRepoPath(#[source] BadPathEncoding),
    #[error(transparent)]
    CheckOutCommit(#[from] CheckOutCommitError),
    #[error(transparent)]
    WorkingCopyState(#[from] WorkingCopyStateError),
    #[error(transparent)]
    Checkout(#[from] CheckoutError),
    #[error(transparent)]
    Path(#[from] PathError),
    #[error(transparent)]
    OpHeadsStore(OpHeadsStoreError),
    #[error(transparent)]
    WorkspaceStore(#[from] WorkspaceStoreError),
    #[error(transparent)]
    Backend(#[from] BackendInitError),
    #[error(transparent)]
    SignInit(#[from] SignInitError),
    #[error(transparent)]
    TransactionCommit(#[from] TransactionCommitError),
}

#[derive(Error, Debug)]
pub enum WorkspaceLoadError {
    #[error("The repo appears to no longer be at {0}")]
    RepoDoesNotExist(PathBuf),
    #[error("There is no Jujutsu repo in {0}")]
    NoWorkspaceHere(PathBuf),
    #[error("Cannot read the repo")]
    StoreLoadError(#[from] StoreLoadError),
    #[error("Repo path could not be decoded")]
    DecodeRepoPath(#[source] BadPathEncoding),
    #[error(transparent)]
    WorkingCopyState(#[from] WorkingCopyStateError),
    #[error(transparent)]
    Path(#[from] PathError),
}

/// The combination of a repo and a working copy.
///
/// Represents the combination of a repo and working copy, i.e. what's typically
/// the .jj/ directory and its parent. See
/// <https://github.com/jj-vcs/jj/blob/main/docs/working-copy.md#workspaces>
/// for more information.
pub struct Workspace {
    // Path to the workspace root (typically the parent of a .jj/ directory), which is where
    // working copy files live.
    workspace_root: PathBuf,
    repo_path: PathBuf,
    repo_loader: RepoLoader,
    working_copy: Box<dyn WorkingCopy>,
}

fn create_jj_dir(workspace_root: &Path) -> Result<PathBuf, WorkspaceInitError> {
    let jj_dir = workspace_root.join(".jj");
    match std::fs::create_dir(&jj_dir).context(&jj_dir) {
        Ok(()) => Ok(jj_dir),
        Err(e) if e.source.kind() == io::ErrorKind::AlreadyExists => {
            Err(WorkspaceInitError::DestinationExists(jj_dir))
        }
        Err(e) => Err(e.into()),
    }
}

async fn init_working_copy(
    repo: &Arc<ReadonlyRepo>,
    workspace_root: &Path,
    jj_dir: &Path,
    working_copy_factory: &dyn WorkingCopyFactory,
    workspace_name: WorkspaceNameBuf,
) -> Result<(Box<dyn WorkingCopy>, Arc<ReadonlyRepo>), WorkspaceInitError> {
    let start_commit = repo.store().root_commit();
    init_working_copy_at(
        repo,
        workspace_root,
        jj_dir,
        working_copy_factory,
        workspace_name,
        &start_commit,
    )
    .await
}

async fn init_working_copy_at(
    repo: &Arc<ReadonlyRepo>,
    workspace_root: &Path,
    jj_dir: &Path,
    working_copy_factory: &dyn WorkingCopyFactory,
    workspace_name: WorkspaceNameBuf,
    start_commit: &Commit,
) -> Result<(Box<dyn WorkingCopy>, Arc<ReadonlyRepo>), WorkspaceInitError> {
    init_working_copy_with_parents(
        repo,
        workspace_root,
        jj_dir,
        working_copy_factory,
        workspace_name,
        std::slice::from_ref(start_commit),
    )
    .await
}

async fn init_working_copy_with_parents(
    repo: &Arc<ReadonlyRepo>,
    workspace_root: &Path,
    jj_dir: &Path,
    working_copy_factory: &dyn WorkingCopyFactory,
    workspace_name: WorkspaceNameBuf,
    start_commits: &[Commit],
) -> Result<(Box<dyn WorkingCopy>, Arc<ReadonlyRepo>), WorkspaceInitError> {
    let working_copy_state_path = jj_dir.join("working_copy");
    std::fs::create_dir(&working_copy_state_path).context(&working_copy_state_path)?;

    let root_commit;
    let start_commits = if start_commits.is_empty() {
        root_commit = repo.store().root_commit();
        std::slice::from_ref(&root_commit)
    } else {
        start_commits
    };

    let mut tx = repo.start_transaction();
    match start_commits {
        [start_commit] => {
            tx.repo_mut()
                .check_out(workspace_name.clone(), start_commit)
                .await?;
        }
        start_commits => {
            let tree = merge_commit_trees(tx.repo(), start_commits)
                .await
                .map_err(CheckOutCommitError::CreateCommit)?;
            let parent_ids = start_commits
                .iter()
                .map(|commit| commit.id().clone())
                .collect();
            let wc_commit = tx
                .repo_mut()
                .new_commit(parent_ids, tree)
                .write()
                .await
                .map_err(CheckOutCommitError::CreateCommit)?;
            tx.repo_mut()
                .edit(workspace_name.clone(), &wc_commit)
                .await
                .map_err(CheckOutCommitError::EditCommit)?;
        }
    }
    let repo = tx
        .commit(format!("add workspace '{}'", workspace_name.as_symbol()))
        .await?;

    let mut working_copy = working_copy_factory.init_working_copy(
        repo.store().clone(),
        workspace_root.to_path_buf(),
        working_copy_state_path.clone(),
        repo.op_id().clone(),
        workspace_name,
        repo.settings(),
    )?;
    if let Some(wc_commit_id) = repo
        .view()
        .get_wc_commit_id(working_copy.workspace_name())
        .cloned()
    {
        let wc_commit = repo
            .store()
            .get_commit_async(&wc_commit_id)
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        let mut locked_wc = working_copy.start_mutation().await?;
        locked_wc.check_out(&wc_commit).await?;
        working_copy = locked_wc.finish(repo.op_id().clone()).await?;
    }
    let working_copy_type_path = working_copy_state_path.join("type");
    fs::write(&working_copy_type_path, working_copy.name()).context(&working_copy_type_path)?;
    Ok((working_copy, repo))
}

fn vex_clone_workspace_name(workspace_root: &Path) -> WorkspaceNameBuf {
    let root_name = workspace_root
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("clone");
    let sanitized_root_name = root_name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .take(32)
        .collect::<String>();
    let root_name = if sanitized_root_name.is_empty() {
        "clone"
    } else {
        &sanitized_root_name
    };
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("vex-{root_name}-{}-{timestamp:x}", std::process::id()).into()
}

impl Workspace {
    pub fn new(
        workspace_root: &Path,
        repo_path: PathBuf,
        working_copy: Box<dyn WorkingCopy>,
        repo_loader: RepoLoader,
    ) -> Result<Self, PathError> {
        let workspace_root = dunce::canonicalize(workspace_root).context(workspace_root)?;
        Ok(Self::new_no_canonicalize(
            workspace_root,
            repo_path,
            working_copy,
            repo_loader,
        ))
    }

    pub fn new_no_canonicalize(
        workspace_root: PathBuf,
        repo_path: PathBuf,
        working_copy: Box<dyn WorkingCopy>,
        repo_loader: RepoLoader,
    ) -> Self {
        Self {
            workspace_root,
            repo_path,
            repo_loader,
            working_copy,
        }
    }

    pub async fn init_simple(
        user_settings: &UserSettings,
        workspace_root: &Path,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let backend_initializer: &BackendInitializer =
            &|_settings, store_path| Ok(Box::new(SimpleBackend::init(store_path)));
        let signer = Signer::from_settings(user_settings)?;
        Self::init_with_backend(user_settings, workspace_root, backend_initializer, signer).await
    }

    /// Initializes a workspace with a new Git backend and bare Git repo in
    /// `.jj/repo/store/git`.
    #[cfg(feature = "git")]
    pub async fn init_internal_git(
        user_settings: &UserSettings,
        workspace_root: &Path,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let backend_initializer: &BackendInitializer = &|settings, store_path| {
            Ok(Box::new(crate::git_backend::GitBackend::init_internal(
                settings, store_path,
            )?))
        };
        let signer = Signer::from_settings(user_settings)?;
        Self::init_with_backend(user_settings, workspace_root, backend_initializer, signer).await
    }

    /// Initializes a workspace with a new Git backend and Git repo that shares
    /// the same working copy.
    #[cfg(feature = "git")]
    pub async fn init_colocated_git(
        user_settings: &UserSettings,
        workspace_root: &Path,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let backend_initializer = |settings: &UserSettings,
                                   store_path: &Path|
         -> Result<Box<dyn crate::backend::Backend>, _> {
            // TODO: Clean up path normalization. store_path is canonicalized by
            // ReadonlyRepo::init(). workspace_root will be canonicalized by
            // Workspace::new(), but it's not yet here.
            let store_relative_workspace_root =
                if let Ok(workspace_root) = dunce::canonicalize(workspace_root) {
                    crate::file_util::relative_path(store_path, &workspace_root)
                } else {
                    workspace_root.to_owned()
                };
            let backend = crate::git_backend::GitBackend::init_colocated(
                settings,
                store_path,
                &store_relative_workspace_root,
            )?;
            Ok(Box::new(backend))
        };
        let signer = Signer::from_settings(user_settings)?;
        Self::init_with_backend(user_settings, workspace_root, &backend_initializer, signer).await
    }

    /// Initializes a workspace with an existing Git repo at the specified path.
    ///
    /// The `git_repo_path` usually ends with `.git`. It's the path to the Git
    /// repo directory, not the working directory.
    #[cfg(feature = "git")]
    pub async fn init_external_git(
        user_settings: &UserSettings,
        workspace_root: &Path,
        git_repo_path: &Path,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let backend_initializer = |settings: &UserSettings,
                                   store_path: &Path|
         -> Result<Box<dyn crate::backend::Backend>, _> {
            // If the git repo is inside the workspace, use a relative path to it so the
            // whole workspace can be moved without breaking.
            // TODO: Clean up path normalization. store_path is canonicalized by
            // ReadonlyRepo::init(). workspace_root will be canonicalized by
            // Workspace::new(), but it's not yet here.
            let store_relative_git_repo_path = match (
                dunce::canonicalize(workspace_root),
                crate::git_backend::canonicalize_git_repo_path(git_repo_path),
            ) {
                (Ok(workspace_root), Ok(git_repo_path))
                    if git_repo_path.starts_with(&workspace_root) =>
                {
                    crate::file_util::relative_path(store_path, &git_repo_path)
                }
                _ => git_repo_path.to_owned(),
            };
            let backend = crate::git_backend::GitBackend::init_external(
                settings,
                store_path,
                &store_relative_git_repo_path,
            )?;
            Ok(Box::new(backend))
        };
        let signer = Signer::from_settings(user_settings)?;
        Self::init_with_backend(user_settings, workspace_root, &backend_initializer, signer).await
    }

    pub async fn init_vex(
        user_settings: &UserSettings,
        workspace_root: &Path,
        config: VexRepoConfig,
        working_copy_factory: &dyn WorkingCopyFactory,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let jj_dir = create_jj_dir(workspace_root)?;
        async {
            let repo_dir = jj_dir.join("repo");
            std::fs::create_dir(&repo_dir).context(&repo_dir)?;
            config
                .write_to_repo_path(&repo_dir)
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;

            let backend_config = config.clone();
            let op_store_config = config.clone();
            let op_heads_config = config.clone();
            let signer = Signer::from_settings(user_settings)?;
            let repo = ReadonlyRepo::init(
                user_settings,
                &repo_dir,
                &move |_settings, _store_path| {
                    Ok(Box::new(VexBackend::init(backend_config.clone())?))
                },
                signer,
                &move |_settings, _store_path, root_data| {
                    Ok(Box::new(VexOpStore::init(
                        op_store_config.clone(),
                        root_data,
                    )?))
                },
                &move |_settings, _store_path| {
                    Ok(Box::new(VexOpHeadsStore::init(op_heads_config.clone())?))
                },
                ReadonlyRepo::default_index_store_initializer(),
                ReadonlyRepo::default_submodule_store_initializer(),
            )
            .await
            .map_err(|repo_init_err| match repo_init_err {
                RepoInitError::Backend(err) => WorkspaceInitError::Backend(err),
                RepoInitError::OpHeadsStore(err) => WorkspaceInitError::OpHeadsStore(err),
                RepoInitError::Path(err) => WorkspaceInitError::Path(err),
            })?;
            let workspace_store = SimpleWorkspaceStore::load(&repo_dir)?;
            let (working_copy, repo) = init_working_copy(
                &repo,
                workspace_root,
                &jj_dir,
                working_copy_factory,
                WorkspaceName::DEFAULT.to_owned(),
            )
            .await?;
            let repo_loader = repo.loader().clone();
            let repo_dir = dunce::canonicalize(&repo_dir).context(&repo_dir)?;
            let workspace = Self::new(workspace_root, repo_dir, working_copy, repo_loader)?;
            workspace_store.add(workspace.workspace_name(), workspace.workspace_root())?;
            Ok((workspace, repo))
        }
        .await
        .inspect_err(|_err| {
            std::fs::remove_dir_all(jj_dir).ok();
        })
    }

    pub async fn clone_vex(
        user_settings: &UserSettings,
        workspace_root: &Path,
        config: VexRepoConfig,
        blob_mode: CloneBlobMode,
        working_copy_factory: &dyn WorkingCopyFactory,
        progress: Option<&crate::vex::CloneProgressFn>,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        if let Some(progress) = progress {
            progress(crate::vex::CloneProgress::Connecting);
        }
        let jj_dir = create_jj_dir(workspace_root)?;
        async {
            let repo_dir = jj_dir.join("repo");
            std::fs::create_dir(&repo_dir).context(&repo_dir)?;
            config
                .write_to_repo_path(&repo_dir)
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;

            let store_path = repo_dir.join("store");
            std::fs::create_dir(&store_path).context(&store_path)?;
            fs::write(store_path.join("type"), VexBackend::name_static())
                .context(store_path.join("type"))?;

            let op_store_path = repo_dir.join("op_store");
            std::fs::create_dir(&op_store_path).context(&op_store_path)?;
            fs::write(op_store_path.join("type"), VexOpStore::name_static())
                .context(op_store_path.join("type"))?;

            let op_heads_path = repo_dir.join("op_heads");
            std::fs::create_dir(&op_heads_path).context(&op_heads_path)?;
            fs::write(op_heads_path.join("type"), VexOpHeadsStore::name_static())
                .context(op_heads_path.join("type"))?;

            let index_path = repo_dir.join("index");
            std::fs::create_dir(&index_path).context(&index_path)?;
            let index_store =
                ReadonlyRepo::default_index_store_initializer()(user_settings, &index_path)
                    .map_err(WorkspaceInitError::Backend)?;
            fs::write(index_path.join("type"), index_store.name())
                .context(index_path.join("type"))?;

            let submodule_store_path = repo_dir.join("submodule_store");
            std::fs::create_dir(&submodule_store_path).context(&submodule_store_path)?;
            let submodule_store = ReadonlyRepo::default_submodule_store_initializer()(
                user_settings,
                &submodule_store_path,
            )
            .map_err(WorkspaceInitError::Backend)?;
            fs::write(submodule_store_path.join("type"), submodule_store.name())
                .context(submodule_store_path.join("type"))?;

            let prefetch_client = crate::vex::VexClient::from_store_path(&store_path)
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            let clone_manifest = prefetch_client
                .get_clone_manifest(blob_mode)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            if let Some(progress) = progress {
                let pack_objects = clone_manifest
                    .packs
                    .iter()
                    .map(|pack| pack.objects.len() as u64)
                    .sum();
                let total_bytes = clone_manifest
                    .packs
                    .iter()
                    .map(|pack| pack.size_bytes)
                    .sum::<u64>()
                    + clone_manifest
                        .objects
                        .iter()
                        .filter_map(|object| object.size_bytes)
                        .sum::<u64>();
                progress(crate::vex::CloneProgress::ManifestReady {
                    packs: clone_manifest.packs.len() as u64,
                    pack_objects,
                    loose_objects: clone_manifest.objects.len() as u64,
                    total_bytes,
                    deferred_objects: clone_manifest.deferred_object_count,
                });
            }
            prefetch_client
                .prefetch_clone_manifest(&clone_manifest, progress)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;

            let mut store_factories = StoreFactories::default();
            store_factories.merge(create_store_factories());
            let repo_loader =
                RepoLoader::init_from_file_system(user_settings, &repo_dir, &store_factories)
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            let repo = repo_loader
                .load_at_head()
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            let workspace_store = SimpleWorkspaceStore::load(&repo_dir)?;
            let start_commit = clone_vex_start_commit(&repo).await?;
            if let Some(progress) = progress {
                progress(crate::vex::CloneProgress::CheckingOut);
            }
            let workspace_name = vex_clone_workspace_name(workspace_root);
            let (working_copy, repo) = init_working_copy_at(
                &repo,
                workspace_root,
                &jj_dir,
                working_copy_factory,
                workspace_name,
                &start_commit,
            )
            .await?;
            let repo_loader = repo.loader().clone();
            let repo_dir = dunce::canonicalize(&repo_dir).context(&repo_dir)?;
            let workspace = Self::new(workspace_root, repo_dir, working_copy, repo_loader)?;
            workspace_store.add(workspace.workspace_name(), workspace.workspace_root())?;
            if let Some(progress) = progress {
                progress(crate::vex::CloneProgress::Done);
            }
            Ok((workspace, repo))
        }
        .await
        .inspect_err(|_err| {
            std::fs::remove_dir_all(jj_dir).ok();
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn init_with_factories(
        user_settings: &UserSettings,
        workspace_root: &Path,
        backend_initializer: &BackendInitializer<'_>,
        signer: Signer,
        op_store_initializer: &OpStoreInitializer<'_>,
        op_heads_store_initializer: &OpHeadsStoreInitializer<'_>,
        index_store_initializer: &IndexStoreInitializer<'_>,
        submodule_store_initializer: &SubmoduleStoreInitializer<'_>,
        working_copy_factory: &dyn WorkingCopyFactory,
        workspace_name: WorkspaceNameBuf,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let jj_dir = create_jj_dir(workspace_root)?;
        async {
            let repo_dir = jj_dir.join("repo");
            std::fs::create_dir(&repo_dir).context(&repo_dir)?;
            let repo = ReadonlyRepo::init(
                user_settings,
                &repo_dir,
                backend_initializer,
                signer,
                op_store_initializer,
                op_heads_store_initializer,
                index_store_initializer,
                submodule_store_initializer,
            )
            .await
            .map_err(|repo_init_err| match repo_init_err {
                RepoInitError::Backend(err) => WorkspaceInitError::Backend(err),
                RepoInitError::OpHeadsStore(err) => WorkspaceInitError::OpHeadsStore(err),
                RepoInitError::Path(err) => WorkspaceInitError::Path(err),
            })?;
            let workspace_store = SimpleWorkspaceStore::load(&repo_dir)?;
            let (working_copy, repo) = init_working_copy(
                &repo,
                workspace_root,
                &jj_dir,
                working_copy_factory,
                workspace_name,
            )
            .await?;
            let repo_loader = repo.loader().clone();
            let repo_dir = dunce::canonicalize(&repo_dir).context(&repo_dir)?;
            let workspace = Self::new(workspace_root, repo_dir, working_copy, repo_loader)?;
            workspace_store.add(workspace.workspace_name(), workspace.workspace_root())?;
            Ok((workspace, repo))
        }
        .await
        .inspect_err(|_err| {
            std::fs::remove_dir_all(jj_dir).ok();
        })
    }

    pub async fn init_with_backend(
        user_settings: &UserSettings,
        workspace_root: &Path,
        backend_initializer: &BackendInitializer<'_>,
        signer: Signer,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        Self::init_with_factories(
            user_settings,
            workspace_root,
            backend_initializer,
            signer,
            ReadonlyRepo::default_op_store_initializer(),
            ReadonlyRepo::default_op_heads_store_initializer(),
            ReadonlyRepo::default_index_store_initializer(),
            ReadonlyRepo::default_submodule_store_initializer(),
            &*default_working_copy_factory(),
            WorkspaceName::DEFAULT.to_owned(),
        )
        .await
    }

    pub async fn init_workspace_with_existing_repo(
        workspace_root: &Path,
        repo_path: &Path,
        repo: &Arc<ReadonlyRepo>,
        working_copy_factory: &dyn WorkingCopyFactory,
        workspace_name: WorkspaceNameBuf,
    ) -> Result<(Self, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let jj_dir = create_jj_dir(workspace_root)?;

        let repo_dir = dunce::canonicalize(repo_path).context(repo_path)?;
        let jj_dir_abs = dunce::canonicalize(&jj_dir).context(&jj_dir)?;
        let path_to_store = file_util::relative_path(&jj_dir_abs, &repo_dir);
        let path_to_store = if path_to_store.is_relative() {
            file_util::slash_path(&path_to_store).into_owned()
        } else {
            path_to_store
        };
        let repo_dir_bytes =
            file_util::path_to_bytes(&path_to_store).map_err(WorkspaceInitError::EncodeRepoPath)?;
        let repo_file_path = jj_dir.join("repo");
        fs::write(&repo_file_path, repo_dir_bytes).context(&repo_file_path)?;

        let workspace_store = SimpleWorkspaceStore::load(repo_path)?;
        let (working_copy, repo) = init_working_copy(
            repo,
            workspace_root,
            &jj_dir,
            working_copy_factory,
            workspace_name,
        )
        .await?;
        let workspace = Self::new(
            workspace_root,
            repo_dir,
            working_copy,
            repo.loader().clone(),
        )?;
        workspace_store.add(workspace.workspace_name(), workspace.workspace_root())?;
        Ok((workspace, repo))
    }

    pub fn load(
        user_settings: &UserSettings,
        workspace_path: &Path,
        store_factories: &StoreFactories,
        working_copy_factories: &WorkingCopyFactories,
    ) -> Result<Self, WorkspaceLoadError> {
        let loader = DefaultWorkspaceLoader::new(workspace_path)?;
        let workspace = loader.load(user_settings, store_factories, working_copy_factories)?;
        Ok(workspace)
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub fn workspace_name(&self) -> &WorkspaceName {
        self.working_copy.workspace_name()
    }

    pub fn repo_path(&self) -> &Path {
        &self.repo_path
    }

    pub fn repo_loader(&self) -> &RepoLoader {
        &self.repo_loader
    }

    /// Settings for this workspace.
    pub fn settings(&self) -> &UserSettings {
        self.repo_loader.settings()
    }

    pub fn working_copy(&self) -> &dyn WorkingCopy {
        self.working_copy.as_ref()
    }

    pub async fn start_working_copy_mutation(
        &mut self,
    ) -> Result<LockedWorkspace<'_>, WorkingCopyStateError> {
        let locked_wc = self.working_copy.start_mutation().await?;
        Ok(LockedWorkspace {
            base: self,
            locked_wc,
        })
    }

    pub async fn check_out(
        &mut self,
        operation_id: OperationId,
        old_tree: Option<&MergedTree>,
        commit: &Commit,
    ) -> Result<CheckoutStats, CheckoutError> {
        let mut locked_ws = self.start_working_copy_mutation().await?;
        // Check if the current working-copy commit has changed on disk compared to what
        // the caller expected. It's safe to check out another commit
        // regardless, but it's probably not what  the caller wanted, so we let
        // them know.
        if let Some(old_tree) = old_tree
            && old_tree.tree_ids_and_labels()
                != locked_ws.locked_wc().old_tree().tree_ids_and_labels()
        {
            return Err(CheckoutError::ConcurrentCheckout);
        }
        let stats = locked_ws.locked_wc().check_out(commit).await?;
        locked_ws
            .finish(operation_id)
            .await
            .map_err(|err| CheckoutError::Other {
                message: "Failed to save the working copy state".to_string(),
                err: err.into(),
            })?;
        Ok(stats)
    }
}

pub struct LockedWorkspace<'a> {
    base: &'a mut Workspace,
    locked_wc: Box<dyn LockedWorkingCopy>,
}

impl LockedWorkspace<'_> {
    pub fn locked_wc(&mut self) -> &mut dyn LockedWorkingCopy {
        self.locked_wc.as_mut()
    }

    pub async fn finish(self, operation_id: OperationId) -> Result<(), WorkingCopyStateError> {
        let new_wc = self.locked_wc.finish(operation_id).await?;
        self.base.working_copy = new_wc;
        Ok(())
    }
}

// Factory trait to build WorkspaceLoaders given the workspace root.
pub trait WorkspaceLoaderFactory {
    fn create(&self, workspace_root: &Path)
    -> Result<Box<dyn WorkspaceLoader>, WorkspaceLoadError>;
}

pub fn get_working_copy_factory<'a>(
    workspace_loader: &dyn WorkspaceLoader,
    working_copy_factories: &'a WorkingCopyFactories,
) -> Result<&'a dyn WorkingCopyFactory, StoreLoadError> {
    let working_copy_type = workspace_loader.get_working_copy_type()?;

    if let Some(factory) = working_copy_factories.get(&working_copy_type) {
        Ok(factory.as_ref())
    } else {
        Err(StoreLoadError::UnsupportedType {
            store: "working copy",
            store_type: working_copy_type.clone(),
        })
    }
}

async fn clone_vex_start_commit(repo: &Arc<ReadonlyRepo>) -> Result<Commit, WorkspaceInitError> {
    let mut head_ids = repo.view().heads().iter().cloned().collect::<Vec<_>>();
    if head_ids.is_empty() {
        return Ok(repo.store().root_commit());
    }
    head_ids.sort();
    let head_id_set = head_ids.iter().cloned().collect::<HashSet<_>>();
    for bookmark_name in ["main", "master", "trunk"] {
        let target = repo.view().get_local_bookmark(bookmark_name.as_ref());
        if let Some(head_id) = target.as_normal().filter(|id| head_id_set.contains(*id)) {
            return repo
                .store()
                .get_commit_async(head_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())));
        }
    }
    for (_, target) in repo.view().local_bookmarks() {
        if let Some(head_id) = target.as_normal().filter(|id| head_id_set.contains(*id)) {
            return repo
                .store()
                .get_commit_async(head_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())));
        }
    }
    if let Some(head_id) = repo
        .view()
        .git_head()
        .as_normal()
        .filter(|id| head_id_set.contains(*id))
    {
        return repo
            .store()
            .get_commit_async(head_id)
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())));
    }
    if let Some(commit) = clone_vex_recent_workspace_commit_from_ops(repo).await? {
        return Ok(commit);
    }
    for head_id in repo.view().wc_commit_ids().values() {
        if head_id_set.contains(head_id) {
            let commit = repo
                .store()
                .get_commit_async(head_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            return clone_vex_peel_discardable_wc_commit(repo, commit).await;
        }
    }
    let mut selected_commit = None;
    for head_id in head_ids {
        let commit = repo
            .store()
            .get_commit_async(&head_id)
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        let should_replace = selected_commit.as_ref().is_none_or(|selected: &Commit| {
            commit.committer().timestamp.timestamp > selected.committer().timestamp.timestamp
        });
        if should_replace {
            selected_commit = Some(commit);
        }
    }
    Ok(selected_commit.expect("non-empty heads should produce a checkout target"))
}

async fn clone_vex_recent_workspace_commit_from_ops(
    repo: &Arc<ReadonlyRepo>,
) -> Result<Option<Commit>, WorkspaceInitError> {
    let mut to_visit = vec![repo.operation().clone()];
    let mut visited = HashSet::new();
    let mut selected_commit = None;
    let mut selected_operation = None;
    let mut selected_timestamp = None;
    while let Some(operation) = to_visit.pop() {
        if !visited.insert(operation.id().clone()) {
            continue;
        }
        if let Some(workspace_name) = operation.metadata().workspace_name.as_ref() {
            let view = operation
                .view()
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            if let Some(commit_id) = view.get_wc_commit_id(workspace_name.as_ref()) {
                let commit = repo
                    .store()
                    .get_commit_async(commit_id)
                    .await
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
                let commit = clone_vex_peel_discardable_wc_commit(repo, commit).await?;
                let operation_id = operation.id().clone();
                let timestamp = operation.metadata().time.end.timestamp;
                let should_replace = selected_timestamp
                    .zip(selected_operation.as_ref())
                    .is_none_or(|(selected_timestamp, selected_operation)| {
                        (timestamp, &operation_id) > (selected_timestamp, selected_operation)
                    });
                if should_replace {
                    selected_commit = Some(commit);
                    selected_operation = Some(operation_id);
                    selected_timestamp = Some(timestamp);
                }
            }
        }
        let parents = operation
            .parents()
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        to_visit.extend(parents);
    }
    Ok(selected_commit)
}

async fn clone_vex_peel_discardable_wc_commit(
    repo: &Arc<ReadonlyRepo>,
    mut commit: Commit,
) -> Result<Commit, WorkspaceInitError> {
    loop {
        let discardable = commit
            .is_discardable(repo.as_ref())
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        if !discardable {
            return Ok(commit);
        }
        let [parent_id] = commit.parent_ids() else {
            return Ok(commit);
        };
        let parent = repo
            .store()
            .get_commit_async(parent_id)
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        if parent.id() == repo.store().root_commit_id() {
            return Ok(commit);
        }
        commit = parent;
    }
}

// Loader assigned to a specific workspace root that knows how to load a
// Workspace object for that path.
pub trait WorkspaceLoader {
    // The root of the Workspace to be loaded.
    fn workspace_root(&self) -> &Path;

    // The path to the repo/ dir for this Workspace.
    fn repo_path(&self) -> &Path;

    // Loads the specified Workspace with the provided factories.
    fn load(
        &self,
        user_settings: &UserSettings,
        store_factories: &StoreFactories,
        working_copy_factories: &WorkingCopyFactories,
    ) -> Result<Workspace, WorkspaceLoadError>;

    // Returns the type identifier for the WorkingCopy trait in this Workspace.
    fn get_working_copy_type(&self) -> Result<String, StoreLoadError>;
}

pub struct DefaultWorkspaceLoaderFactory;

impl WorkspaceLoaderFactory for DefaultWorkspaceLoaderFactory {
    fn create(
        &self,
        workspace_root: &Path,
    ) -> Result<Box<dyn WorkspaceLoader>, WorkspaceLoadError> {
        Ok(Box::new(DefaultWorkspaceLoader::new(workspace_root)?))
    }
}

/// Helps create a `Workspace` instance by reading `.jj/repo/` and
/// `.jj/working_copy/` from the file system.
#[derive(Clone, Debug)]
struct DefaultWorkspaceLoader {
    workspace_root: PathBuf,
    repo_path: PathBuf,
    working_copy_state_path: PathBuf,
}

pub type WorkingCopyFactories = HashMap<String, Box<dyn WorkingCopyFactory>>;

impl DefaultWorkspaceLoader {
    pub fn new(workspace_root: &Path) -> Result<Self, WorkspaceLoadError> {
        let jj_dir = workspace_root.join(".jj");
        if !jj_dir.is_dir() {
            return Err(WorkspaceLoadError::NoWorkspaceHere(
                workspace_root.to_owned(),
            ));
        }
        let mut repo_dir = jj_dir.join("repo");
        // If .jj/repo is a file, then we interpret its contents as a relative path to
        // the actual repo directory (typically in another workspace).
        if repo_dir.is_file() {
            let buf = fs::read(&repo_dir).context(&repo_dir)?;
            let repo_path =
                file_util::path_from_bytes(&buf).map_err(WorkspaceLoadError::DecodeRepoPath)?;
            repo_dir = dunce::canonicalize(jj_dir.join(repo_path)).context(repo_path)?;
            if !repo_dir.is_dir() {
                return Err(WorkspaceLoadError::RepoDoesNotExist(repo_dir));
            }
        }
        let working_copy_state_path = jj_dir.join("working_copy");
        Ok(Self {
            workspace_root: workspace_root.to_owned(),
            repo_path: repo_dir,
            working_copy_state_path,
        })
    }
}

impl WorkspaceLoader for DefaultWorkspaceLoader {
    fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    fn repo_path(&self) -> &Path {
        &self.repo_path
    }

    fn load(
        &self,
        user_settings: &UserSettings,
        store_factories: &StoreFactories,
        working_copy_factories: &WorkingCopyFactories,
    ) -> Result<Workspace, WorkspaceLoadError> {
        let repo_loader =
            RepoLoader::init_from_file_system(user_settings, &self.repo_path, store_factories)?;
        let working_copy_factory = get_working_copy_factory(self, working_copy_factories)?;
        let working_copy = working_copy_factory.load_working_copy(
            repo_loader.store().clone(),
            self.workspace_root.clone(),
            self.working_copy_state_path.clone(),
            user_settings,
        )?;
        let workspace = Workspace::new(
            &self.workspace_root,
            self.repo_path.clone(),
            working_copy,
            repo_loader,
        )?;
        Ok(workspace)
    }

    fn get_working_copy_type(&self) -> Result<String, StoreLoadError> {
        read_store_type("working copy", self.working_copy_state_path.join("type"))
    }
}

pub fn default_working_copy_factories() -> WorkingCopyFactories {
    let mut factories = WorkingCopyFactories::new();
    factories.insert(
        LocalWorkingCopy::name().to_owned(),
        Box::new(LocalWorkingCopyFactory {}),
    );
    factories.insert(
        VirtualWorkingCopy::name().to_owned(),
        Box::new(VirtualWorkingCopyFactory),
    );
    factories
}

pub fn default_working_copy_factory() -> Box<dyn WorkingCopyFactory> {
    Box::new(LocalWorkingCopyFactory {})
}

#[cfg(test)]
mod tests {
    use pollster::FutureExt as _;
    use tempfile::TempDir;

    use super::*;
    use crate::config::ConfigLayer;
    use crate::config::ConfigSource;
    use crate::config::StackedConfig;

    fn user_settings() -> UserSettings {
        let config_text = r#"
            user.name = "Test User"
            user.email = "test.user@example.com"
            operation.username = "test-username"
            operation.hostname = "host.example.com"
            debug.randomness-seed = 42
        "#;
        let mut config = StackedConfig::with_defaults();
        config.add_layer(ConfigLayer::parse(ConfigSource::User, config_text).unwrap());
        UserSettings::from_config(config).unwrap()
    }

    fn init_test_repo(
        settings: &UserSettings,
    ) -> Result<(TempDir, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let temp_dir = tempfile::Builder::new()
            .prefix("jj-test-")
            .tempdir()
            .unwrap();
        let repo_dir = temp_dir.path().join("repo");
        std::fs::create_dir(&repo_dir).unwrap();
        let signer = Signer::from_settings(settings)?;
        let repo = ReadonlyRepo::init(
            settings,
            &repo_dir,
            &|_settings, store_path| Ok(Box::new(SimpleBackend::init(store_path))),
            signer,
            ReadonlyRepo::default_op_store_initializer(),
            ReadonlyRepo::default_op_heads_store_initializer(),
            ReadonlyRepo::default_index_store_initializer(),
            ReadonlyRepo::default_submodule_store_initializer(),
        )
        .block_on()
        .map_err(|repo_init_err| match repo_init_err {
            RepoInitError::Backend(err) => WorkspaceInitError::Backend(err),
            RepoInitError::OpHeadsStore(err) => WorkspaceInitError::OpHeadsStore(err),
            RepoInitError::Path(err) => WorkspaceInitError::Path(err),
        })?;
        Ok((temp_dir, repo))
    }

    #[test]
    fn test_clone_vex_start_commit_prefers_main_bookmark() -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let fallback_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("fallback")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let main_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("main")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "main".as_ref(),
            crate::op_store::RefTarget::normal(main_head.id().clone()),
        );
        let repo = tx.commit("create multiple heads").block_on()?;

        let start_commit = clone_vex_start_commit(&repo).block_on()?;
        assert_eq!(start_commit.id(), main_head.id());
        assert_ne!(start_commit.id(), fallback_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_prefers_recent_workspace_operation()
    -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let default_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("default")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let other_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("other")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut()
            .set_wc_commit(WorkspaceName::DEFAULT.to_owned(), default_head.id().clone())
            .map_err(|err| CheckOutCommitError::EditCommit(err.into()))?;
        tx.set_workspace_name(WorkspaceName::DEFAULT);
        let repo = tx.commit("record default workspace").block_on()?;
        std::thread::sleep(std::time::Duration::from_millis(1));

        let mut tx = repo.start_transaction();
        tx.repo_mut()
            .set_wc_commit("secondary".into(), other_head.id().clone())
            .map_err(|err| CheckOutCommitError::EditCommit(err.into()))?;
        tx.set_workspace_name("secondary".as_ref());
        let repo = tx.commit("record secondary workspace").block_on()?;

        let start_commit = clone_vex_start_commit(&repo).block_on()?;
        assert_eq!(start_commit.id(), other_head.id());
        assert_ne!(start_commit.id(), default_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_peels_discardable_workspace_heads()
    -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let base = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("base")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let empty_wc = tx
            .repo_mut()
            .new_commit(vec![base.id().clone()], base.tree())
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut()
            .set_wc_commit(WorkspaceName::DEFAULT.to_owned(), empty_wc.id().clone())
            .map_err(|err| CheckOutCommitError::EditCommit(err.into()))?;
        let repo = tx.commit("record discardable workspace").block_on()?;

        let start_commit = clone_vex_start_commit(&repo).block_on()?;
        assert_eq!(start_commit.id(), base.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_uses_newest_head_without_refs() -> Result<(), WorkspaceInitError>
    {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let older_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("older")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        std::thread::sleep(std::time::Duration::from_millis(1));
        let newer_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("newer")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let repo = tx.commit("create anonymous heads").block_on()?;

        let start_commit = clone_vex_start_commit(&repo).block_on()?;
        assert_eq!(start_commit.id(), newer_head.id());
        assert_ne!(start_commit.id(), older_head.id());
        Ok(())
    }

    #[test]
    fn test_init_working_copy_with_parents_creates_merge_wc_commit()
    -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let parent1 = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("parent1")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let parent2 = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("parent2")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let repo = tx.commit("create clone heads").block_on()?;

        let temp_dir = tempfile::Builder::new()
            .prefix("jj-test-")
            .tempdir()
            .unwrap();
        let workspace_root = temp_dir.path().join("clone");
        std::fs::create_dir(&workspace_root).unwrap();
        let jj_dir = create_jj_dir(&workspace_root)?;

        let (_working_copy, repo) = init_working_copy_with_parents(
            &repo,
            &workspace_root,
            &jj_dir,
            &*default_working_copy_factory(),
            WorkspaceName::DEFAULT.to_owned(),
            &[parent1.clone(), parent2.clone()],
        )
        .block_on()?;

        let wc_commit_id = repo
            .view()
            .get_wc_commit_id(WorkspaceName::DEFAULT)
            .unwrap();
        let wc_commit = repo
            .store()
            .get_commit(wc_commit_id)
            .map_err(CheckOutCommitError::CreateCommit)?;
        let expected_tree = merge_commit_trees(repo.as_ref(), &[parent1.clone(), parent2.clone()])
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        assert_eq!(
            wc_commit.parent_ids(),
            [parent1.id().clone(), parent2.id().clone()]
        );
        assert_eq!(
            wc_commit.tree().tree_ids_and_labels(),
            expected_tree.tree_ids_and_labels()
        );
        Ok(())
    }
}
