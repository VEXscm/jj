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
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering as AtomicOrdering;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use sha2::Digest as _;
use sha2::Sha256;
use thiserror::Error;

use crate::backend::BackendInitError;
use crate::backend::ChangeId;
use crate::backend::CommitId;
use crate::backend::TreeValue;
use crate::commit::Commit;
use crate::file_util;
use crate::file_util::BadPathEncoding;
use crate::file_util::IoResultExt as _;
use crate::file_util::PathError;
use crate::local_working_copy::LocalWorkingCopy;
use crate::local_working_copy::LocalWorkingCopyFactory;
use crate::merge::Merge;
use crate::merged_tree::MergedTree;
use crate::object_id::ObjectId as _;
use crate::op_heads_store::OpHeadsStoreError;
use crate::op_store::OperationId;
use crate::op_store::RefTarget;
use crate::op_store::RemoteRef;
use crate::op_store::RemoteRefState;
use crate::ref_name::RefNameBuf;
use crate::ref_name::RemoteName;
use crate::ref_name::RemoteRefSymbol;
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
use crate::repo_path::RepoPathBuf;
use crate::rewrite::merge_commit_trees;
use crate::settings::UserSettings;
use crate::signing::SignInitError;
use crate::signing::Signer;
use crate::simple_backend::SimpleBackend;
use crate::transaction::TransactionCommitError;
use crate::tree_builder::TreeBuilder;
use crate::vex::CloneBlobMode;
use crate::vex::VexClient;
use crate::vex::VexContentId;
use crate::vex::VexFederatedHomeConfig;
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
    /// The server advertised a default branch (`server_trunk`) but no native
    /// local or remote-tracking bookmark of that name exists. `vex clone` is
    /// native-only and fails closed here, before working-copy creation: it
    /// never falls back to another branch, an arbitrary head, or `git/ref/*`
    /// (roadmap/066).
    #[error(
        "Server-advertised native trunk bookmark \"{trunk}\" was not found among this \
         repository's native bookmarks. `vex clone` is native-only and does not fall back to \
         Git refs. Complete the repository's native conversion (or repair its default branch), \
         or use `vex git clone` for a Git-protocol clone."
    )]
    NativeTrunkMissing { trunk: String },
}

fn federated_home_init_error(message: impl Into<String>) -> WorkspaceInitError {
    WorkspaceInitError::Backend(BackendInitError(Box::new(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Invalid Home checkout: {}", message.into()),
    ))))
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

/// Commits the operation that adds a workspace to `repo`.
///
/// `local_bookmark` is only used by Vex clones to create the local trunk
/// bookmark in the same operation as the workspace. Keeping those mutations
/// together means the clone can retry their single op-head CAS as one atomic
/// view update.
async fn commit_workspace_operation(
    repo: &Arc<ReadonlyRepo>,
    workspace_name: &WorkspaceNameBuf,
    start_commits: &[Commit],
    local_bookmark: Option<(&str, &CommitId)>,
) -> Result<Arc<ReadonlyRepo>, WorkspaceInitError> {
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
    if let Some((bookmark_name, commit_id)) = local_bookmark {
        tx.repo_mut().set_local_bookmark_target(
            bookmark_name.as_ref(),
            crate::op_store::RefTarget::normal(commit_id.clone()),
        );
    }
    tx.commit(format!("add workspace '{}'", workspace_name.as_symbol()))
        .await
        .map_err(Into::into)
}

async fn finish_init_working_copy(
    repo: &Arc<ReadonlyRepo>,
    workspace_root: &Path,
    working_copy_state_path: &Path,
    working_copy_factory: &dyn WorkingCopyFactory,
    workspace_name: WorkspaceNameBuf,
) -> Result<Box<dyn WorkingCopy>, WorkspaceInitError> {
    let mut working_copy = working_copy_factory.init_working_copy(
        repo.store().clone(),
        workspace_root.to_path_buf(),
        working_copy_state_path.to_path_buf(),
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
    Ok(working_copy)
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

    let repo = commit_workspace_operation(repo, &workspace_name, start_commits, None).await?;
    let working_copy = finish_init_working_copy(
        &repo,
        workspace_root,
        &working_copy_state_path,
        working_copy_factory,
        workspace_name,
    )
    .await?;
    Ok((working_copy, repo))
}

fn vex_clone_local_bookmark_to_set<'name, 'commit>(
    repo: &ReadonlyRepo,
    resolved_trunk: Option<&'name str>,
    initial_target: Option<&crate::op_store::RefTarget>,
    start_commit: &'commit Commit,
) -> Option<(&'name str, &'commit CommitId)> {
    let (name, initial_target) = resolved_trunk.zip(initial_target)?;
    let bookmark_name: &crate::ref_name::RefName = name.as_ref();
    (repo.view().get_local_bookmark(bookmark_name) == initial_target)
        .then_some((name, start_commit.id()))
}

async fn reload_vex_clone_repo_at_head(
    repo: &Arc<ReadonlyRepo>,
) -> Result<Arc<ReadonlyRepo>, WorkspaceInitError> {
    repo.reload_at_head()
        .await
        .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))
}

async fn commit_vex_clone_workspace_operation(
    repo: &Arc<ReadonlyRepo>,
    workspace_name: &WorkspaceNameBuf,
    start_commit: &Commit,
    resolved_trunk: Option<&str>,
) -> Result<Arc<ReadonlyRepo>, WorkspaceInitError> {
    let initial_resolved_trunk_target = resolved_trunk.map(|name| {
        let bookmark_name: &crate::ref_name::RefName = name.as_ref();
        repo.view().get_local_bookmark(bookmark_name).clone()
    });
    // The clone's first `load_at_head()` happens before manifest/prefetch work,
    // so refresh immediately before constructing the write transaction.
    //
    // There is no retry ladder here any more (roadmap/088 Stage 7, S11). It
    // existed because the op head was a strict single-head CAS on the server,
    // which a slow clone lost to any concurrent writer. Op heads are local now:
    // the write cannot be refused for concurrency reasons at all, so a loop
    // would have nothing to retry.
    let repo = reload_vex_clone_repo_at_head(repo).await?;
    // A clone's workspace name is fresh, but the resolved trunk bookmark is
    // shared state. If another operation moved it while this clone was
    // fetching, preserve that newer value instead of resetting it to the
    // clone's earlier checkout target.
    let local_bookmark = vex_clone_local_bookmark_to_set(
        &repo,
        resolved_trunk,
        initial_resolved_trunk_target.as_ref(),
        start_commit,
    );
    commit_workspace_operation(
        &repo,
        workspace_name,
        std::slice::from_ref(start_commit),
        local_bookmark,
    )
    .await
}

#[expect(clippy::too_many_arguments)]
async fn init_vex_clone_working_copy_at(
    repo: &Arc<ReadonlyRepo>,
    workspace_root: &Path,
    jj_dir: &Path,
    working_copy_factory: &dyn WorkingCopyFactory,
    workspace_name: WorkspaceNameBuf,
    start_commit: &Commit,
    resolved_trunk: Option<&str>,
    // Whether the workspace operation CASes the server op heads during the
    // clone. When false, the armed op-heads-store marker makes the commit below
    // record the operation locally instead of publishing it.
    register_workspace: bool,
    progress: Option<&crate::vex::CloneProgressFn>,
) -> Result<(Box<dyn WorkingCopy>, Arc<ReadonlyRepo>), WorkspaceInitError> {
    let working_copy_state_path = jj_dir.join("working_copy");
    std::fs::create_dir(&working_copy_state_path).context(&working_copy_state_path)?;

    let working_copy = working_copy_factory.init_working_copy(
        repo.store().clone(),
        workspace_root.to_path_buf(),
        working_copy_state_path.clone(),
        repo.op_id().clone(),
        workspace_name.clone(),
        repo.settings(),
    )?;
    let locked_wc = working_copy.start_mutation().await?;

    if register_workspace && let Some(progress) = progress {
        progress(crate::vex::CloneProgress::WorkspacePublish);
    }
    let publish =
        commit_vex_clone_workspace_operation(repo, &workspace_name, start_commit, resolved_trunk);
    if let Some(progress) = progress {
        progress(crate::vex::CloneProgress::CheckingOut);
    }

    // The single-parent workspace operation deterministically checks out
    // `start_commit`, so materialize that known tree while the server publishes
    // the operation. Keep the mutation locked and unfinished until the
    // published operation id is available for the working-copy state.
    let mut locked_wc = locked_wc;
    let checkout = locked_wc.check_out(start_commit);
    let (publish_result, checkout_result) = futures::join!(publish, checkout);
    let repo = match publish_result {
        Ok(repo) => repo,
        Err(error) => {
            // Unlike the old sequential order, files may now exist when
            // publishing fails. Restore the empty tree when checkout completed
            // so callers that supplied an existing empty directory don't keep
            // an unpublished worktree.
            if checkout_result.is_ok() {
                locked_wc.check_out(&repo.store().root_commit()).await.ok();
            }
            return Err(error);
        }
    };
    checkout_result?;

    // A CAS retry reloads the repo before rebuilding the workspace operation.
    // The normal single-parent result still points at `start_commit`, but
    // reconcile against the published view before finalizing in case that ever
    // changes. A missing workspace is equivalent to the old empty checkout.
    let published_wc_commit_id = repo
        .view()
        .get_wc_commit_id(workspace_name.as_ref())
        .cloned();
    match published_wc_commit_id {
        Some(commit_id) if commit_id != *start_commit.id() => {
            let wc_commit = repo
                .store()
                .get_commit_async(&commit_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            locked_wc.check_out(&wc_commit).await?;
        }
        None => {
            locked_wc.check_out(&repo.store().root_commit()).await?;
        }
        Some(_) => {}
    }
    let working_copy = locked_wc.finish(repo.op_id().clone()).await?;
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

fn skip_vex_clone_prefetch() -> bool {
    matches!(
        std::env::var("VEX_SKIP_CLONE_PREFETCH").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// Whether a lazy clone bulk-hydrates the start commit's file/symlink contents
/// before checkout. On by default; `VEX_CLONE_HYDRATION=0` restores the pure
/// per-file lazy fetch path (rollback / bench control).
fn vex_clone_hydration_enabled() -> bool {
    !matches!(
        std::env::var("VEX_CLONE_HYDRATION").ok().as_deref(),
        Some("0") | Some("false") | Some("no")
    )
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
                &move |_settings, store_path| {
                    Ok(Box::new(VexBackend::init_at(
                        backend_config.clone(),
                        store_path,
                    )?))
                },
                signer,
                &move |_settings, store_path, root_data| {
                    Ok(Box::new(VexOpStore::init(
                        op_store_config.clone(),
                        store_path,
                        root_data,
                    )?))
                },
                &move |_settings, store_path| {
                    Ok(Box::new(VexOpHeadsStore::init(
                        op_heads_config.clone(),
                        store_path,
                    )?))
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

    #[expect(clippy::too_many_arguments)]
    pub async fn clone_vex(
        user_settings: &UserSettings,
        workspace_root: &Path,
        config: VexRepoConfig,
        blob_mode: CloneBlobMode,
        // When `Some`, the working copy is checked out at this exact commit
        // instead of the native bookmark target `clone_vex_native_target`
        // would pick. CI runners use this to materialize the pipeline's
        // `commit_sha` directly.
        target_commit: Option<&CommitId>,
        // The trunk the server registered for this repo (`default_branch` from
        // the repo-access catalog): the authoritative native bookmark name.
        // On the default (`target_commit == None`) path this selects the start
        // commit through native bookmarks only; if the bookmark is absent the
        // clone fails closed with `WorkspaceInitError::NativeTrunkMissing`
        // (never `git/ref/*`), except an entirely empty native repository
        // starts at root so it can receive its first bookmark. Ignored when
        // `target_commit` is `Some`. `None`
        // falls back to the native-only main/master/trunk heuristic.
        server_trunk: Option<&str>,
        // Whether to bulk-hydrate the start commit's file/symlink contents into
        // the local cache before checkout (lazy clones only). Callers pass
        // `false` for virtual working copies, which materialize nothing — the
        // factory itself can't tell us (no identity on `WorkingCopyFactory`).
        hydrate_blobs: bool,
        // Whether to publish the workspace operation to the server during the
        // clone. When false, the workspace operation is committed locally only
        // and published transparently by the first mutating operation. Ignored
        // (always local) for `local_writes` repos.
        register_workspace: bool,
        working_copy_factory: &dyn WorkingCopyFactory,
        progress: Option<&crate::vex::CloneProgressFn>,
    ) -> Result<(Self, Arc<ReadonlyRepo>, Option<String>), WorkspaceInitError> {
        Self::clone_vex_inner(
            user_settings,
            workspace_root,
            config,
            blob_mode,
            target_commit,
            server_trunk,
            hydrate_blobs,
            register_workspace,
            None,
            working_copy_factory,
            progress,
        )
        .await
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn clone_federated_home(
        user_settings: &UserSettings,
        workspace_root: &Path,
        config: VexRepoConfig,
        blob_mode: CloneBlobMode,
        target_commit: Option<&CommitId>,
        server_trunk: Option<&str>,
        hydrate_blobs: bool,
        register_workspace: bool,
        federated_home: VexFederatedHomeConfig,
        working_copy_factory: &dyn WorkingCopyFactory,
        progress: Option<&crate::vex::CloneProgressFn>,
    ) -> Result<(Self, Arc<ReadonlyRepo>, Option<String>), WorkspaceInitError> {
        Self::clone_vex_inner(
            user_settings,
            workspace_root,
            config,
            blob_mode,
            target_commit,
            server_trunk,
            hydrate_blobs,
            register_workspace,
            Some(federated_home),
            working_copy_factory,
            progress,
        )
        .await
    }

    #[expect(clippy::too_many_arguments)]
    async fn clone_vex_inner(
        user_settings: &UserSettings,
        workspace_root: &Path,
        config: VexRepoConfig,
        blob_mode: CloneBlobMode,
        target_commit: Option<&CommitId>,
        server_trunk: Option<&str>,
        hydrate_blobs: bool,
        register_workspace: bool,
        federated_home: Option<VexFederatedHomeConfig>,
        working_copy_factory: &dyn WorkingCopyFactory,
        progress: Option<&crate::vex::CloneProgressFn>,
    ) -> Result<(Self, Arc<ReadonlyRepo>, Option<String>), WorkspaceInitError> {
        if let Some(federated_home) = federated_home.as_ref() {
            federated_home
                .validate()
                .map_err(|error| federated_home_init_error(error.to_string()))?;
            if register_workspace {
                return Err(federated_home_init_error(
                    "a synthesized flat Home snapshot cannot register a backend workspace during clone",
                ));
            }
            if skip_vex_clone_prefetch() {
                return Err(federated_home_init_error(
                    "Home clone requires snapshot prefetch; VEX_SKIP_CLONE_PREFETCH is not supported",
                ));
            }
        }
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
            // roadmap/088 Stage 7: a clone starts a *fresh local* operation
            // log, rooted at the synthetic root operation, rather than
            // inheriting the server's. Creating the directory here is also what
            // permanently disqualifies a clone from the one-time bootstrap in
            // `VexOpHeadsStore`, so a fresh clone can never read op heads from
            // the backend. The view is rebuilt from `list_refs` by the clone
            // transaction below, which is the only thing that was ever read out
            // of the published log.
            let heads_path = op_heads_path.join("heads");
            std::fs::create_dir(&heads_path).context(&heads_path)?;
            // The root operation id is 32 all-zero bytes (see `VexOpStore`).
            let root_op_hex = crate::op_store::OperationId::from_bytes(&[0; 32]).hex();
            fs::write(heads_path.join(&root_op_hex), "").context(heads_path.join(&root_op_hex))?;

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

            if config.repository_scope_kind.as_deref() != Some("virtual_repository")
                && !skip_vex_clone_prefetch()
            {
                let mut prefetch_client = crate::vex::VexClient::from_store_path(&store_path)
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
                // The `.jj` scaffold above is brand new (`create_jj_dir` fails
                // if one exists) and is removed wholesale if the clone fails,
                // so a repo-local cache dir was created by this process: the
                // unpack's loose writes may take the direct-create fast path.
                // Shared cache dirs keep atomic temp+rename writes — the
                // client checks which kind it has.
                prefetch_client.mark_fresh_clone_cache();
                let clone_manifest = prefetch_client
                    .get_clone_manifest(blob_mode, progress)
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
                if let Some(federated_home) = federated_home.as_ref() {
                    let cache_root = prefetch_client
                        .cache_root()
                        .ok_or_else(|| {
                            federated_home_init_error(
                                "flat Home clone did not initialize its single object cache",
                            )
                        })?
                        .to_path_buf();
                    for repository in federated_home.repositories.iter().skip(1) {
                        let component_config =
                            federated_home.repository_config(&config, repository);
                        let mut component_client =
                            crate::vex::VexClient::from_config_with_cache_root(
                                component_config,
                                cache_root.clone(),
                            )
                            .map_err(|err| {
                                WorkspaceInitError::Backend(BackendInitError(err.into()))
                            })?;
                        component_client.mark_fresh_clone_cache();
                        let component_manifest = component_client
                            .get_clone_manifest(CloneBlobMode::Eager, progress)
                            .await
                            .map_err(|err| {
                                WorkspaceInitError::Backend(BackendInitError(err.into()))
                            })?;
                        component_client
                            .prefetch_clone_manifest(&component_manifest, progress)
                            .await
                            .map_err(|err| {
                                WorkspaceInitError::Backend(BackendInitError(err.into()))
                            })?;
                    }
                }
            }

            if let Some(progress) = progress {
                progress(crate::vex::CloneProgress::LoadingRepo);
            }
            let mut store_factories = StoreFactories::default();
            store_factories.merge(create_store_factories());
            let repo_loader =
                RepoLoader::init_from_file_system(user_settings, &repo_dir, &store_factories)
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            // Bootstrap the commit index from the head view only (roadmap
            // 069): a fresh clone inherits the server's full op log, and the
            // full build fetches every historical op/view one RPC at a time
            // (~7.3 s of the fixture clone's 7.9 s indexing phase). Historical
            // ops remain lazily indexable. `VEX_CLONE_FULL_INDEX=1` restores
            // the exhaustive build.
            let full_index = std::env::var("VEX_CLONE_FULL_INDEX")
                .is_ok_and(|value| !value.is_empty() && value != "0");
            let before_index = || {
                if let Some(progress) = progress {
                    progress(crate::vex::CloneProgress::Indexing);
                }
            };
            let repo = if full_index {
                repo_loader
                    .load_at_head_with_before_index(before_index)
                    .await
            } else {
                repo_loader
                    .load_at_head_with_bootstrap_index(before_index)
                    .await
            }
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            let workspace_store = SimpleWorkspaceStore::load(&repo_dir)?;
            let repo = match federated_home.as_ref() {
                Some(flat_home) => seed_federated_home_clone_view(repo, flat_home).await?,
                None => seed_clone_view_from_refs(repo, &store_path).await?,
            };
            let checkout_trunk = federated_home
                .as_ref()
                .map(|flat_home| flat_home.manifest.home_bookmark.as_str())
                .or(server_trunk);
            let (mut start_commit, mut resolved_trunk) =
                clone_vex_checkout_target(&repo, target_commit, checkout_trunk).await?;
            if let Some(mut flat_home) = federated_home.clone() {
                start_commit =
                    synthesize_federated_home_base(&repo, &flat_home, &start_commit).await?;
                flat_home.aggregate_base_commit_id = Some(content_id_from_commit_id(
                    start_commit.id(),
                    "flat Home base commit",
                )?);
                // Persist routing only after the selected physical snapshots
                // have produced a real, readable aggregate native commit.
                flat_home
                    .write_to_repo_path(&repo_dir)
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
                resolved_trunk = Some(flat_home.manifest.home_bookmark.clone());
            }
            // Pre-checkout hydration: a lazy manifest defers every blob and
            // symlink, so materialization would otherwise pay one RPC per
            // file. Batch-fetch the start commit's contents into the cache
            // first (the tree metadata is already warm from the prefetch).
            // Best-effort — checkout still hydrates on demand if this fails —
            // and skipped wherever the prefetch is skipped (virtual-repository
            // scope, VEX_SKIP_CLONE_PREFETCH), where the walk itself would
            // become per-object RPCs.
            let mut hydration_file_count: Option<u64> = None;
            if hydrate_blobs
                && blob_mode == CloneBlobMode::Lazy
                && config.repository_scope_kind.as_deref() != Some("virtual_repository")
                && !skip_vex_clone_prefetch()
                && vex_clone_hydration_enabled()
            {
                hydration_file_count =
                    hydrate_start_commit_blobs(&repo, &store_path, &start_commit, progress).await;
            }
            let workspace_name = vex_clone_workspace_name(workspace_root);
            let init_working_copy = init_vex_clone_working_copy_at(
                &repo,
                workspace_root,
                &jj_dir,
                working_copy_factory,
                workspace_name,
                &start_commit,
                resolved_trunk.as_deref(),
                register_workspace,
                progress,
            );
            let (working_copy, repo) = match progress {
                // Materializing progress: the checkout has no progress channel
                // of its own, so while it runs poll the process-global
                // `files_written` counter every ~200ms and report the delta.
                // `files_total` comes from the hydration walk when it ran
                // (0 = unknown; sinks omit the total then). The ticker's timer
                // is parked on the shared gRPC runtime (this executor has no
                // timer driver) and the ticker stops within one tick of the
                // checkout future finishing, so `join!` cannot hang on it.
                Some(progress) => {
                    let files_total = hydration_file_count.unwrap_or(0);
                    let files_written_base = crate::vex::vex_client_stats_snapshot().files_written;
                    let checkout_done = AtomicBool::new(false);
                    let checkout = async {
                        let result = init_working_copy.await;
                        checkout_done.store(true, AtomicOrdering::Relaxed);
                        result
                    };
                    let ticker = async {
                        while !checkout_done.load(AtomicOrdering::Relaxed) {
                            crate::vex::shared_runtime_sleep(Duration::from_millis(200)).await;
                            if checkout_done.load(AtomicOrdering::Relaxed) {
                                break;
                            }
                            let files_done = crate::vex::vex_client_stats_snapshot()
                                .files_written
                                .saturating_sub(files_written_base);
                            progress(crate::vex::CloneProgress::Materializing {
                                files_done,
                                files_total,
                            });
                        }
                    };
                    let (checkout_result, ()) = futures::join!(checkout, ticker);
                    checkout_result?
                }
                None => init_working_copy.await?,
            };
            let repo_loader = repo.loader().clone();
            let repo_dir = dunce::canonicalize(&repo_dir).context(&repo_dir)?;
            let workspace = Self::new(workspace_root, repo_dir, working_copy, repo_loader)?;
            workspace_store.add(workspace.workspace_name(), workspace.workspace_root())?;
            if let Some(progress) = progress {
                if let Some(name) = resolved_trunk.as_ref() {
                    progress(crate::vex::CloneProgress::TrunkResolved { name: name.clone() });
                }
                progress(crate::vex::CloneProgress::Done);
            }
            Ok((workspace, repo, resolved_trunk))
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

async fn synthesize_federated_home_base(
    repo: &Arc<ReadonlyRepo>,
    config: &VexFederatedHomeConfig,
    home_commit: &Commit,
) -> Result<Commit, WorkspaceInitError> {
    if home_commit.id().hex() != config.manifest.home_revision.to_hex() {
        return Err(federated_home_init_error(format!(
            "selected Home revision {} does not match manifest revision {}",
            home_commit.id().hex(),
            config.manifest.home_revision
        )));
    }
    let mut selected_ids = Vec::with_capacity(config.manifest.components.len() + 1);
    selected_ids.push(home_commit.id().clone());
    selected_ids.extend(
        config
            .manifest
            .components
            .iter()
            .map(|component| CommitId::new(component.selected_revision.as_bytes().to_vec())),
    );
    // This walk proves every selected snapshot is an ordinary native tree and
    // rejects gitlinks and nested repository artifacts before checkout.
    crate::vex_backend::collect_commit_object_closure(repo.store(), &selected_ids)
        .await
        .map_err(|error| federated_home_init_error(error.to_string()))?;

    let home_root_tree_id = home_commit
        .tree_ids()
        .as_resolved()
        .cloned()
        .ok_or_else(|| {
            federated_home_init_error("the selected Home revision has a conflicted root tree")
        })?;
    let home_tree = home_commit.tree();
    let mut builder = TreeBuilder::new(repo.store().clone(), home_root_tree_id);
    for component in &config.manifest.components {
        let root =
            RepoPathBuf::from_internal_string(component.root_path.clone()).map_err(|error| {
                federated_home_init_error(format!(
                    "invalid Home manifest path {}: {error}",
                    component.root_path
                ))
            })?;
        for ancestor in component_root_ancestors(&component.root_path)? {
            let value = home_tree
                .path_value(&ancestor)
                .await
                .map_err(|error| federated_home_init_error(error.to_string()))?
                .into_resolved()
                .map_err(|_| {
                    federated_home_init_error(format!(
                        "Home tree has a conflict above manifest path {}",
                        component.root_path
                    ))
                })?;
            if value.is_some_and(|value| !matches!(value, TreeValue::Tree(_))) {
                return Err(federated_home_init_error(format!(
                    "Home tree entry above manifest path {} is not a directory",
                    component.root_path
                )));
            }
        }
        let occupied = home_tree
            .path_value(&root)
            .await
            .map_err(|error| federated_home_init_error(error.to_string()))?
            .into_resolved()
            .map_err(|_| {
                federated_home_init_error(format!(
                    "Home tree has a conflict at manifest path {}",
                    component.root_path
                ))
            })?;
        if occupied.is_some() {
            return Err(federated_home_init_error(format!(
                "Home tree already occupies manifest path {}",
                component.root_path
            )));
        }
        let component_id = CommitId::new(component.selected_revision.as_bytes().to_vec());
        let component_commit =
            repo.store()
                .get_commit_async(&component_id)
                .await
                .map_err(|error| {
                    federated_home_init_error(format!(
                        "cannot read a selected Home snapshot: {error}"
                    ))
                })?;
        let component_tree_id = component_commit
            .tree_ids()
            .as_resolved()
            .cloned()
            .ok_or_else(|| {
                federated_home_init_error("a selected Home snapshot has a conflicted root tree")
            })?;
        builder
            .set(root, TreeValue::Tree(component_tree_id))
            .map_err(|error| {
                federated_home_init_error(format!(
                    "invalid Home manifest path {}: {error}",
                    component.root_path
                ))
            })?;
    }
    let flat_tree_id = builder.write_tree().await.map_err(|error| {
        federated_home_init_error(format!("cannot synthesize flat Home tree: {error}"))
    })?;
    let mut change_hasher = Sha256::new();
    change_hasher.update(b"vex-flat-home-base-v1\0");
    change_hasher.update(config.manifest_content_sha256.as_bytes());
    let change_digest = change_hasher.finalize();
    let mut commit = home_commit.store_commit().as_ref().clone();
    commit.parents = vec![home_commit.id().clone()];
    commit.predecessors.clear();
    commit.root_tree = Merge::resolved(flat_tree_id);
    commit.conflict_labels = Merge::resolved(String::new());
    commit.change_id = ChangeId::from_bytes(&change_digest[..repo.store().change_id_length()]);
    commit.description = format!(
        "Vex flat Home base {}\n",
        &config.manifest_content_sha256[..12]
    );
    commit.secure_sig = None;
    repo.store()
        .write_commit(commit, None)
        .await
        .map_err(|error| {
            federated_home_init_error(format!("cannot write flat Home base commit: {error}"))
        })
}

/// Hydrate the exact snapshots in a newer signed Home manifest and compose
/// their local-only flat facade commit. The caller persists the returned
/// routing metadata only after it has recorded the facade in the local view.
pub async fn refresh_federated_home_base(
    repo: &Arc<ReadonlyRepo>,
    repo_path: &Path,
    home_config: &VexRepoConfig,
    mut flat_home: VexFederatedHomeConfig,
) -> Result<(Commit, VexFederatedHomeConfig), WorkspaceInitError> {
    flat_home
        .validate()
        .map_err(|error| federated_home_init_error(error.to_string()))?;
    let store_path = repo_path.join("store");
    let cache_root = VexClient::from_store_path(&store_path)
        .map_err(|error| WorkspaceInitError::Backend(BackendInitError(error.into())))?
        .cache_root()
        .ok_or_else(|| federated_home_init_error("flat Home pull has no object cache"))?
        .to_path_buf();

    for repository in &flat_home.repositories {
        let config = flat_home.repository_config(home_config, repository);
        let client = VexClient::from_config_with_cache_root(config, cache_root.clone())
            .map_err(|error| WorkspaceInitError::Backend(BackendInitError(error.into())))?;
        let manifest = client
            .get_clone_manifest(CloneBlobMode::Lazy, None)
            .await
            .map_err(|error| WorkspaceInitError::Backend(BackendInitError(error.into())))?;
        client
            .prefetch_clone_manifest(&manifest, None)
            .await
            .map_err(|error| WorkspaceInitError::Backend(BackendInitError(error.into())))?;
    }

    let home_id = CommitId::new(flat_home.manifest.home_revision.as_bytes().to_vec());
    let home_commit = repo
        .store()
        .get_commit_async(&home_id)
        .await
        .map_err(|error| federated_home_init_error(format!("cannot read Home root: {error}")))?;
    let facade = synthesize_federated_home_base(repo, &flat_home, &home_commit).await?;
    flat_home.aggregate_base_commit_id = Some(content_id_from_commit_id(
        facade.id(),
        "flat Home base commit",
    )?);
    Ok((facade, flat_home))
}

fn component_root_ancestors(root: &str) -> Result<Vec<RepoPathBuf>, WorkspaceInitError> {
    let segments = root.split('/').collect::<Vec<_>>();
    (1..segments.len())
        .map(|end| {
            RepoPathBuf::from_internal_string(segments[..end].join("/")).map_err(|error| {
                federated_home_init_error(format!("invalid Home manifest path {root}: {error}"))
            })
        })
        .collect()
}

fn content_id_from_commit_id(
    commit_id: &CommitId,
    label: &str,
) -> Result<VexContentId, WorkspaceInitError> {
    let bytes: [u8; 32] = commit_id.as_bytes().try_into().map_err(|_| {
        federated_home_init_error(format!("{label} is not a 32-byte native content id"))
    })?;
    Ok(VexContentId::from_bytes(bytes))
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

/// Resolve the head commit for the bookmark `bookmark_name`, considering both
/// the local bookmark and every remote-tracking bookmark of that name. After
/// `vex clone` the trunk is typically a remote-tracking bookmark (e.g.
/// `master@vex`), so a local-only check would miss it. Within a single name,
/// when multiple candidates exist, pick the newest by committer timestamp.
///
/// When `require_head` is true, only candidates whose target is in
/// `head_id_set` are considered (used by the local main/master/trunk fallback,
/// which legitimately wants a current DAG tip). When `require_head` is false the
/// `head_id_set` filter is skipped, so a bookmark that points at an ancestor of
/// a head still resolves — this is what the authoritative server-trunk lookup
/// needs, since the server can register a trunk (e.g. `master`) that already has
/// descendant commits and is therefore not a view head.
/// Returns `None` when no candidate matches.
async fn clone_vex_bookmark_head(
    repo: &Arc<ReadonlyRepo>,
    bookmark_name: &str,
    head_id_set: &HashSet<CommitId>,
    require_head: bool,
) -> Result<Option<Commit>, WorkspaceInitError> {
    let mut candidate_ids: Vec<&CommitId> = Vec::new();
    if let Some(head_id) = repo
        .view()
        .get_local_bookmark(bookmark_name.as_ref())
        .as_normal()
        .filter(|id| !require_head || head_id_set.contains(*id))
    {
        candidate_ids.push(head_id);
    }
    for (symbol, remote_ref) in repo.view().all_remote_bookmarks() {
        if symbol.name.as_str() != bookmark_name {
            continue;
        }
        if let Some(head_id) = remote_ref
            .target
            .as_normal()
            .filter(|id| !require_head || head_id_set.contains(*id))
        {
            candidate_ids.push(head_id);
        }
    }
    candidate_ids.sort();
    candidate_ids.dedup();
    let mut selected_commit: Option<Commit> = None;
    for head_id in candidate_ids {
        let commit = repo
            .store()
            .get_commit_async(head_id)
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        let should_replace = selected_commit.as_ref().is_none_or(|selected: &Commit| {
            commit.committer().timestamp.timestamp > selected.committer().timestamp.timestamp
        });
        if should_replace {
            selected_commit = Some(commit);
        }
    }
    Ok(selected_commit)
}

/// Rebuild a fresh clone's view from the server's refs (roadmap/088 Stage 7).
///
/// Before Stage 7 a clone inherited the server's operation log, and the view —
/// which bookmarks exist and where they point — came with it. The log is local
/// now and a clone's is empty, so **refs are the only inheritance**: without
/// this, `clone_vex_checkout_target` looks at a view with no bookmarks in it,
/// fails to resolve the server's trunk, and the clone materializes nothing.
///
/// For each server bookmark this writes both the local bookmark and the
/// `name@vex` remote-tracking bookmark, exactly as `vex pull` and `vex push`
/// do — so the clone is immediately a valid base for the three-way merge in
/// [`crate::vex_ref_sync`] rather than looking like a repository that has never
/// heard of the server.
///
/// Zero `get_op_heads` calls, by construction: the only RPCs are `list_refs`
/// and the commit reads it implies. That is what makes a repository whose
/// `jj_op_heads` rows have all been deleted still clone (roadmap/088 metric
/// S2).
async fn seed_clone_view_from_refs(
    repo: Arc<ReadonlyRepo>,
    store_path: &Path,
) -> Result<Arc<ReadonlyRepo>, WorkspaceInitError> {
    let client = crate::vex::VexClient::from_store_path(store_path)
        .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
    let refs = client
        .list_refs(crate::vex::REF_FRESHNESS_PREFIX)
        .await
        .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
    let mut targets: Vec<(RefNameBuf, CommitId)> = Vec::new();
    for value in refs {
        let Some(name) = value.name.strip_prefix(crate::vex::REF_FRESHNESS_PREFIX) else {
            continue;
        };
        // An unparsable target is not information. Skipping the bookmark leaves
        // it absent, which is honest; inventing a target would not be.
        let Some(id) = CommitId::try_from_hex(&value.target_commit_id) else {
            tracing::warn!(
                ref_name = %value.name,
                target = %value.target_commit_id,
                "skipping a server bookmark with an unparsable target while seeding the clone view"
            );
            continue;
        };
        targets.push((RefNameBuf::from(name), id));
    }
    if targets.is_empty() {
        // A repository with no bookmarks yet (a fresh `vex init` that has never
        // been pushed). Nothing to seed, and nothing is wrong.
        return Ok(repo);
    }
    targets.sort();

    let remote = RemoteName::new(crate::vex_ref_sync::VEX_REMOTE);
    let mut tx = repo.start_transaction();
    let mut seeded = 0_usize;
    for (name, id) in targets {
        // Hydrate before pointing anything at the commit: an unindexed target
        // would fail the whole transaction rather than the one bookmark it
        // belongs to.
        let commit = match tx.repo().store().get_commit_async(&id).await {
            Ok(commit) => commit,
            Err(err) => {
                tracing::warn!(
                    bookmark = name.as_str(),
                    error = %err,
                    "skipping a server bookmark whose target commit could not be read"
                );
                continue;
            }
        };
        if let Err(err) = tx.repo_mut().add_head(&commit).await {
            tracing::warn!(
                bookmark = name.as_str(),
                error = %err,
                "skipping a server bookmark whose target could not be indexed"
            );
            continue;
        }
        let target = RefTarget::normal(id);
        tx.repo_mut()
            .set_local_bookmark_target(name.as_ref(), target.clone());
        tx.repo_mut().set_remote_bookmark(
            RemoteRefSymbol {
                name: name.as_ref(),
                remote,
            },
            RemoteRef {
                target,
                state: RemoteRefState::Tracked,
            },
        );
        seeded += 1;
    }
    if !tx.repo().has_changes() {
        return Ok(repo);
    }
    let repo = tx
        .commit("seed bookmarks from the server's refs")
        .await
        .map_err(WorkspaceInitError::TransactionCommit)?;
    tracing::debug!(bookmarks = seeded, "seeded the clone view from server refs");
    Ok(repo)
}

/// Seed a flat Home clone from its signed selection without reading raw refs.
async fn seed_federated_home_clone_view(
    repo: Arc<ReadonlyRepo>,
    config: &VexFederatedHomeConfig,
) -> Result<Arc<ReadonlyRepo>, WorkspaceInitError> {
    let bookmark = RefNameBuf::from(config.manifest.home_bookmark.as_str());
    let revision = CommitId::new(config.manifest.home_revision.as_bytes().to_vec());
    let commit = repo
        .store()
        .get_commit_async(&revision)
        .await
        .map_err(|error| {
            federated_home_init_error(format!(
                "cannot read selected Home revision {}: {error}",
                config.manifest.home_revision
            ))
        })?;

    let mut tx = repo.start_transaction();
    tx.repo_mut().add_head(&commit).await.map_err(|error| {
        federated_home_init_error(format!(
            "cannot index selected Home revision {}: {error}",
            config.manifest.home_revision
        ))
    })?;
    let target = RefTarget::normal(revision);
    tx.repo_mut()
        .set_local_bookmark_target(bookmark.as_ref(), target.clone());
    tx.repo_mut().set_remote_bookmark(
        RemoteRefSymbol {
            name: bookmark.as_ref(),
            remote: RemoteName::new(crate::vex_ref_sync::VEX_REMOTE),
        },
        RemoteRef {
            target,
            state: RemoteRefState::Tracked,
        },
    );
    tx.commit("seed Home bookmark from the signed manifest")
        .await
        .map_err(WorkspaceInitError::TransactionCommit)
}

/// A native bookmark resolved as the `vex clone` checkout target: the bookmark
/// name that drove selection plus its target commit. Built only from native
/// local/remote-tracking bookmark state, never from `git/ref/*`.
#[derive(Clone, Debug)]
pub struct NativeBookmarkTarget {
    pub name: String,
    pub commit: Commit,
}

/// Typed result of native `vex clone` target selection (roadmap/066 Stage 1).
///
/// `vex clone` is native-only: selection reads only native view state and
/// never resolves `git/ref/*` or raw Git objects. With a server-advertised
/// trunk the only outcomes are a resolved [`NativeBookmarkTarget`] or the
/// typed [`WorkspaceInitError::NativeTrunkMissing`] error — except an entirely
/// empty native repository, which starts at root so it can receive its first
/// bookmark.
#[derive(Clone, Debug)]
pub enum NativeCloneTarget {
    /// The server-advertised trunk resolved through native local or
    /// remote-tracking bookmarks. Authoritative: the target does not need to
    /// be a current view head (the trunk may already have descendant commits).
    ServerTrunk(NativeBookmarkTarget),
    /// No server trunk was supplied (legacy catalog metadata): a native-only
    /// heuristic over the view picked the target. `bookmark` is `None` for
    /// the bookmark-less fallbacks (native-view `git_head`, recent workspace,
    /// working copy, newest head).
    LegacyNative {
        bookmark: Option<String>,
        commit: Commit,
    },
}

impl NativeCloneTarget {
    pub fn commit(&self) -> &Commit {
        match self {
            Self::ServerTrunk(target) => &target.commit,
            Self::LegacyNative { commit, .. } => commit,
        }
    }

    pub fn bookmark(&self) -> Option<&str> {
        match self {
            Self::ServerTrunk(target) => Some(&target.name),
            Self::LegacyNative { bookmark, .. } => bookmark.as_deref(),
        }
    }

    fn into_parts(self) -> (Commit, Option<String>) {
        match self {
            Self::ServerTrunk(target) => (target.commit, Some(target.name)),
            Self::LegacyNative { bookmark, commit } => (commit, bookmark),
        }
    }
}

/// Select the commit `clone_vex` checks out. An explicit `target_commit`
/// (CI/tests) is loaded as an exact native commit ID and bypasses bookmark
/// selection entirely; otherwise native target selection runs.
async fn clone_vex_checkout_target(
    repo: &Arc<ReadonlyRepo>,
    target_commit: Option<&CommitId>,
    server_trunk: Option<&str>,
) -> Result<(Commit, Option<String>), WorkspaceInitError> {
    match target_commit {
        Some(commit_id) => Ok((
            repo.store()
                .get_commit_async(commit_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?,
            None,
        )),
        None => Ok(clone_vex_native_target(repo, server_trunk)
            .await?
            .into_parts()),
    }
}

/// Native target selection for `vex clone`. See [`NativeCloneTarget`].
async fn clone_vex_native_target(
    repo: &Arc<ReadonlyRepo>,
    server_trunk: Option<&str>,
) -> Result<NativeCloneTarget, WorkspaceInitError> {
    match server_trunk {
        Some(server_trunk) => Ok(NativeCloneTarget::ServerTrunk(
            clone_vex_server_trunk_target(repo, server_trunk).await?,
        )),
        None => {
            // Legacy catalog metadata supplied no trunk; the heuristic below
            // stays native-only. Counted and logged so the server metadata
            // can be repaired (roadmap/066 Stage 3 observability).
            crate::vex::vex_client_stats().record_native_trunk_missing();
            tracing::debug!(
                "vex clone: no server trunk advertised; using native view fallback selection"
            );
            let (commit, bookmark) = clone_vex_legacy_start_commit(repo).await?;
            Ok(NativeCloneTarget::LegacyNative { bookmark, commit })
        }
    }
}

/// Resolve the server-advertised trunk (`Repository#default_branch`, surfaced
/// via the repo-access catalog `default_branch`) through native local and
/// remote-tracking bookmarks only. The name is authoritative: when the
/// bookmark exists we check out its target regardless of whether that target
/// is a current view head — the server may register a trunk that already has
/// descendant commits. When the bookmark is absent this fails closed with
/// [`WorkspaceInitError::NativeTrunkMissing`]; it never consults another
/// branch, an arbitrary head, `git_head`, or `git/ref/*`.
async fn clone_vex_server_trunk_target(
    repo: &Arc<ReadonlyRepo>,
    server_trunk: &str,
) -> Result<NativeBookmarkTarget, WorkspaceInitError> {
    // `require_head: false` disables the head-set filter, so the set contents
    // are irrelevant here.
    if let Some(commit) =
        clone_vex_bookmark_head(repo, server_trunk, &HashSet::new(), false).await?
    {
        crate::vex::vex_client_stats().record_native_trunk_resolution();
        return Ok(NativeBookmarkTarget {
            name: server_trunk.to_owned(),
            commit,
        });
    }
    if clone_vex_native_view_is_empty(repo) {
        // The catalog's conventional `main` fallback is useful for the UI,
        // but a newly provisioned native repository has no bookmark until its
        // first push. Only this zero-state case may begin at root; any native
        // state with a missing advertised trunk still fails closed below.
        return Ok(NativeBookmarkTarget {
            name: server_trunk.to_owned(),
            commit: repo.store().root_commit(),
        });
    }
    crate::vex::vex_client_stats().record_native_trunk_missing();
    Err(WorkspaceInitError::NativeTrunkMissing {
        trunk: server_trunk.to_owned(),
    })
}

fn clone_vex_native_view_is_empty(repo: &ReadonlyRepo) -> bool {
    repo.view()
        .heads()
        .iter()
        .all(|head| head == repo.store().root_commit().id())
        && repo.view().local_bookmarks().next().is_none()
        && repo.view().all_remote_bookmarks().next().is_none()
}

/// Legacy native start-commit selection, used only when the server supplied no
/// trunk. Reads native view state exclusively: native-only main/master/trunk
/// bookmarks, local bookmarks, the native view's `git_head`, the most recent
/// workspace operation, working-copy commits, and finally the newest head.
async fn clone_vex_legacy_start_commit(
    repo: &Arc<ReadonlyRepo>,
) -> Result<(Commit, Option<String>), WorkspaceInitError> {
    let mut head_ids = repo.view().heads().iter().cloned().collect::<Vec<_>>();
    if head_ids.is_empty() {
        return Ok((repo.store().root_commit(), None));
    }
    head_ids.sort();
    let head_id_set = head_ids.iter().cloned().collect::<HashSet<_>>();
    // Prefer trunk bookmarks (main, then master, then trunk) that are
    // current heads. This mirrors the default `trunk()` revset alias (see
    // jj/cli/src/config/revsets.toml).
    for bookmark_name in ["main", "master", "trunk"] {
        if let Some(commit) =
            clone_vex_bookmark_head(repo, bookmark_name, &head_id_set, true).await?
        {
            return Ok((commit, Some(bookmark_name.to_owned())));
        }
    }
    for (name, target) in repo.view().local_bookmarks() {
        if let Some(head_id) = target.as_normal().filter(|id| head_id_set.contains(*id)) {
            let commit = repo
                .store()
                .get_commit_async(head_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            return Ok((commit, Some(name.as_str().to_owned())));
        }
    }
    if let Some(head_id) = repo
        .view()
        .git_head()
        .as_normal()
        .filter(|id| head_id_set.contains(*id))
    {
        let commit = repo
            .store()
            .get_commit_async(head_id)
            .await
            .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
        return Ok((commit, None));
    }
    if let Some(commit) = clone_vex_recent_workspace_commit_from_ops(repo).await? {
        return Ok((commit, None));
    }
    for head_id in repo.view().wc_commit_ids().values() {
        if head_id_set.contains(head_id) {
            let commit = repo
                .store()
                .get_commit_async(head_id)
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            return Ok((
                clone_vex_peel_discardable_wc_commit(repo, commit).await?,
                None,
            ));
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
    Ok((
        selected_commit.expect("non-empty heads should produce a checkout target"),
        None,
    ))
}

/// Bulk-fetch the file/symlink contents of `start_commit` into the local
/// vex-cache before checkout. Prefer dedicated hydration packs for sequential
/// transfer and unpack, falling back to batched `GetObjectsInline` reads when
/// the server does not implement the pack RPC or any pack step fails.
/// Best-effort: on any failure it logs and returns, and checkout falls back to
/// hydrating the remaining files on demand exactly as before.
/// Returns the number of file/symlink tree entries in the start commit when
/// the hydration walk ran (used as the materializing-progress total), or
/// `None` when the walk was skipped or failed.
async fn hydrate_start_commit_blobs(
    repo: &Arc<ReadonlyRepo>,
    store_path: &Path,
    start_commit: &Commit,
    progress: Option<&crate::vex::CloneProgressFn>,
) -> Option<u64> {
    let started = std::time::Instant::now();
    let client = match crate::vex::VexClient::from_store_path(store_path) {
        Ok(client) => client,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "clone hydration: failed to build client; falling back to lazy per-file fetch"
            );
            return None;
        }
    };
    let walk_started = std::time::Instant::now();
    let walk = clone_vex_hydration_ids(repo, start_commit).await;
    crate::vex::vex_client_stats().hydration_walk_ms.fetch_add(
        walk_started.elapsed().as_millis() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    let (ids, file_count) = match walk {
        Ok(walk) => walk,
        Err(err) => {
            tracing::warn!(
                error = %err,
                "clone hydration: tree walk failed; falling back to lazy per-file fetch"
            );
            return None;
        }
    };
    if ids.is_empty() {
        return Some(file_count);
    }
    let total = ids.len();
    let hydration_objects = ids
        .iter()
        .map(|(kind, content_id, _)| (*kind, *content_id))
        .collect::<Vec<_>>();
    if let Some(progress) = progress {
        progress(crate::vex::CloneProgress::Hydrating {
            done: 0,
            total: total as u64,
        });
    }
    let packs_started = std::time::Instant::now();
    match client.get_hydration_packs(&hydration_objects).await {
        Ok(manifest) if hydration_pack_manifest_covers(&manifest, &hydration_objects) => {
            match client.prefetch_hydration_packs(&manifest, progress).await {
                Ok(()) => {
                    if let Some(progress) = progress {
                        progress(crate::vex::CloneProgress::Hydrating {
                            done: total as u64,
                            total: total as u64,
                        });
                    }
                    tracing::debug!(
                        total,
                        packs = manifest.packs.len(),
                        pack_bytes = manifest.total_bytes,
                        elapsed_ms = packs_started.elapsed().as_millis(),
                        "clone hydration packs complete"
                    );
                    return Some(file_count);
                }
                Err(err) => {
                    tracing::debug!(
                        error = %err,
                        "clone hydration pack prefetch failed; falling back to inline hydration"
                    );
                }
            }
        }
        Ok(manifest) => {
            tracing::debug!(
                expected_objects = total,
                manifest_objects = manifest.object_count,
                described_objects = manifest
                    .packs
                    .iter()
                    .map(|pack| pack.objects.len())
                    .sum::<usize>(),
                "clone hydration pack manifest did not cover the request; falling back to inline hydration"
            );
        }
        Err(err) => {
            tracing::debug!(
                error = %err,
                "clone hydration packs unavailable; falling back to inline hydration"
            );
        }
    }
    match client.get_objects_inline_batched(ids, progress).await {
        Ok(hydrated) => {
            tracing::debug!(
                total,
                hydrated,
                elapsed_ms = started.elapsed().as_millis(),
                "clone hydration complete"
            );
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                "clone hydration failed; falling back to lazy per-file fetch"
            );
        }
    }
    Some(file_count)
}

fn hydration_pack_manifest_covers(
    manifest: &jj_backend_types::HydrationPackManifest,
    requested: &[(jj_backend_types::ObjectKind, jj_backend_types::ContentId)],
) -> bool {
    if manifest.object_count != requested.len() as u64 {
        return false;
    }
    let described = manifest
        .packs
        .iter()
        .flat_map(|pack| pack.objects.iter())
        .map(|object| (object.kind, object.content_id))
        .collect::<HashSet<_>>();
    described.len() == requested.len() && requested.iter().all(|object| described.contains(object))
}

/// Result of the hydration tree walk: the object ids to hydrate and the
/// number of file/symlink tree entries encountered.
type HydrationWalk = (
    Vec<(
        jj_backend_types::ObjectKind,
        jj_backend_types::ContentId,
        Option<u64>,
    )>,
    u64,
);

/// Walk `commit`'s root trees (all conflict terms, so merged/conflicted
/// commits are covered) and collect the content ids of every file blob and
/// symlink target for pre-checkout hydration, plus the number of file/symlink
/// tree entries encountered (per path, before content dedup — the count of
/// working-copy entries a full checkout will materialize). Tree metadata is
/// warm in the local cache after the clone prefetch, so the walk itself stays
/// local. Implicit objects (all-zeros, empty content) are skipped, matching
/// the server's snapshot closure definition.
async fn clone_vex_hydration_ids(
    repo: &Arc<ReadonlyRepo>,
    commit: &Commit,
) -> Result<HydrationWalk, crate::backend::BackendError> {
    use jj_backend_types::ContentId;
    use jj_backend_types::ObjectKind;

    fn vex_content_id(bytes: &[u8]) -> Option<ContentId> {
        <[u8; 32]>::try_from(bytes).ok().map(ContentId::from_bytes)
    }

    let store = repo.store();
    let empty_id = ContentId::hash_bytes(b"");
    let zeros_id = ContentId::from_bytes([0; 32]);
    let mut visited_trees: HashSet<crate::backend::TreeId> = HashSet::new();
    let mut queue: Vec<(crate::repo_path::RepoPathBuf, crate::backend::TreeId)> = Vec::new();
    for tree_id in commit.tree_ids().iter() {
        if visited_trees.insert(tree_id.clone()) {
            queue.push((crate::repo_path::RepoPathBuf::root(), tree_id.clone()));
        }
    }
    let mut seen: HashSet<(ObjectKind, ContentId)> = HashSet::new();
    let mut ids = Vec::new();
    let mut file_count = 0_u64;
    let mut push_content = |kind: ObjectKind, id_bytes: &[u8], ids: &mut Vec<_>| {
        if let Some(content_id) = vex_content_id(id_bytes) {
            if content_id != empty_id && content_id != zeros_id && seen.insert((kind, content_id)) {
                ids.push((kind, content_id, None));
            }
        }
    };
    while let Some((dir, tree_id)) = queue.pop() {
        let tree = store.get_tree(dir, &tree_id).await?;
        for entry in tree.entries_non_recursive() {
            match entry.value() {
                crate::backend::TreeValue::File { id, .. } => {
                    file_count += 1;
                    push_content(ObjectKind::Blob, id.as_bytes(), &mut ids);
                }
                crate::backend::TreeValue::Symlink(id) => {
                    file_count += 1;
                    push_content(ObjectKind::Symlink, id.as_bytes(), &mut ids);
                }
                crate::backend::TreeValue::Tree(sub_tree_id) => {
                    if visited_trees.insert(sub_tree_id.clone()) {
                        queue.push((tree.dir().join(entry.name()), sub_tree_id.clone()));
                    }
                }
                crate::backend::TreeValue::GitSubmodule(_) => {}
            }
        }
    }
    Ok((ids, file_count))
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
    use jj_backend_types::ClonePackScope;
    use jj_backend_types::ContentId;
    use jj_backend_types::FederatedHomeComponent;
    use jj_backend_types::FederatedHomeManifest;
    use jj_backend_types::FederatedHomePathOwner;
    use jj_backend_types::FederatedHomePathOwnerKind;
    use jj_backend_types::HydrationPackManifest;
    use jj_backend_types::ObjectDescriptor;
    use jj_backend_types::ObjectKind;
    use jj_backend_types::PackDescriptor;
    use pollster::FutureExt as _;
    use tempfile::TempDir;

    use super::*;
    use crate::backend::CopyId;
    use crate::backend::FileId;
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

    fn init_test_vex_repo(
        settings: &UserSettings,
    ) -> Result<(TempDir, Arc<ReadonlyRepo>), WorkspaceInitError> {
        let temp_dir = tempfile::Builder::new()
            .prefix("jj-vex-test-")
            .tempdir()
            .unwrap();
        let repo_dir = temp_dir.path().join("repo");
        std::fs::create_dir(&repo_dir).unwrap();
        let signer = Signer::from_settings(settings)?;
        let config = VexRepoConfig {
            endpoint: "http://127.0.0.1:1".to_string(),
            tenant_id: "tenant-backend".to_string(),
            tenant_slug: "acme".to_string(),
            repo_id: "9001".to_string(),
            repo_slug: "home".to_string(),
            repository_scope_kind: Some("repository".to_string()),
            virtual_repository_id: None,
            backing_repo_slug: None,
            virtual_root_path: None,
            virtual_mounts: Vec::new(),
            access_token: Some("vexrt_test".to_string()),
            local_writes: false,
            object_read_mode: crate::vex::VexObjectReadMode::NativeOnly,
        };
        let repo = ReadonlyRepo::init(
            settings,
            &repo_dir,
            &move |_settings, store_path| {
                Ok(Box::new(VexBackend::init_at(config.clone(), store_path)?))
            },
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

    fn hydration_object(byte: u8) -> (jj_backend_types::ObjectKind, jj_backend_types::ContentId) {
        (
            jj_backend_types::ObjectKind::Blob,
            jj_backend_types::ContentId::from_bytes([byte; 32]),
        )
    }

    fn repo_path(path: &str) -> RepoPathBuf {
        RepoPathBuf::from_internal_string(path.to_string()).unwrap()
    }

    fn content_id(bytes: &[u8]) -> ContentId {
        ContentId::from_bytes(bytes.try_into().expect("native ids are 32 bytes"))
    }

    fn flat_home_config(
        home: &Commit,
        components: &[(&str, &str, &Commit)],
    ) -> VexFederatedHomeConfig {
        let manifest = FederatedHomeManifest {
            format_version: 1,
            home_repository_id: "9001".to_string(),
            home_bookmark: "main".to_string(),
            home_revision: content_id(home.id().as_bytes()),
            components: components
                .iter()
                .map(
                    |(repository_id, root_path, commit)| FederatedHomeComponent {
                        repository_id: (*repository_id).to_string(),
                        root_path: (*root_path).to_string(),
                        selected_bookmark: "main".to_string(),
                        selected_revision: content_id(commit.id().as_bytes()),
                    },
                )
                .collect(),
            path_owners: std::iter::once(FederatedHomePathOwner {
                path: String::new(),
                owner: FederatedHomePathOwnerKind::HomeRoot,
            })
            .chain(
                components
                    .iter()
                    .enumerate()
                    .map(|(index, (_, root_path, _))| FederatedHomePathOwner {
                        path: (*root_path).to_string(),
                        owner: FederatedHomePathOwnerKind::Component {
                            component_index: index,
                        },
                    }),
            )
            .collect(),
        };
        let repositories = std::iter::once(crate::vex::VexFederatedHomeRepository {
            repository_id: "9001".to_string(),
            repository_public_id: "repository_home".to_string(),
            repository_slug: "home".to_string(),
            root_path: String::new(),
            endpoint: "http://127.0.0.1:1".to_string(),
        })
        .chain(components.iter().map(|(repository_id, root_path, _)| {
            crate::vex::VexFederatedHomeRepository {
                repository_id: (*repository_id).to_string(),
                repository_public_id: format!("repository_{repository_id}"),
                repository_slug: format!("repo-{repository_id}"),
                root_path: (*root_path).to_string(),
                endpoint: "http://127.0.0.1:1".to_string(),
            }
        }))
        .collect();
        VexFederatedHomeConfig {
            format_version: 1,
            manifest_artifact_suffix: manifest.artifact_suffix().unwrap(),
            manifest_content_sha256: manifest.content_sha256().unwrap(),
            manifest_generation: 1,
            manifest,
            aggregate_access_token: "vexhome_test".to_string(),
            repositories,
            aggregate_base_commit_id: None,
        }
    }

    async fn write_test_file(
        store: &Arc<crate::store::Store>,
        path: &RepoPathBuf,
        contents: &[u8],
    ) -> FileId {
        store.write_file(path, &mut &contents[..]).await.unwrap()
    }

    async fn write_test_commit(
        store: &Arc<crate::store::Store>,
        parent: CommitId,
        root_tree: crate::backend::TreeId,
        change_byte: u8,
    ) -> Commit {
        let mut commit = store.root_commit().store_commit().as_ref().clone();
        commit.parents = vec![parent];
        commit.predecessors.clear();
        commit.root_tree = Merge::resolved(root_tree);
        commit.conflict_labels = Merge::resolved(String::new());
        commit.change_id = ChangeId::from_bytes(&vec![change_byte; store.change_id_length()]);
        commit.description = format!("test commit {change_byte}");
        commit.secure_sig = None;
        store.write_commit(commit, None).await.unwrap()
    }

    #[test]
    fn delta_closure_skips_an_unchanged_large_subtree() {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_vex_repo(&settings).unwrap();
        let store = repo.store().clone();
        let (base, target, unchanged_tree, unchanged_blob, changed_blob) = async {
            let unchanged_path = repo_path("large/unchanged.bin");
            let second_unchanged_path = repo_path("large/also-unchanged.bin");
            let changed_path = repo_path("changed.txt");
            let unchanged_blob = write_test_file(&store, &unchanged_path, b"large-a").await;
            let second_blob = write_test_file(&store, &second_unchanged_path, b"large-b").await;
            let old_changed = write_test_file(&store, &changed_path, b"before").await;
            let mut base_builder = TreeBuilder::new(store.clone(), store.empty_tree_id().clone());
            for (path, id) in [
                (unchanged_path.clone(), unchanged_blob.clone()),
                (second_unchanged_path, second_blob),
                (changed_path.clone(), old_changed),
            ] {
                base_builder
                    .set(
                        path,
                        TreeValue::File {
                            id,
                            executable: false,
                            copy_id: CopyId::placeholder(),
                        },
                    )
                    .unwrap();
            }
            let base_tree = base_builder.write_tree().await.unwrap();
            let base =
                write_test_commit(&store, store.root_commit_id().clone(), base_tree, 1).await;
            let unchanged_tree = match base
                .tree()
                .path_value(&repo_path("large"))
                .await
                .unwrap()
                .into_resolved()
                .unwrap()
                .unwrap()
            {
                TreeValue::Tree(id) => id,
                value => panic!("expected unchanged subtree, got {value:?}"),
            };
            let changed_blob = write_test_file(&store, &changed_path, b"after").await;
            let mut target_builder = TreeBuilder::new(
                store.clone(),
                base.tree_ids().as_resolved().unwrap().clone(),
            );
            target_builder
                .set(
                    changed_path,
                    TreeValue::File {
                        id: changed_blob.clone(),
                        executable: false,
                        copy_id: CopyId::placeholder(),
                    },
                )
                .unwrap();
            let target_tree = target_builder.write_tree().await.unwrap();
            let target = write_test_commit(&store, base.id().clone(), target_tree, 2).await;
            (base, target, unchanged_tree, unchanged_blob, changed_blob)
        }
        .block_on();

        let closure = crate::vex_backend::collect_commit_delta_object_closure(
            &store,
            &[(target.id().clone(), Some(base.id().clone()))],
        )
        .block_on()
        .unwrap();

        assert!(closure.contains(&(ObjectKind::Commit, content_id(target.id().as_bytes()))));
        assert!(closure.contains(&(
            ObjectKind::Tree,
            content_id(target.tree_ids().as_resolved().unwrap().as_bytes())
        )));
        assert!(closure.contains(&(ObjectKind::Blob, content_id(changed_blob.as_bytes()))));
        assert!(!closure.contains(&(ObjectKind::Tree, content_id(unchanged_tree.as_bytes()))));
        assert!(!closure.contains(&(ObjectKind::Blob, content_id(unchanged_blob.as_bytes()))));
    }

    #[test]
    fn composition_marker_closure_stops_at_selected_component_snapshot() {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_vex_repo(&settings).unwrap();
        let store = repo.store().clone();
        let (home, synthetic, component_tree, component_blob, home_blob, overlay_ancestor) =
            async {
                let home_path = repo_path("README.md");
                let home_blob = write_test_file(&store, &home_path, b"home").await;
                let mut home_builder =
                    TreeBuilder::new(store.clone(), store.empty_tree_id().clone());
                home_builder
                    .set(
                        home_path,
                        TreeValue::File {
                            id: home_blob.clone(),
                            executable: false,
                            copy_id: CopyId::placeholder(),
                        },
                    )
                    .unwrap();
                let home_tree = home_builder.write_tree().await.unwrap();
                let home =
                    write_test_commit(&store, store.root_commit_id().clone(), home_tree, 3).await;

                let component_path = repo_path("unchanged-component.bin");
                let component_blob = write_test_file(&store, &component_path, b"component").await;
                let mut component_builder =
                    TreeBuilder::new(store.clone(), store.empty_tree_id().clone());
                component_builder
                    .set(
                        component_path,
                        TreeValue::File {
                            id: component_blob.clone(),
                            executable: false,
                            copy_id: CopyId::placeholder(),
                        },
                    )
                    .unwrap();
                let component_tree = component_builder.write_tree().await.unwrap();

                let mut aggregate_builder = TreeBuilder::new(
                    store.clone(),
                    home.tree_ids().as_resolved().unwrap().clone(),
                );
                aggregate_builder
                    .set(
                        repo_path("apps/web"),
                        TreeValue::Tree(component_tree.clone()),
                    )
                    .unwrap();
                let aggregate_tree = aggregate_builder.write_tree().await.unwrap();
                let synthetic =
                    write_test_commit(&store, home.id().clone(), aggregate_tree, 4).await;
                let overlay_ancestor = match synthetic
                    .tree()
                    .path_value(&repo_path("apps"))
                    .await
                    .unwrap()
                    .into_resolved()
                    .unwrap()
                    .unwrap()
                {
                    TreeValue::Tree(id) => id,
                    value => panic!("expected overlay ancestor, got {value:?}"),
                };
                (
                    home,
                    synthetic,
                    component_tree,
                    component_blob,
                    home_blob,
                    overlay_ancestor,
                )
            }
            .block_on();

        let closure = crate::vex_backend::collect_composed_overlay_object_closure(
            &store,
            synthetic.id(),
            home.id(),
            [component_tree.clone()],
        )
        .block_on()
        .unwrap();

        assert!(closure.contains(&(ObjectKind::Commit, content_id(synthetic.id().as_bytes()))));
        assert!(closure.contains(&(
            ObjectKind::Tree,
            content_id(synthetic.tree_ids().as_resolved().unwrap().as_bytes())
        )));
        assert!(closure.contains(&(ObjectKind::Tree, content_id(overlay_ancestor.as_bytes()))));
        assert!(!closure.contains(&(ObjectKind::Tree, content_id(component_tree.as_bytes()))));
        assert!(!closure.contains(&(ObjectKind::Blob, content_id(component_blob.as_bytes()))));
        assert!(!closure.contains(&(ObjectKind::Blob, content_id(home_blob.as_bytes()))));
    }

    #[test]
    fn flat_snapshot_validation_rejects_nested_repository_artifacts() {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_vex_repo(&settings).unwrap();
        let store = repo.store().clone();
        let commit = async {
            let path = repo_path("component/.jj/config");
            let file = write_test_file(&store, &path, b"nested metadata").await;
            let mut builder = TreeBuilder::new(store.clone(), store.empty_tree_id().clone());
            builder
                .set(
                    path,
                    TreeValue::File {
                        id: file,
                        executable: false,
                        copy_id: CopyId::placeholder(),
                    },
                )
                .unwrap();
            let tree = builder.write_tree().await.unwrap();
            write_test_commit(&store, store.root_commit_id().clone(), tree, 5).await
        }
        .block_on();

        let error = crate::vex_backend::collect_commit_object_closure(
            &store,
            std::slice::from_ref(commit.id()),
        )
        .block_on()
        .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("Home trees cannot contain reserved metadata `.jj`"),
            "unexpected validation error: {message}"
        );
    }

    #[test]
    fn composition_rejects_a_home_entry_at_the_exact_component_root() {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_vex_repo(&settings).unwrap();
        let store = repo.store().clone();
        let (home, component) = async {
            let occupied = repo_path("apps/web");
            let file = write_test_file(&store, &occupied, b"Home collision").await;
            let mut home_builder = TreeBuilder::new(store.clone(), store.empty_tree_id().clone());
            home_builder
                .set(
                    occupied,
                    TreeValue::File {
                        id: file,
                        executable: false,
                        copy_id: CopyId::placeholder(),
                    },
                )
                .unwrap();
            let home_tree = home_builder.write_tree().await.unwrap();
            let home =
                write_test_commit(&store, store.root_commit_id().clone(), home_tree, 6).await;
            let component = write_test_commit(
                &store,
                store.root_commit_id().clone(),
                store.empty_tree_id().clone(),
                7,
            )
            .await;
            (home, component)
        }
        .block_on();
        let config = flat_home_config(&home, &[("9002", "apps/web", &component)]);

        let error = synthesize_federated_home_base(&repo, &config, &home)
            .block_on()
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("already occupies manifest path apps/web")
        );
    }

    #[test]
    fn flat_config_rejects_overlapping_component_roots() {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_vex_repo(&settings).unwrap();
        let store = repo.store().clone();
        let (home, first, second) = async {
            let home = write_test_commit(
                &store,
                store.root_commit_id().clone(),
                store.empty_tree_id().clone(),
                8,
            )
            .await;
            let first = write_test_commit(
                &store,
                store.root_commit_id().clone(),
                store.empty_tree_id().clone(),
                9,
            )
            .await;
            let second = write_test_commit(
                &store,
                store.root_commit_id().clone(),
                store.empty_tree_id().clone(),
                10,
            )
            .await;
            (home, first, second)
        }
        .block_on();
        let manifest = FederatedHomeManifest {
            format_version: 1,
            home_repository_id: "9001".to_string(),
            home_bookmark: "main".to_string(),
            home_revision: content_id(home.id().as_bytes()),
            components: vec![
                FederatedHomeComponent {
                    repository_id: "9002".to_string(),
                    root_path: "apps".to_string(),
                    selected_bookmark: "main".to_string(),
                    selected_revision: content_id(first.id().as_bytes()),
                },
                FederatedHomeComponent {
                    repository_id: "9003".to_string(),
                    root_path: "apps/web".to_string(),
                    selected_bookmark: "main".to_string(),
                    selected_revision: content_id(second.id().as_bytes()),
                },
            ],
            path_owners: vec![
                FederatedHomePathOwner {
                    path: String::new(),
                    owner: FederatedHomePathOwnerKind::HomeRoot,
                },
                FederatedHomePathOwner {
                    path: "apps".to_string(),
                    owner: FederatedHomePathOwnerKind::Component { component_index: 0 },
                },
                FederatedHomePathOwner {
                    path: "apps/web".to_string(),
                    owner: FederatedHomePathOwnerKind::Component { component_index: 1 },
                },
            ],
        };

        let error = manifest.validate().unwrap_err();

        assert!(error.to_string().contains("component roots collide"));
    }

    #[test]
    fn hydration_pack_manifest_must_describe_every_requested_object() {
        let first = hydration_object(1);
        let second = hydration_object(2);
        let manifest = HydrationPackManifest {
            packs: vec![PackDescriptor {
                content_id: jj_backend_types::ContentId::from_bytes([3; 32]),
                size_bytes: 42,
                scope: ClonePackScope::Full,
                chunk_frames: false,
                chunks: Vec::new(),
                objects: vec![
                    ObjectDescriptor {
                        kind: first.0,
                        content_id: first.1,
                        size_bytes: Some(10),
                    },
                    ObjectDescriptor {
                        kind: second.0,
                        content_id: second.1,
                        size_bytes: Some(20),
                    },
                ],
            }],
            object_count: 2,
            total_bytes: 42,
        };

        assert!(hydration_pack_manifest_covers(&manifest, &[first, second]));
    }

    #[test]
    fn hydration_pack_manifest_rejects_missing_descriptor_even_when_count_matches() {
        let first = hydration_object(1);
        let second = hydration_object(2);
        let manifest = HydrationPackManifest {
            packs: vec![PackDescriptor {
                content_id: jj_backend_types::ContentId::from_bytes([3; 32]),
                size_bytes: 42,
                scope: ClonePackScope::Full,
                chunk_frames: false,
                chunks: Vec::new(),
                objects: vec![ObjectDescriptor {
                    kind: first.0,
                    content_id: first.1,
                    size_bytes: Some(10),
                }],
            }],
            object_count: 2,
            total_bytes: 42,
        };

        assert!(!hydration_pack_manifest_covers(&manifest, &[first, second]));
    }

    #[test]
    fn federated_home_clone_view_uses_only_the_manifest_bookmark_and_revision()
    -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_vex_repo(&settings)?;
        let home = write_test_commit(
            repo.store(),
            repo.store().root_commit_id().clone(),
            repo.store().empty_tree_id().clone(),
            11,
        )
        .block_on();
        let mut config = flat_home_config(&home, &[]);
        config.manifest.home_bookmark = "release".to_string();
        config.manifest_content_sha256 = config.manifest.content_sha256().unwrap();
        config.manifest_artifact_suffix = config.manifest.artifact_suffix().unwrap();
        config.validate().unwrap();

        let repo = seed_federated_home_clone_view(repo, &config).block_on()?;
        let (start_commit, resolved_trunk) =
            clone_vex_checkout_target(&repo, None, Some(config.manifest.home_bookmark.as_str()))
                .block_on()?;

        assert_eq!(start_commit.id(), home.id());
        assert_eq!(resolved_trunk.as_deref(), Some("release"));
        assert_eq!(
            repo.view()
                .get_local_bookmark("release".as_ref())
                .as_normal(),
            Some(home.id())
        );
        let remote = repo.view().get_remote_bookmark(RemoteRefSymbol {
            name: "release".as_ref(),
            remote: RemoteName::new(crate::vex_ref_sync::VEX_REMOTE),
        });
        assert_eq!(remote.target.as_normal(), Some(home.id()));
        assert_eq!(remote.state, RemoteRefState::Tracked);
        assert_eq!(repo.view().local_bookmarks().count(), 1);
        assert_eq!(repo.view().all_remote_bookmarks().count(), 1);
        assert!(repo.view().heads().contains(home.id()));
        Ok(())
    }

    #[test]
    fn test_clone_workspace_operation_adds_local_trunk_in_same_view_update()
    -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let start_commit = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("clone start")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let repo = tx.commit("create clone start").block_on()?;

        let workspace_name: WorkspaceNameBuf = "vex-clone-test".into();
        let repo = commit_workspace_operation(
            &repo,
            &workspace_name,
            std::slice::from_ref(&start_commit),
            Some(("main", start_commit.id())),
        )
        .block_on()?;

        assert!(
            repo.view()
                .get_wc_commit_id(workspace_name.as_ref())
                .is_some()
        );
        assert_eq!(
            repo.view().get_local_bookmark("main".as_ref()).as_normal(),
            Some(start_commit.id())
        );
        Ok(())
    }

    #[test]
    fn test_vex_clone_workspace_preserves_concurrently_moved_trunk()
    -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let clone_start = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("clone start")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "main".as_ref(),
            crate::op_store::RefTarget::normal(clone_start.id().clone()),
        );
        let repo = tx.commit("create clone start").block_on()?;
        let initial_trunk_target = repo.view().get_local_bookmark("main".as_ref()).clone();

        let mut tx = repo.start_transaction();
        let advanced_trunk = tx
            .repo_mut()
            .new_commit(vec![clone_start.id().clone()], clone_start.tree())
            .set_description("concurrent trunk advance")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "main".as_ref(),
            crate::op_store::RefTarget::normal(advanced_trunk.id().clone()),
        );
        let repo = tx.commit("advance trunk concurrently").block_on()?;

        assert_eq!(
            vex_clone_local_bookmark_to_set(
                &repo,
                Some("main"),
                Some(&initial_trunk_target),
                &clone_start,
            ),
            None
        );
        assert_eq!(
            repo.view().get_local_bookmark("main".as_ref()).as_normal(),
            Some(advanced_trunk.id())
        );
        Ok(())
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

        let target = clone_vex_native_target(&repo, None).block_on()?;
        assert!(matches!(target, NativeCloneTarget::LegacyNative { .. }));
        assert_eq!(target.commit().id(), main_head.id());
        assert_eq!(target.bookmark(), Some("main"));
        assert_ne!(target.commit().id(), fallback_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_prefers_remote_trunk_bookmark() -> Result<(), WorkspaceInitError>
    {
        // Regression: after `vex clone`, the trunk is a remote-tracking bookmark
        // (e.g. `master@vex`), while unrelated local bookmarks may exist. The
        // start commit must be the remote trunk head, not an arbitrary local one.
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let codex_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("codex")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let master_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("master")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "codex/dev-agent-local-guidance".as_ref(),
            crate::op_store::RefTarget::normal(codex_head.id().clone()),
        );
        tx.repo_mut().set_remote_bookmark(
            crate::ref_name::RemoteRefSymbol {
                name: "master".as_ref(),
                remote: "vex".as_ref(),
            },
            crate::op_store::RemoteRef {
                target: crate::op_store::RefTarget::normal(master_head.id().clone()),
                state: crate::op_store::RemoteRefState::Tracked,
            },
        );
        let repo = tx
            .commit("create remote trunk and local bookmark")
            .block_on()?;

        let target = clone_vex_native_target(&repo, None).block_on()?;
        assert!(matches!(target, NativeCloneTarget::LegacyNative { .. }));
        assert_eq!(target.commit().id(), master_head.id());
        assert_eq!(target.bookmark(), Some("master"));
        assert_ne!(target.commit().id(), codex_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_uses_server_trunk() -> Result<(), WorkspaceInitError> {
        // The server registers the trunk (`Repository#default_branch`), surfaced
        // to clone via the repo-access catalog `default_branch`. Given a remote
        // `master@vex` head and an unrelated local `codex/...` head, passing
        // `server_trunk = Some("master")` must check out the remote master head,
        // not the arbitrary local bookmark.
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let codex_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("codex")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let master_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("master")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "codex/dev-agent-local-guidance".as_ref(),
            crate::op_store::RefTarget::normal(codex_head.id().clone()),
        );
        tx.repo_mut().set_remote_bookmark(
            crate::ref_name::RemoteRefSymbol {
                name: "master".as_ref(),
                remote: "vex".as_ref(),
            },
            crate::op_store::RemoteRef {
                target: crate::op_store::RefTarget::normal(master_head.id().clone()),
                state: crate::op_store::RemoteRefState::Tracked,
            },
        );
        let repo = tx
            .commit("create remote trunk and local bookmark")
            .block_on()?;

        let target = clone_vex_native_target(&repo, Some("master")).block_on()?;
        assert!(matches!(target, NativeCloneTarget::ServerTrunk(_)));
        assert_eq!(target.commit().id(), master_head.id());
        assert_eq!(target.bookmark(), Some("master"));
        assert_ne!(target.commit().id(), codex_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_empty_native_repo_uses_root() -> Result<(), WorkspaceInitError> {
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let target = clone_vex_native_target(&repo, Some("main")).block_on()?;
        assert_eq!(target.commit().id(), repo.store().root_commit().id());
        assert_eq!(target.bookmark(), Some("main"));
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_missing_server_trunk_fails_closed()
    -> Result<(), WorkspaceInitError> {
        // The server advertised `main` but no native `main` bookmark exists.
        // Native clone must fail with the typed `NativeTrunkMissing` error
        // (before any working-copy creation) instead of falling through to a
        // differently named bookmark, an arbitrary head, the view's git_head,
        // or `git/ref/*`. A same-name raw Git ref and a git_head are present
        // in the view to prove they are not consulted; the selector takes no
        // client/store handle, so no `git/ref/*` RPC is even reachable.
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let master_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("master")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let git_only_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("git-only main")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_remote_bookmark(
            crate::ref_name::RemoteRefSymbol {
                name: "master".as_ref(),
                remote: "vex".as_ref(),
            },
            crate::op_store::RemoteRef {
                target: crate::op_store::RefTarget::normal(master_head.id().clone()),
                state: crate::op_store::RemoteRefState::Tracked,
            },
        );
        // Raw Git view state for the advertised name must NOT rescue the clone.
        tx.repo_mut().set_git_ref_target(
            "refs/heads/main".as_ref(),
            crate::op_store::RefTarget::normal(git_only_head.id().clone()),
        );
        tx.repo_mut()
            .set_git_head_target(crate::op_store::RefTarget::normal(
                git_only_head.id().clone(),
            ));
        let repo = tx.commit("create master without main").block_on()?;

        let err = clone_vex_native_target(&repo, Some("main"))
            .block_on()
            .expect_err("missing advertised native trunk must fail closed");
        match &err {
            WorkspaceInitError::NativeTrunkMissing { trunk } => assert_eq!(trunk, "main"),
            other => panic!("expected NativeTrunkMissing, got {other:?}"),
        }
        // The operator guidance must be actionable: native conversion or the
        // explicit Git clone surface.
        let message = err.to_string();
        assert!(message.contains("native conversion"), "{message}");
        assert!(message.contains("vex git clone"), "{message}");
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_native_bookmark_wins_over_git_ref()
    -> Result<(), WorkspaceInitError> {
        // Mixed converted state: the native `master@vex` bookmark and a raw
        // Git ref `refs/heads/master` (plus git_head) point at different
        // commits. The server-advertised trunk must resolve to the native
        // bookmark target; the Git-side commits stay untouched.
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let native_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("native master")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let git_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("git master")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_remote_bookmark(
            crate::ref_name::RemoteRefSymbol {
                name: "master".as_ref(),
                remote: "vex".as_ref(),
            },
            crate::op_store::RemoteRef {
                target: crate::op_store::RefTarget::normal(native_head.id().clone()),
                state: crate::op_store::RemoteRefState::Tracked,
            },
        );
        tx.repo_mut().set_git_ref_target(
            "refs/heads/master".as_ref(),
            crate::op_store::RefTarget::normal(git_head.id().clone()),
        );
        tx.repo_mut()
            .set_git_head_target(crate::op_store::RefTarget::normal(git_head.id().clone()));
        let repo = tx
            .commit("create conflicting native and git master")
            .block_on()?;

        let target = clone_vex_native_target(&repo, Some("master")).block_on()?;
        assert!(matches!(target, NativeCloneTarget::ServerTrunk(_)));
        assert_eq!(target.commit().id(), native_head.id());
        assert_eq!(target.bookmark(), Some("master"));
        assert_ne!(target.commit().id(), git_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_no_trunk_stays_native_only() -> Result<(), WorkspaceInitError> {
        // Legacy path (no server trunk): with no native bookmarks, selection
        // must not adopt a raw Git ref name from the view. The newest native
        // head wins; the git-ref'd commit is not treated as a trunk.
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let git_reffed_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("git-ref'd older head")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        std::thread::sleep(std::time::Duration::from_millis(1));
        let newer_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("newer anonymous head")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_git_ref_target(
            "refs/heads/main".as_ref(),
            crate::op_store::RefTarget::normal(git_reffed_head.id().clone()),
        );
        let repo = tx.commit("create git ref without bookmarks").block_on()?;

        let target = clone_vex_native_target(&repo, None).block_on()?;
        assert!(matches!(target, NativeCloneTarget::LegacyNative { .. }));
        assert_eq!(target.commit().id(), newer_head.id());
        assert_eq!(target.bookmark(), None);
        Ok(())
    }

    #[test]
    fn test_clone_vex_checkout_target_exact_target_commit_bypasses_bookmarks()
    -> Result<(), WorkspaceInitError> {
        // An explicit `target_commit` (CI runners) is authoritative: it is
        // loaded as an exact native commit and bookmark selection never runs,
        // even when the advertised server trunk is missing (which would
        // otherwise fail closed with `NativeTrunkMissing`).
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        let exact_commit = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("exact target")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let other_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("other head")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "main".as_ref(),
            crate::op_store::RefTarget::normal(other_head.id().clone()),
        );
        let repo = tx.commit("create exact target and main").block_on()?;

        let (start_commit, resolved_trunk) =
            clone_vex_checkout_target(&repo, Some(exact_commit.id()), Some("does-not-exist"))
                .block_on()?;
        assert_eq!(start_commit.id(), exact_commit.id());
        assert_eq!(resolved_trunk, None);
        assert_ne!(start_commit.id(), other_head.id());
        Ok(())
    }

    #[test]
    fn test_clone_vex_start_commit_server_trunk_not_a_head() -> Result<(), WorkspaceInitError> {
        // The server-registered trunk is authoritative even when its target is
        // not a current view head. Real clones see `master` already carrying
        // descendant commits (so it is an ancestor, not a DAG tip), while an
        // unrelated branch (e.g. `codex/...`) IS a tip. Passing
        // `server_trunk = Some("master")` must still check out the master head,
        // not the arbitrary head branch.
        let settings = user_settings();
        let (_temp_dir, repo) = init_test_repo(&settings)?;

        let mut tx = repo.start_transaction();
        let root = repo.store().root_commit();
        // `master` points here, but a child is committed on top so this commit is
        // an ancestor (not a head) of the visible DAG.
        let master_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("master")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        let master_child = tx
            .repo_mut()
            .new_commit(vec![master_head.id().clone()], root.tree())
            .set_description("master child")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        // Unrelated branch that IS a view head.
        let codex_head = tx
            .repo_mut()
            .new_commit(vec![root.id().clone()], root.tree())
            .set_description("codex")
            .write()
            .block_on()
            .map_err(CheckOutCommitError::CreateCommit)?;
        tx.repo_mut().set_local_bookmark_target(
            "codex/feat-workspace-spaces".as_ref(),
            crate::op_store::RefTarget::normal(codex_head.id().clone()),
        );
        // The server trunk is a remote-tracking bookmark pointing at the
        // non-head `master_head`.
        tx.repo_mut().set_remote_bookmark(
            crate::ref_name::RemoteRefSymbol {
                name: "master".as_ref(),
                remote: "vex".as_ref(),
            },
            crate::op_store::RemoteRef {
                target: crate::op_store::RefTarget::normal(master_head.id().clone()),
                state: crate::op_store::RemoteRefState::Tracked,
            },
        );
        let repo = tx
            .commit("server trunk behind a descendant commit")
            .block_on()?;

        // Sanity: master_head is NOT a view head; master_child and codex_head are.
        let heads = repo.view().heads();
        assert!(!heads.contains(master_head.id()));
        assert!(heads.contains(master_child.id()));
        assert!(heads.contains(codex_head.id()));

        let target = clone_vex_native_target(&repo, Some("master")).block_on()?;
        assert!(matches!(target, NativeCloneTarget::ServerTrunk(_)));
        assert_eq!(target.commit().id(), master_head.id());
        assert_eq!(target.bookmark(), Some("master"));
        assert_ne!(target.commit().id(), codex_head.id());
        assert_ne!(target.commit().id(), master_child.id());
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

        let target = clone_vex_native_target(&repo, None).block_on()?;
        assert_eq!(target.commit().id(), other_head.id());
        assert_ne!(target.commit().id(), default_head.id());
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

        let target = clone_vex_native_target(&repo, None).block_on()?;
        assert_eq!(target.commit().id(), base.id());
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

        let target = clone_vex_native_target(&repo, None).block_on()?;
        assert_eq!(target.commit().id(), newer_head.id());
        assert_ne!(target.commit().id(), older_head.id());
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
