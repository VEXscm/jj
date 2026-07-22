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

use thiserror::Error;

use crate::backend::BackendInitError;
use crate::backend::CommitId;
use crate::commit::Commit;
use crate::file_util;
use crate::file_util::BadPathEncoding;
use crate::file_util::IoResultExt as _;
use crate::file_util::PathError;
use crate::local_working_copy::LocalWorkingCopy;
use crate::local_working_copy::LocalWorkingCopyFactory;
use crate::merged_tree::MergedTree;
use crate::object_id::ObjectId as _;
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

/// Vex's production op-head store uses a strict single-head CAS. A large
/// clone can spend many minutes fetching metadata after loading its initial
/// operation, so its final workspace transaction is often based on an old
/// operation head. Retrying the already-written operation cannot work: the
/// server validates that the operation's parent ids equal the CAS expected
/// head. Instead, reload the current operation and rebuild the workspace view
/// mutation on top of it. The workspace name is unique to this clone, so this
/// preserves concurrent view changes rather than replacing them.
const MAX_VEX_CLONE_WORKSPACE_OPERATION_ATTEMPTS: u32 = 3;

fn is_vex_op_heads_cas_conflict(error: &TransactionCommitError) -> bool {
    matches!(
        error,
        TransactionCommitError::OpHeadsStore(OpHeadsStoreError::Write { source, .. })
            if source.to_string().contains("CAS conflict on op heads")
    )
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
    // The clone's first `load_at_head()` happens before manifest/prefetch work.
    // Refresh immediately before constructing the write transaction, then do
    // so again after an exact CAS rejection. Each attempt writes a new
    // operation whose parent is the currently published head.
    let mut repo = reload_vex_clone_repo_at_head(repo).await?;
    let mut attempt = 1;
    loop {
        // A clone's workspace name is fresh, but the resolved trunk bookmark
        // is shared state. If another operation moved it while this clone was
        // fetching, preserve that newer value instead of resetting it to the
        // clone's earlier checkout target.
        let local_bookmark = vex_clone_local_bookmark_to_set(
            &repo,
            resolved_trunk,
            initial_resolved_trunk_target.as_ref(),
            start_commit,
        );
        match commit_workspace_operation(
            &repo,
            workspace_name,
            std::slice::from_ref(start_commit),
            local_bookmark,
        )
        .await
        {
            Ok(repo) => return Ok(repo),
            Err(WorkspaceInitError::TransactionCommit(error))
                if attempt < MAX_VEX_CLONE_WORKSPACE_OPERATION_ATTEMPTS
                    && is_vex_op_heads_cas_conflict(&error) =>
            {
                tracing::warn!(
                    attempt,
                    max_attempts = MAX_VEX_CLONE_WORKSPACE_OPERATION_ATTEMPTS,
                    "vex clone workspace op-head CAS conflict; reloading and retrying"
                );
                repo = reload_vex_clone_repo_at_head(&repo).await?;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn init_vex_clone_working_copy_at(
    repo: &Arc<ReadonlyRepo>,
    workspace_root: &Path,
    jj_dir: &Path,
    working_copy_factory: &dyn WorkingCopyFactory,
    workspace_name: WorkspaceNameBuf,
    start_commit: &Commit,
    resolved_trunk: Option<&str>,
    progress: Option<&crate::vex::CloneProgressFn>,
) -> Result<(Box<dyn WorkingCopy>, Arc<ReadonlyRepo>), WorkspaceInitError> {
    let working_copy_state_path = jj_dir.join("working_copy");
    std::fs::create_dir(&working_copy_state_path).context(&working_copy_state_path)?;

    if let Some(progress) = progress {
        progress(crate::vex::CloneProgress::WorkspacePublish);
    }
    let repo =
        commit_vex_clone_workspace_operation(repo, &workspace_name, start_commit, resolved_trunk)
            .await?;
    if let Some(progress) = progress {
        progress(crate::vex::CloneProgress::CheckingOut);
    }
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
        // (never `git/ref/*`). Ignored when `target_commit` is `Some`. `None`
        // falls back to the native-only main/master/trunk heuristic.
        server_trunk: Option<&str>,
        // Whether to bulk-hydrate the start commit's file/symlink contents into
        // the local cache before checkout (lazy clones only). Callers pass
        // `false` for virtual working copies, which materialize nothing — the
        // factory itself can't tell us (no identity on `WorkingCopyFactory`).
        // Also gates snapshot-pack consumption (roadmap/032): a clone that
        // won't materialize files must not download working-tree snapshots.
        hydrate_blobs: bool,
        // Extra snapshot commit ids (64-char hex) the caller already holds
        // fully unpacked, sent as `have_snapshot_commit_ids` on the clone
        // manifest request on top of the shared cache's `.snapshots` markers
        // (`vex bench clone --have`).
        have_snapshot_commit_ids: &[String],
        working_copy_factory: &dyn WorkingCopyFactory,
        progress: Option<&crate::vex::CloneProgressFn>,
    ) -> Result<(Self, Arc<ReadonlyRepo>, Option<String>), WorkspaceInitError> {
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
                    .get_clone_manifest(blob_mode, have_snapshot_commit_ids, progress)
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
                // Snapshot packs carry the trunk working tree's blob closure;
                // they are only worth downloading when this clone will
                // materialize files from the cache (lazy + non-virtual — the
                // same conditions as the hydration step below).
                let fetch_snapshot_packs = hydrate_blobs && blob_mode == CloneBlobMode::Lazy;
                prefetch_client
                    .prefetch_clone_manifest(&clone_manifest, fetch_snapshot_packs, progress)
                    .await
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            }

            if let Some(progress) = progress {
                progress(crate::vex::CloneProgress::LoadingRepo);
            }
            let mut store_factories = StoreFactories::default();
            store_factories.merge(create_store_factories());
            let repo_loader =
                RepoLoader::init_from_file_system(user_settings, &repo_dir, &store_factories)
                    .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            let repo = repo_loader
                .load_at_head_with_before_index(|| {
                    if let Some(progress) = progress {
                        progress(crate::vex::CloneProgress::Indexing);
                    }
                })
                .await
                .map_err(|err| WorkspaceInitError::Backend(BackendInitError(err.into())))?;
            let workspace_store = SimpleWorkspaceStore::load(&repo_dir)?;
            let (start_commit, resolved_trunk) =
                clone_vex_checkout_target(&repo, target_commit, server_trunk).await?;
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
/// typed [`WorkspaceInitError::NativeTrunkMissing`] error — there is no
/// fallback chain.
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
    crate::vex::vex_client_stats().record_native_trunk_missing();
    Err(WorkspaceInitError::NativeTrunkMissing {
        trunk: server_trunk.to_owned(),
    })
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
/// vex-cache before checkout, replacing thousands of per-file `GetObject` RPCs
/// during materialization with a few batched `GetObjectsInline` reads.
/// Best-effort: on any failure it logs and returns, and checkout falls back to
/// hydrating the remaining files on demand exactly as before.
/// Returns the number of file/symlink tree entries in the start commit when
/// the hydration walk ran (used as the materializing-progress total), or
/// `None` when the walk was skipped (snapshot packs already cover the commit)
/// or failed.
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
    // Snapshot fast path (roadmap/032 Stage 4): a fully-unpacked snapshot set
    // covering the start commit means every blob/symlink in its tree is
    // already in the local cache — skip the hydration tree walk entirely.
    // Recorded in the `snapshot_walk_skips` counter so `vex bench clone` can
    // tell which path ran.
    if client.has_unpacked_snapshot(&start_commit.id().hex()) {
        crate::vex::vex_client_stats()
            .snapshot_walk_skips
            .fetch_add(1, AtomicOrdering::Relaxed);
        tracing::debug!(
            commit_id = %start_commit.id().hex(),
            "clone hydration: snapshot packs cover the start commit; skipping hydration walk"
        );
        return None;
    }
    let (ids, file_count) = match clone_vex_hydration_ids(repo, start_commit).await {
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

    fn vex_op_heads_cas_conflict_error() -> TransactionCommitError {
        TransactionCommitError::OpHeadsStore(OpHeadsStoreError::Write {
            new_op_id: OperationId::new(vec![7; 32]),
            source: Box::new(std::io::Error::other("CAS conflict on op heads")),
        })
    }

    #[test]
    fn test_vex_clone_workspace_retry_only_matches_op_head_cas_conflicts() {
        assert!(is_vex_op_heads_cas_conflict(
            &vex_op_heads_cas_conflict_error()
        ));

        let unrelated_write = TransactionCommitError::OpHeadsStore(OpHeadsStoreError::Write {
            new_op_id: OperationId::new(vec![7; 32]),
            source: Box::new(std::io::Error::other("connection reset by peer")),
        });
        assert!(!is_vex_op_heads_cas_conflict(&unrelated_write));

        let read_error = TransactionCommitError::OpHeadsStore(OpHeadsStoreError::Read(Box::new(
            std::io::Error::other("CAS conflict on op heads"),
        )));
        assert!(!is_vex_op_heads_cas_conflict(&read_error));
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
