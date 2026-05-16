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

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use pollster::FutureExt as _;

use crate::commit::Commit;
use crate::local_working_copy::LocalWorkingCopy;
use crate::merged_tree::MergedTree;
use crate::op_store::OperationId;
use crate::ref_name::WorkspaceName;
use crate::ref_name::WorkspaceNameBuf;
use crate::repo_path::RepoPathBuf;
use crate::settings::UserSettings;
use crate::store::Store;
use crate::working_copy::CheckoutError;
use crate::working_copy::CheckoutStats;
use crate::working_copy::LockedWorkingCopy;
use crate::working_copy::ResetError;
use crate::working_copy::SnapshotError;
use crate::working_copy::SnapshotOptions;
use crate::working_copy::SnapshotStats;
use crate::working_copy::WorkingCopy;
use crate::working_copy::WorkingCopyFactory;
use crate::working_copy::WorkingCopyStateError;

/// Agent-oriented working copy that starts with no materialized paths.
///
/// This is intentionally implemented as a thin wrapper around
/// `LocalWorkingCopy`. The important behavior is the empty sparse set at
/// initialization time, which keeps clone/open metadata-only until a command
/// explicitly materializes paths.
pub struct VirtualWorkingCopy {
    inner: Box<dyn WorkingCopy>,
}

impl VirtualWorkingCopy {
    pub fn name() -> &'static str {
        "vex-virtual"
    }

    fn init(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        operation_id: OperationId,
        workspace_name: WorkspaceNameBuf,
        user_settings: &UserSettings,
    ) -> Result<Self, WorkingCopyStateError> {
        let inner = Box::new(LocalWorkingCopy::init(
            store,
            working_copy_path,
            state_path,
            operation_id,
            workspace_name,
            user_settings,
        )?);
        let inner = configure_virtual_defaults(inner)?;
        Ok(Self { inner })
    }

    fn load(
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        user_settings: &UserSettings,
    ) -> Result<Self, WorkingCopyStateError> {
        Ok(Self {
            inner: Box::new(LocalWorkingCopy::load(
                store,
                working_copy_path,
                state_path,
                user_settings,
            )?),
        })
    }
}

fn configure_virtual_defaults(
    inner: Box<dyn WorkingCopy>,
) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
    if inner.sparse_patterns()?.is_empty() {
        return Ok(inner);
    }
    async {
        let mut locked = inner.start_mutation().await?;
        locked
            .set_sparse_patterns(Vec::new())
            .await
            .map_err(|err| WorkingCopyStateError {
                message: "Failed to initialize virtual working copy sparsity".to_string(),
                err: Box::new(err),
            })?;
        let operation_id = locked.old_operation_id().clone();
        locked.finish(operation_id).await
    }
    .block_on()
}

#[async_trait(?Send)]
impl WorkingCopy for VirtualWorkingCopy {
    fn name(&self) -> &str {
        Self::name()
    }

    fn workspace_name(&self) -> &WorkspaceName {
        self.inner.workspace_name()
    }

    fn operation_id(&self) -> &OperationId {
        self.inner.operation_id()
    }

    fn tree(&self) -> Result<&MergedTree, WorkingCopyStateError> {
        self.inner.tree()
    }

    fn sparse_patterns(&self) -> Result<&[RepoPathBuf], WorkingCopyStateError> {
        self.inner.sparse_patterns()
    }

    async fn start_mutation(&self) -> Result<Box<dyn LockedWorkingCopy>, WorkingCopyStateError> {
        Ok(Box::new(LockedVirtualWorkingCopy {
            inner: self.inner.start_mutation().await?,
        }))
    }
}

pub struct VirtualWorkingCopyFactory;

impl WorkingCopyFactory for VirtualWorkingCopyFactory {
    fn init_working_copy(
        &self,
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        operation_id: OperationId,
        workspace_name: WorkspaceNameBuf,
        settings: &UserSettings,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        Ok(Box::new(VirtualWorkingCopy::init(
            store,
            working_copy_path,
            state_path,
            operation_id,
            workspace_name,
            settings,
        )?))
    }

    fn load_working_copy(
        &self,
        store: Arc<Store>,
        working_copy_path: PathBuf,
        state_path: PathBuf,
        settings: &UserSettings,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        Ok(Box::new(VirtualWorkingCopy::load(
            store,
            working_copy_path,
            state_path,
            settings,
        )?))
    }
}

struct LockedVirtualWorkingCopy {
    inner: Box<dyn LockedWorkingCopy>,
}

#[async_trait]
impl LockedWorkingCopy for LockedVirtualWorkingCopy {
    fn old_operation_id(&self) -> &OperationId {
        self.inner.old_operation_id()
    }

    fn old_tree(&self) -> &MergedTree {
        self.inner.old_tree()
    }

    async fn snapshot(
        &mut self,
        options: &SnapshotOptions,
    ) -> Result<(MergedTree, SnapshotStats), SnapshotError> {
        self.inner.snapshot(options).await
    }

    async fn check_out(&mut self, commit: &Commit) -> Result<CheckoutStats, CheckoutError> {
        self.inner.check_out(commit).await
    }

    fn rename_workspace(&mut self, new_workspace_name: WorkspaceNameBuf) {
        self.inner.rename_workspace(new_workspace_name);
    }

    async fn reset(&mut self, commit: &Commit) -> Result<(), ResetError> {
        self.inner.reset(commit).await
    }

    async fn recover(&mut self, commit: &Commit) -> Result<(), ResetError> {
        self.inner.recover(commit).await
    }

    fn sparse_patterns(&self) -> Result<&[RepoPathBuf], WorkingCopyStateError> {
        self.inner.sparse_patterns()
    }

    async fn set_sparse_patterns(
        &mut self,
        new_sparse_patterns: Vec<RepoPathBuf>,
    ) -> Result<CheckoutStats, CheckoutError> {
        self.inner.set_sparse_patterns(new_sparse_patterns).await
    }

    async fn finish(
        self: Box<Self>,
        operation_id: OperationId,
    ) -> Result<Box<dyn WorkingCopy>, WorkingCopyStateError> {
        let inner = self.inner.finish(operation_id).await?;
        Ok(Box::new(VirtualWorkingCopy { inner }))
    }
}
