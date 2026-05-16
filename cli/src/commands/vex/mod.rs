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

mod clone;
mod init;
mod repo_auth;

use clap::Subcommand;
use jj_lib::local_working_copy::LocalWorkingCopyFactory;
use jj_lib::vex::CloneBlobMode;
use jj_lib::virtual_working_copy::VirtualWorkingCopyFactory;
use jj_lib::working_copy::WorkingCopyFactory;

use self::clone::VexCloneArgs;
use self::clone::cmd_vex_clone;
use self::init::VexInitArgs;
use self::init::cmd_vex_init;
use crate::cli_util::CommandHelper;
use crate::command_error::CommandError;
use crate::command_error::user_error;
use crate::ui::Ui;

/// Commands for working with Vex-backed JJ repositories
#[derive(Subcommand, Clone, Debug)]
pub enum VexCommand {
    Clone(VexCloneArgs),
    Init(VexInitArgs),
}

/// Working tree and clone-time blob strategy for `jj vex clone` / `jj vex init`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum VexFsMode {
    /// Full materialization on the local filesystem; clone prefetches blob bodies (default).
    #[default]
    System,
    /// Virtual working copy (`vex-virtual`); clone uses lazy blob prefetch.
    Virtual,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VexWorkingCopyMode {
    Local,
    Virtual,
}

pub(crate) fn resolve_clone_blob_mode(
    fs: VexFsMode,
    blob_mode: Option<CloneBlobModeArg>,
) -> CloneBlobMode {
    match blob_mode {
        Some(CloneBlobModeArg::Eager) => CloneBlobMode::Eager,
        Some(CloneBlobModeArg::Lazy) => CloneBlobMode::Lazy,
        None => match fs {
            VexFsMode::Virtual => CloneBlobMode::Lazy,
            VexFsMode::System => CloneBlobMode::Eager,
        },
    }
}

pub(crate) fn resolve_working_copy_mode(
    fs: VexFsMode,
    working_copy_mode: Option<VexWorkingCopyMode>,
) -> VexWorkingCopyMode {
    match working_copy_mode {
        Some(mode) => mode,
        None => match fs {
            VexFsMode::Virtual => VexWorkingCopyMode::Virtual,
            VexFsMode::System => VexWorkingCopyMode::Local,
        },
    }
}

pub(crate) fn working_copy_factory(mode: VexWorkingCopyMode) -> Box<dyn WorkingCopyFactory> {
    match mode {
        VexWorkingCopyMode::Local => Box::new(LocalWorkingCopyFactory {}),
        VexWorkingCopyMode::Virtual => Box::new(VirtualWorkingCopyFactory),
    }
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CloneBlobModeArg {
    Eager,
    Lazy,
}

pub async fn cmd_vex(
    ui: &mut Ui,
    command: &CommandHelper,
    subcommand: &VexCommand,
) -> Result<(), CommandError> {
    match subcommand {
        VexCommand::Clone(args) => cmd_vex_clone(ui, command, args).await,
        VexCommand::Init(args) => cmd_vex_init(ui, command, args).await,
    }
}

pub(crate) fn parse_repo_path(spec: &str) -> Result<(&str, &str), CommandError> {
    let (tenant, repo) = spec
        .split_once('/')
        .ok_or_else(|| user_error("Vex repository must be specified as <tenant>/<repo>"))?;
    if tenant.is_empty() || repo.is_empty() {
        return Err(user_error(
            "Vex repository must be specified as <tenant>/<repo>",
        ));
    }
    Ok((tenant, repo))
}

#[cfg(test)]
mod tests {
    use jj_lib::vex::CloneBlobMode;

    use super::CloneBlobModeArg;
    use super::VexFsMode;
    use super::VexWorkingCopyMode;
    use super::parse_repo_path;
    use super::resolve_clone_blob_mode;
    use super::resolve_working_copy_mode;

    #[test]
    fn parse_repo_path_accepts_tenant_and_repo() {
        let (tenant, repo) = parse_repo_path("acme/project").unwrap();
        assert_eq!(tenant, "acme");
        assert_eq!(repo, "project");
    }

    #[test]
    fn parse_repo_path_rejects_invalid_specs() {
        assert!(parse_repo_path("acme").is_err());
        assert!(parse_repo_path("/project").is_err());
        assert!(parse_repo_path("acme/").is_err());
    }

    #[test]
    fn virtual_fs_profile_defaults_to_lazy_virtual() {
        assert_eq!(
            resolve_clone_blob_mode(VexFsMode::Virtual, None),
            CloneBlobMode::Lazy
        );
        assert_eq!(
            resolve_working_copy_mode(VexFsMode::Virtual, None),
            VexWorkingCopyMode::Virtual
        );
    }

    #[test]
    fn system_fs_profile_defaults_to_eager_local() {
        assert_eq!(
            resolve_clone_blob_mode(VexFsMode::System, None),
            CloneBlobMode::Eager
        );
        assert_eq!(
            resolve_working_copy_mode(VexFsMode::System, None),
            VexWorkingCopyMode::Local
        );
    }

    #[test]
    fn explicit_modes_override_fs_profile_defaults() {
        assert_eq!(
            resolve_clone_blob_mode(VexFsMode::Virtual, Some(CloneBlobModeArg::Eager)),
            CloneBlobMode::Eager
        );
        assert_eq!(
            resolve_working_copy_mode(VexFsMode::Virtual, Some(VexWorkingCopyMode::Local)),
            VexWorkingCopyMode::Local
        );
    }
}
