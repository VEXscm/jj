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

use std::fs;
use std::io::Write as _;

use jj_lib::file_util;
use jj_lib::vex::VexClient;
use jj_lib::workspace::Workspace;
use tracing::instrument;

use super::CloneBlobModeArg;
use super::VexFsMode;
use super::VexWorkingCopyMode;
use super::parse_repo_path;
use super::repo_auth::RepoAuthArgs;
use super::repo_auth::resolve_repo_auth;
use super::resolve_clone_blob_mode;
use super::resolve_working_copy_mode;
use super::working_copy_factory;
use crate::cli_util::CommandHelper;
use crate::command_error::CommandError;
use crate::command_error::cli_error;
use crate::command_error::user_error;
use crate::command_error::user_error_with_message;
use crate::ui::Ui;

/// Clone an existing repo backed directly by a Vex backend.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct VexCloneArgs {
    /// Vex repository in the form `<tenant>/<repo>`
    repo: String,

    /// The destination directory for the cloned repository.
    #[arg(value_hint = clap::ValueHint::DirPath)]
    destination: Option<String>,

    #[command(flatten)]
    auth: RepoAuthArgs,

    /// Working tree layout and clone-time blob prefetch: `system` (default) or `virtual`.
    #[arg(long = "fs", value_enum, default_value_t = VexFsMode::System)]
    fs: VexFsMode,

    /// Clone-time blob hydration policy.
    #[arg(long, value_enum)]
    blob_mode: Option<CloneBlobModeArg>,

    /// Working-copy implementation to initialize.
    #[arg(long, value_enum)]
    working_copy: Option<VexWorkingCopyMode>,
}

#[instrument(skip_all)]
pub(crate) async fn cmd_vex_clone(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &VexCloneArgs,
) -> Result<(), CommandError> {
    if command.global_args().at_operation.is_some() {
        return Err(cli_error("--at-op is not respected"));
    }

    let (tenant_slug, repo_slug) = parse_repo_path(&args.repo)?;
    let cwd = command.cwd();
    let destination = args.destination.as_deref().unwrap_or(repo_slug);
    let wc_path = cwd.join(destination);

    let wc_path_existed = wc_path.exists();
    if wc_path_existed && !file_util::is_empty_dir(&wc_path)? {
        return Err(user_error(
            "Destination path exists and is not an empty directory",
        ));
    }

    fs::create_dir_all(&wc_path)
        .map_err(|err| user_error_with_message(format!("Failed to create {destination}"), err))?;
    let canonical_wc_path = dunce::canonicalize(&wc_path)
        .map_err(|err| user_error_with_message(format!("Failed to create {destination}"), err))?;

    let clone_result: Result<(), CommandError> = async {
        let auth = resolve_repo_auth(&args.auth, tenant_slug, repo_slug, "JJ native clone").await?;
        let config = VexClient::get_repo(
            &auth.endpoint,
            tenant_slug,
            &auth.repository_slug,
            auth.access_token.as_deref(),
        )
        .await
        .map_err(user_error)?;
        let settings = command
            .settings_for_new_workspace(ui, &canonical_wc_path)?
            .0;
        let blob_mode = resolve_clone_blob_mode(args.fs, args.blob_mode);
        let working_copy_mode = resolve_working_copy_mode(args.fs, args.working_copy);
        let working_copy_factory = working_copy_factory(working_copy_mode);
        // Virtual working copies materialize nothing at checkout, so skip the
        // pre-checkout blob hydration for them.
        let hydrate_blobs = !matches!(working_copy_mode, super::VexWorkingCopyMode::Virtual);
        drop(
            Workspace::clone_vex(
                &settings,
                &canonical_wc_path,
                config,
                blob_mode,
                None,
                None,
                hydrate_blobs,
                &[],
                &*working_copy_factory,
                None,
            )
            .await?,
        );
        Ok(())
    }
    .await;

    if clone_result.is_err() && !wc_path_existed {
        fs::remove_dir_all(&canonical_wc_path).ok();
    }
    clone_result?;

    writeln!(
        ui.status(),
        "Cloned Vex-backed repo into \"{}\"",
        file_util::relative_path(cwd, &canonical_wc_path).display()
    )?;
    Ok(())
}
