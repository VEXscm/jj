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

use std::io::Write as _;

use jj_lib::file_util;
use jj_lib::vex::VexClient;
use jj_lib::workspace::Workspace;
use tracing::instrument;

use super::VexFsMode;
use super::VexWorkingCopyMode;
use super::parse_repo_path;
use super::repo_auth::RepoAuthArgs;
use super::repo_auth::resolve_repo_auth;
use super::resolve_working_copy_mode;
use super::working_copy_factory;
use crate::cli_util::CommandHelper;
use crate::command_error::CommandError;
use crate::command_error::cli_error;
use crate::command_error::user_error;
use crate::command_error::user_error_with_message;
use crate::ui::Ui;

/// Create a new repo backed directly by a Vex backend.
#[derive(clap::Args, Clone, Debug)]
pub(crate) struct VexInitArgs {
    /// Vex repository in the form `<tenant>/<repo>`
    repo: String,

    /// The destination directory where the `jj` repo will be created.
    #[arg(default_value = ".", value_hint = clap::ValueHint::DirPath)]
    destination: String,

    #[command(flatten)]
    auth: RepoAuthArgs,

    /// Working tree layout: `system` (default, on-disk) or `virtual` (`vex-virtual`).
    #[arg(long = "fs", value_enum, default_value_t = VexFsMode::System)]
    fs: VexFsMode,

    /// Working-copy implementation to initialize.
    #[arg(long, value_enum)]
    working_copy: Option<VexWorkingCopyMode>,
}

#[instrument(skip_all)]
pub(crate) async fn cmd_vex_init(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &VexInitArgs,
) -> Result<(), CommandError> {
    if command.global_args().no_integrate_operation {
        return Err(cli_error("--no-integrate-operation is not respected"));
    }
    if command.global_args().ignore_working_copy {
        return Err(cli_error("--ignore-working-copy is not respected"));
    }
    if command.global_args().at_operation.is_some() {
        return Err(cli_error("--at-op is not respected"));
    }

    let cwd = command.cwd();
    let wc_path = cwd.join(&args.destination);
    let wc_path = file_util::create_or_reuse_dir(&wc_path)
        .and_then(|_| dunce::canonicalize(wc_path))
        .map_err(|e| user_error_with_message("Failed to create workspace", e))?;
    let (tenant_slug, repo_slug) = parse_repo_path(&args.repo)?;
    let auth = resolve_repo_auth(&args.auth, tenant_slug, repo_slug, "JJ native init").await?;
    let config = VexClient::init_repo(
        &auth.endpoint,
        tenant_slug,
        &auth.repository_slug,
        auth.access_token.as_deref(),
    )
    .await
    .map_err(user_error)?;
    let settings = command.settings_for_new_workspace(ui, &wc_path)?.0;
    let working_copy_mode = resolve_working_copy_mode(args.fs, args.working_copy);
    let working_copy_factory = working_copy_factory(working_copy_mode);

    Workspace::init_vex(&settings, &wc_path, config, &*working_copy_factory).await?;

    let relative_wc_path = file_util::relative_path(cwd, &wc_path);
    writeln!(
        ui.status(),
        "Initialized Vex-backed repo in \"{}\"",
        relative_wc_path.display()
    )?;
    Ok(())
}
