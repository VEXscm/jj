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

use jj_lib::vex_publish::PublishOutcome;
use jj_lib::vex_publish::ensure_published_at;
use jj_lib::vex_publish::find_op_heads_dir;
use jj_lib::vex_publish::read_pending_publish;
use jj_lib::vex_publish::read_server_heads;

use crate::cli_util::CommandHelper;
use crate::command_error::CommandError;
use crate::command_error::user_error;
use crate::command_error::user_error_with_message;
use crate::ui::Ui;

/// Force-drain the local-first publish queue and report the CAS outcome
#[derive(clap::Args, Clone, Debug)]
pub struct DebugPublishArgs {
    /// Report the queue without publishing anything
    #[arg(long)]
    dry_run: bool,
}

pub async fn cmd_debug_publish(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &DebugPublishArgs,
) -> Result<(), CommandError> {
    let Some(dir) = find_op_heads_dir(command.cwd()) else {
        return Err(user_error("no repository found for the current directory"));
    };
    let queued = read_pending_publish(&dir)
        .map_err(|err| user_error_with_message("cannot read the publish queue", err))?;
    let depth = queued.as_ref().map_or(0, |chain| chain.ops.len());
    writeln!(ui.stdout(), "queue: {depth} operation(s)")?;
    if let Some(chain) = &queued {
        writeln!(ui.stdout(), "base:  {}", chain.base_heads.join(", "))?;
        for entry in &chain.ops {
            writeln!(ui.stdout(), "  op {} ({} objects)", entry.op, entry.objects.len())?;
        }
    }
    if let Some(server) = read_server_heads(&dir)
        .map_err(|err| user_error_with_message("cannot read the server head marker", err))?
    {
        writeln!(ui.stdout(), "server heads: {}", server.heads.join(", "))?;
        if let Some(local) = &server.published_local_head {
            writeln!(ui.stdout(), "standing for local head: {local}")?;
        }
    }
    if args.dry_run {
        return Ok(());
    }

    match ensure_published_at(&dir)
        .map_err(|err| user_error_with_message("publish failed", err))?
    {
        PublishOutcome::Idle => writeln!(ui.stdout(), "nothing to publish")?,
        PublishOutcome::Published {
            ops,
            coalesced,
            head,
            elapsed_ms,
        } => writeln!(
            ui.stdout(),
            "published {ops} operation(s) as {head} in {elapsed_ms}ms{}",
            if coalesced { " (coalesced)" } else { "" }
        )?,
        PublishOutcome::ServerHeadMoved { server_heads } => {
            let heads: Vec<String> = server_heads.iter().map(ToString::to_string).collect();
            return Err(user_error(format!(
                "the server operation head moved to {}; reload the repository to merge it, then \
                 publish again",
                heads.join(", ")
            )));
        }
    }
    Ok(())
}
