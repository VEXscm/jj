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
    // Resolve through the workspace loader so a top-level `-R <path>` selects
    // the repository exactly as it does for every other command; the cwd walk
    // is only the fallback for an invocation that has no loadable workspace.
    let dir = match command.workspace_loader() {
        Ok(loader) => loader.repo_path().join("op_heads"),
        Err(err) => find_op_heads_dir(command.cwd()).ok_or(err)?,
    };
    let queued = read_pending_publish(&dir)
        .map_err(|err| user_error_with_message("cannot read the publish queue", err))?;
    let depth = queued.as_ref().map_or(0, |chain| chain.ops.len());
    writeln!(ui.stdout(), "queue: {depth} operation(s)")?;
    if let Some(chain) = &queued {
        writeln!(ui.stdout(), "base:  {}", chain.base_heads.join(", "))?;
        for entry in &chain.ops {
            writeln!(
                ui.stdout(),
                "  op {} ({} objects)",
                entry.op,
                entry.objects.len()
            )?;
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

    // A moved server head is the ordinary concurrent case, not a failure: the
    // repository serves both heads, loading it makes jj's own op-head
    // resolution merge them into a merge operation, and that merge publishes
    // against the moved head. Do that here instead of asking the caller to
    // reload and re-run.
    const RELOAD_MERGE_ATTEMPTS: usize = 3;
    let mut moved_to: Vec<String> = Vec::new();
    for attempt in 0..RELOAD_MERGE_ATTEMPTS {
        match ensure_published_at(&dir)
            .map_err(|err| user_error_with_message("publish failed", err))?
        {
            PublishOutcome::Idle => {
                writeln!(ui.stdout(), "nothing to publish")?;
                return Ok(());
            }
            PublishOutcome::Published {
                ops,
                coalesced,
                head,
                elapsed_ms,
            } => {
                writeln!(
                    ui.stdout(),
                    "published {ops} operation(s) as {head} in {elapsed_ms}ms{}",
                    if coalesced { " (coalesced)" } else { "" }
                )?;
                return Ok(());
            }
            PublishOutcome::ServerHeadMoved { server_heads } => {
                moved_to = server_heads.iter().map(ToString::to_string).collect();
            }
        }
        if attempt + 1 == RELOAD_MERGE_ATTEMPTS {
            break;
        }
        writeln!(
            ui.status(),
            "server operation head moved to {}; merging it and retrying",
            moved_to.join(", ")
        )?;
        // `no_snapshot`: this is a publish barrier, not a working-copy mutation.
        command.workspace_helper_no_snapshot(ui).await?;
    }
    // A moved head is no longer a refusal in itself (roadmap 093): this publish
    // and the writer that beat us would both have landed. Exhausting the
    // attempts means every merge we built was outraced before we could publish
    // it, so report the sustained race rather than the moved head.
    Err(user_error(format!(
        "every attempt to merge and republish was outraced (heads now {}); this repository is \
         under sustained concurrent writes — the queued operations are safe on disk and the \
         next command will retry them",
        moved_to.join(", ")
    )))
}
