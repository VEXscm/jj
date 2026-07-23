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

use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use itertools::Itertools as _;
use jj_lib::backend::CommitId;
use jj_lib::backend::MillisSinceEpoch;
use jj_lib::backend::Timestamp;
use jj_lib::ref_name::WorkspaceName;
use jj_lib::ref_name::WorkspaceNameBuf;
use jj_lib::workspace_store::SimpleWorkspaceStore;
use jj_lib::workspace_store::WorkspaceStore as _;
use tracing::instrument;

use crate::cli_util::CommandHelper;
use crate::command_error::CommandError;
use crate::command_error::user_error;
use crate::ui::Ui;

const DEFAULT_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Remove stale, empty working-copy commits from no-longer-used workspaces.
#[derive(clap::Args, Clone, Debug)]
pub struct WorkspaceGcArgs {
    /// Only clean workspaces older than this duration (for example, 30d, 24h, or 60m)
    #[arg(long, value_name = "DURATION")]
    older_than: Option<String>,

    /// Show which workspaces would be cleaned without changing anything
    #[arg(long)]
    dry_run: bool,
}

#[instrument(skip_all)]
pub async fn cmd_workspace_gc(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &WorkspaceGcArgs,
) -> Result<(), CommandError> {
    let max_age = args
        .older_than
        .as_deref()
        .map(parse_duration)
        .transpose()
        .map_err(user_error)?
        .unwrap_or(DEFAULT_MAX_AGE);
    let cutoff = cutoff_timestamp(max_age).map_err(user_error)?;

    // A dry run must never snapshot the working copy, since snapshotting can
    // itself create an operation.
    let mut workspace_command = if args.dry_run {
        command.workspace_helper_no_snapshot(ui).await?
    } else {
        command.workspace_helper(ui).await?
    };
    let candidates = select_workspace_gc_candidates(
        workspace_command.repo().as_ref(),
        workspace_command.workspace_name(),
        cutoff,
    )
    .await;

    if candidates.is_empty() {
        writeln!(ui.status(), "Nothing to clean up.")?;
        return Ok(());
    }

    let noun = if candidates.len() == 1 {
        "workspace"
    } else {
        "workspaces"
    };
    if args.dry_run {
        writeln!(ui.status(), "Would clean up {} {noun}:", candidates.len(),)?;
        for workspace in &candidates {
            writeln!(ui.status(), "  {}", workspace.as_symbol())?;
        }
        writeln!(
            ui.status(),
            "Dry-run requested; no workspaces were removed."
        )?;
        return Ok(());
    }

    writeln!(ui.status(), "Cleaning up {} {noun}:", candidates.len())?;
    for workspace in &candidates {
        writeln!(ui.status(), "  {}", workspace.as_symbol())?;
    }

    let workspace_store = SimpleWorkspaceStore::load(workspace_command.repo_path())?;

    // Keep every removal in one operation, so an undo restores the complete
    // compaction instead of only a subset of its workspaces.
    let mut tx = workspace_command.start_transaction();
    for workspace in &candidates {
        tx.repo_mut().remove_wc_commit(workspace).await?;
    }
    let workspace_refs: Vec<&WorkspaceName> = candidates
        .iter()
        .map(|workspace| workspace.as_ref())
        .collect();
    workspace_store.forget(&workspace_refs)?;

    let description = if let [workspace] = candidates.as_slice() {
        format!("clean up workspace {}", workspace.as_symbol())
    } else {
        format!(
            "clean up workspaces {}",
            candidates
                .iter()
                .map(|workspace| workspace.as_symbol())
                .join(", ")
        )
    };
    tx.finish(ui, description).await?;
    Ok(())
}

fn parse_duration(value: &str) -> Result<Duration, String> {
    let Some(unit) = value.chars().last() else {
        return Err(invalid_duration_error());
    };
    let Some(number) = value.strip_suffix(unit) else {
        return Err(invalid_duration_error());
    };
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(invalid_duration_error());
    }
    let quantity = number
        .parse::<u64>()
        .map_err(|_| invalid_duration_error())?;
    if quantity == 0 {
        return Err(invalid_duration_error());
    }

    let seconds_per_unit = match unit {
        'd' => 24 * 60 * 60,
        'h' => 60 * 60,
        'm' => 60,
        _ => return Err(invalid_duration_error()),
    };
    let seconds = quantity
        .checked_mul(seconds_per_unit)
        .ok_or_else(invalid_duration_error)?;
    Ok(Duration::from_secs(seconds))
}

fn invalid_duration_error() -> String {
    "--older-than must be a positive duration like 30d, 24h, or 60m".to_string()
}

fn cutoff_timestamp(max_age: Duration) -> Result<MillisSinceEpoch, String> {
    let age_millis =
        i64::try_from(max_age.as_millis()).map_err(|_| "--older-than is too large".to_string())?;
    Timestamp::now()
        .timestamp
        .0
        .checked_sub(age_millis)
        .map(MillisSinceEpoch)
        .ok_or_else(|| "--older-than is too large".to_string())
}

async fn select_workspace_gc_candidates(
    repo: &dyn jj_lib::repo::Repo,
    current_workspace: &WorkspaceName,
    cutoff: MillisSinceEpoch,
) -> Vec<WorkspaceNameBuf> {
    let view = repo.view();
    let workspaces: Vec<(WorkspaceNameBuf, CommitId)> = view
        .wc_commit_ids()
        .iter()
        .map(|(name, commit_id)| (name.clone(), commit_id.clone()))
        .collect();
    let workspace_ref_counts = workspaces
        .iter()
        .fold(HashMap::new(), |mut counts, (_, id)| {
            *counts.entry(id.clone()).or_insert(0usize) += 1;
            counts
        });
    let local_ref_ids: HashSet<CommitId> = view
        .local_bookmarks()
        .flat_map(|(_, target)| target.added_ids())
        .chain(view.local_tags().flat_map(|(_, target)| target.added_ids()))
        .cloned()
        .collect();

    let mut candidates = Vec::new();
    for (workspace, wc_commit_id) in workspaces {
        let workspace_name: &WorkspaceName = workspace.as_ref();
        if workspace_name == WorkspaceName::DEFAULT || workspace_name == current_workspace {
            continue;
        }
        if workspace_ref_counts[&wc_commit_id] != 1 || local_ref_ids.contains(&wc_commit_id) {
            continue;
        }

        let wc_commit = match repo.store().get_commit_async(&wc_commit_id).await {
            Ok(commit) => commit,
            Err(error) => {
                tracing::debug!(
                    workspace = %workspace.as_symbol(),
                    %error,
                    "skipping workspace GC candidate because its working-copy commit could not be read"
                );
                continue;
            }
        };
        if wc_commit.parent_ids().len() != 1
            || !wc_commit.description().is_empty()
            || wc_commit.committer().timestamp.timestamp >= cutoff
        {
            continue;
        }

        let parent = match repo
            .store()
            .get_commit_async(&wc_commit.parent_ids()[0])
            .await
        {
            Ok(commit) => commit,
            Err(error) => {
                tracing::debug!(
                    workspace = %workspace.as_symbol(),
                    %error,
                    "skipping workspace GC candidate because its parent commit could not be read"
                );
                continue;
            }
        };
        if wc_commit.tree_ids() == parent.tree_ids() {
            candidates.push(workspace);
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_duration_accepts_supported_positive_units() {
        assert_eq!(
            parse_duration("2d"),
            Ok(Duration::from_secs(2 * 24 * 60 * 60))
        );
        assert_eq!(parse_duration("3h"), Ok(Duration::from_secs(3 * 60 * 60)));
        assert_eq!(parse_duration("4m"), Ok(Duration::from_secs(4 * 60)));
    }

    #[test]
    fn parse_duration_rejects_invalid_values() {
        for value in ["", "0d", "-1d", "1s", "d", "1.5h"] {
            assert_eq!(parse_duration(value), Err(invalid_duration_error()));
        }
    }
}
