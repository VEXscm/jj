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

//! `vex hint` — enable or disable the occasional nudges the CLI prints.
//!
//! Hints are backed by `hints.<name>` boolean config keys (see
//! `src/config/hints.toml` for the defaults). This command is a friendly
//! wrapper around those keys so users can run, e.g.:
//!
//! ```text
//! vex hint disable watchman
//! ```

use clap::Subcommand;
use itertools::Itertools as _;
use jj_lib::config::ConfigNamePathBuf;
use jj_lib::config::ConfigValue;
use tracing::instrument;

use crate::cli_util::CommandHelper;
use crate::command_error::CommandError;
use crate::command_error::user_error;
use crate::command_error::user_error_with_message;
use crate::ui::Ui;

/// A hint the CLI may print, exposed through `vex hint`.
struct KnownHint {
    /// Primary user-facing name accepted on the command line.
    name: &'static str,
    /// Extra names that also resolve to this hint.
    aliases: &'static [&'static str],
    /// Backing `hints.*` config key.
    config_key: &'static str,
    /// One-line description shown by `vex hint list`.
    description: &'static str,
}

/// The hints that `vex hint` knows how to toggle.
const KNOWN_HINTS: &[KnownHint] = &[
    KnownHint {
        name: "watchman",
        aliases: &["fsmonitor"],
        config_key: "hints.fsmonitor",
        description: "Suggest installing a filesystem monitor when snapshots are slow",
    },
    KnownHint {
        name: "resolving-conflicts",
        aliases: &["conflicts"],
        config_key: "hints.resolving-conflicts",
        description: "Explain how to resolve conflicts after an operation creates them",
    },
];

fn find_hint(name: &str) -> Result<&'static KnownHint, CommandError> {
    KNOWN_HINTS
        .iter()
        .find(|hint| hint.name == name || hint.aliases.contains(&name))
        .ok_or_else(|| {
            let known = KNOWN_HINTS.iter().map(|hint| hint.name).join(", ");
            user_error(format!("Unknown hint \"{name}\". Known hints: {known}"))
        })
}

/// Enable or disable the CLI's occasional hints
#[derive(Subcommand, Clone, Debug)]
pub enum HintCommand {
    /// Stop printing a hint
    Disable(HintToggleArgs),
    /// Resume printing a previously disabled hint
    Enable(HintToggleArgs),
    /// List the hints that can be toggled and their current state
    List(HintListArgs),
}

/// Selects which config file the preference is written to.
#[derive(clap::Args, Clone, Debug)]
struct HintLevelArgs {
    /// Write the preference to the user-level config (default)
    #[arg(long)]
    user: bool,

    /// Write the preference to the repo-level config
    #[arg(long, conflicts_with = "user")]
    repo: bool,
}

/// Toggle a single hint on or off.
#[derive(clap::Args, Clone, Debug)]
pub struct HintToggleArgs {
    /// Name of the hint to toggle (see `vex hint list`)
    name: String,

    #[command(flatten)]
    level: HintLevelArgs,
}

/// List the hints that can be toggled and their current state.
#[derive(clap::Args, Clone, Debug)]
pub struct HintListArgs {}

#[instrument(skip_all)]
pub async fn cmd_hint(
    ui: &mut Ui,
    command: &CommandHelper,
    subcommand: &HintCommand,
) -> Result<(), CommandError> {
    match subcommand {
        HintCommand::Disable(args) => set_hint(ui, command, args, false),
        HintCommand::Enable(args) => set_hint(ui, command, args, true),
        HintCommand::List(_) => list_hints(ui, command),
    }
}

fn set_hint(
    ui: &mut Ui,
    command: &CommandHelper,
    args: &HintToggleArgs,
    enabled: bool,
) -> Result<(), CommandError> {
    let hint = find_hint(&args.name)?;
    let config_env = command.config_env();
    let raw_config = command.raw_config();

    let mut file = if args.level.repo {
        config_env
            .repo_config_files(ui, raw_config)?
            .pop()
            .ok_or_else(|| {
                user_error("No repo config path found; run inside a Vex repository or omit --repo")
            })?
    } else {
        config_env
            .user_config_files(raw_config)?
            .pop()
            .ok_or_else(|| user_error("No user config path found"))?
    };

    let name = ConfigNamePathBuf::from_iter(hint.config_key.split('.'));
    file.set_value(&name, &ConfigValue::from(enabled))
        .map_err(|err| user_error_with_message(format!("Failed to set {name}"), err))?;
    file.save()?;

    writeln!(
        ui.status(),
        "{} hint \"{}\".",
        if enabled { "Enabled" } else { "Disabled" },
        hint.name,
    )?;
    Ok(())
}

fn list_hints(ui: &mut Ui, command: &CommandHelper) -> Result<(), CommandError> {
    let settings = command.settings();
    let mut formatter = ui.stdout_formatter();
    for hint in KNOWN_HINTS {
        // Default to enabled if the key is unset or unreadable, matching the
        // `hints.toml` defaults.
        let enabled = settings.get_bool(hint.config_key).unwrap_or(true);
        writeln!(
            formatter,
            "{:<20} {:<9} {}",
            hint.name,
            if enabled { "enabled" } else { "disabled" },
            hint.description,
        )?;
    }
    Ok(())
}
