//! Bounds Vex's automatic stale-working-copy recovery.
//!
//! A stale working copy normally means a transient lost op-head race, and
//! recovering in place is the right move. When the backend is unhealthy the
//! same command goes stale over and over, and each recovery re-snapshots
//! whatever is on disk at that moment — which can split one logical change
//! across several commits or leave a half-written file committed. After a
//! couple of consecutive recoveries the odds that another retry helps are low
//! and the cost of being wrong is high, so Vex stops and hands control back.

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use serde::Deserialize;
use serde::Serialize;

/// Recoveries allowed before Vex refuses to keep going on its own.
const MAX_CONSECUTIVE_RECOVERIES: u32 = 2;

/// How long a recovery counts toward the streak. Long enough to cover an
/// agent retrying a slow command, short enough that unrelated staleness weeks
/// apart never trips the guard.
const RECOVERY_WINDOW_SECONDS: i64 = 15 * 60;

const STATE_FILE: &str = "vex-stale-recovery.json";

#[derive(Default, Deserialize, Serialize)]
struct RecoveryState {
    #[serde(default)]
    consecutive_recoveries: u32,
    #[serde(default)]
    last_recovery_unix_seconds: Option<i64>,
}

/// Records one automatic recovery and reports whether the caller may proceed.
///
/// Returns `false` once the streak is exhausted, meaning the caller should
/// surface the staleness instead of recovering again.
pub fn record_recovery(workspace_root: &Path) -> bool {
    let Some(state_path) = state_path(workspace_root) else {
        // Without somewhere to keep the count the guard cannot be enforced;
        // preserve the previous always-recover behavior rather than blocking.
        return true;
    };
    let now = match unix_seconds() {
        Some(now) => now,
        None => return true,
    };

    let mut state = read_state(&state_path);
    if !within_window(&state, now) {
        state.consecutive_recoveries = 0;
    }
    state.consecutive_recoveries = state.consecutive_recoveries.saturating_add(1);
    state.last_recovery_unix_seconds = Some(now);
    let may_recover = state.consecutive_recoveries <= MAX_CONSECUTIVE_RECOVERIES;
    write_state(&state_path, &state);
    may_recover
}

/// Clears the streak after a snapshot that needed no recovery.
pub fn clear(workspace_root: &Path) {
    let Some(state_path) = state_path(workspace_root) else {
        return;
    };
    if state_path.exists() {
        drop(fs::remove_file(state_path));
    }
}

/// Operator-facing explanation for a refused recovery.
pub fn exhausted_message() -> String {
    format!(
        "the working copy went stale {MAX_CONSECUTIVE_RECOVERIES} times in a row, so Vex stopped \
         recovering automatically. Repeated recovery under an unhealthy backend can split one \
         change across several commits or commit a partially written file."
    )
}

/// Recovery steps to attach to [`exhausted_message`].
pub fn exhausted_hint() -> &'static str {
    "This usually means the Vex backend is degraded or another session is writing to this \
     repository. Check `vex status` and `vex log` before retrying. Recover manually with `vex \
     workspace update-stale`, and use `vex op log` / `vex op restore <operation_id>` to undo a \
     recovery that took the wrong state."
}

fn state_path(workspace_root: &Path) -> Option<PathBuf> {
    let jj_dir = workspace_root.join(".jj");
    jj_dir.is_dir().then(|| jj_dir.join(STATE_FILE))
}

fn within_window(state: &RecoveryState, now: i64) -> bool {
    state
        .last_recovery_unix_seconds
        .is_some_and(|last| now.saturating_sub(last) < RECOVERY_WINDOW_SECONDS)
}

fn unix_seconds() -> Option<i64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_secs()).ok())
}

fn read_state(state_path: &Path) -> RecoveryState {
    fs::read(state_path)
        .ok()
        .and_then(|contents| serde_json::from_slice(&contents).ok())
        .unwrap_or_default()
}

fn write_state(state_path: &Path, state: &RecoveryState) {
    let Ok(contents) = serde_json::to_vec(state) else {
        return;
    };
    let temporary_path = state_path.with_extension(format!("json.{}", std::process::id()));
    if fs::write(&temporary_path, contents).is_ok() && fs::rename(&temporary_path, state_path).is_err()
    {
        drop(fs::remove_file(temporary_path));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> tempfile::TempDir {
        let temp_dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(temp_dir.path().join(".jj")).unwrap();
        temp_dir
    }

    #[test]
    fn allows_the_first_recoveries_then_refuses() {
        let temp_dir = workspace();
        for _ in 0..MAX_CONSECUTIVE_RECOVERIES {
            assert!(record_recovery(temp_dir.path()));
        }
        assert!(!record_recovery(temp_dir.path()));
    }

    #[test]
    fn clearing_the_streak_restores_the_full_allowance() {
        let temp_dir = workspace();
        for _ in 0..MAX_CONSECUTIVE_RECOVERIES {
            assert!(record_recovery(temp_dir.path()));
        }
        clear(temp_dir.path());
        assert!(record_recovery(temp_dir.path()));
    }

    #[test]
    fn recoveries_outside_the_window_do_not_accumulate() {
        let now = 1_000_000_000;
        let stale = RecoveryState {
            consecutive_recoveries: MAX_CONSECUTIVE_RECOVERIES,
            last_recovery_unix_seconds: Some(now - RECOVERY_WINDOW_SECONDS),
        };
        assert!(!within_window(&stale, now));

        let recent = RecoveryState {
            consecutive_recoveries: 1,
            last_recovery_unix_seconds: Some(now - 1),
        };
        assert!(within_window(&recent, now));
        assert!(!within_window(&RecoveryState::default(), now));
    }

    #[test]
    fn a_workspace_without_metadata_never_blocks_recovery() {
        let temp_dir = tempfile::tempdir().unwrap();
        assert_eq!(state_path(temp_dir.path()), None);
        for _ in 0..(MAX_CONSECUTIVE_RECOVERIES + 2) {
            assert!(record_recovery(temp_dir.path()));
        }
    }
}
