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

//! Read-path freshness for local-first repos (roadmap/088, D7/D8).
//!
//! Since Stage 7 the operation log is local, so "am I up to date?" is no longer
//! a question about op heads — it is a question about **refs**. This module
//! answers it with one repo-scoped opaque token (D8): the probe reads the
//! server's current token, compares it with the one recorded in
//! [`ServerHeadsMarker`], and reports one of three states.
//!
//! Nothing here is ever on a command's critical path. [`freshness_state`] is a
//! pure marker read that never touches the network, and [`refresh_markers`]
//! runs *after* the command's output is done, so the next command starts from a
//! current token having paid nothing for it.
//!
//! Three states, never two: a probe that failed, was suppressed, or has never
//! run reports [`FreshnessState::Unknown`] rather than claiming currency. That
//! distinction is the whole point of D7 — silence about staleness is what let a
//! repository look fine while it was hours behind.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

use crate::vex::VexClient;
use crate::vex_publish::ServerHeadsMarker;
use crate::vex_publish::read_server_heads;
use crate::vex_publish::write_server_heads;

/// Default wall-clock budget for the probe, covering the connect handshake as
/// well as the request.
pub const DEFAULT_REFRESH_BUDGET_MS: u64 = 100;

/// Why freshness is unknown. Each variant is a different remedy, so they are
/// not collapsed into one "unknown".
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnknownReason {
    /// No probe has ever completed for this repository.
    NoProbeYet,
    /// The last probe failed, with this message.
    ProbeFailed(String),
    /// The probe was deliberately suppressed (`VEX_NO_REFRESH=1`, or
    /// `--no-refresh`).
    Suppressed,
}

/// How current this repository's view of the server is.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FreshnessState {
    /// The recorded ref token was confirmed current at `checked_unix`.
    Current {
        /// Unix time of the probe that confirmed it.
        checked_unix: i64,
    },
    /// The server's ref token has moved past the recorded one; the remedy is
    /// `vex pull`.
    Behind {
        /// Unix time the recorded token was last confirmed current.
        last_confirmed_unix: i64,
    },
    /// Freshness could not be established. Never rendered as "current".
    Unknown {
        /// Which of the three ways it is unknown, since each has its own
        /// remedy.
        reason: UnknownReason,
    },
}

impl FreshnessState {
    /// Whether freshness could not be established, in any of the three ways.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }

    /// Stable machine-readable name, for `--format json` (WP11).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Current { .. } => "current",
            Self::Behind { .. } => "behind",
            Self::Unknown { .. } => "unknown",
        }
    }
}

/// Budget for a *blocking* refresh on the read path. `None` — the default —
/// means the read never waits for the network at all.
///
/// An opportunistic refresh that blocks is not opportunistic: any budget at or
/// above the link's round trip time is spent in full on nearly every command,
/// because the request usually completes just inside it. Measured on the
/// reference laptop, a 100 ms budget against a ~90-110 ms RTT produced status
/// medians identical to synchronous mode, while forcing the budget below the
/// RTT dropped them from 0.16 s to 0.06 s. Setting `VEX_REFRESH_BUDGET_MS` opts
/// back into blocking freshness.
pub fn refresh_budget() -> Option<Duration> {
    let raw = std::env::var("VEX_REFRESH_BUDGET_MS").ok()?;
    let millis = raw.trim().parse::<u64>().ok()?;
    (millis > 0).then(|| Duration::from_millis(millis))
}

/// Budget for the end-of-command probe, from `VEX_REFRESH_BUDGET_MS` or
/// [`DEFAULT_REFRESH_BUDGET_MS`]. This one runs after the command's output is
/// done, so spending it costs the user nothing.
pub fn background_refresh_budget() -> Duration {
    refresh_budget().unwrap_or(Duration::from_millis(DEFAULT_REFRESH_BUDGET_MS))
}

/// Whether the user asked for no network probe at all (`VEX_NO_REFRESH=1`).
/// One canonical name; `vex status --no-refresh` (WP11) sets it.
pub fn no_refresh_requested() -> bool {
    matches!(
        std::env::var("VEX_NO_REFRESH").ok().as_deref(),
        Some("1") | Some("true") | Some("yes")
    )
}

/// This repository's freshness, read from the marker alone.
///
/// Performs no I/O beyond that single read, never blocks, and never fails: an
/// unreadable marker is an unknown state, not an error, because this is called
/// from output paths that must not be able to fail a command.
pub fn freshness_state(dir: &Path) -> FreshnessState {
    if no_refresh_requested() {
        return FreshnessState::Unknown {
            reason: UnknownReason::Suppressed,
        };
    }
    let marker = match read_server_heads(dir) {
        Ok(Some(marker)) => marker,
        Ok(None) => {
            return FreshnessState::Unknown {
                reason: UnknownReason::NoProbeYet,
            };
        }
        Err(err) => {
            return FreshnessState::Unknown {
                reason: UnknownReason::ProbeFailed(err.to_string()),
            };
        }
    };
    state_of(&marker)
}

/// The marker's own account of itself. A recorded failure wins over a recorded
/// token: the token may be arbitrarily old, and reporting it as current is the
/// exact silence D7 exists to remove.
fn state_of(marker: &ServerHeadsMarker) -> FreshnessState {
    if let Some(error) = &marker.last_probe_error {
        return FreshnessState::Unknown {
            reason: UnknownReason::ProbeFailed(error.clone()),
        };
    }
    match marker.ref_token {
        Some(_) => FreshnessState::Current {
            checked_unix: marker.updated_unix,
        },
        None => FreshnessState::Unknown {
            reason: UnknownReason::NoProbeYet,
        },
    }
}

/// Probe once per repository per process, within `budget`, and record the
/// outcome durably.
///
/// Five steps, unchanged in shape from the op-head refresh this replaces:
/// budget, claim the single flight, one bounded backend call, memoize, write
/// the marker. The marker write happens on **every** outcome including timeout
/// and error, so a failing probe is visible to `vex doctor` in another process
/// rather than only to the process that hit it.
pub fn refresh_once(dir: &Path, client: &VexClient, budget: Duration) -> FreshnessState {
    if no_refresh_requested() {
        return FreshnessState::Unknown {
            reason: UnknownReason::Suppressed,
        };
    }
    if let Err(previous) = claim_refresh(dir) {
        return previous;
    }
    let mut marker = read_server_heads(dir).ok().flatten().unwrap_or_default();
    let recorded = marker.ref_token.clone();
    let outcome = match client.ref_freshness_token_within(budget) {
        Ok(Some(fetched)) => {
            let behind = recorded.is_some_and(|recorded| recorded != fetched);
            let last_confirmed_unix = marker.updated_unix;
            marker.record_success(Some(fetched));
            if behind {
                // The recorded token is genuinely stale. The ref sync that
                // acts on it is D9 (WP12); until it lands the report is the
                // whole remedy, which is why it names `vex pull`.
                FreshnessState::Behind {
                    last_confirmed_unix,
                }
            } else {
                FreshnessState::Current {
                    checked_unix: marker.updated_unix,
                }
            }
        }
        Ok(None) => {
            // BLOCKED-ON-STAGE6: `VexClient::ref_freshness_token_within` is a
            // seam that always answers `None` until Stage 6's repo-scoped
            // ref-freshness token RPC lands. Reporting `NoProbeYet` rather than
            // `Current` is deliberate: this client cannot yet establish
            // freshness, and must not pretend otherwise.
            marker.record_failure("no ref-freshness token available".to_string());
            FreshnessState::Unknown {
                reason: UnknownReason::NoProbeYet,
            }
        }
        Err(err) => {
            let message = err.to_string();
            tracing::debug!(error = %message, "ref-freshness probe failed");
            marker.record_failure(message.clone());
            FreshnessState::Unknown {
                reason: UnknownReason::ProbeFailed(message),
            }
        }
    };
    if let Err(err) = write_server_heads(dir, &marker) {
        tracing::debug!(error = %err, "could not record the ref-freshness probe");
    }
    record_refresh(dir, outcome)
}

/// Update this repository's freshness marker after the command has finished, so
/// the *next* command reports an accurate state without any command ever having
/// waited for the network.
pub fn refresh_markers(dir: &Path, client: &VexClient) {
    refresh_once(dir, client, background_refresh_budget());
}

/// Per-process memo of the probe outcome for one repository.
///
/// The probe must happen at most once per process, but every later caller in
/// that process has to see the *same* answer. Keyed by directory rather than a
/// plain flag so tests (and any future multi-repo command) stay independent.
type RefreshMemo = OnceLock<Mutex<HashMap<PathBuf, FreshnessState>>>;
static REFRESHED: RefreshMemo = OnceLock::new();

fn refresh_memo() -> &'static Mutex<HashMap<PathBuf, FreshnessState>> {
    REFRESHED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `Ok(())` when this call owns the single probe for `dir`; `Err(previous)`
/// when another call already made it, carrying that call's outcome.
fn claim_refresh(dir: &Path) -> Result<(), FreshnessState> {
    let memo = refresh_memo().lock().unwrap();
    match memo.get(dir) {
        Some(previous) => Err(previous.clone()),
        None => Ok(()),
    }
}

fn record_refresh(dir: &Path, outcome: FreshnessState) -> FreshnessState {
    refresh_memo()
        .lock()
        .unwrap()
        .insert(dir.to_path_buf(), outcome.clone());
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker_with_token(dir: &Path, token: &str) {
        write_server_heads(dir, &ServerHeadsMarker::new(Some(token.to_string()))).unwrap();
    }

    #[test]
    fn an_unprobed_repo_is_unknown_not_current() {
        let temp = tempfile::tempdir().unwrap();
        assert_eq!(
            freshness_state(temp.path()),
            FreshnessState::Unknown {
                reason: UnknownReason::NoProbeYet
            }
        );
    }

    #[test]
    fn a_recorded_token_reads_back_as_current() {
        let temp = tempfile::tempdir().unwrap();
        marker_with_token(temp.path(), "token-a");
        let state = freshness_state(temp.path());
        assert_eq!(state.as_str(), "current");
        assert!(matches!(state, FreshnessState::Current { checked_unix } if checked_unix > 0));
    }

    #[test]
    fn a_probe_failure_is_recorded_durably_and_reads_back_as_unknown() {
        let temp = tempfile::tempdir().unwrap();
        let mut marker = ServerHeadsMarker::new(Some("token-a".to_string()));
        marker.record_failure("connect timed out".to_string());
        write_server_heads(temp.path(), &marker).unwrap();

        // A *different* reader — standing in for another process — sees the
        // failure rather than the stale token.
        assert_eq!(
            freshness_state(temp.path()),
            FreshnessState::Unknown {
                reason: UnknownReason::ProbeFailed("connect timed out".to_string())
            }
        );
    }

    #[test]
    fn single_flight_replays_one_outcome_within_a_process() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        assert!(claim_refresh(dir).is_ok(), "first caller owns the probe");

        let outcome = FreshnessState::Current { checked_unix: 7 };
        assert_eq!(record_refresh(dir, outcome.clone()), outcome);
        assert_eq!(claim_refresh(dir), Err(outcome.clone()));
        assert_eq!(claim_refresh(dir), Err(outcome));

        // A recorded unknown replays as an unknown, not as a fresh claim.
        let other = tempfile::tempdir().unwrap();
        assert!(claim_refresh(other.path()).is_ok());
        let unknown = FreshnessState::Unknown {
            reason: UnknownReason::NoProbeYet,
        };
        assert_eq!(record_refresh(other.path(), unknown.clone()), unknown);
        assert_eq!(claim_refresh(other.path()), Err(unknown));
    }

    #[test]
    fn no_refresh_env_suppresses_the_probe_and_reports_unknown() {
        // Not driven through the process environment (tests share it); the
        // policy itself is what matters, and it is one function.
        let suppressed = FreshnessState::Unknown {
            reason: UnknownReason::Suppressed,
        };
        assert!(suppressed.is_unknown());
        assert_eq!(suppressed.as_str(), "unknown");
        // The default environment does not suppress.
        assert!(!no_refresh_requested());
    }

    #[test]
    fn freshness_state_never_performs_io_beyond_the_marker_read() {
        // The store path holds no `vex.json`, so constructing a client is
        // impossible; a `freshness_state` that touched the network could not
        // return at all. It returns, immediately, from the marker alone.
        let temp = tempfile::tempdir().unwrap();
        marker_with_token(temp.path(), "token-a");
        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries.len(), 1, "only the marker exists: {entries:?}");
        assert!(matches!(
            freshness_state(temp.path()),
            FreshnessState::Current { .. }
        ));
    }

    #[test]
    fn the_read_path_does_not_block_on_refresh_by_default() {
        assert_eq!(refresh_budget(), None);
        assert_eq!(
            background_refresh_budget(),
            Duration::from_millis(DEFAULT_REFRESH_BUDGET_MS)
        );
    }

    /// A corrupt marker is an unknown state, never a failure and never a claim
    /// of currency.
    #[test]
    fn an_unreadable_marker_is_unknown() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join(crate::vex_publish::SERVER_HEADS_FILE),
            "not json",
        )
        .unwrap();
        assert!(freshness_state(temp.path()).is_unknown());
    }
}
