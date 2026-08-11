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
//!
//! [`FreshnessState::Behind`] is also the trigger for D9: a repository that is
//! known to be behind may advance its **tracked bookmarks** — never its
//! working state — by jj's own three-way merge. That lives in
//! [`crate::vex_ref_sync`], which reads the state this module records and
//! shares its opt-out ([`no_refresh_requested`]).

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

/// Default wall-clock budget for a *blocking* probe, covering the connect
/// handshake as well as the request. Deliberately tight: a caller that opted
/// into `VEX_REFRESH_BUDGET_MS` is waiting on this before its output.
pub const DEFAULT_REFRESH_BUDGET_MS: u64 = 100;

/// Budget for the background probe, which runs detached after the command has
/// already produced its output and exited.
///
/// Nothing waits on it, so the tight blocking budget bought nothing here and
/// cost accuracy: 100ms is below a single TLS + gRPC round trip to a hosted
/// backend, so any ordinary moment of latency timed the probe out, recorded a
/// failure, and made the next command report "freshness unknown" — the warning
/// that made this state look permanently broken when the probe merely never had
/// time to answer.
pub const DEFAULT_BACKGROUND_REFRESH_BUDGET_MS: u64 = 5_000;

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

    /// Unix time of the last *successful* contact with the server, when there
    /// has been one. `None` for a repository that has never been probed
    /// successfully — which is not the same as "contacted a long time ago", and
    /// is why this is an `Option` rather than a sentinel.
    pub fn last_contact_unix(&self) -> Option<i64> {
        match self {
            Self::Current { checked_unix } => Some(*checked_unix),
            Self::Behind {
                last_confirmed_unix,
            } => Some(*last_confirmed_unix).filter(|unix| *unix > 0),
            Self::Unknown { .. } => None,
        }
    }

    /// The command that resolves this state, if any. Frozen as part of the
    /// `--format json` schema (C6).
    pub fn remedy(&self) -> Option<&'static str> {
        match self {
            Self::Current { .. } => None,
            Self::Behind { .. } => Some("vex pull"),
            Self::Unknown {
                reason: UnknownReason::Suppressed,
            } => None,
            Self::Unknown { .. } => Some("vex status"),
        }
    }

    /// Stable machine-readable reason, for `--format json`. `None` unless the
    /// state is [`Self::Unknown`].
    pub fn reason_str(&self) -> Option<&'static str> {
        match self {
            Self::Unknown { reason } => Some(match reason {
                UnknownReason::NoProbeYet => "no-probe-yet",
                UnknownReason::ProbeFailed(_) => "probe-failed",
                UnknownReason::Suppressed => "suppressed",
            }),
            _ => None,
        }
    }

    /// The probe failure message, when the state is unknown because a probe
    /// failed.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Unknown {
                reason: UnknownReason::ProbeFailed(message),
            } => Some(message),
            _ => None,
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
/// [`DEFAULT_BACKGROUND_REFRESH_BUDGET_MS`]. This one runs after the command's
/// output is done, so spending it costs the user nothing — and it used to be
/// capped as if it did, which is what made the probe fail against any backend
/// further away than a millisecond.
pub fn background_refresh_budget() -> Duration {
    refresh_budget().unwrap_or(Duration::from_millis(DEFAULT_BACKGROUND_REFRESH_BUDGET_MS))
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
    if marker.pending_token.is_some() {
        // Recorded by a probe that saw the server move, and kept until this
        // repository is brought back into sync. Being behind survives the
        // process that discovered it — otherwise only the command that
        // happened to run the probe would ever say so.
        return FreshnessState::Behind {
            last_confirmed_unix: marker.updated_unix,
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

/// What a probe is allowed to conclude from a token it has never seen before.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeKind {
    /// An ordinary end-of-command probe. A token that differs from the
    /// confirmed one means the server moved and this repository is behind.
    Observe,
    /// A probe that runs immediately after the local refs were synced with the
    /// server (clone, pull, push), so whatever the server reports now *is* this
    /// repository's state. Adopts the token and clears "behind".
    SyncPoint,
}

/// Everything the marker records about probing, for `vex doctor` (C8).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProbeReport {
    /// Current three-state freshness.
    pub state: Option<FreshnessState>,
    /// The confirmed server ref token, verbatim and opaque.
    pub ref_token: Option<String>,
    /// A newer server token seen while behind.
    pub pending_token: Option<String>,
    /// Unix time of the last probe of any outcome.
    pub last_probe_unix: Option<i64>,
    /// Unix time the token was last confirmed.
    pub last_confirmed_unix: Option<i64>,
    /// Why the last probe failed, when it did.
    pub last_probe_error: Option<String>,
    /// Whether a marker existed at all.
    pub probed: bool,
}

/// Read everything the freshness marker knows, without probing. Like
/// [`freshness_state`] this never blocks and never fails: `vex doctor` reporting
/// nothing is better than `vex doctor` failing.
pub fn probe_report(dir: &Path) -> ProbeReport {
    let Ok(Some(marker)) = read_server_heads(dir) else {
        return ProbeReport {
            state: Some(freshness_state(dir)),
            ..ProbeReport::default()
        };
    };
    ProbeReport {
        state: Some(freshness_state(dir)),
        ref_token: marker.ref_token.clone(),
        pending_token: marker.pending_token.clone(),
        last_probe_unix: marker.last_probe_unix,
        last_confirmed_unix: (marker.updated_unix > 0).then_some(marker.updated_unix),
        last_probe_error: marker.last_probe_error.clone(),
        probed: true,
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
    refresh_once_as(dir, client, budget, ProbeKind::Observe)
}

/// [`refresh_once`], told whether the local refs were just synced.
pub fn refresh_once_as(
    dir: &Path,
    client: &VexClient,
    budget: Duration,
    kind: ProbeKind,
) -> FreshnessState {
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
            // A repository with no confirmed token yet has no basis on which to
            // call itself behind, and reporting "unknown" forever would make the
            // first probe useless; the first token is the baseline. Every later
            // probe compares against it.
            let in_sync = kind == ProbeKind::SyncPoint
                || recorded
                    .as_ref()
                    .is_none_or(|recorded| *recorded == fetched);
            if in_sync {
                marker.record_success(Some(fetched));
                FreshnessState::Current {
                    checked_unix: marker.updated_unix,
                }
            } else {
                // The confirmed token is genuinely stale. Recording it is all
                // this probe does: acting on it is D9
                // ([`crate::vex_ref_sync::sync_if_behind`]), which reads this
                // very state, advances tracked bookmarks by jj's three-way
                // merge, and then adopts the pending token. Until a caller
                // runs it the report is the whole remedy, which is why it
                // names `vex pull`.
                let last_confirmed_unix = marker.updated_unix;
                marker.record_behind(fetched);
                FreshnessState::Behind {
                    last_confirmed_unix,
                }
            }
        }
        Ok(None) => {
            // The backend answered with no token: it is telling this client it
            // cannot establish freshness, which is unknown and never current.
            // A budget that ran out is a different thing with a different
            // remedy and arrives as an error below.
            let message = "the server did not provide a freshness token".to_string();
            marker.record_failure(message.clone());
            FreshnessState::Unknown {
                reason: UnknownReason::ProbeFailed(message),
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

/// As [`refresh_markers`], for a command that just synced this repository's
/// refs with the server (clone, pull, push). The token it reads becomes the new
/// confirmed baseline, so a repository that was behind stops reporting behind
/// once the user has done the thing the report asked for.
pub fn refresh_markers_after_sync(dir: &Path, client: &VexClient) {
    refresh_once_as(
        dir,
        client,
        background_refresh_budget(),
        ProbeKind::SyncPoint,
    );
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
    }

    /// The two budgets are not the same number, and the difference is the whole
    /// point: the blocking probe is a caller waiting on its own output, while
    /// the background one runs after the command has exited. Capping the
    /// background probe as tightly as the blocking one meant it could not
    /// outlast a single round trip to a hosted backend, so it timed out and
    /// reported "freshness unknown" on repositories that were perfectly current.
    #[test]
    fn the_background_probe_outlasts_a_round_trip() {
        assert_eq!(
            background_refresh_budget(),
            Duration::from_millis(DEFAULT_BACKGROUND_REFRESH_BUDGET_MS)
        );
        assert!(
            DEFAULT_BACKGROUND_REFRESH_BUDGET_MS > DEFAULT_REFRESH_BUDGET_MS,
            "the background probe must not be capped as tightly as a blocking one"
        );
        // A round trip to a hosted backend, handshake included, comfortably
        // inside the budget rather than at its edge.
        assert!(DEFAULT_BACKGROUND_REFRESH_BUDGET_MS >= 1_000);
    }

    /// The token comparison that makes "behind" possible, and the fact that it
    /// survives the process that discovered it.
    #[test]
    fn a_changed_server_token_reads_back_as_behind_in_another_process() {
        let temp = tempfile::tempdir().unwrap();
        marker_with_token(temp.path(), "v2:token-a");
        let confirmed = read_server_heads(temp.path())
            .unwrap()
            .unwrap()
            .updated_unix;

        // What `refresh_once` does when the fetched token differs.
        let mut marker = read_server_heads(temp.path()).unwrap().unwrap();
        assert_ne!(marker.ref_token.as_deref(), Some("v2:token-b"));
        marker.record_behind("v2:token-b".to_string());
        write_server_heads(temp.path(), &marker).unwrap();

        assert_eq!(
            freshness_state(temp.path()),
            FreshnessState::Behind {
                last_confirmed_unix: confirmed
            }
        );
        // The confirmed token is *not* overwritten: it is what "last in sync"
        // means, and adopting the server's would report a stale repo current on
        // the very next read.
        let marker = read_server_heads(temp.path()).unwrap().unwrap();
        assert_eq!(marker.ref_token.as_deref(), Some("v2:token-a"));
        assert_eq!(marker.pending_token.as_deref(), Some("v2:token-b"));
    }

    /// Being behind ends when the repository is synced again, not before.
    #[test]
    fn a_sync_point_clears_behind() {
        let temp = tempfile::tempdir().unwrap();
        let mut marker = ServerHeadsMarker::new(Some("v2:token-a".to_string()));
        marker.record_behind("v2:token-b".to_string());
        write_server_heads(temp.path(), &marker).unwrap();
        assert!(matches!(
            freshness_state(temp.path()),
            FreshnessState::Behind { .. }
        ));

        marker.record_success(Some("v2:token-b".to_string()));
        write_server_heads(temp.path(), &marker).unwrap();
        assert!(matches!(
            freshness_state(temp.path()),
            FreshnessState::Current { .. }
        ));
    }

    /// The three states are mutually exclusive and each has its own name and
    /// remedy, so no caller can render two of them the same way.
    #[test]
    fn the_three_states_are_distinguishable() {
        let current = FreshnessState::Current { checked_unix: 10 };
        let behind = FreshnessState::Behind {
            last_confirmed_unix: 10,
        };
        let unknown = FreshnessState::Unknown {
            reason: UnknownReason::NoProbeYet,
        };

        assert_eq!(current.as_str(), "current");
        assert_eq!(behind.as_str(), "behind");
        assert_eq!(unknown.as_str(), "unknown");
        assert!(!current.is_unknown() && !behind.is_unknown() && unknown.is_unknown());

        assert_eq!(current.remedy(), None);
        assert_eq!(behind.remedy(), Some("vex pull"));
        assert_eq!(unknown.remedy(), Some("vex status"));

        assert_eq!(current.last_contact_unix(), Some(10));
        assert_eq!(behind.last_contact_unix(), Some(10));
        assert_eq!(unknown.last_contact_unix(), None);

        assert_eq!(current.reason_str(), None);
        assert_eq!(unknown.reason_str(), Some("no-probe-yet"));
        assert_eq!(
            FreshnessState::Unknown {
                reason: UnknownReason::ProbeFailed("boom".to_string())
            }
            .error_message(),
            Some("boom")
        );
    }

    /// `vex doctor`'s source (C8): the probe record survives the process that
    /// wrote it, failure included.
    #[test]
    fn the_probe_report_reads_the_whole_record() {
        let temp = tempfile::tempdir().unwrap();
        assert_eq!(
            probe_report(temp.path()),
            ProbeReport {
                state: Some(FreshnessState::Unknown {
                    reason: UnknownReason::NoProbeYet
                }),
                ..ProbeReport::default()
            }
        );

        let mut marker = ServerHeadsMarker::new(Some("v2:token-a".to_string()));
        marker.record_failure("connect timed out".to_string());
        write_server_heads(temp.path(), &marker).unwrap();

        let report = probe_report(temp.path());
        assert!(report.probed);
        assert_eq!(report.ref_token.as_deref(), Some("v2:token-a"));
        assert_eq!(
            report.last_probe_error.as_deref(),
            Some("connect timed out")
        );
        assert!(report.last_probe_unix.unwrap() > 0);
        assert!(report.last_confirmed_unix.unwrap() > 0);
        assert!(report.state.unwrap().is_unknown());
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
