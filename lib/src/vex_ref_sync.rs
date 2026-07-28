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

//! Auto fast-forward of tracked bookmarks (roadmap/088, D9).
//!
//! When the freshness probe ([`crate::vex_freshness`]) reports
//! [`FreshnessState::Behind`], this module brings the repository's *knowledge*
//! of the server forward. It does not move the user's working state.
//!
//! # Why this is safe
//!
//! jj has no "fast-forward" special case, and neither does this module. For
//! each tracked bookmark it computes jj's three-way merge
//! `[local, remote] - [known_remote]` (`docs/design/tracking-branches.md:162`)
//! by calling jj's own [`crate::refs::merge_ref_targets`] through
//! [`MutableRepo::merge_local_bookmark`]. A fast-forward is simply the
//! *trivial* resolution of that merge, which is exactly why it is safe: the
//! last-known server target is the merge base, so a local bookmark that has
//! moved cannot be silently overwritten — it produces a **conflicted
//! bookmark** instead (`refs.rs:146-148`: a "fast-forwardable move can be
//! safely tracked", and everything else requires user intervention). A
//! conflicted bookmark is jj's designed outcome for divergence: not an error,
//! not a clobber, and not a blocked command.
//!
//! The last-known server target is not re-derived or guessed: it is the
//! `name@vex` remote bookmark in the view, which is a full [`RemoteRef`] (a
//! `RefTarget` that may itself be conflicted, which is why it is not collapsed
//! to a bare `CommitId`) persisted in the local operation log. It is advanced
//! *unconditionally* after every merge, exactly as [`crate::git`]'s
//! `import_refs_inner` does, so the base moves forward even when the local
//! merge conflicted and the same divergence is not reported twice.
//!
//! # Where Vex is more conservative than jj
//!
//! jj fast-forwards during an explicit `jj git fetch`. Vex may do it as a side
//! effect of an unrelated command, so:
//!
//! * **The working copy is never moved.** `@` and the working-copy commit are
//!   not touched — advancing what the user is *working on* stays with explicit
//!   `vex pull`. This is structural (nothing here can reach a `WorkingCopy`;
//!   the only handle taken is a [`ReadonlyRepo`]) and re-checked at runtime by
//!   [`RefSyncError::WorkingCopyMoved`].
//! * **Nothing is created or deleted.** Only bookmarks this repository already
//!   tracks are considered, and a name the server did not report is left
//!   alone: silence from a prefix listing is missing information, never a
//!   deletion.
//! * **Every advance is reported**, both to the caller ([`RefSyncReport`]) and
//!   durably to the next command ([`record_report`]), so the change is never
//!   invisible.
//! * **One opt-out**, the same one that disables the probe: `--no-refresh` /
//!   `VEX_NO_REFRESH=1` ([`crate::vex_freshness::no_refresh_requested`]).

use std::collections::BTreeMap;
use std::fmt;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use pollster::FutureExt as _;
use serde::Deserialize;
use serde::Serialize;
use tempfile::NamedTempFile;

use crate::backend::CommitId;
use crate::object_id::ObjectId as _;
use crate::op_store::RefTarget;
use crate::op_store::RemoteRef;
use crate::op_store::RemoteRefState;
use crate::ref_name::RefName;
use crate::ref_name::RefNameBuf;
use crate::ref_name::RemoteName;
use crate::ref_name::RemoteRefSymbol;
use crate::repo::MutableRepo;
use crate::repo::ReadonlyRepo;
use crate::repo::Repo as _;
use crate::vex::REF_FRESHNESS_PREFIX;
use crate::vex::VexClient;
use crate::vex_freshness::FreshnessState;
use crate::vex_freshness::freshness_state;
use crate::vex_freshness::no_refresh_requested;

/// The remote name Vex records server bookmarks under. Matches `vex pull` and
/// `vex push`, which write the same `name@vex` remote bookmarks.
pub const VEX_REMOTE: &str = "vex";

/// Durable record of the last fast-forward, so the note survives the process
/// that made it. Lives beside the other `vex-*` markers in `op_heads/`.
pub const REF_SYNC_FILE: &str = "vex-ref-sync";

/// Marker schema version written by this client.
pub const REF_SYNC_MARKER_VERSION: u32 = 1;

/// What the three-way merge did to one bookmark.
///
/// These are outcomes *observed* from the `RefTarget` the merge produced, not
/// decisions made ahead of it — there is no code path that chooses to
/// fast-forward.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum BookmarkUpdate {
    /// The merge changed nothing: the server target was already known.
    Unchanged,
    /// The trivial resolution moved the local bookmark forward.
    FastForwarded {
        /// The local target before the merge; `None` when the bookmark had no
        /// local target (so this is a first adoption, still trivial).
        from: Option<String>,
        /// The local target after the merge.
        to: String,
    },
    /// The merge was non-trivial, so the local bookmark is now conflicted —
    /// jj's designed outcome for divergence, and never an error.
    Conflicted {
        /// Number of positive terms in the conflicted target.
        adds: usize,
        /// Number of negative terms in the conflicted target.
        removes: usize,
    },
    /// The bookmark was left untouched, with the reason. A skip is never a
    /// silent no-op: the commit could not be hydrated, or the server reported
    /// something unusable.
    Skipped {
        /// Why this bookmark was not considered.
        reason: String,
    },
}

impl BookmarkUpdate {
    /// Whether this outcome changed the local bookmark.
    pub fn changed(&self) -> bool {
        matches!(self, Self::FastForwarded { .. } | Self::Conflicted { .. })
    }
}

/// One bookmark's outcome.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BookmarkOutcome {
    /// Bookmark name, without any remote qualification.
    pub name: String,
    /// What happened to it.
    pub update: BookmarkUpdate,
}

/// Everything one fast-forward pass did, in the order the bookmarks were
/// considered.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefSyncReport {
    /// Per-bookmark outcomes.
    pub outcomes: Vec<BookmarkOutcome>,
    /// Whether the pass was skipped wholesale (`--no-refresh`), which is
    /// distinct from a pass that ran and found nothing to do.
    pub suppressed: bool,
}

impl RefSyncReport {
    /// A report for a pass that never ran because the user opted out.
    pub fn suppressed() -> Self {
        Self {
            outcomes: Vec::new(),
            suppressed: true,
        }
    }

    /// Bookmarks that moved forward, as `(name, from, to)`.
    pub fn fast_forwarded(&self) -> impl Iterator<Item = (&str, Option<&str>, &str)> {
        self.outcomes
            .iter()
            .filter_map(|outcome| match &outcome.update {
                BookmarkUpdate::FastForwarded { from, to } => {
                    Some((outcome.name.as_str(), from.as_deref(), to.as_str()))
                }
                _ => None,
            })
    }

    /// Bookmarks the merge left conflicted.
    pub fn conflicted(&self) -> impl Iterator<Item = &str> {
        self.outcomes.iter().filter_map(|outcome| {
            matches!(outcome.update, BookmarkUpdate::Conflicted { .. })
                .then_some(outcome.name.as_str())
        })
    }

    /// Whether anything at all changed. A report with nothing in it is not
    /// worth printing or recording.
    pub fn changed(&self) -> bool {
        self.outcomes.iter().any(|outcome| outcome.update.changed())
    }

    /// Human-readable lines for the command that performed the sync (D9 rule
    /// 4). Empty when nothing changed, so callers can print unconditionally.
    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (name, from, to) in self.fast_forwarded() {
            let short = |id: &str| id.chars().take(12).collect::<String>();
            lines.push(match from {
                Some(from) => format!(
                    "Fast-forwarded bookmark {name}: {} -> {}",
                    short(from),
                    short(to)
                ),
                None => format!("Adopted bookmark {name} at {}", short(to)),
            });
        }
        let conflicted = self.conflicted().collect::<Vec<_>>();
        if !conflicted.is_empty() {
            lines.push(format!(
                "Bookmark{} {} diverged from the server and {} now conflicted; resolve with `vex \
                 pull`.",
                if conflicted.len() == 1 { "" } else { "s" },
                conflicted.join(", "),
                if conflicted.len() == 1 { "is" } else { "are" },
            ));
        }
        lines
    }
}

impl fmt::Display for RefSyncReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in self.summary_lines() {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

/// The durable form of a [`RefSyncReport`], for the staleness note.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefSyncRecord {
    /// Schema version.
    pub v: u32,
    /// When the pass ran.
    pub synced_unix: i64,
    /// What it did.
    pub report: RefSyncReport,
}

/// A fast-forward pass could not be completed. Never fatal to a command: every
/// caller reports it and carries on, because a command that fails because its
/// *background knowledge* could not be refreshed is worse than a stale one.
#[derive(Debug)]
pub enum RefSyncError {
    /// The server refs could not be listed.
    Fetch(String),
    /// The merge needed index information that was not available.
    Merge(String),
    /// The operation could not be committed.
    Commit(String),
    /// The pass would have moved the working copy. Structurally unreachable —
    /// this is the assertion, not the handler, and the transaction is dropped
    /// rather than committed when it fires.
    WorkingCopyMoved,
}

impl fmt::Display for RefSyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fetch(message) => write!(f, "cannot list server bookmarks: {message}"),
            Self::Merge(message) => write!(f, "cannot merge tracked bookmarks: {message}"),
            Self::Commit(message) => write!(f, "cannot record the bookmark update: {message}"),
            Self::WorkingCopyMoved => write!(
                f,
                "refusing to record a bookmark update that would move the working copy"
            ),
        }
    }
}

impl std::error::Error for RefSyncError {}

/// Where the server's current bookmark targets come from.
///
/// A trait so the merge rules can be tested against jj's real merge machinery
/// without a backend: the wire format is not what D9 is about.
pub trait ServerRefSource {
    /// The server's current target for each requested bookmark. A name the
    /// server does not report must be **omitted**, never mapped to an absent
    /// target — the difference is "no information" versus "deleted", and only
    /// the first is safe to infer from a prefix listing.
    fn bookmark_targets(
        &self,
        names: &[RefNameBuf],
    ) -> Result<BTreeMap<RefNameBuf, CommitId>, RefSyncError>;
}

/// The real source: one `list_refs` over the same ref namespace the freshness
/// token is derived from ([`REF_FRESHNESS_PREFIX`]), so the token that said
/// "behind" and the refs fetched here describe the same scope.
pub struct VexServerRefs<'a> {
    client: &'a VexClient,
}

impl<'a> VexServerRefs<'a> {
    /// Read server bookmarks through `client`.
    pub fn new(client: &'a VexClient) -> Self {
        Self { client }
    }
}

impl ServerRefSource for VexServerRefs<'_> {
    fn bookmark_targets(
        &self,
        names: &[RefNameBuf],
    ) -> Result<BTreeMap<RefNameBuf, CommitId>, RefSyncError> {
        let wanted: std::collections::BTreeSet<&str> =
            names.iter().map(|name| name.as_str()).collect();
        let refs = self
            .client
            .list_refs(REF_FRESHNESS_PREFIX)
            .block_on()
            .map_err(|err| RefSyncError::Fetch(err.to_string()))?;
        let mut targets = BTreeMap::new();
        for value in refs {
            let Some(name) = value.name.strip_prefix(REF_FRESHNESS_PREFIX) else {
                continue;
            };
            if !wanted.contains(name) {
                continue;
            }
            let Some(id) = CommitId::try_from_hex(&value.target_commit_id) else {
                // An unparsable target is not information; leaving the
                // bookmark out means it is skipped, not clobbered.
                continue;
            };
            targets.insert(RefNameBuf::from(name), id);
        }
        Ok(targets)
    }
}

/// Merge the server's bookmark targets into this repository's tracked
/// bookmarks, using jj's own three-way merge.
///
/// `repo` must already have the target commits in its index (see
/// [`sync_tracked_bookmarks`], which hydrates them); the merge consults the
/// index for the non-trivial cases.
///
/// This function is the whole of D9's rule 1: for each bookmark it takes the
/// persisted last-known server target as the **base**, the local bookmark as
/// **left**, and the new server target as **right**, and hands all three to
/// [`MutableRepo::merge_local_bookmark`]. There is no branch on "did the local
/// bookmark move" — the merge answers that, and a moved local bookmark comes
/// back as a conflicted target (rule 2).
pub fn fast_forward_tracked_bookmarks(
    repo: &mut MutableRepo,
    remote: &RemoteName,
    server: &BTreeMap<RefNameBuf, CommitId>,
) -> Result<RefSyncReport, RefSyncError> {
    // Rule 3, as an assertion rather than a promise: nothing below can reach a
    // working copy, and this proves it. The comparison is against the
    // operation the transaction *started* from rather than the state on entry,
    // so it refuses a whole transaction that moves the working copy — a probe
    // pass has no business committing one, whichever line of it did the
    // moving.
    let wc_before = repo.base_repo().view().wc_commit_ids().clone();

    let mut outcomes = Vec::new();
    for (name, target_id) in server {
        let name: &RefName = name.as_ref();
        let symbol = RemoteRefSymbol { name, remote };
        let known = repo.get_remote_bookmark(symbol);
        // Only bookmarks this repository already tracks. An untracked remote
        // bookmark is one the user chose not to follow, and an unknown name is
        // not ours to create behind their back.
        if !known.is_tracked() {
            continue;
        }
        let new_target = RefTarget::normal(target_id.clone());
        let before = repo.view().get_local_bookmark(name).clone();
        // The base. `tracked_target()` is absent for an untracked ref, which is
        // why the tracking check above comes first.
        let base = known.tracked_target().clone();
        if base == new_target && before == new_target {
            outcomes.push(BookmarkOutcome {
                name: name.as_str().to_owned(),
                update: BookmarkUpdate::Unchanged,
            });
            continue;
        }
        repo.merge_local_bookmark(name, &base, &new_target)
            .map_err(|err| RefSyncError::Merge(err.to_string()))?;
        let after = repo.view().get_local_bookmark(name).clone();
        // The remote-tracking bookmark records the last known state of the
        // bookmark on the server, so it advances whether or not the local
        // merge was trivial. Skipping it on conflict would re-merge the same
        // divergence on every later pass.
        repo.set_remote_bookmark(
            symbol,
            RemoteRef {
                target: new_target,
                state: RemoteRefState::Tracked,
            },
        );
        outcomes.push(BookmarkOutcome {
            name: name.as_str().to_owned(),
            update: classify(&before, &after),
        });
    }

    if repo.view().wc_commit_ids() != &wc_before {
        return Err(RefSyncError::WorkingCopyMoved);
    }
    Ok(RefSyncReport {
        outcomes,
        suppressed: false,
    })
}

/// Read the outcome off the `RefTarget` the merge produced. Nothing here
/// decides anything; it names what jj already did.
fn classify(before: &RefTarget, after: &RefTarget) -> BookmarkUpdate {
    if before == after {
        return BookmarkUpdate::Unchanged;
    }
    if after.has_conflict() {
        return BookmarkUpdate::Conflicted {
            adds: after.added_ids().count(),
            removes: after.removed_ids().count(),
        };
    }
    match after.as_normal() {
        Some(to) => BookmarkUpdate::FastForwarded {
            from: before.as_normal().map(|id| id.hex()),
            to: to.hex(),
        },
        // An absent result cannot arise from a present `right`, but reporting
        // it as a fast-forward would be a lie, so it is a skip with a reason.
        None => BookmarkUpdate::Skipped {
            reason: "merge produced no target".to_owned(),
        },
    }
}

/// Fast-forward this repository's tracked bookmarks from `source`.
///
/// Returns the report and the repository at the new operation (unchanged when
/// nothing moved). Takes an [`Arc<ReadonlyRepo>`] and nothing else: there is no
/// `Workspace` and no `WorkingCopy` in reach, which is how rule 3 is enforced
/// rather than promised.
pub async fn sync_tracked_bookmarks(
    repo: &Arc<ReadonlyRepo>,
    source: &dyn ServerRefSource,
) -> Result<(RefSyncReport, Arc<ReadonlyRepo>), RefSyncError> {
    // Rule 5: the same opt-out as the probe. Checked here as well as at the
    // call site so no future caller can route around it.
    if no_refresh_requested() {
        return Ok((RefSyncReport::suppressed(), repo.clone()));
    }
    let remote = RemoteName::new(VEX_REMOTE);
    let tracked: Vec<RefNameBuf> = repo
        .view()
        .remote_bookmarks(remote)
        .filter(|(_, remote_ref)| remote_ref.is_tracked())
        .map(|(name, _)| name.to_owned())
        .collect();
    if tracked.is_empty() {
        return Ok((RefSyncReport::default(), repo.clone()));
    }
    let server = source.bookmark_targets(&tracked)?;
    if server.is_empty() {
        return Ok((RefSyncReport::default(), repo.clone()));
    }

    let mut tx = repo.start_transaction();
    // Hydrate before merging: `merge_ref_targets` asks the index whether one
    // target is an ancestor of another, and an unindexed commit would fail the
    // whole pass rather than the one bookmark it belongs to.
    let mut usable = BTreeMap::new();
    let mut skipped = Vec::new();
    for (name, id) in server {
        match tx.repo().store().get_commit_async(&id).await {
            Ok(commit) => match tx.repo_mut().add_head(&commit).await {
                Ok(()) => {
                    usable.insert(name, id);
                }
                Err(err) => skipped.push((name, err.to_string())),
            },
            Err(err) => skipped.push((name, err.to_string())),
        }
    }
    let mut report = fast_forward_tracked_bookmarks(tx.repo_mut(), remote, &usable)?;
    for (name, reason) in skipped {
        report.outcomes.push(BookmarkOutcome {
            name: name.as_str().to_owned(),
            update: BookmarkUpdate::Skipped { reason },
        });
    }
    if !tx.repo().has_changes() {
        drop(tx);
        return Ok((report, repo.clone()));
    }
    let repo = tx
        .commit("fast-forward tracked bookmarks from the server")
        .await
        .map_err(|err| RefSyncError::Commit(err.to_string()))?;
    Ok((report, repo))
}

/// What a [`sync_if_behind`] pass concluded.
pub struct RefSyncOutcome {
    /// The freshness state that triggered (or did not trigger) the pass.
    pub state: FreshnessState,
    /// The pass's report, when one ran.
    pub report: Option<RefSyncReport>,
    /// The repository at the new operation. The caller's own handle when
    /// nothing moved.
    pub repo: Arc<ReadonlyRepo>,
    /// Why the pass could not run, when it could not. Never fatal.
    pub error: Option<RefSyncError>,
}

impl RefSyncOutcome {
    /// Lines the command should print (rule 4). Empty when nothing moved.
    pub fn summary_lines(&self) -> Vec<String> {
        self.report
            .as_ref()
            .map(RefSyncReport::summary_lines)
            .unwrap_or_default()
    }
}

/// Advance tracked bookmarks when the freshness probe says this repository is
/// behind (D9).
///
/// `dir` is the repository's `op_heads` directory, where the freshness marker
/// lives. The probe's own verdict is the trigger: any state other than
/// [`FreshnessState::Behind`] means there is nothing known to be newer on the
/// server, and this returns without a single RPC.
///
/// On success the recorded ref token is advanced to the token the probe found,
/// because this repository has now seen the server's ref state — the
/// divergence that remains, if any, is a conflicted bookmark the report names,
/// not staleness.
pub fn sync_if_behind(dir: &Path, client: &VexClient, repo: &Arc<ReadonlyRepo>) -> RefSyncOutcome {
    let state = freshness_state(dir);
    if !matches!(state, FreshnessState::Behind { .. }) {
        return RefSyncOutcome {
            state,
            report: None,
            repo: repo.clone(),
            error: None,
        };
    }
    let source = VexServerRefs::new(client);
    match sync_tracked_bookmarks(repo, &source).block_on() {
        Ok((report, repo)) => {
            if report.changed() {
                if let Err(err) = record_report(dir, &report) {
                    tracing::debug!(error = %err, "could not record the bookmark update");
                }
            }
            if !report.suppressed {
                adopt_pending_token(dir);
            }
            RefSyncOutcome {
                state: freshness_state(dir),
                report: Some(report),
                repo,
                error: None,
            }
        }
        Err(err) => {
            tracing::debug!(error = %err, "bookmark fast-forward failed");
            RefSyncOutcome {
                state,
                report: None,
                repo: repo.clone(),
                error: Some(err),
            }
        }
    }
}

/// Promote the token the probe saw to the confirmed one, so a repository that
/// has just taken the server's refs stops reporting itself behind. Best
/// effort: a marker that cannot be written costs one repeated pass, never a
/// command.
fn adopt_pending_token(dir: &Path) {
    let Ok(Some(mut marker)) = crate::vex_publish::read_server_heads(dir) else {
        return;
    };
    let Some(pending) = marker.pending_token.clone() else {
        return;
    };
    marker.record_success(Some(pending));
    if let Err(err) = crate::vex_publish::write_server_heads(dir, &marker) {
        tracing::debug!(error = %err, "could not record the adopted ref token");
    }
}

/// Record the last fast-forward durably, so the next command's staleness note
/// can report a change made by a previous one (rule 4).
pub fn record_report(dir: &Path, report: &RefSyncReport) -> Result<(), std::io::Error> {
    let record = RefSyncRecord {
        v: REF_SYNC_MARKER_VERSION,
        synced_unix: now_unix(),
        report: report.clone(),
    };
    let data = serde_json::to_vec_pretty(&record)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    std::fs::create_dir_all(dir)?;
    let mut temporary = NamedTempFile::new_in(dir)?;
    temporary.write_all(&data)?;
    temporary.flush()?;
    temporary
        .persist(dir.join(REF_SYNC_FILE))
        .map_err(|err| err.error)?;
    Ok(())
}

/// The last recorded fast-forward, if any. Never fails: an unreadable record
/// is no record, because this is read from output paths.
pub fn last_report(dir: &Path) -> Option<RefSyncRecord> {
    let data = std::fs::read(dir.join(REF_SYNC_FILE)).ok()?;
    let record: RefSyncRecord = serde_json::from_slice(&data).ok()?;
    (record.v <= REF_SYNC_MARKER_VERSION).then_some(record)
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_with_nothing_in_it_prints_nothing() {
        let report = RefSyncReport::default();
        assert!(!report.changed());
        assert!(report.summary_lines().is_empty());
    }

    #[test]
    fn a_fast_forward_and_a_conflict_are_both_reported() {
        let report = RefSyncReport {
            outcomes: vec![
                BookmarkOutcome {
                    name: "main".to_owned(),
                    update: BookmarkUpdate::FastForwarded {
                        from: Some("a".repeat(40)),
                        to: "b".repeat(40),
                    },
                },
                BookmarkOutcome {
                    name: "topic".to_owned(),
                    update: BookmarkUpdate::Conflicted {
                        adds: 2,
                        removes: 1,
                    },
                },
            ],
            suppressed: false,
        };
        let lines = report.summary_lines();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("Fast-forwarded bookmark main: aaaaaaaaaaaa -> bbbbbbbbbbbb"));
        assert!(lines[1].contains("topic"));
        assert!(lines[1].contains("conflicted"));
        assert!(report.changed());
    }

    #[test]
    fn a_record_round_trips_through_the_marker() {
        let dir = tempfile::tempdir().unwrap();
        assert!(last_report(dir.path()).is_none());
        let report = RefSyncReport {
            outcomes: vec![BookmarkOutcome {
                name: "main".to_owned(),
                update: BookmarkUpdate::FastForwarded {
                    from: None,
                    to: "c".repeat(40),
                },
            }],
            suppressed: false,
        };
        record_report(dir.path(), &report).unwrap();
        let record = last_report(dir.path()).unwrap();
        assert_eq!(record.report, report);
        assert_eq!(record.v, REF_SYNC_MARKER_VERSION);
    }

    #[test]
    fn a_record_from_a_newer_client_is_ignored_rather_than_misread() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(REF_SYNC_FILE),
            r#"{"v":99,"synced_unix":1,"report":{"outcomes":[],"suppressed":false}}"#,
        )
        .unwrap();
        assert!(last_report(dir.path()).is_none());
    }
}
