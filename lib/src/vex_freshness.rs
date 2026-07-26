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

//! Read-path freshness for local-first repos (roadmap/088).
//!
//! Local op heads are authoritative for the session, and a read never waits
//! for the network to confirm them. Freshness happens *after* the command:
//! [`refresh_markers`] runs one budgeted `GetOpHeads` at the end of the
//! process, alongside the publisher drain, so the next command starts from a
//! current server head having paid nothing for it. Setting
//! `VEX_REFRESH_BUDGET_MS` opts back into a blocking check on the read path.
//!
//! A refresh never rewrites history; it either fast-forwards a local head that
//! is literally the server's own operation, or hands both heads to jj, whose
//! op-head resolution drops ancestors and merges anything genuinely divergent.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::Duration;

use jj_backend_types::ContentId;

use crate::vex::VexClient;
use crate::vex::vex_client_stats;
use crate::vex_publish::MarkerError;
use crate::vex_publish::PendingPublishMarker;
use crate::vex_publish::ServerHeadsMarker;
use crate::vex_publish::write_server_heads;

/// Default wall-clock budget for the opportunistic refresh, covering the
/// connect handshake as well as the request.
pub const DEFAULT_REFRESH_BUDGET_MS: u64 = 100;
/// Comfortably above a normal `GetOpHeads` round trip (measured 110-165 ms in
/// production) so the read actually completes instead of timing out.
pub const UNFOLDED_CLONE_REFRESH_BUDGET_MS: u64 = 3_000;

/// What to do with the local heads once the server's have been read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RefreshDecision {
    /// Server and local agree; keep serving local heads.
    Unchanged,
    /// The local head is the server's own operation and the server has moved
    /// on: adopt the newer heads outright.
    FastForward(Vec<ContentId>),
    /// The local head is not on the server (unpublished work, or a coalesced
    /// publish under a different id): serve both and let jj merge.
    Merge(Vec<ContentId>),
}

/// Budget for a *blocking* refresh on the read path. `None` — the default —
/// means the read never waits for the network at all.
///
/// An opportunistic refresh that blocks is not opportunistic: any budget at or
/// above the link's round trip time is spent in full on nearly every command,
/// because the request usually completes just inside it. Measured on the
/// reference laptop, a 100 ms budget against a ~90-110 ms RTT produced status
/// medians identical to synchronous mode, while forcing the budget below the
/// RTT dropped them from 0.16 s to 0.06 s. So the default is to serve local
/// heads immediately and let [`refresh_markers`] update the markers at the end
/// of the command, for the *next* one to use. Setting
/// `VEX_REFRESH_BUDGET_MS` opts back into blocking freshness.
pub fn refresh_budget() -> Option<Duration> {
    let raw = std::env::var("VEX_REFRESH_BUDGET_MS").ok()?;
    let millis = raw.trim().parse::<u64>().ok()?;
    (millis > 0).then(|| Duration::from_millis(millis))
}

/// Budget for the end-of-command refresh, from `VEX_REFRESH_BUDGET_MS` or
/// [`DEFAULT_REFRESH_BUDGET_MS`]. This one runs after the command's output is
/// done, so spending it costs the user nothing.
/// Budget for the one server read a clone performs while its registration is
/// still pending. It is on the read path, so it is bounded — but it must be
/// larger than a typical round trip, or the repository silently keeps serving
/// its clone-time head and never shows anyone else's work.
pub fn unfolded_clone_refresh_budget() -> Duration {
    refresh_budget().unwrap_or(Duration::from_millis(UNFOLDED_CLONE_REFRESH_BUDGET_MS))
}

pub fn background_refresh_budget() -> Duration {
    refresh_budget().unwrap_or(Duration::from_millis(DEFAULT_REFRESH_BUDGET_MS))
}

/// Decide what the freshly read server heads mean for this repo.
///
/// `local` is what the local marker serves, `chain` the queued operations, and
/// `server` the last confirmed server head set. Anything ambiguous resolves to
/// [`RefreshDecision::Merge`], which is always safe: jj filters ancestors out
/// of an op-head set before merging, so a strictly newer server head collapses
/// to itself.
pub fn plan_refresh(
    local: &[ContentId],
    chain: Option<&PendingPublishMarker>,
    server: Option<&ServerHeadsMarker>,
    fetched: &[ContentId],
) -> RefreshDecision {
    if fetched.is_empty() {
        return RefreshDecision::Unchanged;
    }
    let local_set: HashSet<&ContentId> = local.iter().collect();
    let fetched_set: HashSet<&ContentId> = fetched.iter().collect();
    if local_set == fetched_set {
        return RefreshDecision::Unchanged;
    }
    let queued = chain.is_some_and(|chain| !chain.is_empty());
    if !queued
        && let Some(server) = server
        && server.published_local_head.is_none()
        && server.stands_for(local)
    {
        // The local head is the server's own operation, so nothing local can
        // be lost by adopting the newer server heads wholesale.
        return RefreshDecision::FastForward(fetched.to_vec());
    }
    let mut merged = local.to_vec();
    for head in fetched {
        if !local_set.contains(head) {
            merged.push(*head);
        }
    }
    RefreshDecision::Merge(merged)
}

/// Divergence this repo can see without any RPC, as an op-head set for jj to
/// resolve. `None` means the markers describe a converged repo.
///
/// There are two ways the markers record divergence, and both have to be read,
/// because a repo hits them in either order.
///
/// *With a chain queued*: the publisher recorded a server head that the queued
/// chain is not parented on, so the chain cannot advance until jj merges that
/// head in locally.
///
/// *With nothing queued*: the repo's own publish is what created the
/// divergence. Under the delta contract a publish that lands beside a sibling
/// head is accepted rather than refused, so the chain drains to empty and the
/// recorded server heads come back holding a head this repo does not have.
/// Nothing else will ever mention it — the CAS refusal that used to force the
/// merge is precisely what the delta contract removes — so a reader that
/// looked only at the queue would keep committing on its own head and silently
/// stop seeing every other workspace's work. The recorded heads are a strict
/// superset of the local ones exactly then, and only then: any weaker
/// relationship (a coalesced publish, a server head this repo never observed)
/// means the marker no longer describes where the local head sits, and that is
/// the [`refresh_once`] path's business, not this one's.
///
/// Serving the union is self-limiting either way. jj's op-head resolution
/// merges the heads into one operation, which is queued and published like any
/// other; the drain then records the resulting single head, local and recorded
/// heads agree again, and reads go straight back to the local fast path.
///
/// The head this returns may not be readable offline, and that is accepted
/// rather than degraded. jj resolves the extra head by reading its operation
/// object, which is on the server; while the backend is unreachable such a
/// read fails, where serving the local head alone would have succeeded. The
/// obvious guard — serve the union only when every head is already in the
/// local object cache — is worse than the problem: a sibling head is learned
/// from a CAS response or a head refresh, never by fetching its object, so it
/// is *never* locally cached the first time it is served, and the guard would
/// suppress this branch in exactly the case it exists for. Telling "offline"
/// apart from "not fetched yet" needs a network probe, which is the one thing
/// a local-first read must not do. The regression is bounded to the window
/// between recording a sibling and merging it, is a plain read error rather
/// than corruption, and one online command clears it for good — whereas
/// hiding the sibling leaves the repository silently forked for as long as it
/// stays hidden.
pub fn known_divergence(
    local: &[ContentId],
    chain: Option<&PendingPublishMarker>,
    server: Option<&ServerHeadsMarker>,
) -> Result<Option<Vec<ContentId>>, MarkerError> {
    let Some(server) = server else {
        return Ok(None);
    };
    let heads = server.head_ids()?;
    match chain.filter(|chain| !chain.is_empty()) {
        Some(chain) => {
            let base = chain.base_ids()?;
            if heads.iter().collect::<HashSet<_>>() == base.iter().collect::<HashSet<_>>() {
                return Ok(None);
            }
        }
        None => {
            if !local.iter().all(|head| heads.contains(head)) {
                return Ok(None);
            }
        }
    }
    let local_set: HashSet<&ContentId> = local.iter().collect();
    let mut merged = local.to_vec();
    for head in &heads {
        if !local_set.contains(head) {
            merged.push(*head);
        }
    }
    Ok((merged.len() > local.len()).then_some(merged))
}

/// Run the refresh at most once per repo per process, within `budget`.
/// Returns the heads to serve, or `None` to keep serving the local ones.
pub fn refresh_once(
    dir: &Path,
    client: &VexClient,
    budget: Option<Duration>,
    local: &[ContentId],
    chain: Option<&PendingPublishMarker>,
    server: Option<&ServerHeadsMarker>,
) -> Option<Vec<ContentId>> {
    let budget = budget?;
    if let Err(previous) = claim_refresh(dir) {
        return previous;
    }
    let stats = vex_client_stats();
    stats.refresh_attempts.fetch_add(1, Ordering::Relaxed);
    let fetched = match client.get_op_heads_within(budget) {
        Ok(Some(fetched)) => fetched,
        Ok(None) => {
            stats.refresh_timeouts.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                ?budget,
                "op-head refresh exceeded its budget; serving local heads"
            );
            return record_refresh(dir, None);
        }
        Err(err) => {
            stats.refresh_timeouts.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(error = %err, "op-head refresh failed; serving local heads");
            return record_refresh(dir, None);
        }
    };
    match plan_refresh(local, chain, server, &fetched) {
        RefreshDecision::Unchanged => record_refresh(dir, None),
        RefreshDecision::FastForward(heads) => {
            if let Err(err) = write_server_heads(dir, &ServerHeadsMarker::new(heads.clone(), None))
            {
                tracing::debug!(error = %err, "could not record refreshed server heads");
                return record_refresh(dir, None);
            }
            if let Err(err) = crate::vex_publish::write_local_heads(dir, &heads) {
                tracing::debug!(error = %err, "could not fast-forward local heads");
                return record_refresh(dir, None);
            }
            record_refresh(dir, Some(heads))
        }
        RefreshDecision::Merge(heads) => {
            if let Err(err) = write_server_heads(dir, &ServerHeadsMarker::new(fetched, None)) {
                tracing::debug!(error = %err, "could not record refreshed server heads");
            }
            record_refresh(dir, Some(heads))
        }
    }
}

/// Update this repo's freshness markers after the command has finished, so the
/// *next* command starts from a current server head without any command ever
/// having waited for the network to read one.
///
/// Only ever advances bookkeeping: it records the server heads it saw, and
/// fast-forwards the local heads solely when they are the server's own
/// operation and nothing is queued. Divergence is left for the read path to
/// surface as a second head, which jj merges.
pub fn refresh_markers(dir: &Path, client: &VexClient) {
    let Ok(Some(chain)) = crate::vex_publish::read_pending_publish(dir) else {
        return refresh_with(dir, client, None);
    };
    refresh_with(dir, client, Some(&chain));
}

fn refresh_with(dir: &Path, client: &VexClient, chain: Option<&PendingPublishMarker>) {
    let Ok(Some(local)) = crate::vex_publish::read_local_heads(dir) else {
        return;
    };
    let local: Vec<ContentId> = local
        .iter()
        .filter_map(crate::vex_publish::content_id_from_op_id)
        .collect();
    let server = crate::vex_publish::read_server_heads(dir).ok().flatten();
    refresh_once(
        dir,
        client,
        Some(background_refresh_budget()),
        &local,
        chain,
        server.as_ref(),
    );
}

/// One refresh per repo per process. Keyed by directory rather than a plain
/// flag so tests (and any future multi-repo command) stay independent.
/// Per-process memo of the refresh outcome for one repository.
///
/// The refresh must happen at most once per process, but every later
/// `get_op_heads` in that process has to see the *same* answer. Returning the
/// stale local head to the second caller is how a repository ends up loaded at
/// its own head while the freshly fetched server head is silently dropped —
/// the reader then never sees another workspace's work.
type RefreshMemo = OnceLock<Mutex<HashMap<PathBuf, Option<Vec<ContentId>>>>>;
static REFRESHED: RefreshMemo = OnceLock::new();

fn refresh_memo() -> &'static Mutex<HashMap<PathBuf, Option<Vec<ContentId>>>> {
    REFRESHED.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `Ok(())` when this call owns the single refresh for `dir`; `Err(previous)`
/// when another call already made it, carrying that call's outcome.
fn claim_refresh(dir: &Path) -> Result<(), Option<Vec<ContentId>>> {
    let memo = refresh_memo().lock().unwrap();
    match memo.get(dir) {
        Some(previous) => Err(previous.clone()),
        None => Ok(()),
    }
}

fn record_refresh(dir: &Path, outcome: Option<Vec<ContentId>>) -> Option<Vec<ContentId>> {
    refresh_memo()
        .lock()
        .unwrap()
        .insert(dir.to_path_buf(), outcome.clone());
    outcome
}

#[cfg(test)]
mod tests {
    use jj_backend_types::ObjectKind;

    use super::*;

    fn id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    fn chain_with(base: ContentId, ops: &[ContentId]) -> PendingPublishMarker {
        let mut chain = PendingPublishMarker::new(&[base]);
        for op in ops {
            chain.push(op, &[(ObjectKind::Op, *op)]);
        }
        chain
    }

    #[test]
    fn identical_heads_need_no_action() {
        let server = ServerHeadsMarker::new(vec![id(1)], None);
        assert_eq!(
            plan_refresh(&[id(1)], None, Some(&server), &[id(1)]),
            RefreshDecision::Unchanged
        );
        assert_eq!(
            plan_refresh(&[id(1)], None, Some(&server), &[]),
            RefreshDecision::Unchanged
        );
    }

    #[test]
    fn a_published_local_head_fast_forwards() {
        let server = ServerHeadsMarker::new(vec![id(1)], None);
        assert_eq!(
            plan_refresh(&[id(1)], None, Some(&server), &[id(2)]),
            RefreshDecision::FastForward(vec![id(2)])
        );
    }

    #[test]
    fn a_coalesced_local_head_merges_instead_of_fast_forwarding() {
        // The server holds a rewrite of local operation id(5).
        let server = ServerHeadsMarker::new(vec![id(1)], Some(id(5)));
        assert_eq!(
            plan_refresh(&[id(5)], None, Some(&server), &[id(2)]),
            RefreshDecision::Merge(vec![id(5), id(2)])
        );
    }

    #[test]
    fn queued_operations_block_a_fast_forward() {
        let server = ServerHeadsMarker::new(vec![id(1)], None);
        let chain = chain_with(id(1), &[id(3)]);
        assert_eq!(
            plan_refresh(&[id(3)], Some(&chain), Some(&server), &[id(2)]),
            RefreshDecision::Merge(vec![id(3), id(2)])
        );
    }

    #[test]
    fn a_repo_without_markers_merges() {
        assert_eq!(
            plan_refresh(&[id(4)], None, None, &[id(2)]),
            RefreshDecision::Merge(vec![id(4), id(2)])
        );
    }

    #[test]
    fn known_divergence_needs_a_recorded_head_off_the_chain_base() {
        let chain = chain_with(id(1), &[id(3)]);
        let converged = ServerHeadsMarker::new(vec![id(1)], None);
        assert_eq!(
            known_divergence(&[id(3)], Some(&chain), Some(&converged)).unwrap(),
            None
        );
        let moved = ServerHeadsMarker::new(vec![id(9)], None);
        assert_eq!(
            known_divergence(&[id(3)], Some(&chain), Some(&moved)).unwrap(),
            Some(vec![id(3), id(9)])
        );
        let empty = PendingPublishMarker::new(&[id(1)]);
        assert_eq!(
            known_divergence(&[id(3)], Some(&empty), Some(&moved)).unwrap(),
            None,
            "with nothing queued a head the local one is not among is the \
             refresh path's business"
        );
    }

    /// A repo whose own publish landed beside a sibling drains to an empty
    /// queue, so the queue can no longer be what makes the divergence visible.
    /// The recorded server heads are a strict superset of the local ones
    /// exactly in that case.
    #[test]
    fn a_drained_repo_still_sees_a_sibling_head_the_server_recorded() {
        let diverged = ServerHeadsMarker::new(vec![id(1), id(2)], None);
        assert_eq!(
            known_divergence(&[id(2)], None, Some(&diverged)).unwrap(),
            Some(vec![id(2), id(1)]),
            "the local head first, then the sibling jj has to merge"
        );
        // An empty chain marker is the same state as no chain at all.
        let empty = PendingPublishMarker::new(&[id(2)]);
        assert_eq!(
            known_divergence(&[id(2)], Some(&empty), Some(&diverged)).unwrap(),
            Some(vec![id(2), id(1)])
        );
        // Once the merge is published the two agree again and the read goes
        // back to the local fast path — no loop, no permanent extra work.
        let converged = ServerHeadsMarker::new(vec![id(3)], None);
        assert_eq!(
            known_divergence(&[id(3)], None, Some(&converged)).unwrap(),
            None
        );
    }

    /// Anything short of a strict superset is left to [`refresh_once`]: the
    /// marker no longer describes where the local head sits, so the union is
    /// not something this function can derive offline.
    #[test]
    fn a_drained_repo_does_not_merge_heads_its_local_head_is_missing_from() {
        // The server moved off the local head entirely — a fast-forward
        // candidate, decided by `plan_refresh` against a fresh read.
        let moved = ServerHeadsMarker::new(vec![id(9)], None);
        assert_eq!(
            known_divergence(&[id(3)], None, Some(&moved)).unwrap(),
            None
        );
        // A coalesced publish: the server holds a rewrite of local id(5), so
        // the local head is legitimately absent from the recorded set.
        let coalesced = ServerHeadsMarker::new(vec![id(1), id(9)], Some(id(5)));
        assert_eq!(
            known_divergence(&[id(5)], None, Some(&coalesced)).unwrap(),
            None
        );
        // No marker at all says nothing about divergence.
        assert_eq!(known_divergence(&[id(3)], None, None).unwrap(), None);
    }

    #[test]
    fn the_read_path_does_not_block_on_refresh_by_default() {
        // Unset by default: a read serves local heads and never waits for the
        // network. Only an explicit VEX_REFRESH_BUDGET_MS opts back into a
        // blocking freshness check.
        assert_eq!(refresh_budget(), None);
        // The end-of-command refresh still has a budget, because spending it
        // costs the user nothing.
        assert_eq!(
            background_refresh_budget(),
            Duration::from_millis(DEFAULT_REFRESH_BUDGET_MS)
        );
    }

    /// One refresh per process per repository, and — critically — every later
    /// caller sees the same answer. Handing the second caller `None` is what
    /// let a repository load at its own stale head after the first call had
    /// already fetched the server's, so the reader never saw other work.
    #[test]
    fn refresh_is_claimed_once_and_replays_its_outcome() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        assert!(claim_refresh(dir).is_ok(), "first caller owns the refresh");

        let heads = vec![ContentId::from_bytes([7; 32])];
        assert_eq!(
            record_refresh(dir, Some(heads.clone())),
            Some(heads.clone())
        );

        // Every later caller in this process replays it rather than falling
        // back to the local head.
        assert_eq!(claim_refresh(dir), Err(Some(heads.clone())));
        assert_eq!(claim_refresh(dir), Err(Some(heads)));

        // A recorded miss replays as a miss, not as a fresh claim.
        let other = tempfile::tempdir().unwrap();
        assert!(claim_refresh(other.path()).is_ok());
        assert_eq!(record_refresh(other.path(), None), None);
        assert_eq!(claim_refresh(other.path()), Err(None));
    }
}
