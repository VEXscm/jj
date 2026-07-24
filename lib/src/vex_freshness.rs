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

/// Divergence this repo can see without any RPC: the publisher recorded a
/// server head that the queued chain is not parented on, so the chain cannot
/// advance until jj merges that head in locally.
pub fn known_divergence(
    local: &[ContentId],
    chain: Option<&PendingPublishMarker>,
    server: Option<&ServerHeadsMarker>,
) -> Result<Option<Vec<ContentId>>, MarkerError> {
    let (Some(chain), Some(server)) = (chain, server) else {
        return Ok(None);
    };
    if chain.is_empty() {
        return Ok(None);
    }
    let base = chain.base_ids()?;
    let heads = server.head_ids()?;
    if heads.iter().collect::<HashSet<_>>() == base.iter().collect::<HashSet<_>>() {
        return Ok(None);
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
    if !claim_refresh(dir) {
        return None;
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
            return None;
        }
        Err(err) => {
            stats.refresh_timeouts.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(error = %err, "op-head refresh failed; serving local heads");
            return None;
        }
    };
    match plan_refresh(local, chain, server, &fetched) {
        RefreshDecision::Unchanged => None,
        RefreshDecision::FastForward(heads) => {
            if let Err(err) = write_server_heads(dir, &ServerHeadsMarker::new(heads.clone(), None))
            {
                tracing::debug!(error = %err, "could not record refreshed server heads");
                return None;
            }
            if let Err(err) = crate::vex_publish::write_local_heads(dir, &heads) {
                tracing::debug!(error = %err, "could not fast-forward local heads");
                return None;
            }
            Some(heads)
        }
        RefreshDecision::Merge(heads) => {
            if let Err(err) = write_server_heads(dir, &ServerHeadsMarker::new(fetched, None)) {
                tracing::debug!(error = %err, "could not record refreshed server heads");
            }
            Some(heads)
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
fn claim_refresh(dir: &Path) -> bool {
    static CLAIMED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    CLAIMED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap()
        .insert(dir.to_path_buf())
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
            None
        );
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

    #[test]
    fn refresh_is_claimed_once_per_directory() {
        let temp = tempfile::tempdir().unwrap();
        assert!(claim_refresh(temp.path()));
        assert!(!claim_refresh(temp.path()));
    }
}
