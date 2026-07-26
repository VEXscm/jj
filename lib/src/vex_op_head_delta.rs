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

//! Client half of the divergence-tolerant op-head publish.
//!
//! `OpHeadsStore::update_op_heads(old_ids, new_id)` is defined by jj as a set
//! *delta*: remove those ids, add this one. The Vex backend historically
//! reinterpreted it as set equality plus full replacement, which forces two
//! writers branching from the same head into a conflict even though jj's own
//! model has room for both. The server now implements the real delta when the
//! request asks for it; this module is the single place that decides whether to
//! ask, and the single place that interprets the structured refusal that comes
//! back.
//!
//! Everything here is deliberately free functions over plain values so the
//! synchronous publish path — which holds a concrete `VexClient` and therefore
//! has no seam for a fake transport — is still testable.

use std::sync::OnceLock;

/// Whether this client asks the backend for the delta semantics. On by default;
/// `VEX_OP_HEADS_DELTA_CAS=0` (or `false`/`off`/`no`) opts back out onto the
/// server's set-equality path without a new release. The server has its own
/// kill switch that overrides this one.
pub(crate) fn divergence_ok() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| divergence_ok_from(std::env::var("VEX_OP_HEADS_DELTA_CAS").ok()))
}

/// [`divergence_ok`] over an explicit value, so the policy is testable without
/// mutating the process environment (the cached read above happens once per
/// process and cannot be re-observed).
fn divergence_ok_from(value: Option<impl AsRef<str>>) -> bool {
    !matches!(
        value.as_ref().map(AsRef::as_ref),
        Some("0") | Some("false") | Some("off") | Some("no")
    )
}

/// The head-set bound this client advertises. Zero means "use whatever bound
/// the server is configured with", which is what we want: the bound is an
/// operational rail on the server side, and a client that pinned its own would
/// only drift from it.
pub(crate) fn max_op_heads() -> u32 {
    0
}

/// The heads a workspace serves when the server holds several: its own, plus
/// every server head it does not already have, in that order.
///
/// Handing all of them to jj is what lets `resolve_op_heads` merge the
/// divergence into one operation and publish it. Serving only the local head
/// instead hides the other heads from jj entirely, so the repository looks
/// converged, no merge is ever built, and the workspace stays pinned to
/// whatever it last saw.
pub(crate) fn heads_with_local(
    local: &[jj_backend_types::ContentId],
    server: &[jj_backend_types::ContentId],
) -> Vec<jj_backend_types::ContentId> {
    let known: std::collections::HashSet<_> = local.iter().collect();
    local
        .iter()
        .copied()
        .chain(server.iter().filter(|id| !known.contains(*id)).copied())
        .collect()
}

/// The refusal a clone reports when it cannot rebuild its deferred workspace
/// registration because the repository is divergent.
///
/// It is a conflict, not a failure: reloading runs jj's op-head resolution,
/// which merges the heads back to one and makes the rebuild well defined. The
/// literal prefix is load bearing —
/// [`crate::transaction::is_op_heads_cas_conflict`] falls back to matching it
/// for refusals that are raised as bare messages, and the CLI retry loops sit
/// on top of that classification.
pub(crate) fn divergent_registration_conflict(server_heads: usize) -> String {
    format!(
        "CAS conflict on op heads: cannot republish the deferred workspace registration while \
         the repository has {server_heads} divergent op heads; reload and retry"
    )
}

/// Why the backend refused to move the op heads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpHeadRefusal {
    /// The publish did not prove that every id it wanted removed is reachable
    /// from the id it wanted added — under the legacy path, that the live head
    /// set did not equal `expected`. Ordinary contention: reload and retry.
    PreconditionUnmet,
    /// The repository is already carrying as many divergent op heads as the
    /// server permits, so adding another was refused. Also resolved by a
    /// reload-and-retry — jj merges the divergent heads on the way back, which
    /// is what brings the count down — but the cause is worth naming, because
    /// it means reconciliation is not keeping up rather than that one other
    /// writer happened to win a race.
    HeadSetSaturated,
}

/// A structured `CommitOperation` refusal, carried as the source of an
/// [`crate::op_heads_store::OpHeadsStoreError::Write`].
///
/// [`crate::transaction::is_op_heads_cas_conflict`] downcasts to this rather
/// than matching on the message. The `Display` text still begins with the
/// literal `CAS conflict on op heads` because older clients, the CLI's
/// user-facing error, and several retry loops outside this crate key off that
/// string.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpHeadCasConflict {
    refusal: OpHeadRefusal,
    /// The op heads the server reported live at the moment it refused. The
    /// caller reconciles from these instead of issuing another read.
    current_heads: Vec<String>,
    /// The bound in force when the refusal was a saturation, for the message.
    max_op_heads: u32,
}

impl OpHeadCasConflict {
    /// Why the server refused.
    pub fn refusal(&self) -> OpHeadRefusal {
        self.refusal
    }

    /// The op heads the server reported live at the moment it refused.
    pub fn current_heads(&self) -> &[String] {
        &self.current_heads
    }
}

impl std::fmt::Display for OpHeadCasConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.refusal {
            OpHeadRefusal::PreconditionUnmet => write!(f, "CAS conflict on op heads"),
            OpHeadRefusal::HeadSetSaturated => write!(
                f,
                "CAS conflict on op heads: the operation head set is saturated ({} divergent \
                 heads, server limit {}); this publish would have added another. Reload and \
                 retry — resolving the divergence merges them back down.",
                self.current_heads.len(),
                self.max_op_heads
            ),
        }
    }
}

impl std::error::Error for OpHeadCasConflict {}

/// Classify a non-`ok` `CommitOperation` response.
///
/// A server that predates the structured reason leaves `failure_reason` at
/// `UNSPECIFIED` and describes itself in `error_message`; that is treated as
/// the ordinary precondition failure it has always been.
pub(crate) fn classify_refusal(
    response: &jj_backend_api::CommitOperationResponse,
) -> OpHeadCasConflict {
    let refusal =
        match jj_backend_api::CommitOperationFailureReason::try_from(response.failure_reason) {
            Ok(jj_backend_api::CommitOperationFailureReason::HeadSetBoundExceeded) => {
                OpHeadRefusal::HeadSetSaturated
            }
            _ => OpHeadRefusal::PreconditionUnmet,
        };
    OpHeadCasConflict {
        refusal,
        current_heads: response.current_op_head_ids.clone(),
        max_op_heads: response.effective_max_op_heads,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(
        reason: i32,
        heads: &[&str],
        limit: u32,
    ) -> jj_backend_api::CommitOperationResponse {
        jj_backend_api::CommitOperationResponse {
            ok: false,
            current_op_head_ids: heads.iter().map(|id| (*id).to_string()).collect(),
            error_message: "CAS conflict on op heads".to_string(),
            failure_reason: reason,
            effective_max_op_heads: limit,
            divergence_applied: false,
        }
    }

    #[test]
    fn divergence_is_requested_by_default() {
        assert!(divergence_ok_from(None::<&str>));
        assert!(divergence_ok_from(Some("1")));
        assert!(divergence_ok_from(Some("")));
    }

    #[test]
    fn the_kill_switch_values_opt_out() {
        assert!(!divergence_ok_from(Some("0")));
        assert!(!divergence_ok_from(Some("false")));
        assert!(!divergence_ok_from(Some("off")));
        assert!(!divergence_ok_from(Some("no")));
    }

    #[test]
    fn the_client_defers_the_bound_to_the_server() {
        assert_eq!(max_op_heads(), 0);
    }

    #[test]
    fn an_unspecified_reason_is_the_ordinary_precondition_failure() {
        let conflict = classify_refusal(&response(0, &["aa"], 0));
        assert_eq!(conflict.refusal(), OpHeadRefusal::PreconditionUnmet);
        assert_eq!(conflict.to_string(), "CAS conflict on op heads");
    }

    #[test]
    fn a_precondition_reason_keeps_the_legacy_message() {
        let conflict = classify_refusal(&response(
            jj_backend_api::CommitOperationFailureReason::PreconditionUnmet as i32,
            &["aa", "bb"],
            32,
        ));
        assert_eq!(conflict.refusal(), OpHeadRefusal::PreconditionUnmet);
        assert_eq!(conflict.to_string(), "CAS conflict on op heads");
        assert_eq!(
            conflict.current_heads(),
            ["aa".to_string(), "bb".to_string()]
        );
    }

    #[test]
    fn saturation_names_itself_and_carries_the_live_heads() {
        let conflict = classify_refusal(&response(
            jj_backend_api::CommitOperationFailureReason::HeadSetBoundExceeded as i32,
            &["aa", "bb"],
            2,
        ));
        assert_eq!(conflict.refusal(), OpHeadRefusal::HeadSetSaturated);
        let message = conflict.to_string();
        assert!(message.starts_with("CAS conflict on op heads"), "{message}");
        assert!(message.contains("saturated"), "{message}");
        assert!(message.contains("server limit 2"), "{message}");
        assert_eq!(conflict.current_heads().len(), 2);
    }

    fn content_id(byte: u8) -> jj_backend_types::ContentId {
        jj_backend_types::ContentId::from_bytes([byte; 32])
    }

    #[test]
    fn a_divergent_server_is_served_alongside_the_local_head() {
        // The local head first, then every server head the workspace does not
        // already have: jj merges what it is given, and it can only merge heads
        // it is shown.
        assert_eq!(
            heads_with_local(&[content_id(1)], &[content_id(2), content_id(3)]),
            vec![content_id(1), content_id(2), content_id(3)]
        );
        // A server head the workspace already holds is not repeated.
        assert_eq!(
            heads_with_local(&[content_id(1)], &[content_id(1), content_id(2)]),
            vec![content_id(1), content_id(2)]
        );
    }

    #[test]
    fn the_divergent_registration_refusal_names_itself_as_a_conflict() {
        let message = divergent_registration_conflict(3);
        assert!(message.starts_with("CAS conflict on op heads"), "{message}");
        assert!(message.contains("3 divergent op heads"), "{message}");
    }

    #[test]
    fn an_unknown_future_reason_degrades_to_a_retryable_conflict() {
        let conflict = classify_refusal(&response(97, &[], 0));
        assert_eq!(conflict.refusal(), OpHeadRefusal::PreconditionUnmet);
    }
}
