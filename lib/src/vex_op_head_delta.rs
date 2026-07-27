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

//! What this client asks of the server's op-head CAS, on the one path that
//! still talks to it.
//!
//! Op heads are stored locally now (roadmap/088 Stage 7): `update_op_heads` is
//! a directory write under a real file lock, so two writers branching from the
//! same head simply both land and the next read merges them. Nothing on the
//! default path can be refused for concurrency reasons, which is why this
//! module no longer carries a refusal type, a classifier, or the retry ladders
//! that sat on top of them — they were deleted with the rest of the client CAS
//! apparatus (Stage 7, D10/S11).
//!
//! What survives is the request-shaping policy for `commit_op_heads`, the
//! server CAS reached only under the `VEX_PUBLISH_OP_LOG=1` escape, where a
//! refusal is logged and the local head stands. Stage 10 removes the server
//! half and this module with it.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
