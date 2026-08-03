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

//! What this client would ask of the server's op-head publication.
//!
//! Op heads are stored locally now (roadmap/088 Stage 7): `update_op_heads` is
//! a directory write under a real file lock, so two writers branching from the
//! same head simply both land and the next read merges them. Nothing on the
//! default path can be refused for concurrency reasons, which is why this
//! module no longer carries a refusal type, a classifier, or the retry ladders
//! that sat on top of them — they were deleted with the rest of the client CAS
//! apparatus (Stage 7, D10/S11).
//!
//! Stage 10 then retired the server's set-equality path, so there is nothing
//! left to opt into or out of either: every `CommitOperation` gets jj's
//! remove-and-add delta whatever the request says. `VEX_OP_HEADS_DELTA_CAS` is
//! gone from both halves rather than left as a knob that reads as live
//! configuration and turns nothing.
//!
//! What survives is the request-shaping policy for `commit_op_heads`, which no
//! client path calls: the compatibility escape that used to reach the server
//! publication was deleted (Stage 9). This module goes with the RPC itself.

/// Whether this client asks the backend for the delta semantics.
///
/// Constant since roadmap 088 Stage 10. The server ignores the field — it has
/// only the delta contract left — and it is still set truthfully rather than
/// dropped so an older server, if one were ever rolled back to, still takes the
/// path this client expects.
pub(crate) fn divergence_ok() -> bool {
    true
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

    /// The field is no longer a choice: there is no server path left for a
    /// `false` to select, so nothing may reintroduce one.
    #[test]
    fn divergence_is_always_requested() {
        assert!(divergence_ok());
    }

    #[test]
    fn the_client_defers_the_bound_to_the_server() {
        assert_eq!(max_op_heads(), 0);
    }
}
