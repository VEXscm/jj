// Copyright 2021 The Jujutsu Authors
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

//! The content diff engine.
//!
//! The implementation moved to the `jj_diff` crate so `jj-backend` can run the
//! same algorithm without depending on all of `jj-lib`; see that crate's docs
//! for why it lives in the `jj-backend` workspace. This module re-exports it
//! unchanged, so `jj_lib::diff::*` still names exactly what it always did.
//!
//! Upstream changes to `jj/lib/src/diff.rs` belong in
//! `jj-backend/crates/jj_diff/src/diff.rs`, which is kept byte identical to
//! upstream so those patches apply with only a path adjustment.

pub use jj_diff::diff::*;
