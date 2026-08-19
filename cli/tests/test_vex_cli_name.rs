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

//! Hints and errors name the binary the user actually has.
//!
//! Run as `jj` the rename in `cli/src/cli_name.rs` substitutes `jj` for `jj`,
//! so the rest of this suite would pass whether or not it works. These tests
//! drive the binary under the name `vex` so the rewriting path is the one
//! being snapshotted.

use crate::common::TestEnvironment;

const VEX: &str = "vex";

#[test]
fn test_error_and_hints_name_the_running_binary() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    let output = work_dir.run_jj_with_executable_name(VEX, ["restore", "-r", "@"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: `vex restore` does not have a `--revision`/`-r` option.
    Hint: To modify the current revision, use `--from`.
    Hint: To undo changes in a revision compared to its parents, use `--changes-in`.
    [EOF]
    [exit status: 1]
    ");
}

/// The config footer names the command to run but links to upstream's docs.
/// Only the first is a command, and only the first is renamed.
#[test]
fn test_config_error_keeps_the_upstream_docs_url() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    let output =
        work_dir.run_jj_with_executable_name(VEX, ["--config", "ui.color=nonsense", "status"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Config error: Invalid type or value for ui.color
    Caused by: unknown variant `nonsense`, expected one of `always`, `never`, `debug`, `auto`

    For help, see https://docs.jj-vcs.dev/latest/config/ or use `vex help -k config`.
    [EOF]
    [exit status: 1]
    ");
}

/// A revision the user named after the tool is data, not advice. Renaming it
/// would send them after a bookmark that does not exist.
#[test]
fn test_a_bookmark_named_jj_is_left_alone() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    let output = work_dir.run_jj_with_executable_name(VEX, ["log", "-r", "jj"]);
    insta::assert_snapshot!(output, @r"
    ------- stderr -------
    Error: Revision `jj` doesn't exist
    [EOF]
    [exit status: 1]
    ");
}
