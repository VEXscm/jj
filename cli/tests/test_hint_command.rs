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

use crate::common::TestEnvironment;

#[test]
fn test_hint_disable_and_enable_user_level() {
    let test_env = TestEnvironment::default();

    // Disabling a hint writes the backing `hints.*` config key to `false`.
    let output = test_env.run_jj_in(".", ["hint", "disable", "watchman"]);
    insta::assert_snapshot!(output, @r#"
    ------- stderr -------
    Disabled hint "watchman".
    [EOF]
    "#);
    let output = test_env.run_jj_in(".", ["config", "get", "hints.fsmonitor"]);
    insta::assert_snapshot!(output, @"
    false
    [EOF]
    ");

    // Re-enabling flips it back to `true`.
    let output = test_env.run_jj_in(".", ["hint", "enable", "watchman"]);
    insta::assert_snapshot!(output, @r#"
    ------- stderr -------
    Enabled hint "watchman".
    [EOF]
    "#);
    let output = test_env.run_jj_in(".", ["config", "get", "hints.fsmonitor"]);
    insta::assert_snapshot!(output, @"
    true
    [EOF]
    ");
}

#[test]
fn test_hint_accepts_aliases() {
    let test_env = TestEnvironment::default();

    let output = test_env.run_jj_in(".", ["hint", "disable", "fsmonitor"]);
    insta::assert_snapshot!(output, @r#"
    ------- stderr -------
    Disabled hint "watchman".
    [EOF]
    "#);
    let output = test_env.run_jj_in(".", ["config", "get", "hints.fsmonitor"]);
    insta::assert_snapshot!(output, @"
    false
    [EOF]
    ");
}

#[test]
fn test_hint_unknown_name_errors() {
    let test_env = TestEnvironment::default();
    let output = test_env.run_jj_in(".", ["hint", "disable", "nonexistent"]);
    insta::assert_snapshot!(output, @r#"
    ------- stderr -------
    Error: Unknown hint "nonexistent". Known hints: watchman, resolving-conflicts
    [EOF]
    [exit status: 1]
    "#);
}

#[test]
fn test_hint_list() {
    let test_env = TestEnvironment::default();

    let output = test_env.run_jj_in(".", ["hint", "list"]);
    insta::assert_snapshot!(output, @r"
    watchman             enabled   Suggest installing a filesystem monitor when snapshots are slow
    resolving-conflicts  enabled   Explain how to resolve conflicts after an operation creates them
    [EOF]
    ");

    test_env
        .run_jj_in(".", ["hint", "disable", "watchman"])
        .success();
    let output = test_env.run_jj_in(".", ["hint", "list"]);
    insta::assert_snapshot!(output, @r"
    watchman             disabled  Suggest installing a filesystem monitor when snapshots are slow
    resolving-conflicts  enabled   Explain how to resolve conflicts after an operation creates them
    [EOF]
    ");
}

#[test]
fn test_hint_disable_repo_level() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "repo"]).success();
    let work_dir = test_env.work_dir("repo");

    let output = work_dir.run_jj(["hint", "disable", "watchman", "--repo"]);
    insta::assert_snapshot!(output, @r#"
    ------- stderr -------
    Disabled hint "watchman".
    [EOF]
    "#);
    let output = work_dir.run_jj(["config", "get", "hints.fsmonitor"]);
    insta::assert_snapshot!(output, @"
    false
    [EOF]
    ");
}
