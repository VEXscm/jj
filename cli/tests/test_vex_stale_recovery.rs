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

//! S13 (roadmap/088 Stage 7): stale-working-copy recovery is jj's merge-based
//! path again, for every CLI name.
//!
//! `recover_stale_working_copy` used to fork on `current_cli_name() != "vex"`.
//! Under the name `vex` it abandoned upstream's recovery — which snapshots on
//! top of the stale operation and lets the op-heads store merge the resulting
//! divergent head — and created a `RECOVERY COMMIT` instead, because the Vex
//! op-heads store was a strict single-head server CAS that rejected the
//! transiently divergent head. Op heads are stored locally now and hold several
//! heads natively, so the fork is deleted and both names take the same path.
//!
//! These tests drive the binary under the name `vex` on purpose: run as `jj`
//! they would have passed before the deletion too.

use crate::common::CommandOutput;
use crate::common::TestEnvironment;
use crate::common::TestWorkDir;

const VEX: &str = "vex";

fn run_vex<I>(work_dir: &TestWorkDir, args: I) -> CommandOutput
where
    I: IntoIterator,
    I::Item: AsRef<std::ffi::OsStr>,
{
    work_dir.run_jj_with_executable_name(VEX, args)
}

fn log_output(work_dir: &TestWorkDir) -> CommandOutput {
    let template = r#"
    separate(" ",
      commit_id.short(),
      working_copies,
      surround('"', '"', description.first_line()),
    )
    "#;
    run_vex(work_dir, ["log", "-T", template, "-r", "all()"])
}

/// A workspace whose recorded operation is no longer the op head, with local
/// changes to preserve, recovers by updating to the fresh commit — not by
/// creating a recovery commit.
#[test]
fn test_vex_stale_working_copy_recovers_by_merge_not_recovery_commit() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "main"]).success();
    let main_dir = test_env.work_dir("main");
    let secondary_dir = test_env.work_dir("secondary");

    main_dir.write_file("file", "contents\n");
    main_dir.run_jj(["new"]).success();
    main_dir
        .run_jj(["workspace", "add", "../secondary"])
        .success();

    // Unsnapshotted work in the secondary workspace, so the recovery has
    // something to preserve and `check_stale` reports `WorkingCopyStale`
    // rather than `Fresh`.
    secondary_dir.write_file("file", "secondary edit\n");

    // Rewrite the secondary workspace's working-copy commit from the main
    // workspace. The secondary's recorded operation is now an ancestor of the
    // head rather than the head itself.
    main_dir.write_file("file", "changed in main\n");
    main_dir.run_jj(["squash"]).success();

    let output = run_vex(&secondary_dir, ["status"]);
    let stderr = output.stderr.raw();
    assert!(
        stderr.contains("Updated working copy to fresh commit"),
        "expected jj's merge-based recovery; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("RECOVERY COMMIT"),
        "recovery must not mint a recovery commit any more; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("CAS conflict on op heads"),
        "the local op-heads store cannot refuse the divergent head; got:\n{stderr}"
    );
    output.success();

    // The recovery is a working-copy update, so no `RECOVERY COMMIT` exists
    // anywhere in the log.
    let log = log_output(&secondary_dir);
    assert!(
        !log.stdout.raw().contains("RECOVERY COMMIT"),
        "unexpected recovery commit in the log:\n{}",
        log.stdout.raw()
    );

    // The unsnapshotted edit was not silently dropped: upstream's recovery
    // snapshots it onto the stale working-copy commit before updating to the
    // fresh one, so it is still reachable in the operation log.
    let op_log = run_vex(&secondary_dir, ["operation", "log", "-T", "description"]);
    assert!(
        op_log.stdout.raw().contains("snapshot working copy"),
        "the stale working copy was never snapshotted:\n{}",
        op_log.stdout.raw()
    );
    op_log.success();

    // Recovery converged: the next command is an ordinary no-op, not another
    // round of recovery.
    let output = run_vex(&secondary_dir, ["status"]);
    let stderr = output.stderr.raw();
    assert!(
        !stderr.contains("Updated working copy to fresh commit"),
        "recovery should not repeat; got:\n{stderr}"
    );
    output.success();
}

/// The "sibling operation" lockup is unreachable.
///
/// When the working copy's operation is a *sibling* of the repo operation
/// (their closest common ancestor is neither), the deleted `vex` arm minted a
/// recovery commit and — because the strict single-head CAS then rejected the
/// operation it produced — could leave the workspace pinned to the same stale
/// operation, so every subsequent command reported "working copy is stale"
/// again. Now the sibling case takes jj's merge-based path, so it recovers once
/// and stays recovered.
#[test]
fn test_vex_sibling_operation_lockup_is_unreachable() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "main"]).success();
    let main_dir = test_env.work_dir("main");
    let secondary_dir = test_env.work_dir("secondary");

    main_dir.write_file("file", "contents\n");
    main_dir.run_jj(["new"]).success();
    main_dir
        .run_jj(["workspace", "add", "../secondary"])
        .success();

    // Advance the secondary workspace so it records an operation of its own,
    // then branch the repository off that operation's *parent* from the main
    // workspace. The two operations are siblings: neither is an ancestor of the
    // other.
    secondary_dir
        .run_jj(["describe", "-m", "secondary work"])
        .success();
    secondary_dir.write_file("file", "secondary edit\n");
    main_dir
        .run_jj(["describe", "-m", "main work", "--at-op", "@-"])
        .success();

    // Recovery must succeed and must not create a recovery commit.
    let output = run_vex(&secondary_dir, ["status"]);
    let stderr = output.stderr.raw();
    assert!(
        !stderr.contains("RECOVERY COMMIT"),
        "the sibling case must not mint a recovery commit; got:\n{stderr}"
    );
    assert!(
        !stderr.contains("CAS conflict on op heads"),
        "the local op-heads store cannot refuse a divergent head; got:\n{stderr}"
    );
    output.success();

    // The lockup was that this stayed stale forever. Two further commands must
    // both succeed without reporting staleness.
    for _ in 0..2 {
        let output = run_vex(&secondary_dir, ["status"]);
        let stderr = output.stderr.raw();
        assert!(
            !stderr.contains("working copy is stale"),
            "the workspace is still pinned to a stale operation:\n{stderr}"
        );
        assert!(
            !stderr.contains("Attempted recovery"),
            "recovery should not repeat once it has converged:\n{stderr}"
        );
        output.success();
    }

    // The unsnapshotted edit was never lost along the way.
    assert_eq!(secondary_dir.read_file("file"), b"secondary edit\n");
}

/// A snapshot whose operation write fails reports the failure instead of
/// retrying. The bounded `MAX_OP_HEAD_SNAPSHOT_ATTEMPTS` ladder is gone (S11):
/// with local op heads there is no concurrency refusal to retry, so a write
/// error is a genuine failure. Asserted at the surface that remains observable
/// from the CLI: a snapshot never prints the retry notice.
#[test]
fn test_snapshot_does_not_retry_on_op_head_write_failure() {
    let test_env = TestEnvironment::default();
    test_env.run_jj_in(".", ["git", "init", "main"]).success();
    let main_dir = test_env.work_dir("main");
    let secondary_dir = test_env.work_dir("secondary");

    main_dir.write_file("file", "contents\n");
    main_dir.run_jj(["new"]).success();
    main_dir
        .run_jj(["workspace", "add", "../secondary"])
        .success();

    // Two workspaces snapshotting against one repo is the case the ladder
    // existed for.
    secondary_dir.write_file("file", "secondary\n");
    main_dir.write_file("file", "main\n");

    for dir in [&main_dir, &secondary_dir] {
        let output = run_vex(dir, ["status"]);
        let stderr = output.stderr.raw();
        assert!(
            !stderr.contains("reloading and retrying the working-copy snapshot"),
            "the snapshot retry ladder should be gone; got:\n{stderr}"
        );
        output.success();
    }
}
