use crate::common::VexLiveTestHarness;

// Run with:
//   cargo test -p jj-cli --test runner test_vex_ -- --ignored --nocapture
#[test]
#[ignore = "requires Docker and a live jj-backend harness"]
fn test_vex_init_commit_log_status_and_reopen() {
    let harness = VexLiveTestHarness::start();
    let repo = harness.repo("vex-init");
    let repo_dir = harness.init_repo(&repo, "repo");
    repo_dir.write_file("hello.txt", "hello vex\n");
    repo_dir.run_jj(["commit", "-m", "initial"]).success();

    harness.assert_clean_status(&repo_dir);
    harness.assert_log_description(&repo_dir, "@-", "initial");
    harness.assert_vex_metadata(&repo_dir, &repo);

    let reopened = harness.work_dir("repo");
    harness.assert_log_description(&reopened, "@-", "initial");
    harness.assert_clean_status(&reopened);
}

#[test]
#[ignore = "requires Docker and a live jj-backend harness"]
fn test_vex_clone_and_bookmark_propagation() {
    let harness = VexLiveTestHarness::start();
    let repo = harness.repo("vex-clone");
    let source = harness.init_repo(&repo, "source");
    source.write_file("hello.txt", "one\n");
    source.run_jj(["commit", "-m", "seed"]).success();
    source
        .run_jj(["bookmark", "create", "-r", "@-", "main"])
        .success();

    let first_clone = harness.clone_repo(&repo, "clone-1");
    harness.assert_log_description(&first_clone, "main", "seed");
    harness.assert_vex_cache_populated(&first_clone);

    source.run_jj(["workspace", "update-stale"]).success();
    source.write_file("hello.txt", "two\n");
    source.run_jj(["commit", "-m", "second"]).success();
    source
        .run_jj(["bookmark", "set", "-r", "@-", "main"])
        .success();

    let second_clone = harness.clone_repo(&repo, "clone-2");
    harness.assert_log_description(&second_clone, "main", "second");
    harness.assert_vex_cache_populated(&second_clone);
}

#[test]
#[ignore = "requires Docker and a live jj-backend harness"]
fn test_vex_backend_restart_preserves_repo_state() {
    let mut harness = VexLiveTestHarness::start();
    let repo = harness.repo("vex-restart");
    {
        let source = harness.init_repo(&repo, "source");
        source.write_file("hello.txt", "persist me\n");
        source.run_jj(["commit", "-m", "seed"]).success();
        source
            .run_jj(["bookmark", "create", "-r", "@-", "main"])
            .success();
    }

    harness.restart_backend();

    let source = harness.work_dir("source");
    harness.assert_log_description(&source, "@-", "seed");
    harness.assert_log_description(&source, "main", "seed");

    let clone = harness.clone_repo(&repo, "clone-after-restart");
    harness.assert_log_description(&clone, "main", "seed");
}

#[test]
#[ignore = "requires Docker and a live jj-backend harness"]
fn test_vex_virtual_fs_clone_uses_virtual_working_copy_and_lazy_blob_prefetch() {
    let harness = VexLiveTestHarness::start();
    let repo = harness.repo("vex-virtual-fs-clone");
    let source = harness.init_repo(&repo, "source");
    source.write_file("hello.txt", "seed\n");
    source.run_jj(["commit", "-m", "seed"]).success();
    source
        .run_jj(["bookmark", "create", "-r", "@-", "main"])
        .success();

    let virtual_clone = harness.clone_repo_with_args(&repo, "virtual-clone", ["--fs=virtual"]);
    harness.assert_log_description(&virtual_clone, "main", "seed");
    harness.assert_working_copy_type(&virtual_clone, "vex-virtual");
    harness.assert_path_missing(&virtual_clone, "hello.txt");
    harness.assert_no_cached_kind_entries(&virtual_clone, "blob");
}
