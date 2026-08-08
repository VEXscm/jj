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

use jj_lib::backend::TreeValue;
use jj_lib::repo::Repo as _;
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::tree_builder::TreeBuilder;
use pollster::FutureExt as _;
use testutils::TestRepo;

/// A `vex materialize` job handed the repository root as its `--root-path` used
/// to abort the whole CI task with `assertion failed: !path.is_root()`, naming
/// no input. The root has no parent tree to hold an entry for it, so it is bad
/// input the caller can report, not a broken internal invariant.
#[test]
fn overriding_the_repository_root_is_a_recoverable_error() {
    let test_repo = TestRepo::init();
    let store = test_repo.repo.store().clone();
    let empty_tree = store.empty_tree_id().clone();
    let root = RepoPathBuf::from_internal_string("").unwrap();
    assert!(root.is_root(), "the empty path must parse to the root");

    let mut builder = TreeBuilder::new(store, empty_tree.clone());
    let set = builder
        .set(root.clone(), TreeValue::Tree(empty_tree.clone()))
        .expect_err("setting the root as a tree entry must fail");
    let remove = builder
        .remove(root.clone())
        .expect_err("removing the root as a tree entry must fail");
    let set_or_remove = builder
        .set_or_remove(root, Some(TreeValue::Tree(empty_tree)))
        .expect_err("set_or_remove on the root must fail");

    for error in [set, remove, set_or_remove] {
        let message = error.to_string();
        assert!(
            message.contains("repository root"),
            "the error must name the offending input, got: {message}"
        );
    }
}

/// The rejection must not leave a half-applied override behind: a builder that
/// refused a bad path still writes the tree its accepted overrides describe.
#[test]
fn a_rejected_root_override_leaves_the_builder_usable() {
    let test_repo = TestRepo::init();
    let store = test_repo.repo.store().clone();
    let empty_tree = store.empty_tree_id().clone();

    let mut builder = TreeBuilder::new(store.clone(), empty_tree.clone());
    builder
        .set(
            RepoPathBuf::from_internal_string("").unwrap(),
            TreeValue::Tree(empty_tree.clone()),
        )
        .unwrap_err();
    builder
        .set(
            RepoPathBuf::from_internal_string("jj").unwrap(),
            TreeValue::Tree(empty_tree),
        )
        .unwrap();

    let names = async {
        let tree_id = builder.write_tree().await.unwrap();
        let tree = store.get_tree(RepoPathBuf::root(), &tree_id).await.unwrap();
        tree.entries_non_recursive()
            .map(|entry| entry.name().as_internal_str().to_string())
            .collect::<Vec<String>>()
    }
    .block_on();
    assert_eq!(names, vec!["jj".to_string()]);
}
