// Copyright 2020 The Jujutsu Authors
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

#![expect(missing_docs)]

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::StreamExt as _;
use futures::TryStreamExt as _;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::FuturesUnordered;

use crate::backend;
use crate::backend::BackendError;
use crate::backend::BackendResult;
use crate::backend::TreeId;
use crate::backend::TreeValue;
use crate::repo_path::RepoPathBuf;
use crate::repo_path::RepoPathComponentBuf;
use crate::repo_path::RepoPathTree;
use crate::store::Store;
use crate::tree::Tree;

#[derive(Debug)]
enum Override {
    Tombstone,
    Replace(TreeValue),
}

#[derive(Debug)]
pub struct TreeBuilder {
    store: Arc<Store>,
    base_tree_id: TreeId,
    overrides: BTreeMap<RepoPathBuf, Override>,
}

impl TreeBuilder {
    pub fn new(store: Arc<Store>, base_tree_id: TreeId) -> Self {
        let overrides = BTreeMap::new();
        Self {
            store,
            base_tree_id,
            overrides,
        }
    }

    pub fn store(&self) -> &Store {
        self.store.as_ref()
    }

    pub fn set(&mut self, path: RepoPathBuf, value: TreeValue) {
        assert!(!path.is_root());
        self.overrides.insert(path, Override::Replace(value));
    }

    pub fn remove(&mut self, path: RepoPathBuf) {
        assert!(!path.is_root());
        self.overrides.insert(path, Override::Tombstone);
    }

    pub fn set_or_remove(&mut self, path: RepoPathBuf, value: Option<TreeValue>) {
        assert!(!path.is_root());
        if let Some(value) = value {
            self.overrides.insert(path, Override::Replace(value));
        } else {
            self.overrides.insert(path, Override::Tombstone);
        }
    }

    pub async fn write_tree(self) -> BackendResult<TreeId> {
        if self.overrides.is_empty() {
            return Ok(self.base_tree_id);
        }

        let mut trees_to_write = self.get_base_trees().await?;

        // Update entries in parent trees for file overrides
        for (path, file_override) in self.overrides {
            let (dir, basename) = path.split().unwrap();
            let tree_entries = trees_to_write.get_mut(dir).unwrap();
            match file_override {
                Override::Replace(value) => {
                    tree_entries.insert(basename.to_owned(), value);
                }
                Override::Tombstone => {
                    tree_entries.remove(basename);
                }
            }
        }

        // Write trees deepest-first, one depth level at a time. Within a level
        // every tree is mutually independent (no same-depth tree is an ancestor
        // of another) and all of its children live in deeper levels that were
        // already written, so the whole level can be written concurrently —
        // which collapses the per-directory round-trip chain on high-latency
        // backends (e.g. the Vex remote object store). Parent-entry updates are
        // applied serially after each level, preserving the child-before-parent
        // ordering the previous serial implementation relied on.
        let concurrency = self.store.concurrency().max(1);

        // Bucket directories by depth (component count); root is depth 0.
        let mut dirs_by_depth: BTreeMap<usize, Vec<RepoPathBuf>> = BTreeMap::new();
        for dir in trees_to_write.keys() {
            dirs_by_depth
                .entry(dir.components().count())
                .or_default()
                .push(dir.clone());
        }

        // Process from the deepest level down to (but excluding) the root.
        let depths: Vec<usize> = dirs_by_depth.keys().copied().filter(|d| *d > 0).rev().collect();
        for depth in depths {
            let dirs = dirs_by_depth.remove(&depth).unwrap();
            let mut pending_writes = Vec::new();
            for dir in dirs {
                let cur_entries = trees_to_write.remove(&dir).unwrap();
                if cur_entries.is_empty() {
                    // Empty subtree: drop it from its parent (unless the entry was
                    // already replaced with a file override above).
                    let (parent, basename) = dir.split().unwrap();
                    let parent_entries = trees_to_write.get_mut(parent).unwrap();
                    if let Some(TreeValue::Tree(_)) = parent_entries.get(basename) {
                        parent_entries.remove(basename);
                    }
                } else {
                    let data =
                        backend::Tree::from_sorted_entries(cur_entries.into_iter().collect());
                    pending_writes.push((dir, data));
                }
            }

            // Write every non-empty tree at this depth concurrently.
            let written: Vec<(RepoPathBuf, TreeId)> = stream::iter(pending_writes.into_iter().map(
                |(dir, data)| {
                    let store = self.store.clone();
                    async move {
                        let tree = store.write_tree(&dir, data).await?;
                        Ok::<(RepoPathBuf, TreeId), BackendError>((dir, tree.id().clone()))
                    }
                },
            ))
            .buffer_unordered(concurrency)
            .try_collect()
            .await?;

            // Now that the level is durable, link each written tree into its parent.
            for (dir, tree_id) in written {
                let (parent, basename) = dir.split().unwrap();
                let parent_entries = trees_to_write.get_mut(parent).unwrap();
                parent_entries.insert(basename.to_owned(), TreeValue::Tree(tree_id));
            }
        }

        // Finally write the root tree (depth 0), even if empty, and return its id.
        let root_entries = trees_to_write
            .remove(&RepoPathBuf::root())
            .expect("trees_to_write must contain the root tree");
        let data = backend::Tree::from_sorted_entries(root_entries.into_iter().collect());
        let written_root = self.store.write_tree(&RepoPathBuf::root(), data).await?;
        Ok(written_root.id().clone())
    }

    async fn get_base_trees(
        &self,
    ) -> BackendResult<BTreeMap<RepoPathBuf, BTreeMap<RepoPathComponentBuf, TreeValue>>> {
        // All base trees we need
        let mut needed_dirs: RepoPathTree<()> = RepoPathTree::default();
        for path in self.overrides.keys() {
            if let Some(dir) = path.parent() {
                needed_dirs.add(dir);
            }
        }

        let mut tree_reads: FuturesUnordered<BoxFuture<'_, BackendResult<(RepoPathBuf, Tree)>>> =
            FuturesUnordered::new();

        // Schedule reading the root tree
        tree_reads.push(Box::pin(async move {
            let root_dir = RepoPathBuf::root();
            let tree = self
                .store
                .get_tree(root_dir.clone(), &self.base_tree_id)
                .await?;
            Ok((root_dir, tree))
        }));

        let mut base_trees = BTreeMap::new();
        while let Some((dir, tree)) = tree_reads.try_next().await? {
            if let Some(node) = needed_dirs.get(&dir) {
                for (basename, _child) in node.children() {
                    let basename = basename.to_owned();
                    let sub_dir = dir.join(&basename);
                    let tree = tree.clone();
                    tree_reads.push(Box::pin(async move {
                        let sub_tree = tree
                            .sub_tree(&basename)
                            .await?
                            .unwrap_or_else(|| Tree::empty(self.store.clone(), sub_dir.clone()));
                        Ok((sub_dir, sub_tree))
                    }));
                }
            }
            base_trees.insert(dir, tree);
        }

        Ok(base_trees
            .into_iter()
            .map(|(dir, tree)| {
                let entries = tree
                    .data()
                    .entries()
                    .map(|entry| (entry.name().to_owned(), entry.value().clone()))
                    .collect();
                (dir, entries)
            })
            .collect())
    }
}
