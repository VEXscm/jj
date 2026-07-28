// Copyright 2025 The Jujutsu Authors
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

//! Utilities to present file diffs to the user
//!
//! The content-only half lives in the shared `jj_diff` crate so `jj-backend`
//! can format the same hunks; it is re-exported below under its original
//! names. What stays here is what needs the repository: reading a
//! materialized file and materializing a conflict.

#![expect(missing_docs)]

use bstr::BString;

use crate::backend::BackendResult;
use crate::conflicts::MaterializedFileValue;
use crate::repo_path::RepoPath;

pub use jj_diff::diff_presentation::DiffTokenType;
pub use jj_diff::diff_presentation::DiffTokenVec;
pub use jj_diff::diff_presentation::LineCompareMode;
pub use jj_diff::diff_presentation::diff_by_line;
pub use jj_diff::diff_presentation::unzip_diff_hunks_to_lines;

pub mod unified;
// TODO: colored_diffs utils should also be moved from `jj_cli::diff_utils` to
// here.

#[derive(Clone, Debug)]
pub struct FileContent<T> {
    /// false if this file is likely text; true if it is likely binary.
    pub is_binary: bool,
    pub contents: T,
}

pub async fn file_content_for_diff<T>(
    path: &RepoPath,
    file: &mut MaterializedFileValue,
    map_resolved: impl FnOnce(BString) -> T,
) -> BackendResult<FileContent<T>> {
    // If this is a binary file, don't show the full contents.
    // Determine whether it's binary by whether the first 8k bytes contain a null
    // character; this is the same heuristic used by git as of writing: https://github.com/git/git/blob/eea0e59ffbed6e33d171ace5be13cde9faa41639/xdiff-interface.c#L192-L198
    const PEEK_SIZE: usize = 8000;
    // TODO: currently we look at the whole file, even though for binary files we
    // only need to know the file size. To change that we'd have to extend all
    // the data backends to support getting the length.
    let contents = BString::new(file.read_all(path).await?);
    let start = &contents[..PEEK_SIZE.min(contents.len())];
    Ok(FileContent {
        is_binary: start.contains(&b'\0'),
        contents: map_resolved(contents),
    })
}
