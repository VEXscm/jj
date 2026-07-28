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

//! Utilities to compute unified (Git-style) diffs of 2 sides
//!
//! The hunk computation and formatting moved to the shared `jj_diff` crate and
//! is re-exported below. `git_diff_part` stays here: it materializes
//! conflicts and needs `BackendError`/`ObjectId`.

use bstr::BString;
use thiserror::Error;

use super::FileContent;
use super::file_content_for_diff;
use crate::backend::BackendError;
use crate::conflicts::ConflictMaterializeOptions;
use crate::conflicts::MaterializedTreeValue;
use crate::conflicts::materialize_merge_result_to_bytes;
use crate::object_id::ObjectId as _;
use crate::repo_path::RepoPath;

pub use jj_diff::diff_presentation::unified::DiffLineType;
pub use jj_diff::diff_presentation::unified::NO_NEWLINE_MARKER;
pub use jj_diff::diff_presentation::unified::UnifiedDiffHunk;
pub use jj_diff::diff_presentation::unified::hunk_header_line_number;
pub use jj_diff::diff_presentation::unified::unified_diff_hunks;

#[derive(Clone, Debug)]
pub struct GitDiffPart {
    /// Octal mode string or `None` if the file is absent.
    pub mode: Option<&'static str>,
    pub hash: String,
    pub content: FileContent<BString>,
}

#[derive(Debug, Error)]
pub enum UnifiedDiffError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("Access denied to {path}")]
    AccessDenied {
        path: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub async fn git_diff_part(
    path: &RepoPath,
    value: MaterializedTreeValue,
    materialize_options: &ConflictMaterializeOptions,
) -> Result<GitDiffPart, UnifiedDiffError> {
    const DUMMY_HASH: &str = "0000000000";
    let mode;
    let mut hash;
    let content;
    match value {
        MaterializedTreeValue::Absent => {
            return Ok(GitDiffPart {
                mode: None,
                hash: DUMMY_HASH.to_owned(),
                content: FileContent {
                    is_binary: false,
                    contents: BString::default(),
                },
            });
        }
        MaterializedTreeValue::AccessDenied(err) => {
            return Err(UnifiedDiffError::AccessDenied {
                path: path.as_internal_file_string().to_owned(),
                source: err,
            });
        }
        MaterializedTreeValue::File(mut file) => {
            mode = if file.executable { "100755" } else { "100644" };
            hash = file.id.hex();
            content = file_content_for_diff(path, &mut file, |content| content).await?;
        }
        MaterializedTreeValue::Symlink { id, target } => {
            mode = "120000";
            hash = id.hex();
            content = FileContent {
                // Unix file paths can't contain null bytes.
                is_binary: false,
                contents: target.into(),
            };
        }
        MaterializedTreeValue::GitSubmodule(id) => {
            // TODO: What should we actually do here?
            mode = "040000";
            hash = id.hex();
            content = FileContent {
                is_binary: false,
                contents: BString::default(),
            };
        }
        MaterializedTreeValue::FileConflict(file) => {
            mode = match file.executable {
                Some(true) => "100755",
                Some(false) | None => "100644",
            };
            hash = DUMMY_HASH.to_owned();
            content = FileContent {
                is_binary: false, // TODO: are we sure this is never binary?
                contents: materialize_merge_result_to_bytes(
                    &file.contents,
                    &file.labels,
                    materialize_options,
                ),
            };
        }
        MaterializedTreeValue::OtherConflict { id, labels } => {
            mode = "100644";
            hash = DUMMY_HASH.to_owned();
            content = FileContent {
                is_binary: false,
                contents: id.describe(&labels).into(),
            };
        }
        MaterializedTreeValue::Tree(_) => {
            panic!("Unexpected tree in diff at path {path:?}");
        }
    }
    hash.truncate(10);
    Ok(GitDiffPart {
        mode: Some(mode),
        hash,
        content,
    })
}
