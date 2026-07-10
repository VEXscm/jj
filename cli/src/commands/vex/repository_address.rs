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

use serde::Deserialize;

const ORGANIZATION_HOME_REPOSITORY_SLUG: &str = "home";
const LEGACY_ORGANIZATION_HOME_REPOSITORY_SLUG: &str = "main";

#[derive(Clone, Debug, Deserialize)]
pub(super) struct RepositoryAccessCatalogEntry {
    /// Control-plane public IDs. These are intentionally not compared with the
    /// native backend IDs stored in `VexRepoConfig`; no cross-domain equality
    /// contract exists yet, so old configs must remain untouched.
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) organization_id: Option<String>,
    pub(super) organization_slug: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) repository_id: Option<String>,
    pub(super) repository_slug: String,
    #[serde(default)]
    pub(super) repository_kind: Option<String>,
    #[serde(default)]
    pub(super) is_organization_home_repository: Option<bool>,
    #[serde(default)]
    pub(super) canonical_repository_slug: Option<String>,
    #[serde(default)]
    pub(super) repository_slug_aliases: Vec<String>,
    #[serde(default)]
    pub(super) repository_scope_kind: Option<String>,
    #[serde(default)]
    pub(super) git_https_supported: Option<bool>,
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) canonical_git_https_clone_url: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) canonical_jj_repo_path: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) canonical_backing_repository_slug: Option<String>,
    #[serde(default)]
    pub(super) jj_grpc_endpoint: Option<String>,
    pub(super) jj_grpc_supported: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ResolvedRepositoryAddress<'a> {
    pub(super) entry: &'a RepositoryAccessCatalogEntry,
}

impl<'a> ResolvedRepositoryAddress<'a> {
    pub(super) fn canonical_slug(self) -> &'a str {
        nonempty(self.entry.canonical_repository_slug.as_deref())
            .unwrap_or(&self.entry.repository_slug)
    }

    pub(super) fn token_slug(self) -> &'a str {
        self.canonical_slug()
    }
}

pub(super) fn resolve_repository_address<'a>(
    catalog: &'a [RepositoryAccessCatalogEntry],
    organization_slug: &str,
    requested_repository_slug: &str,
) -> Result<Option<ResolvedRepositoryAddress<'a>>, String> {
    validate_request_component("organization", organization_slug)?;
    validate_request_component("repository", requested_repository_slug)?;

    let organization_entries = || {
        catalog
            .iter()
            .filter(|entry| entry.organization_slug == organization_slug)
    };
    for entry in organization_entries() {
        validate_catalog_entry(entry)?;
    }

    if is_home_address(requested_repository_slug) {
        return unique_match(
            organization_entries().filter(|entry| {
                is_home_repository(entry)
                    && entry_accepts_home_address(entry, requested_repository_slug)
            }),
            organization_slug,
            requested_repository_slug,
        );
    }

    unique_match(
        organization_entries().filter(|entry| {
            entry.repository_slug == requested_repository_slug
                || entry.canonical_repository_slug.as_deref() == Some(requested_repository_slug)
                || entry
                    .repository_slug_aliases
                    .iter()
                    .any(|alias| alias == requested_repository_slug)
        }),
        organization_slug,
        requested_repository_slug,
    )
}

fn entry_accepts_home_address(
    entry: &RepositoryAccessCatalogEntry,
    requested_repository_slug: &str,
) -> bool {
    if entry.canonical_repository_slug.is_none() {
        return true;
    }

    entry.repository_slug == requested_repository_slug
        || entry.canonical_repository_slug.as_deref() == Some(requested_repository_slug)
        || entry
            .repository_slug_aliases
            .iter()
            .any(|alias| alias == requested_repository_slug)
}

fn is_home_address(repository_slug: &str) -> bool {
    matches!(
        repository_slug,
        ORGANIZATION_HOME_REPOSITORY_SLUG | LEGACY_ORGANIZATION_HOME_REPOSITORY_SLUG
    )
}

fn is_physical(entry: &RepositoryAccessCatalogEntry) -> bool {
    entry.repository_scope_kind.as_deref() != Some("virtual_repository")
        && entry.repository_kind.as_deref() != Some("virtual")
}

fn is_home_repository(entry: &RepositoryAccessCatalogEntry) -> bool {
    entry.is_organization_home_repository == Some(true)
        || entry.repository_kind.as_deref() == Some(LEGACY_ORGANIZATION_HOME_REPOSITORY_SLUG)
}

fn entry_claims_home_address(entry: &RepositoryAccessCatalogEntry) -> bool {
    is_home_repository(entry)
        || is_home_address(&entry.repository_slug)
        || entry
            .canonical_repository_slug
            .as_deref()
            .is_some_and(is_home_address)
        || entry
            .repository_slug_aliases
            .iter()
            .any(|alias| is_home_address(alias))
}

fn validate_catalog_entry(entry: &RepositoryAccessCatalogEntry) -> Result<(), String> {
    validate_catalog_slug(entry, &entry.repository_slug, "repository_slug")?;
    if let Some(canonical) = entry.canonical_repository_slug.as_deref() {
        validate_catalog_slug(entry, canonical, "canonical_repository_slug")?;
    }
    for alias in &entry.repository_slug_aliases {
        validate_catalog_slug(entry, alias, "repository_slug_aliases")?;
    }

    if !entry_claims_home_address(entry) {
        return Ok(());
    }
    if !is_physical(entry) {
        return Err(malformed(
            entry,
            "a virtual repository claims the reserved `main`/`home` address",
        ));
    }
    if !is_home_repository(entry) {
        return Err(malformed(
            entry,
            "the reserved `main`/`home` address is not marked as the aggregate repository",
        ));
    }
    if entry
        .repository_kind
        .as_deref()
        .is_some_and(|kind| kind != LEGACY_ORGANIZATION_HOME_REPOSITORY_SLUG)
    {
        return Err(malformed(
            entry,
            "the aggregate repository must retain its internal `main` kind",
        ));
    }
    if !is_home_address(&entry.repository_slug) {
        return Err(malformed(
            entry,
            "the aggregate repository slug must be `main` or `home`",
        ));
    }
    if entry.is_organization_home_repository == Some(false) {
        return Err(malformed(
            entry,
            "aggregate repository metadata explicitly marks the row as non-Home",
        ));
    }
    let canonical = entry.canonical_repository_slug.as_deref();
    if let Some(canonical) = canonical {
        if !is_home_address(canonical) {
            return Err(malformed(
                entry,
                "canonical_repository_slug must be `main` or `home` for the aggregate repository",
            ));
        }
    }
    if canonical == Some(ORGANIZATION_HOME_REPOSITORY_SLUG)
        || !entry.repository_slug_aliases.is_empty()
    {
        let effective_canonical = canonical.unwrap_or(entry.repository_slug.as_str());
        let required_alias = if effective_canonical == ORGANIZATION_HOME_REPOSITORY_SLUG {
            LEGACY_ORGANIZATION_HOME_REPOSITORY_SLUG
        } else {
            ORGANIZATION_HOME_REPOSITORY_SLUG
        };
        if entry.repository_slug_aliases.as_slice() != [required_alias] {
            return Err(malformed(
                entry,
                format!(
                    "canonical `{effective_canonical}` requires exactly the fixed `{required_alias}` alias"
                ),
            ));
        }
    }
    if entry
        .repository_slug_aliases
        .iter()
        .any(|alias| !is_home_address(alias))
    {
        return Err(malformed(
            entry,
            "aggregate repository aliases may contain only `main` and `home`",
        ));
    }
    Ok(())
}

fn validate_request_component(component: &str, value: &str) -> Result<(), String> {
    if is_valid_slug(value) {
        Ok(())
    } else {
        Err(format!(
            "invalid {component} slug `{value}` in repository address"
        ))
    }
}

fn validate_catalog_slug(
    entry: &RepositoryAccessCatalogEntry,
    value: &str,
    field: &str,
) -> Result<(), String> {
    if is_valid_slug(value) {
        Ok(())
    } else {
        Err(malformed(
            entry,
            format!("{field} must be a non-empty single slug component"),
        ))
    }
}

fn is_valid_slug(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.contains('/')
}

fn malformed(entry: &RepositoryAccessCatalogEntry, reason: impl Into<String>) -> String {
    format!(
        "repository catalog entry `{}/{}` is malformed: {}",
        entry.organization_slug,
        entry.repository_slug,
        reason.into()
    )
}

fn unique_match<'a>(
    mut matches: impl Iterator<Item = &'a RepositoryAccessCatalogEntry>,
    organization_slug: &str,
    requested_repository_slug: &str,
) -> Result<Option<ResolvedRepositoryAddress<'a>>, String> {
    let Some(entry) = matches.next() else {
        return Ok(None);
    };
    let match_count = 1 + matches.count();
    if match_count > 1 {
        return Err(format!(
            "repository address `{organization_slug}/{requested_repository_slug}` is ambiguous across {match_count} catalog entries"
        ));
    }
    Ok(Some(ResolvedRepositoryAddress { entry }))
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use serde_json::json;

    use super::*;

    fn entry(value: Value) -> RepositoryAccessCatalogEntry {
        serde_json::from_value(value).expect("valid repository catalog entry")
    }

    fn base_entry(repository_slug: &str, repository_kind: &str) -> Value {
        json!({
            "organization_id": "org_control_plane_1",
            "organization_slug": "acme",
            "repository_id": "repo_control_plane_1",
            "repository_slug": repository_slug,
            "repository_kind": repository_kind,
            "repository_scope_kind": "repository",
            "jj_grpc_endpoint": "https://legacy-jj.example.test",
            "jj_grpc_supported": true
        })
    }

    #[test]
    fn legacy_catalog_home_request_mints_against_main_projection() {
        let catalog = [entry(base_entry("main", "main"))];
        let resolved = resolve_repository_address(&catalog, "acme", "home")
            .unwrap()
            .unwrap();

        assert_eq!(resolved.canonical_slug(), "main");
        assert_eq!(resolved.token_slug(), "main");
        assert_eq!(
            resolved.entry.repository_id.as_deref(),
            Some("repo_control_plane_1")
        );
    }

    #[test]
    fn home_and_main_select_one_canonical_catalog_identity() {
        let mut value = base_entry("main", "main");
        value["is_organization_home_repository"] = json!(true);
        value["canonical_repository_slug"] = json!("home");
        value["repository_slug_aliases"] = json!(["main"]);
        value["canonical_git_https_clone_url"] = json!("https://vex.example.test/git/acme/home");
        value["canonical_jj_repo_path"] = json!("acme/home");
        value["canonical_backing_repository_slug"] = json!("home");
        let catalog = [entry(value)];

        for requested_slug in ["home", "main"] {
            let resolved = resolve_repository_address(&catalog, "acme", requested_slug)
                .unwrap()
                .unwrap();
            assert_eq!(resolved.canonical_slug(), "home");
            assert_eq!(resolved.token_slug(), "home");
            assert_eq!(
                resolved.entry.organization_id.as_deref(),
                Some("org_control_plane_1")
            );
            assert_eq!(
                resolved.entry.repository_id.as_deref(),
                Some("repo_control_plane_1")
            );
            assert_eq!(
                resolved.entry.canonical_git_https_clone_url.as_deref(),
                Some("https://vex.example.test/git/acme/home")
            );
            assert_eq!(
                resolved.entry.canonical_jj_repo_path.as_deref(),
                Some("acme/home")
            );
            assert_eq!(
                resolved.entry.canonical_backing_repository_slug.as_deref(),
                Some("home")
            );
        }
    }

    #[test]
    fn expand_metadata_selects_canonical_main_with_home_alias() {
        let mut value = base_entry("main", "main");
        value["is_organization_home_repository"] = json!(true);
        value["canonical_repository_slug"] = json!("main");
        value["repository_slug_aliases"] = json!(["home"]);
        value["canonical_git_https_clone_url"] = json!("https://vex.example.test/git/acme/main");
        value["canonical_jj_repo_path"] = json!("acme/main");
        value["canonical_backing_repository_slug"] = json!("main");
        let catalog = [entry(value)];

        for requested_slug in ["home", "main"] {
            let resolved = resolve_repository_address(&catalog, "acme", requested_slug)
                .unwrap()
                .unwrap();
            assert_eq!(resolved.canonical_slug(), "main");
            assert_eq!(resolved.token_slug(), "main");
        }
    }

    #[test]
    fn preseed_expand_metadata_resolves_only_main() {
        let mut value = base_entry("main", "main");
        value["is_organization_home_repository"] = json!(true);
        value["canonical_repository_slug"] = json!("main");
        value["repository_slug_aliases"] = json!([]);
        let catalog = [entry(value)];

        assert!(
            resolve_repository_address(&catalog, "acme", "main")
                .unwrap()
                .is_some()
        );
        assert!(
            resolve_repository_address(&catalog, "acme", "home")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn alias_metadata_requires_exact_fixed_counterpart() {
        let mut missing = base_entry("main", "main");
        missing["is_organization_home_repository"] = json!(true);
        missing["canonical_repository_slug"] = json!("home");
        missing["repository_slug_aliases"] = json!([]);
        assert!(resolve_repository_address(&[entry(missing)], "acme", "home").is_err());

        let mut extra = base_entry("main", "main");
        extra["is_organization_home_repository"] = json!(true);
        extra["canonical_repository_slug"] = json!("main");
        extra["repository_slug_aliases"] = json!(["home", "main"]);
        assert!(resolve_repository_address(&[entry(extra)], "acme", "main").is_err());
    }

    #[test]
    fn virtual_home_collision_fails_closed() {
        let mut virtual_home = base_entry("home", "virtual");
        virtual_home["repository_scope_kind"] = json!("virtual_repository");
        let catalog = [entry(virtual_home), entry(base_entry("main", "main"))];

        assert!(resolve_repository_address(&catalog, "acme", "home").is_err());
    }

    #[test]
    fn duplicate_home_rows_fail_closed() {
        let catalog = [
            entry(base_entry("main", "main")),
            entry(base_entry("home", "main")),
        ];

        assert!(resolve_repository_address(&catalog, "acme", "home").is_err());
    }

    #[test]
    fn component_and_virtual_rows_remain_valid() {
        let mut virtual_entry = base_entry("docs", "virtual");
        virtual_entry["repository_scope_kind"] = json!("virtual_repository");
        let catalog = [
            entry(base_entry("widgets", "component")),
            entry(virtual_entry),
        ];

        let component = resolve_repository_address(&catalog, "acme", "widgets")
            .unwrap()
            .unwrap();
        let virtual_repo = resolve_repository_address(&catalog, "acme", "docs")
            .unwrap()
            .unwrap();

        assert_eq!(component.canonical_slug(), "widgets");
        assert_eq!(virtual_repo.canonical_slug(), "docs");
    }
}
