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

use std::env;
use std::fs;
use std::path::PathBuf;

use etcetera::BaseStrategy;
use reqwest::Method;
use reqwest::StatusCode;
use reqwest::header::AUTHORIZATION;
use serde::Deserialize;
use serde_json::Value;

use crate::command_error::CommandError;
use crate::command_error::user_error;

const DEFAULT_API_BASE_URL: &str = "https://vex.sc";

#[derive(clap::Args, Clone, Debug, Default)]
pub(crate) struct RepoAuthArgs {
    /// gRPC endpoint for the Vex backend.
    #[arg(long, value_hint = clap::ValueHint::Url)]
    pub endpoint: Option<String>,

    /// Repository access token issued by the Vex control plane.
    #[arg(long)]
    pub token: Option<String>,

    /// Vex control-plane API base URL used for auto-authenticated clone/init.
    #[arg(long)]
    pub api_base_url: Option<String>,

    /// Vex control-plane API token used for auto-authenticated clone/init.
    #[arg(long)]
    pub api_token: Option<String>,
}

pub(crate) struct ResolvedRepoAuth {
    pub endpoint: String,
    pub access_token: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FileConfig {
    api: Option<FileApiConfig>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FileApiConfig {
    base_url: Option<String>,
    token: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct RepositoryAccessCatalogResponse {
    repositories: Vec<RepositoryAccessCatalogEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct RepositoryAccessCatalogEntry {
    organization_slug: String,
    repository_slug: String,
    #[serde(default)]
    repository_scope_kind: Option<String>,
    #[serde(default)]
    git_https_supported: Option<bool>,
    jj_grpc_endpoint: Option<String>,
    jj_grpc_supported: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct RepositoryAccessTokenCreateResponse {
    plain_text_token: String,
}

pub(crate) async fn resolve_repo_auth(
    args: &RepoAuthArgs,
    tenant_slug: &str,
    repo_slug: &str,
    token_name: &str,
) -> Result<ResolvedRepoAuth, CommandError> {
    let explicit_access_token = args
        .token
        .clone()
        .or_else(|| env::var("VEX_ACCESS_TOKEN").ok());
    if let Some(access_token) = explicit_access_token {
        return Ok(ResolvedRepoAuth {
            endpoint: args
                .endpoint
                .clone()
                .unwrap_or_else(|| jj_lib::vex::DEFAULT_ENDPOINT.to_string()),
            access_token: Some(access_token),
        });
    }

    let (base_url, api_token) = resolve_api_auth(args)?;
    let catalog = fetch_repository_access_catalog(&base_url, &api_token).await?;
    let entry = catalog
        .into_iter()
        .find(|entry| entry.organization_slug == tenant_slug && entry.repository_slug == repo_slug)
        .ok_or_else(|| {
            user_error(format!(
                "repository `{tenant_slug}/{repo_slug}` was not found in your accessible Vex catalog"
            ))
        })?;

    if entry.repository_scope_kind.as_deref() == Some("virtual_repository") {
        return Err(user_error(
            "virtual repositories are VEX path-scoped views today; browse them through VEX APIs or GitHub sync while native clone transport is wired up",
        ));
    }

    if !entry.jj_grpc_supported && args.endpoint.is_none() {
        let message = if entry.git_https_supported.unwrap_or(true) {
            "this environment does not advertise JJ-native clone access; run `vex login` for the same Vex base URL or pass --endpoint with a repo token"
        } else {
            "this repository is not available over JJ or Git transport in this environment yet"
        };
        return Err(user_error(message));
    }

    let endpoint = args
        .endpoint
        .clone()
        .or(entry.jj_grpc_endpoint)
        .unwrap_or_else(|| jj_lib::vex::DEFAULT_ENDPOINT.to_string());
    let access_token =
        create_repository_access_token(&base_url, &api_token, tenant_slug, repo_slug, token_name)
            .await?;

    Ok(ResolvedRepoAuth {
        endpoint,
        access_token: Some(access_token),
    })
}

fn resolve_api_auth(args: &RepoAuthArgs) -> Result<(String, String), CommandError> {
    let file = load_file_config().map_err(|err| user_error(err.to_string()))?;
    let base_url = args
        .api_base_url
        .clone()
        .or_else(|| env::var("VEX_API_BASE_URL").ok())
        .or_else(|| file.api.as_ref().and_then(|api| api.base_url.clone()))
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_API_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string();
    let api_token = args
        .api_token
        .clone()
        .or_else(|| env::var("VEX_API_TOKEN").ok())
        .or_else(|| file.api.as_ref().and_then(|api| api.token.clone()))
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            user_error(
                "missing repository auth; run `vex login` or pass --token / VEX_ACCESS_TOKEN",
            )
        })?;

    Ok((base_url, api_token))
}

fn load_file_config() -> Result<FileConfig, std::io::Error> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(FileConfig::default());
    }

    let raw = fs::read_to_string(path)?;
    Ok(toml::from_str(&raw).unwrap_or_default())
}

fn config_path() -> Result<PathBuf, std::io::Error> {
    let strategy = etcetera::choose_base_strategy()
        .map_err(|_| std::io::Error::other("failed to determine vex config directory"))?;
    Ok(strategy.config_dir().join("vex").join("config.toml"))
}

async fn fetch_repository_access_catalog(
    base_url: &str,
    api_token: &str,
) -> Result<Vec<RepositoryAccessCatalogEntry>, CommandError> {
    let response = authenticated_request(
        base_url,
        api_token,
        Method::GET,
        "/api/v1/repository_access/catalog",
        None,
    )
    .await?;
    let parsed: RepositoryAccessCatalogResponse = parse_json_response(response).await?;
    Ok(parsed.repositories)
}

async fn create_repository_access_token(
    base_url: &str,
    api_token: &str,
    org_slug: &str,
    repo_slug: &str,
    name: &str,
) -> Result<String, CommandError> {
    let response = authenticated_request(
        base_url,
        api_token,
        Method::POST,
        &format!("/api/v1/organizations/{org_slug}/repos/{repo_slug}/access_tokens"),
        Some(serde_json::json!({
            "name": name,
            "permission_level": "read_write",
            "expires_in_days": 30,
        })),
    )
    .await?;
    let parsed: RepositoryAccessTokenCreateResponse = parse_json_response(response).await?;
    Ok(parsed.plain_text_token)
}

async fn authenticated_request(
    base_url: &str,
    api_token: &str,
    method: Method,
    path: &str,
    body: Option<Value>,
) -> Result<reqwest::Response, CommandError> {
    let client = reqwest::Client::new();
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let mut request = client
        .request(method, url)
        .header(AUTHORIZATION, format!("Bearer {api_token}"));
    if let Some(body) = body {
        request = request
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string());
    }
    request
        .send()
        .await
        .map_err(|err| user_error(err.to_string()))
}

async fn parse_json_response<T: for<'de> Deserialize<'de>>(
    response: reqwest::Response,
) -> Result<T, CommandError> {
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|err| user_error(err.to_string()))?;
    let body = if text.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&text).unwrap_or(Value::String(text.clone()))
    };

    if !status.is_success() {
        let message = body
            .get("error")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                if status == StatusCode::UNAUTHORIZED {
                    "Unauthorized".to_string()
                } else {
                    text.clone()
                }
            });
        return Err(user_error(message));
    }

    serde_json::from_value(body).map_err(|err| user_error(err.to_string()))
}
