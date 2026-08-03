use std::sync::{LazyLock, Mutex};

use jj_lib::vex::VexClient;
use jj_lib::vex::VexConfigError;
use jj_lib::vex::VexObjectReadMode;
use jj_lib::vex::VexRepoConfig;

static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn sample_config() -> VexRepoConfig {
    VexRepoConfig {
        endpoint: "http://127.0.0.1:50051".to_string(),
        tenant_id: "tenant-id".to_string(),
        tenant_slug: "acme".to_string(),
        repo_id: "repo-id".to_string(),
        repo_slug: "widget".to_string(),
        repository_scope_kind: Some("repository".to_string()),
        virtual_repository_id: None,
        backing_repo_slug: None,
        virtual_root_path: None,
        virtual_mounts: Vec::new(),
        access_token: None,
        local_writes: false,
        object_read_mode: VexObjectReadMode::NativeOnly,
    }
}

#[test]
fn test_vex_repo_config_round_trips_from_repo_and_store_paths() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    let store_dir = repo_dir.join("store");
    std::fs::create_dir_all(&store_dir).unwrap();

    let config = sample_config();
    config.write_to_repo_path(&repo_dir).unwrap();

    assert_eq!(
        VexRepoConfig::load_from_repo_path(&repo_dir).unwrap(),
        config
    );
    assert_eq!(
        VexRepoConfig::load_from_store_path(&store_dir).unwrap(),
        config
    );
    assert_eq!(
        VexClient::from_store_path(&store_dir).unwrap().config(),
        &config
    );
}

#[test]
fn test_legacy_main_vex_repo_config_loads_without_rewrite() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    let metadata_path = repo_dir.join("vex.json");
    let legacy_json = r#"{
  "endpoint": "http://127.0.0.1:50051",
  "tenant_id": "backend-tenant-id",
  "tenant_slug": "acme",
  "repo_id": "backend-repo-id",
  "repo_slug": "main"
}"#;
    std::fs::write(&metadata_path, legacy_json).unwrap();

    let config = VexRepoConfig::load_from_repo_path(&repo_dir).unwrap();

    assert_eq!(config.repo_slug, "main");
    assert_eq!(config.tenant_id, "backend-tenant-id");
    assert_eq!(config.repo_id, "backend-repo-id");
    // Legacy files carry no object-read-mode field; they must load as
    // native-only (roadmap/066).
    assert_eq!(config.object_read_mode, VexObjectReadMode::NativeOnly);
    assert_eq!(std::fs::read_to_string(metadata_path).unwrap(), legacy_json);
}

/// A `vex.json` written before the object-read-mode field existed (or by a
/// current clone, which never writes the field) deserializes to
/// `NativeOnly` — never to Git compatibility.
#[test]
fn test_vex_repo_config_missing_object_read_mode_defaults_to_native_only() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    let json_without_mode = r#"{
  "endpoint": "http://127.0.0.1:50051",
  "tenant_id": "tenant-id",
  "tenant_slug": "acme",
  "repo_id": "repo-id",
  "repo_slug": "widget",
  "repository_scope_kind": "repository",
  "local_writes": false
}"#;
    std::fs::write(repo_dir.join("vex.json"), json_without_mode).unwrap();

    let config = VexRepoConfig::load_from_repo_path(&repo_dir).unwrap();

    assert_eq!(config.object_read_mode, VexObjectReadMode::NativeOnly);
    assert!(!config.object_read_mode.allows_git_compatibility());
}

/// Compatibility mode is an in-memory, explicit opt-in for conversion/Git
/// bridge callers: persisting a config never writes the mode field, so a
/// round trip through `vex.json` always loads back as `NativeOnly`.
#[test]
fn test_vex_repo_config_never_persists_git_compatibility_mode() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();

    let mut config = sample_config();
    config.object_read_mode = VexObjectReadMode::GitCompatibility;
    config.write_to_repo_path(&repo_dir).unwrap();

    let written = std::fs::read_to_string(repo_dir.join("vex.json")).unwrap();
    assert!(
        !written.contains("object_read_mode"),
        "mode must never be persisted: {written}"
    );

    let reloaded = VexRepoConfig::load_from_repo_path(&repo_dir).unwrap();
    assert_eq!(reloaded.object_read_mode, VexObjectReadMode::NativeOnly);
}

/// An explicit mode field (hand-written test fixtures; nothing in the product
/// writes one) still deserializes, and unknown values are rejected instead of
/// being coerced.
#[test]
fn test_vex_repo_config_explicit_object_read_mode_field_deserializes() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();
    let json_with_mode = r#"{
  "endpoint": "http://127.0.0.1:50051",
  "tenant_id": "tenant-id",
  "tenant_slug": "acme",
  "repo_id": "repo-id",
  "repo_slug": "widget",
  "object_read_mode": "git_compatibility"
}"#;
    std::fs::write(repo_dir.join("vex.json"), json_with_mode).unwrap();

    let config = VexRepoConfig::load_from_repo_path(&repo_dir).unwrap();
    assert_eq!(config.object_read_mode, VexObjectReadMode::GitCompatibility);

    std::fs::write(
        repo_dir.join("vex.json"),
        json_with_mode.replace("git_compatibility", "mystery_mode"),
    )
    .unwrap();
    assert!(VexRepoConfig::load_from_repo_path(&repo_dir).is_err());
}

/// The `durability` key is **retired but tolerated**.
///
/// Roadmap/088 Stage 9 (C9) removed the `VexDurability` lattice: since Stage 7
/// the operation log is local, so there is no queue for a deferred mode to
/// defer into and the field controlled nothing. But `vex.json` files in the
/// wild still carry `"durability": "sync" | "flush-on-exit" | "local-first"`,
/// and `local-first` was the *serialized* default, so many files omit the key
/// entirely. All four shapes must keep deserializing.
///
/// This works only because serde ignores unknown fields by default. Do **not**
/// add `#[serde(deny_unknown_fields)]` to `VexRepoConfig`: it would make every
/// pre-Stage-9 clone fail to load, and the failure surfaces to users as
/// "repository config is corrupt", not as an obvious regression. This test is
/// the pin that catches that.
#[test]
fn test_vex_repo_config_tolerates_retired_durability_key() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    std::fs::create_dir_all(&repo_dir).unwrap();

    let payload = |durability: Option<&str>| {
        let line = durability
            .map(|value| format!("\n  \"durability\": \"{value}\","))
            .unwrap_or_default();
        format!(
            r#"{{
  "endpoint": "http://127.0.0.1:50051",
  "tenant_id": "tenant-id",
  "tenant_slug": "acme",
  "repo_id": "repo-id",{line}
  "repo_slug": "widget"
}}"#
        )
    };

    for durability in [
        Some("sync"),
        Some("flush-on-exit"),
        Some("local-first"),
        None,
    ] {
        let json = payload(durability);
        std::fs::write(repo_dir.join("vex.json"), &json).unwrap();
        let config = VexRepoConfig::load_from_repo_path(&repo_dir).unwrap_or_else(|err| {
            panic!("durability={durability:?} must still deserialize, got {err}: {json}")
        });
        assert_eq!(config.repo_slug, "widget");
        assert_eq!(config.tenant_id, "tenant-id");
    }

    // Rewriting a config loaded from a legacy file is safe: the file is
    // replaced wholesale with the current shape, which simply no longer
    // carries the retired key. Nothing is left half-written or ambiguous.
    std::fs::write(repo_dir.join("vex.json"), payload(Some("local-first"))).unwrap();
    let config = VexRepoConfig::load_from_repo_path(&repo_dir).unwrap();
    config.write_to_repo_path(&repo_dir).unwrap();
    let rewritten = std::fs::read_to_string(repo_dir.join("vex.json")).unwrap();
    assert!(
        !rewritten.contains("durability"),
        "the retired key must not be written back: {rewritten}"
    );
    assert_eq!(
        VexRepoConfig::load_from_repo_path(&repo_dir).unwrap(),
        config
    );
}

#[test]
fn test_vex_repo_config_missing_store_metadata_reports_repo_metadata_path() {
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    let store_dir = repo_dir.join("store");
    std::fs::create_dir_all(&store_dir).unwrap();

    let err = VexRepoConfig::load_from_store_path(&store_dir).unwrap_err();
    let missing_path = repo_dir.join("vex.json");
    assert!(matches!(
        err,
        VexConfigError::MissingMetadata(ref path) if path == &missing_path
    ));
}

#[test]
fn test_vex_client_uses_shared_cache_root_when_configured() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp_dir = tempfile::tempdir().unwrap();
    let repo_dir = temp_dir.path().join("repo");
    let store_dir = repo_dir.join("store");
    let shared_cache_dir = temp_dir.path().join("shared-cache");
    std::fs::create_dir_all(&store_dir).unwrap();

    let config = sample_config();
    config.write_to_repo_path(&repo_dir).unwrap();

    unsafe {
        std::env::set_var("JJ_VEX_SHARED_CACHE_DIR", &shared_cache_dir);
        std::env::remove_var("JJ_VEX_CACHE_MAX_BYTES");
    }
    let client = VexClient::from_store_path(&store_dir).unwrap();
    let expected_root = shared_cache_dir
        .join(&config.tenant_id)
        .join(&config.repo_id);
    assert!(
        expected_root.is_dir(),
        "expected {}",
        expected_root.display()
    );
    assert_eq!(client.config(), &config);
    unsafe {
        std::env::remove_var("JJ_VEX_SHARED_CACHE_DIR");
    }
}
