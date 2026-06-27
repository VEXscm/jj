use std::sync::{LazyLock, Mutex};

use jj_lib::vex::VexClient;
use jj_lib::vex::VexConfigError;
use jj_lib::vex::VexRepoConfig;

static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

fn sample_config() -> VexRepoConfig {
    VexRepoConfig {
        endpoint: "http://127.0.0.1:50051".to_string(),
        tenant_id: "tenant-id".to_string(),
        tenant_slug: "acme".to_string(),
        repo_id: "repo-id".to_string(),
        repo_slug: "widget".to_string(),
        access_token: None,
        local_writes: false,
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
