use jj_lib::config::ConfigLayer;
use jj_lib::config::ConfigSource;
use jj_lib::config::StackedConfig;
use jj_lib::ref_name::WorkspaceName;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::StoreFactories;
use jj_lib::settings::UserSettings;
use jj_lib::signing::Signer;
use jj_lib::simple_backend::SimpleBackend;
use jj_lib::virtual_working_copy::VirtualWorkingCopyFactory;
use jj_lib::workspace::Workspace;
use jj_lib::workspace::default_working_copy_factories;

fn user_settings() -> UserSettings {
    let config_text = r#"
        user.name = "Test User"
        user.email = "test.user@example.com"
        operation.username = "test-username"
        operation.hostname = "host.example.com"
        debug.randomness-seed = 42
    "#;
    let mut config = StackedConfig::with_defaults();
    config.add_layer(ConfigLayer::parse(ConfigSource::User, config_text).unwrap());
    UserSettings::from_config(config).unwrap()
}

#[tokio::test]
async fn virtual_working_copy_starts_with_empty_sparse_patterns() {
    let settings = user_settings();
    let temp_dir = tempfile::tempdir().unwrap();
    let workspace_root = temp_dir.path();
    let backend_initializer = |_settings: &UserSettings, store_path: &std::path::Path| {
        Ok::<Box<dyn jj_lib::backend::Backend>, jj_lib::backend::BackendInitError>(Box::new(
            SimpleBackend::init(store_path),
        ))
    };

    let (workspace, _repo) = Workspace::init_with_factories(
        &settings,
        workspace_root,
        &backend_initializer,
        Signer::from_settings(&settings).unwrap(),
        &ReadonlyRepo::default_op_store_initializer(),
        &ReadonlyRepo::default_op_heads_store_initializer(),
        &ReadonlyRepo::default_index_store_initializer(),
        &ReadonlyRepo::default_submodule_store_initializer(),
        &VirtualWorkingCopyFactory,
        WorkspaceName::DEFAULT.to_owned(),
    )
    .await
    .unwrap();

    assert!(
        workspace
            .working_copy()
            .sparse_patterns()
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        std::fs::read_to_string(workspace_root.join(".jj/working_copy/type")).unwrap(),
        "vex-virtual"
    );

    let reopened = Workspace::load(
        &settings,
        workspace_root,
        &StoreFactories::default(),
        &default_working_copy_factories(),
    )
    .unwrap();
    assert!(
        reopened
            .working_copy()
            .sparse_patterns()
            .unwrap()
            .is_empty()
    );
}
