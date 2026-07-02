use std::ffi::OsString;
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use jj_backend_test_support::LiveBackendHarness;
use serde_json::Value;

use super::TestEnvironment;
use super::TestWorkDir;

const TENANT: &str = "acme";
static LIVE_VEX_TEST_GUARD: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[derive(Clone, Debug)]
pub struct VexRepoSpec {
    repo_slug: String,
}

impl VexRepoSpec {
    pub fn full_name(&self) -> String {
        format!("{TENANT}/{}", self.repo_slug)
    }

    pub fn repo_slug(&self) -> &str {
        &self.repo_slug
    }
}

pub struct VexLiveTestHarness {
    _guard: MutexGuard<'static, ()>,
    backend: LiveBackendHarness,
    test_env: TestEnvironment,
}

impl VexLiveTestHarness {
    pub fn start() -> Self {
        // The live harness backend runs with no repo-access authorizer
        // (allow-all), but the CLI's repo-auth resolution requires a token
        // (`resolve_repo_auth` falls back to the Vex API catalog without
        // one), and `TestEnvironment` spawns `jj` with a cleared env. Inject
        // a dummy token so tests stay hermetic.
        let mut test_env = TestEnvironment::default();
        test_env.add_env_var("VEX_ACCESS_TOKEN", "vex-live-test-token");
        Self {
            _guard: LIVE_VEX_TEST_GUARD
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            backend: LiveBackendHarness::start().expect("live backend harness"),
            test_env,
        }
    }

    pub fn repo(&self, prefix: &str) -> VexRepoSpec {
        VexRepoSpec {
            repo_slug: unique_repo_slug(prefix),
        }
    }

    pub fn endpoint(&self) -> String {
        self.backend.grpc_endpoint()
    }

    pub fn init_repo<'a>(&'a self, repo: &VexRepoSpec, destination: &str) -> TestWorkDir<'a> {
        let endpoint = self.endpoint();
        let repo_spec = repo.full_name();
        let output = self
            .test_env
            .run_jj_in(
                ".",
                [
                    "vex",
                    "init",
                    repo_spec.as_str(),
                    destination,
                    "--endpoint",
                    endpoint.as_str(),
                ],
            )
            .success();
        assert_contains(output.stderr.raw(), "Initialized Vex-backed repo");
        self.work_dir(destination)
    }

    pub fn clone_repo<'a>(&'a self, repo: &VexRepoSpec, destination: &str) -> TestWorkDir<'a> {
        self.clone_repo_with_args(repo, destination, std::iter::empty::<&str>())
    }

    pub fn clone_repo_with_args<'a, I, S>(
        &'a self,
        repo: &VexRepoSpec,
        destination: &str,
        extra_args: I,
    ) -> TestWorkDir<'a>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let endpoint = self.endpoint();
        let repo_spec = repo.full_name();
        let mut args = vec![
            "vex".to_string(),
            "clone".to_string(),
            repo_spec,
            destination.to_string(),
            "--endpoint".to_string(),
            endpoint,
        ];
        args.extend(extra_args.into_iter().map(|arg| arg.as_ref().to_string()));
        let output = self.test_env.run_jj_in(".", args).success();
        assert_contains(output.stderr.raw(), "Cloned Vex-backed repo");
        self.work_dir(destination)
    }

    pub fn work_dir(&self, destination: &str) -> TestWorkDir<'_> {
        self.test_env.work_dir(destination)
    }

    pub fn add_env_var(&mut self, key: impl Into<OsString>, value: impl Into<OsString>) {
        self.test_env.add_env_var(key, value);
    }

    pub fn restart_backend(&mut self) {
        self.backend.restart_backend().expect("restart backend");
    }

    pub fn assert_clean_status(&self, work_dir: &TestWorkDir<'_>) {
        let status = work_dir.run_jj(["status"]).success();
        assert_contains(status.stdout.raw(), "The working copy has no changes.");
    }

    pub fn assert_log_description(&self, work_dir: &TestWorkDir<'_>, revset: &str, expected: &str) {
        let output = work_dir
            .run_jj(["log", "-r", revset, "-T", "description", "--no-graph"])
            .success();
        assert_eq!(output.stdout.raw().trim(), expected, "{output}");
    }

    pub fn assert_vex_metadata(&self, work_dir: &TestWorkDir<'_>, repo: &VexRepoSpec) {
        let metadata_path = work_dir.root().join(".jj/repo/vex.json");
        let metadata = std::fs::read_to_string(&metadata_path).expect("read vex metadata");
        let metadata: Value = serde_json::from_str(&metadata).expect("parse vex metadata");
        let endpoint = self.endpoint();
        assert_eq!(metadata["endpoint"].as_str(), Some(endpoint.as_str()));
        assert_eq!(metadata["tenant_slug"].as_str(), Some(TENANT));
        assert_eq!(metadata["repo_slug"].as_str(), Some(repo.repo_slug()));
    }

    pub fn assert_vex_cache_populated(&self, work_dir: &TestWorkDir<'_>) {
        let cache_root = work_dir.root().join(".jj/repo/vex-cache");
        let entries = std::fs::read_dir(&cache_root).expect("read vex cache root");
        let file_count = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .flat_map(|dir| std::fs::read_dir(dir).into_iter().flatten())
            .filter_map(Result::ok)
            .count();
        assert!(
            file_count > 0,
            "expected populated vex cache at {}",
            cache_root.display()
        );
    }

    pub fn assert_working_copy_type(&self, work_dir: &TestWorkDir<'_>, expected: &str) {
        let working_copy_type =
            std::fs::read_to_string(work_dir.root().join(".jj/working_copy/type"))
                .expect("read working copy type");
        assert_eq!(working_copy_type, expected);
    }

    pub fn assert_path_missing(&self, work_dir: &TestWorkDir<'_>, path: &str) {
        assert!(
            !work_dir.root().join(path).exists(),
            "expected {} to be absent",
            work_dir.root().join(path).display()
        );
    }

    pub fn assert_no_cached_kind_entries(&self, work_dir: &TestWorkDir<'_>, kind: &str) {
        let kind_dir = work_dir.root().join(".jj/repo/vex-cache").join(kind);
        if !kind_dir.exists() {
            return;
        }
        let count = std::fs::read_dir(&kind_dir)
            .expect("read cache kind dir")
            .filter_map(Result::ok)
            .count();
        assert_eq!(count, 0, "expected no cached {} entries", kind);
    }
}

fn assert_contains(haystack: &str, needle: &str) {
    assert!(
        haystack.contains(needle),
        "expected `{needle}` in output:\n{haystack}"
    );
}

fn unique_repo_slug(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before unix epoch")
        .as_nanos();
    format!("{prefix}-{nanos}")
}
