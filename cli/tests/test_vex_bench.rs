use std::time::{Duration, Instant};

use crate::common::TestWorkDir;
use crate::common::VexLiveTestHarness;

const SMALL_FILE_COUNT: usize = 256;
const SMALL_FILE_BYTES: usize = 4096;
const LARGE_FILE_BYTES: usize = 512 * 1024;

#[test]
#[ignore = "requires Docker and a live jj-backend harness; prints benchmark timings"]
fn test_vex_bench_clone_profiles_and_file_show() {
    let mut harness = VexLiveTestHarness::start();
    let repo = harness.repo("vex-bench");
    let source = harness.init_repo(&repo, "source");
    seed_benchmark_repo(&source);
    source.run_jj(["commit", "-m", "seed"]).success();
    source
        .run_jj(["bookmark", "create", "-r", "@-", "main"])
        .success();

    let shared_cache_root = harness.work_dir("shared-cache");
    std::fs::create_dir_all(shared_cache_root.root()).unwrap();
    let shared_cache_root_path = shared_cache_root.root().to_owned();
    harness.add_env_var("JJ_VEX_SHARED_CACHE_DIR", shared_cache_root_path);

    let eager_samples = (0..3)
        .map(|index| {
            let destination = format!("eager-clone-{index}");
            measure(|| {
                let clone = harness.clone_repo(&repo, &destination);
                harness.assert_working_copy_type(&clone, "local");
            })
        })
        .collect::<Vec<_>>();

    let cold_virtual_clone_ms = measure(|| {
        let clone = harness.clone_repo_with_args(&repo, "virtual-cold", ["--fs=virtual"]);
        harness.assert_working_copy_type(&clone, "vex-virtual");
        harness.assert_no_cached_kind_entries(&clone, "blob");
    });

    let warm_virtual_clone_samples = (0..2)
        .map(|index| {
            let destination = format!("virtual-warm-{index}");
            measure(|| {
                let clone = harness.clone_repo_with_args(&repo, &destination, ["--fs=virtual"]);
                harness.assert_working_copy_type(&clone, "vex-virtual");
            })
        })
        .collect::<Vec<_>>();

    let virtual_cold = harness.work_dir("virtual-cold");
    let cold_file_show_ms = measure(|| {
        virtual_cold
            .run_jj([
                "--ignore-working-copy",
                "file",
                "show",
                "-r",
                "main",
                "large.txt",
            ])
            .success();
    });
    let warm_file_show_same_clone_ms = measure(|| {
        virtual_cold
            .run_jj([
                "--ignore-working-copy",
                "file",
                "show",
                "-r",
                "main",
                "large.txt",
            ])
            .success();
    });

    let virtual_warm = harness.work_dir("virtual-warm-0");
    let warm_file_show_cross_clone_ms = measure(|| {
        virtual_warm
            .run_jj([
                "--ignore-working-copy",
                "file",
                "show",
                "-r",
                "main",
                "large.txt",
            ])
            .success();
    });

    println!("Vex benchmark workload:");
    println!(
        "  small files: {SMALL_FILE_COUNT} x {SMALL_FILE_BYTES} bytes = {:.2} MiB",
        (SMALL_FILE_COUNT * SMALL_FILE_BYTES) as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  large file:  1 x {LARGE_FILE_BYTES} bytes = {:.2} MiB",
        LARGE_FILE_BYTES as f64 / (1024.0 * 1024.0)
    );
    println!();
    print_stat("eager clone (local working copy)", &eager_samples);
    print_stat(
        "virtual fs clone cold (--fs=virtual)",
        &[cold_virtual_clone_ms],
    );
    print_stat(
        "virtual fs clone warm shared-cache (--fs=virtual)",
        &warm_virtual_clone_samples,
    );
    print_stat("virtual fs file show cold", &[cold_file_show_ms]);
    print_stat(
        "virtual fs file show warm same clone",
        &[warm_file_show_same_clone_ms],
    );
    print_stat(
        "virtual fs file show warm cross-clone shared cache",
        &[warm_file_show_cross_clone_ms],
    );
}

fn seed_benchmark_repo(work_dir: &TestWorkDir<'_>) {
    for index in 0..SMALL_FILE_COUNT {
        let content = format!(
            "{index:04}:{}\n",
            "x".repeat(SMALL_FILE_BYTES.saturating_sub(6))
        );
        work_dir.write_file(format!("tree/file-{index:04}.txt"), content);
    }
    work_dir.write_file("large.txt", "L".repeat(LARGE_FILE_BYTES));
}

fn measure(f: impl FnOnce()) -> Duration {
    let started = Instant::now();
    f();
    started.elapsed()
}

fn print_stat(label: &str, samples: &[Duration]) {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let total = sorted.iter().copied().sum::<Duration>();
    let mean = total.div_f64(sorted.len() as f64);
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    println!(
        "{label}: mean={:.1} ms median={:.1} ms min={:.1} ms max={:.1} ms samples={}",
        mean.as_secs_f64() * 1000.0,
        median.as_secs_f64() * 1000.0,
        min.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
        sorted.len()
    );
}
