# Vex lazy clone and virtual filesystem mode — benchmarks

This document records how to reproduce performance measurements for the Vex-backed JJ path: **eager (`--fs=system`, default)** clone vs **virtual working tree (`--fs=virtual`)** clone, plus cold/warm blob reads via `jj file show`. It complements backend semantics in [`jj-backend/docs/architecture.md`](../../jj-backend/docs/architecture.md).

## What is being measured

| Scenario | Meaning |
| -------- | ------- |
| Eager clone | Default `jj vex clone` (`--fs=system`): **local** working copy; clone prefetch includes blob bodies in the pack path. |
| Virtual-fs clone | `jj vex clone --fs=virtual`: **lazy** blob policy in the clone manifest and **`vex-virtual`** working copy (metadata-first bootstrap). |
| `file show` cold | First read of a large blob from a lazily cloned repo (blob not in per-repo cache before read). |
| `file show` warm (same clone) | Second read in the same workspace; blob should hit local `vex-cache`. |
| `file show` warm (cross-clone) | Read from another clone that shares `JJ_VEX_SHARED_CACHE_DIR`. |

**Not measured here:** kernel or FUSE path-faulting latency. The current `vex-virtual` working copy is **not** a full virtual filesystem; it avoids materializing the tree on disk. File access in this benchmark goes through **`jj file show`** (repo/tree + backend), not `open(2)` on workspace paths.

## Where the harness lives

- Integration benchmark (ignored by default): [`cli/tests/test_vex_bench.rs`](../cli/tests/test_vex_bench.rs)
- Live harness (Docker Postgres + Valkey + `jj-backend`): [`jj-backend/crates/jj_backend_test_support`](../../jj-backend/crates/jj_backend_test_support) (used via `VexLiveTestHarness` in [`cli/tests/common/vex_live.rs`](../cli/tests/common/vex_live.rs))

## Prerequisites

- Docker available (Postgres and Valkey containers).
- Network allowed for image pulls if needed.

## How to run

From the `jj/` workspace root:

```bash
cargo test -p jj-cli --test runner test_vex_bench_clone_profiles_and_file_show -- --ignored --nocapture
```

The test prints a **Vex benchmark workload** summary and timing lines to stdout (`--nocapture` is required to see them in the terminal).

## Workload (as implemented in the test)

Constants in `test_vex_bench.rs` (adjust there to scale the repo):

| Item | Default |
| ---- | ------- |
| Small files | `256` × `4096` bytes ≈ **1.0 MiB** total |
| Large file | `1` × `524288` bytes = **0.5 MiB** |
| Shared cache | `JJ_VEX_SHARED_CACHE_DIR` set to a directory under the test env for `--fs=virtual` clones |

`file show` uses `--ignore-working-copy` so timings reflect **repository read / blob fetch** without working-copy snapshot or op-head CAS noise.

## Sample results (reference only)

These numbers are **environment-specific**. One run on a developer machine (Apple Silicon, OrbStack Docker, debug `jj-backend` via `cargo run`) produced:

| Metric | Mean (ms) | Notes |
| ------ | ---------: | ----- |
| Eager clone (3 samples) | ~284 | Local working copy, full prefetch path |
| Virtual-fs clone cold (1 sample) | ~104 | Lazy blobs + virtual WC |
| Virtual-fs clone warm, shared cache (2 samples) | ~104 | Small repo; dominated by fixed clone cost |
| `file show` cold | ~42 | First touch of `large.txt` blob |
| `file show` warm, same clone | ~35 | Cache hit |
| `file show` warm, cross-clone | ~43 | Shared cache; overhead similar to cold at this size |

**Takeaway for this workload:** virtual-fs clone was roughly **2.7×** faster than eager clone, driven by skipping blob hydration at clone time. Shared-cache effects on clone time were negligible at this scale; larger repos or higher latency to the object store will separate cold vs warm more clearly.

## Caveats

1. **Backend build profile:** The live harness starts `jj-backend` with Cargo’s **dev** profile unless you change the harness. Release builds will shift absolute numbers down.
2. **Loopback:** gRPC and HTTP object edge traffic stay on localhost; real deployments add RTT and TLS.
3. **Synthetic content:** Highly compressible or deduplicated data would behave differently than random-like payloads.
4. **CLI overhead:** `jj file show` includes process startup, config load, and template rendering; micro-benchmarks on `VexClient::get_object` alone would isolate network/cache more.
5. **Stale working copy:** Benchmarks intentionally avoid `jj workspace update-stale` in the timed `file show` path to prevent op-head CAS conflicts when measuring reads.

## Related implementation

- CLI: `jj vex clone --fs=system|virtual`, `jj vex init --fs=system|virtual` (see `jj/cli/src/commands/vex/`). Explicit `--blob-mode` / `--working-copy` still override the defaults implied by `--fs`.
- Clone blob mode and manifest: `jj/lib/src/vex.rs`, `jj/lib/src/workspace.rs`, `jj-backend` `GetCloneManifest` / `CloneManifest` types.
- Shared cache env: `JJ_VEX_SHARED_CACHE_DIR`, `JJ_VEX_CACHE_MAX_BYTES` (see `VexClient` in `jj/lib/src/vex.rs`).
- Object download hints: `JJ_OBJECT_BASE_URL` and `GET /objects/{kind}/{content_id}` on the git HTTP server (`jj-backend`).

## Future benchmarks (suggested)

- Scale total blob volume (e.g. tens or hundreds of MiB) to stress pack fetch vs lazy clone.
- Inject latency or point at a remote staging backend.
- Compare **release** `jj` + **release** `jj-backend`.
- After a true path-faulting read API exists, add benchmarks for **per-path open/read** comparable to Eden/Sapling-style access.

## Changelog

- **2026-04-03:** Initial document; matches `test_vex_bench_clone_profiles_and_file_show` and one recorded sample run.
- **2026-04-03:** Renamed user-facing “agent” profile to **`--fs=system` / `--fs=virtual`**; benchmarks doc updated accordingly.
