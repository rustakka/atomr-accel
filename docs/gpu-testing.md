# GPU Integration Testing

This page covers the **opt-in GPU integration suite** — tests that exercise
real CUDA kernels and observability backends. They are **not part of CI**:
`cargo xtask verify` (the release-pipeline gate) explicitly runs `cargo test
--no-default-features` and `cargo check --features full-cuda`, neither of
which trips the suite. CI hosts don't have a GPU; running the suite there
would either silently skip or noisily fail in ways that obscure real
breakage.

The suite is for **local development on a CUDA-equipped host**: a quick
smoke confirmation that a feature still launches end-to-end, plus
performance regression tracking via Criterion benches.

## Quick start

```bash
# 1. Confirm the local box has what the tests need.
cargo xtask gpu-probe

# 2. Run all GPU integration suites.
cargo xtask gpu-test

# 3. Run a single suite (faster).
cargo xtask gpu-test cublas
cargo xtask gpu-test telemetry

# 4. Run the perf-regression benches.
cargo xtask gpu-bench
```

## How the gating works

Every GPU-touching test is gated by **two layers**:

1. **Cargo feature**: `cuda-runtime-tests` (or `nvtx`/`nvml`/`cupti` for the
   telemetry crate). Without these features the test source is
   `#[cfg]`-stripped and never reaches the linker. This keeps default
   `cargo build` host-clean.

2. **`#[ignore]` attribute**: Even with the feature on, the test is
   `#[ignore = "requires CUDA driver"]` so plain `cargo test` skips it.
   Run with `-- --ignored` (or via `cargo xtask gpu-test`).

When the feature *and* `--ignored` flags are both supplied, each test
performs a runtime probe (`CudaContext::new(0)`, `NvmlActor::try_new`,
`TrtActor::ensure_runtime`, …) and **skips with a logged message** if the
local environment lacks the required driver/library. Tests never crash —
they pass with a `[skip] …` log line.

This three-layer pattern (feature flag + `#[ignore]` + runtime probe) means
the same test code is safe to invoke on:

| environment | result |
|---|---|
| no CUDA installed | `cargo test` passes (cfg-stripped) |
| no CUDA, but `--features cuda-runtime-tests --ignored` | passes with `[skip]` logs |
| CUDA driver present, libraries match cudarc | actually runs |
| CUDA driver present but older than cudarc bindings | passes with `[skip] cudarc panicked on dlsym (driver likely older than its bindings)` |

## Available suites

`cargo xtask gpu-test <suite>` — `<suite>` is one of:

| Suite | Crate | What it exercises |
|---|---|---|
| `cublas` | atomr-accel-cuda | `BlasActor::Sgemm` against a real cuBLAS handle (existing `sgemm_e2e.rs`) |
| `cublaslt` | atomr-accel-cuda | cuBLASLt matmul + epilogue dispatch |
| `cudnn` | atomr-accel-cuda | cuDNN frontend graph: conv-fwd, layernorm, MHA |
| `cufft` | atomr-accel-cuda | cuFFT 1D/2D/3D R2C round-trips |
| `curand` | atomr-accel-cuda | cuRAND fill_uniform + statistical sanity checks |
| `cusolver` | atomr-accel-cuda | cuSOLVER QR / SVD on small matrices |
| `cusparse` | atomr-accel-cuda | cuSPARSE SpMV CSR (existing `spmv_e2e.rs`) |
| `cutensor` | atomr-accel-cuda | cuTENSOR contraction (existing `contract_e2e.rs`) |
| `nccl` | atomr-accel-cuda | single-rank NCCL world; multi-rank tests skip if `<2` GPUs |
| `nvrtc` | atomr-accel-cuda | NVRTC compile + launch round-trip |
| `graph` | atomr-accel-cuda | CUDA graph capture/replay |
| `event` | atomr-accel-cuda | `EventActor` record/wait/elapsed_time + IPC |
| `memory` | atomr-accel-cuda | pinned-pool memcpy (existing `pinned_memcpy_e2e.rs`), managed-memory advise |
| `cub` | atomr-accel-cub | dispatch surface + kernel-source cache round-trip |
| `cutlass` | atomr-accel-cutlass | arch×dtype support matrix (host-side; full e2e via NVRTC pending) |
| `flashattn` | atomr-accel-flashattn | dispatch table covers canonical (arch, dtype, head_dim, causal, varlen) configurations |
| `tensorrt` | atomr-accel-tensorrt | `TrtActor::ensure_runtime` lazy-load against `libnvinfer.so` |
| `telemetry` | atomr-accel-telemetry | NVML snapshot returns real device 0 name + memory |

## Adding a new GPU test

1. Decide which crate owns it. Per-library tests live in
   `crates/atomr-accel-cuda/tests/` (the actor crate); per-sibling-crate
   tests live in their own `tests/` directory.

2. Top of file:

   ```rust
   #![cfg(feature = "cuda-runtime-tests")]
   ```

   For tests that need additional library features (`cudnn`, `cublaslt`,
   etc.), use `#![cfg(all(feature = "cuda-runtime-tests", feature = "..."))]`.

3. Each test gets `#[ignore]`:

   ```rust
   #[test]
   #[ignore = "requires CUDA driver"]
   fn my_kernel_runs_e2e() { ... }
   ```

4. Probe and skip when the driver is missing:

   ```rust
   let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
   match probe {
       Ok(Ok(_)) => {}
       Ok(Err(e)) => { eprintln!("[skip] CUDA init failed: {e}"); return; }
       Err(_)    => { eprintln!("[skip] cudarc panicked on dlsym (older driver)"); return; }
   }
   ```

   (Tokio-based tests should put this inside the `#[tokio::test]` body.)

5. Add the suite to `xtask/src/main.rs`'s `gpu_test_plan` if it's a new
   crate or feature combination. The xtask drives `cargo test -p <crate>
   --features <feats> -- --ignored --nocapture`.

6. Document it in the suite table above.

## Performance benches

`cargo xtask gpu-bench` runs Criterion benchmarks. Existing benches:

- `sgemm_overhead` — measures cuBLAS SGEMM actor-pipeline overhead vs raw
  cudarc. Phase 0 success criterion: ≤5% overhead at N≥2048.
- `rng_throughput` — measures cuRAND `FillUniformF32` throughput.

Add a new bench by dropping a Criterion harness into
`crates/atomr-accel-cuda/benches/`, registering it in `Cargo.toml` with
`required-features = ["cuda-runtime-tests", "<lib>"]`, and adding it to
the `gpu-bench` entry list in `xtask/src/main.rs`.

Track regressions by saving Criterion's `target/criterion/` baselines
between runs (e.g. with `--save-baseline before` and
`--baseline before` on the next run).

## What gets exercised

When you have the full CUDA toolkit installed locally (cuBLAS / cuDNN /
cuFFT / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR / NCCL / NVRTC) plus the
matching `libcuda.so`, `cargo xtask gpu-test` will run:

- ~10 `tests/*_e2e.rs` integration tests in atomr-accel-cuda (Phases 1, 2, 4)
- 1 dispatch-table smoke per Phase 5/6/7/8/9 sibling crate (cub, cutlass,
  flashattn, tensorrt, telemetry)
- The Phase 0.7 NVTX/NVML/CUPTI hooks in `atomr-accel-telemetry/tests/`

Coverage gaps (TODO):

- A real CUTLASS GEMM JIT round-trip: requires nvcc on PATH and the
  vendored CUTLASS template subset to be expanded beyond placeholders.
- A real fa2 forward correctness check: needs the vendored fa2 csrc
  populated with a working kernel + a reference attention impl.
- TensorRT engine-build: the `tensorrt-link` feature is currently
  fenced off (compile_error!) until `nvinfer_shim.cpp` lands —
  https://github.com/rustakka/atomr-accel/issues/6. Once the shim
  ships this re-enables a libnvinfer link probe.

## Why this isn't in CI

1. **No GPU runners.** GitHub Actions standard pool has no CUDA. Self-hosted
   GPU runners exist but fold a hardware dependency into the merge gate
   that locks out contributors without that hardware.

2. **Driver version skew.** cudarc 0.19.4 binds against CUDA 12.x; older
   drivers are missing symbols cudarc tries to dlsym at startup.
   Production driver fleets lag toolkit versions, so the same test code
   that works on developer A's box panics on developer B's. The
   `catch_unwind` skip pattern handles this for opt-in runs but isn't
   appropriate for a green-or-red CI gate.

3. **Flake budget.** GPU tests have non-determinism around scheduling,
   driver state across runs, and ECC retries. Acceptable on a developer's
   workstation; corrosive in a green/red CI signal.

4. **Build farm hygiene.** Linking `libnvinfer.so` (TensorRT) requires
   accepting NVIDIA's EULA per host. Routing that through CI runners is
   organizationally heavier than it's worth for a smoke gate.

When a code change demonstrably needs GPU validation, the contract is:
the author runs `cargo xtask gpu-test` locally, attaches the full output
to the PR, and (if a perf-sensitive change) attaches a
before/after `cargo xtask gpu-bench` comparison.
