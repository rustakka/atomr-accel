# Feature matrix

Pick the smallest dependency footprint that does what you need. Every
feature is independent unless explicitly listed as an aggregate.

```
                                    ┌─────────────┐
                                    │  atomr-core │   ← always pulled
                                    └──────┬──────┘
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                  ┌───────┴────────┐               ┌────────┴────────┐
                  │ atomr-config   │               │  atomr-macros   │
                  └────────────────┘               └─────────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │  atomr-accel    │   ← portable trait surface (AccelDtype, AccelBackend, …)
                  └────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ atomr-accel-cuda│   ← NVIDIA CUDA implementation (DeviceActor, GpuRef, …)
                  └────────┬───────┘
                           │
       ┌──────┬──────┬─────┼─────┬──────┬──────┬──────┬──────┐
       ▼      ▼      ▼     ▼     ▼      ▼      ▼      ▼      ▼
   patterns  train agents  py  cub   cutlass flash  trt   telemetry
                          + realtime         attn        + nvtx/nvml/cupti
                                                          probes
                                                  + extensions on cuda crate
                                                  (replay, cluster, streams)
```

Sub-crates `patterns` / `train` / `agents` / `cuda-realtime` / `py`
all depend on `atomr-accel-cuda` and **nothing else from this
workspace**. Reach for one without inheriting the others.

The Phase 5–9 sibling crates (`cub`, `cutlass`, `flashattn`,
`tensorrt`, `telemetry`) are **opt-in**: enable the matching feature
on `atomr-accel-cuda` and the actor surface re-exports through
`atomr_accel_cuda::prelude`. They link nothing extra when their
feature is off.

## Pick by goal

### "I just want to run a cuBLAS kernel"

```toml
[dependencies]
atomr-accel-cuda = "0.1"
```

Runtime cost: atomr-core + atomr-config + atomr-macros + cudarc +
tokio. cuBLAS is the always-on library; everything else is gated.

### "I want a batching server in front of my GPU model"

```toml
[dependencies]
atomr-accel-cuda     = "0.1"
atomr-accel-patterns = "0.1"
```

Adds: nothing beyond the patterns crate itself.

### "I want JIT-compiled kernels"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = ["nvrtc"] }
```

Adds: cudarc nvrtc bindings (zero new transitive deps; just unlocks
`NvrtcActor::Compile` / `Launch`). With `nvrtc-lto` the persistent
disk cache compiles `-dlto` PTX once and reuses across runs; with
`nvrtc-async` long compiles dispatch onto a Tokio blocking pool.

### "I want training across multiple GPUs"

```toml
[dependencies]
atomr-accel-cuda  = { version = "0.1", features = ["full-cuda", "f16"] }
atomr-accel-train = "0.1"
```

`full-cuda` aggregate bundles cuDNN + cuFFT + cuRAND + cuSPARSE +
cuTENSOR + cuda-managed + cuSOLVER + cuBLASLt + NVRTC + NCCL +
cuda-ipc + graphs-conditional. Add `cublas-fp8` / `nccl-fp8` for
Hopper-class fp8 numerics.

### "I want fast attention"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = ["flashattn", "flashattn-paged", "f16"] }
```

Pulls `atomr-accel-flashattn` for fa2 + fa3 forward/backward, paged
KV-cache, varlen, MQA/GQA. Add `flashattn-fp8` on Hopper.

### "I want CUTLASS-grade GEMM templates"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = ["cutlass", "cutlass-grouped", "cutlass-evt", "f16"] }
```

NVRTC-instantiates CUTLASS templates against vendored headers.
`cutlass-prebuilt` flips to Strategy B (build.rs runs `nvcc` to
produce a static archive at build time — faster startup, slower
iteration; opt-in).

### "I want TensorRT inference"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = ["tensorrt", "tensorrt-onnx", "tensorrt-int8"] }
```

Pulls `atomr-accel-tensorrt`. The `libnvinfer.so` library itself is
**not** vendored — it links at runtime via
`atomr-accel-tensorrt/tensorrt-link`. INT8/FP8 PTQ helpers are
behind matching features.

### "I want a deterministic replay journal"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = ["replay"] }
```

Adds: atomr-persistence (Journal trait), serde, serde_json. The
`Journal` trait is generic over the backend — pair with
`atomr-persistence-redis` / `atomr-persistence-sql` / etc. depending
on where you want events to land.

### "I want metrics in a dashboard"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = ["telemetry"] }
```

Adds: atomr-telemetry. Run `atomr-dashboard` as a sidecar to
visualize. For NVIDIA-specific signals layer the Phase 9 backends:

```toml
features = ["observability-full"]   # = telemetry + nvtx-trace + nvml + cupti
```

### "I want all of it"

```toml
[dependencies]
atomr-accel-cuda = { version = "0.1", features = [
    "full-cuda",
    "f16", "f8", "f4",
    "cublas-fp8", "nccl-fp8", "cusparse-lt", "cutensor-autotune",
    "cublas-fp8", "cudnn-frontend", "nvrtc-lto", "nvrtc-async",
    "hopper", "blackwell",
    "cub", "cutlass", "cutlass-grouped", "cutlass-evt",
    "flashattn", "flashattn-fp8", "flashattn-paged",
    "tensorrt", "tensorrt-onnx", "tensorrt-int8", "tensorrt-fp8", "tensorrt-plugin",
    "observability-full",
    "replay", "cluster", "streams",
] }
atomr-accel-patterns      = "0.1"
atomr-accel-train         = "0.1"
atomr-accel-agents        = "0.1"
atomr-accel-cuda-realtime = { version = "0.1", features = ["nvrtc"] }
```

## Feature reference

### `atomr-accel` (portable trait surface)

| Feature   | Adds                                                  |
|-----------|-------------------------------------------------------|
| (default) | `AccelBackend`, `AccelDevice`, `AccelStream`, `AccelRef`, `AccelError`, `CompletionStrategy`, `AccelDtype`, `KernelOp` |
| `f16`     | `half::f16` / `half::bf16` impls of `AccelDtype` |
| `f8`      | `F8E4m3` / `F8E5m2` newtype impls of `AccelDtype` |
| `f4`      | `F4E2m1` newtype impls of `AccelDtype` |

### `atomr-accel-cuda` — per-library actors

| Feature              | Adds                                                                 |
|----------------------|----------------------------------------------------------------------|
| (default)            | cuBLAS via `BlasActor`                                               |
| `cudnn`              | `CudnnActor` + cudarc cudnn bindings                                 |
| `cudnn-frontend`     | cuDNN v9 backend Graph API (conv/norm/MHA/RNN); requires `cudnn`     |
| `cufft`              | `FftActor` (1D/2D/3D R2C/C2R/C2C, f32/f64 + plan_many batched)       |
| `curand`             | `RngActor` (Philox, XORWOW, MTGP32, …)                               |
| `curand-host`        | host-API `curandGenerate*` filling host buffers                      |
| `curand-quasirandom` | Sobol32 / ScrambledSobol64 quasi-random generators                   |
| `cusolver`           | `SolverActor` (QR/LU/Chol/SVD/Syevd, dense + batched)                |
| `cusolver-sp`        | sparse Cholesky/QR/LU via cusolverSp                                 |
| `cusparse`           | `SparseActor` (CSR SpMV/SpMM)                                        |
| `cusparse-generic`   | descriptor-based generic API (SpMM/SpMV/SpGEMM/SpSV); on with `cusparse` |
| `cusparse-lt`        | structured 2:4 sparsity via cuSPARSELt (off-by-default; loaded via libloading) |
| `cutensor`           | `TensorActor` (Einstein contractions, reduce, elementwise)           |
| `cutensor-autotune`  | top-k algo probe + LRU cache for contractions                        |
| `cublaslt`           | `BlasLtActor` (fused matmul + epilogue)                              |
| `cublas-fp8`         | fp8 GEMM via cuBLASLt + scaling helpers (Hopper sm_90+)              |
| `nvrtc`              | `NvrtcActor` (JIT compile + launch) + persistent on-disk cache       |
| `nvrtc-lto`          | `-dlto` link-time-opt flags on `NvrtcOpts`                           |
| `nvrtc-async`        | `NvrtcMsg::CompileAsync` via Tokio blocking pool                     |
| `nccl`               | `CollectiveActor` + `NcclWorldActor`                                 |
| `nccl-fp8`           | fp8 reduce ops (NCCL ≥ 2.20; degrades gracefully)                    |
| `nccl-nvls`          | NVLink-Sharp opt-in (NCCL ≥ 2.18, surfaced via `NcclCapabilities`)   |
| `cuda-managed`       | `ManagedAllocatorActor` + `cudaMemPrefetchAsync` / `cuMemAdvise`     |
| `cuda-ipc`           | `EventActor` IPC variants + `memory::ipc` wrappers                   |
| `graphs-conditional` | `cudaGraphConditionalNode` (CUDA ≥ 12.4)                             |
| `hopper`             | TMA descriptor builder, cluster launch, wgmma / cp.async macros (sm_90a) |
| `blackwell`          | Blackwell-only intrinsics + sm_100 / sm_120 targets (implies `hopper`) |
| `f16`                | F16 / Bf16 alloc + tensor-descriptor variants                        |
| `f8`                 | F8E4m3 / F8E5m2 wrapper-type variants                                |
| `f4`                 | F4E2m1 wrapper-type variant                                          |
| `cuda-runtime-tests` | unlocks GPU integration tests + examples (gated `#[ignore]`)         |

### `atomr-accel-cuda` — sibling-crate gates

| Feature              | Pulls in                          | Adds                                                              |
|----------------------|-----------------------------------|-------------------------------------------------------------------|
| `cub`                | `atomr-accel-cub`                 | `CubActor` for reduce / scan / sort / histogram / select / segmented |
| `cutlass`            | `atomr-accel-cutlass`             | `CutlassActor` for GEMM / grouped-GEMM / implicit-GEMM conv       |
| `cutlass-grouped`    | `atomr-accel-cutlass/grouped`     | grouped-GEMM dispatch surface                                     |
| `cutlass-evt`        | `atomr-accel-cutlass/evt`         | epilogue visitor tree (EVT) emitter                               |
| `cutlass-prebuilt`   | `atomr-accel-cutlass/cutlass-prebuilt` | Strategy B — build.rs runs nvcc                              |
| `flashattn`          | `atomr-accel-flashattn`           | `FlashAttnActor` (fa2 + fa3 fwd/bwd, varlen, ALiBi, sliding-window, MQA/GQA) |
| `flashattn-fp8`      | `atomr-accel-flashattn/fp8`       | fp8 e4m3 / e5m2 paths in fa3 (sm_90a only)                        |
| `flashattn-paged`    | `atomr-accel-flashattn/paged`     | paged KV-cache + chunked prefill                                  |
| `tensorrt`           | `atomr-accel-tensorrt`            | `TrtActor` + `IBuilderConfig` + engine runtime                    |
| `tensorrt-onnx`      | `atomr-accel-tensorrt/tensorrt-onnx` | ONNX import via libnvonnxparser                                |
| `tensorrt-plugin`    | `atomr-accel-tensorrt/tensorrt-plugin` | IPluginV3 Rust trampolines                                  |
| `tensorrt-int8`      | `atomr-accel-tensorrt/tensorrt-int8` | INT8 entropy / minmax PTQ                                     |
| `tensorrt-fp8`       | `atomr-accel-tensorrt/tensorrt-fp8`  | FP8 PTQ helpers (Hopper)                                      |
| `nvtx`               | cudarc/nvtx                       | NVTX range/event annotations on `KernelEnvelope`                  |
| `nvtx-trace`         | `atomr-accel-telemetry/nvtx`      | automatic kernel-range markers via `NvtxKernelTrace`              |
| `nvml`               | `atomr-accel-telemetry/nvml`      | `NvmlActor` polling power / temp / ECC / clocks / mem util        |
| `cupti`              | `atomr-accel-telemetry/cupti`     | `CuptiSession` for activity tracing + range profiler              |

### `atomr-accel-cuda` — atomr integrations

| Feature      | Adds                                                              | Transitive deps |
|--------------|-------------------------------------------------------------------|-----------------|
| `replay`     | `ReplayHarness::with_journal(...)`                                | atomr-persistence, serde_json |
| `cluster`    | `placement::sharded::PlacementShardingAdapter`                    | atomr-cluster-sharding |
| `streams`    | `streams_pipeline` helpers                                        | atomr-streams |
| `telemetry`  | `observability::install` + GPU probes                             | atomr-telemetry |

### `atomr-accel-cuda` — aggregates

| Aggregate            | Expands to |
|----------------------|------------|
| `core-libs`          | `cudnn` + `cufft` + `curand` + `cusparse` + `cutensor` + `cuda-managed` |
| `training-libs`      | `core-libs` + `cusolver` + `cublaslt` + `nvrtc` |
| `full-cuda`          | `training-libs` + `nccl` + `cuda-ipc` + `graphs-conditional` |
| `observability-full` | `telemetry` + `nvtx-trace` + `nvml` + `cupti` |

### `atomr-accel-cub` (Phase 5)

| Feature   | Adds                                                  |
|-----------|-------------------------------------------------------|
| (default) | host-side dispatcher surface + `KernelSourceCache`    |
| `cuda-runtime-tests` | NVRTC-emitting kernels + GPU integration tests |

CUB headers are vendored under `vendor/cub/` (BSD-3-Clause) and
NVRTC-instantiated against the Phase 0 persistent kernel cache.

### `atomr-accel-cutlass` (Phase 6)

| Feature             | Adds                                                  |
|---------------------|-------------------------------------------------------|
| (default)           | `CutlassActor` + plan-cache + GEMM dispatcher         |
| `grouped`           | grouped-GEMM dispatcher (heterogeneous batch)         |
| `evt`               | epilogue visitor tree emitter (post-GEMM ops)         |
| `cutlass-prebuilt`  | build.rs runs nvcc to produce a static archive        |
| `cuda-runtime-tests` | arch×dtype matrix smoke + NVRTC e2e (when nvcc present) |

### `atomr-accel-flashattn` (Phase 7)

| Feature   | Adds                                                       |
|-----------|------------------------------------------------------------|
| (default) | `FlashAttnActor` + dispatch table (fa2 fp16/bf16 fwd+bwd)  |
| `fp8`     | fa3 fp8 e4m3 / e5m2 paths (sm_90a only)                    |
| `paged`   | paged KV-cache + chunked prefill                           |
| `cuda-runtime-tests` | dispatch-table smoke + e2e launch (when csrc populated) |

### `atomr-accel-tensorrt` (Phase 8)

| Feature             | Adds                                                  |
|---------------------|-------------------------------------------------------|
| (default)           | `TrtActor` host-side surface + `IBuilderConfig` types |
| `tensorrt-link`     | _disabled_ — pending `nvinfer_shim.cpp`; see [#6][trt-shim] |
| `tensorrt-onnx`     | `OnnxParser` wrapper + `onnx_resnet50_int8` example   |
| `tensorrt-int8`     | INT8 entropy/minmax calibrator helpers                |
| `tensorrt-fp8`      | FP8 PTQ helpers (Hopper)                              |
| `tensorrt-plugin`   | IPluginV3 Rust trampoline trait surface               |
| `cuda-runtime-tests` | lazy-load smoke (skips cleanly when libnvinfer missing) |

`libnvinfer.so` is **not** vendored — link only.

[trt-shim]: https://github.com/rustakka/atomr-accel/issues/6

### `atomr-accel-telemetry` (Phase 9)

| Feature   | Adds                                                       |
|-----------|------------------------------------------------------------|
| (default) | `NvtxKernelTrace` (no-op when `nvtx` off) + probe traits   |
| `nvtx`    | NVTX range push/pop wired through `KernelEnvelope::with_nvtx` |
| `nvml`    | `NvmlActor` (power / temp / ECC / clocks / mem / processes) |
| `cupti`   | `CuptiSession` (activity tracing + range profiler)         |
| `cuda-runtime-tests` | NVML / CUPTI smoke tests (skip without driver)  |

### `atomr-accel-patterns`

No optional features. Pulls in `atomr-accel-cuda` (default features).

### `atomr-accel-train`

No optional features. Pulls in `atomr-accel-cuda` (default features) and
`atomr-accel-patterns` (for replica routing).

### `atomr-accel-agents`

No optional features. Pulls in `atomr-accel-cuda` (default features).

### `atomr-accel-cuda-realtime`

| Feature   | Adds                                                       |
|-----------|------------------------------------------------------------|
| (default) | CPU reference implementations of every actor               |
| `cudnn`   | pass-through to `atomr-accel-cuda/cudnn`                   |
| `nvrtc`   | pass-through to `atomr-accel-cuda/nvrtc`; enables `with_nvrtc(...)` constructors |

### `atomr-accel-py` (Python bindings)

| Feature                      | Adds                                |
|------------------------------|-------------------------------------|
| `extension-module` (default) | PyO3 cdylib build flag              |
| `curand`                     | `RngGenerator` Python class         |
| `nvrtc`                      | `NvrtcKernel` Python class          |
| (the atomr-accel-cuda library aggregates) | matching cudarc bindings |

## Reading transitive deps

Run `cargo tree -p <crate> --no-default-features` to see exactly what
a default-feature dependency pulls in, then add features one at a
time and re-run. The workspace is structured so each feature's
transitive footprint is short and easy to inspect.

```bash
cargo tree -p atomr-accel-cuda --no-default-features --depth 2
cargo tree -p atomr-accel-cuda --features replay --depth 2
cargo tree -p atomr-accel-cuda --features cluster --depth 3
cargo tree -p atomr-accel-cuda --features full-cuda --depth 1
cargo tree -p atomr-accel-cuda --features flashattn --depth 1
```
