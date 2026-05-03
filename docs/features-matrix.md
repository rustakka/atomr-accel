# Feature matrix

Pick the smallest dependency footprint that does what you need. Every
feature is independent unless explicitly listed as an aggregate.

```
                                    ┌─────────────┐
                                    │  rakka-core │   ← always pulled
                                    └──────┬──────┘
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                  ┌───────┴────────┐               ┌────────┴────────┐
                  │ rakka-config   │               │  rakka-macros   │
                  └────────────────┘               └─────────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │   rakka-accel   │   ← foundation (DeviceActor, GpuRef, …)
                  └────────────────┘
                          │
       ┌──────────┬───────┼───────┬──────────┬──────────┐
       ▼          ▼       ▼       ▼          ▼          ▼
   patterns    train   agents  realtime     py     extensions
                                                  (replay,
                                                   cluster,
                                                   streams,
                                                   telemetry)
```

Sub-crates (`patterns`, `train`, `agents`, `realtime`, `py`) all path-
or version-depend on `rakka-accel-cuda` and **nothing else from this
workspace**. Reach for one without inheriting the others.

## Pick by goal

### "I just want to run a cuBLAS kernel"

```toml
[dependencies]
rakka-accel = "0.0"
```

Runtime cost: rakka-core + rakka-config + rakka-macros + cudarc +
tokio. cuBLAS is the always-on library; everything else is gated.

### "I want a batching server in front of my GPU model"

```toml
[dependencies]
rakka-accel          = "0.0"
rakka-accel-patterns = "0.0"
```

Adds: nothing beyond the patterns crate itself.

### "I want JIT-compiled kernels"

```toml
[dependencies]
rakka-accel = { version = "0.0", features = ["nvrtc"] }
```

Adds: cudarc nvrtc bindings (zero new transitive deps; just unlocks
`NvrtcActor::Compile` / `Launch`).

### "I want training across multiple GPUs"

```toml
[dependencies]
rakka-accel       = { version = "0.0", features = ["full-cuda"] }
rakka-accel-train = "0.0"
```

`full-cuda` aggregate bundles cuDNN + cuFFT + cuRAND + cuSPARSE +
cuSOLVER + cuBLASLt + NVRTC + cuTENSOR + NCCL.

### "I want a deterministic replay journal"

```toml
[dependencies]
rakka-accel = { version = "0.0", features = ["replay"] }
```

Adds: rakka-persistence (Journal trait), serde, serde_json. The
`Journal` trait is generic over the backend — pair with
`rakka-persistence-redis` / `rakka-persistence-sql` / etc. depending
on where you want events to land.

### "I want metrics in a dashboard"

```toml
[dependencies]
rakka-accel = { version = "0.0", features = ["telemetry"] }
```

Adds: rakka-telemetry. Run `rakka-dashboard` as a sidecar to
visualize.

### "I want all of it"

```toml
[dependencies]
rakka-accel          = { version = "0.0", features = ["full-cuda", "replay", "cluster", "streams", "telemetry"] }
rakka-accel-patterns = "0.0"
rakka-accel-train    = "0.0"
rakka-accel-agents   = "0.0"
rakka-accel-cuda-realtime = { version = "0.0", features = ["nvrtc"] }
```

## Feature reference

### `rakka-accel-cuda` (foundation)

| Feature              | Adds                                          | Transitive deps |
|----------------------|-----------------------------------------------|-----------------|
| (default)            | cuBLAS via `BlasActor`                        | rakka-core, rakka-config, rakka-macros, cudarc, tokio |
| `cudnn`              | `CudnnActor` + cudarc cudnn bindings          | (cudnn lib, dlopen) |
| `cufft`              | `FftActor`                                    | (cufft lib) |
| `curand`             | `RngActor`                                    | (curand lib) |
| `cusolver`           | `SolverActor` (QR/LU/Chol/SVD/Syevd)          | (cusolver lib) |
| `cusparse`           | `SparseActor` (CSR SpMv/SpMm)                 | (cusparse lib) |
| `cutensor`           | `TensorActor` (Einstein contractions)         | (cutensor lib) |
| `cublaslt`           | `BlasLtActor` (fused matmul + epilogue)       | — |
| `nvrtc`              | `NvrtcActor` (JIT compile + launch)           | — |
| `nccl`               | `CollectiveActor` + `NcclWorldActor`          | (nccl lib) |
| `f16`                | F16 / Bf16 dtype variants                     | half crate |
| `cuda-runtime-tests` | unlocks GPU integration tests + examples      | — |
| **`core-libs`**      | `cudnn` + `cufft` + `curand` + `cusparse`     | aggregate |
| **`training-libs`**  | `core-libs` + `cusolver` + `cublaslt` + `nvrtc` + `cutensor` | aggregate |
| **`full-cuda`**      | `training-libs` + `nccl`                      | aggregate |
| `replay`             | `ReplayHarness::with_journal(...)`            | rakka-persistence, serde, serde_json |
| `cluster`            | `placement::sharded::PlacementShardingAdapter` | rakka-cluster-sharding (and its transitive cluster crates) |
| `streams`            | `streams_pipeline` helpers                    | rakka-streams |
| `telemetry`          | `observability::install` + GPU probes         | rakka-telemetry |

### `rakka-accel-patterns`

No optional features. Pulls in `rakka-accel-cuda` (default features).

### `rakka-accel-train`

No optional features. Pulls in `rakka-accel-cuda` (default features) and
`rakka-accel-patterns` (for replica routing).

### `rakka-accel-agents`

No optional features. Pulls in `rakka-accel-cuda` (default features).

### `rakka-accel-cuda-realtime`

| Feature   | Adds                                                       |
|-----------|------------------------------------------------------------|
| (default) | CPU reference implementations of every actor               |
| `cudnn`   | pass-through to `rakka-accel-cuda/cudnn`                         |
| `nvrtc`   | pass-through to `rakka-accel-cuda/nvrtc`; enables `with_nvrtc(...)` constructors |

### `rakka-accel-py` (Python bindings)

| Feature           | Adds                                       |
|-------------------|--------------------------------------------|
| `extension-module` (default) | PyO3 cdylib build flag         |
| `curand`          | `RngGenerator` Python class                |
| `nvrtc`           | `NvrtcKernel` Python class                 |
| (the rakka-accel library aggregates) | matching cudarc bindings  |

## Reading transitive deps

Run `cargo tree -p <crate> --no-default-features` to see exactly what
a default-feature dependency pulls in, then add features one at a
time and re-run. The workspace is structured so each feature's
transitive footprint is short and easy to inspect.

```bash
cargo tree -p rakka-accel --no-default-features --depth 2
cargo tree -p rakka-accel --features replay --depth 2
cargo tree -p rakka-accel --features cluster --depth 3
```
