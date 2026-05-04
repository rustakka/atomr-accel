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
                  │   atomr-accel   │   ← foundation (DeviceActor, GpuRef, …)
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
or version-depend on `atomr-accel-cuda` and **nothing else from this
workspace**. Reach for one without inheriting the others.

## Pick by goal

### "I just want to run a cuBLAS kernel"

```toml
[dependencies]
atomr-accel = "0.0"
```

Runtime cost: atomr-core + atomr-config + atomr-macros + cudarc +
tokio. cuBLAS is the always-on library; everything else is gated.

### "I want a batching server in front of my GPU model"

```toml
[dependencies]
atomr-accel          = "0.0"
atomr-accel-patterns = "0.0"
```

Adds: nothing beyond the patterns crate itself.

### "I want JIT-compiled kernels"

```toml
[dependencies]
atomr-accel = { version = "0.0", features = ["nvrtc"] }
```

Adds: cudarc nvrtc bindings (zero new transitive deps; just unlocks
`NvrtcActor::Compile` / `Launch`).

### "I want training across multiple GPUs"

```toml
[dependencies]
atomr-accel       = { version = "0.0", features = ["full-cuda"] }
atomr-accel-train = "0.0"
```

`full-cuda` aggregate bundles cuDNN + cuFFT + cuRAND + cuSPARSE +
cuSOLVER + cuBLASLt + NVRTC + cuTENSOR + NCCL.

### "I want a deterministic replay journal"

```toml
[dependencies]
atomr-accel = { version = "0.0", features = ["replay"] }
```

Adds: atomr-persistence (Journal trait), serde, serde_json. The
`Journal` trait is generic over the backend — pair with
`atomr-persistence-redis` / `atomr-persistence-sql` / etc. depending
on where you want events to land.

### "I want metrics in a dashboard"

```toml
[dependencies]
atomr-accel = { version = "0.0", features = ["telemetry"] }
```

Adds: atomr-telemetry. Run `atomr-dashboard` as a sidecar to
visualize.

### "I want all of it"

```toml
[dependencies]
atomr-accel          = { version = "0.0", features = ["full-cuda", "replay", "cluster", "streams", "telemetry"] }
atomr-accel-patterns = "0.0"
atomr-accel-train    = "0.0"
atomr-accel-agents   = "0.0"
atomr-accel-cuda-realtime = { version = "0.0", features = ["nvrtc"] }
```

## Feature reference

### `atomr-accel-cuda` (foundation)

| Feature              | Adds                                          | Transitive deps |
|----------------------|-----------------------------------------------|-----------------|
| (default)            | cuBLAS via `BlasActor`                        | atomr-core, atomr-config, atomr-macros, cudarc, tokio |
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
| `replay`             | `ReplayHarness::with_journal(...)`            | atomr-persistence, serde, serde_json |
| `cluster`            | `placement::sharded::PlacementShardingAdapter` | atomr-cluster-sharding (and its transitive cluster crates) |
| `streams`            | `streams_pipeline` helpers                    | atomr-streams |
| `telemetry`          | `observability::install` + GPU probes         | atomr-telemetry |

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
| `cudnn`   | pass-through to `atomr-accel-cuda/cudnn`                         |
| `nvrtc`   | pass-through to `atomr-accel-cuda/nvrtc`; enables `with_nvrtc(...)` constructors |

### `atomr-accel-py` (Python bindings)

| Feature           | Adds                                       |
|-------------------|--------------------------------------------|
| `extension-module` (default) | PyO3 cdylib build flag         |
| `curand`          | `RngGenerator` Python class                |
| `nvrtc`           | `NvrtcKernel` Python class                 |
| (the atomr-accel library aggregates) | matching cudarc bindings  |

## Reading transitive deps

Run `cargo tree -p <crate> --no-default-features` to see exactly what
a default-feature dependency pulls in, then add features one at a
time and re-run. The workspace is structured so each feature's
transitive footprint is short and easy to inspect.

```bash
cargo tree -p atomr-accel --no-default-features --depth 2
cargo tree -p atomr-accel --features replay --depth 2
cargo tree -p atomr-accel --features cluster --depth 3
```
