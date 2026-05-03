# rakka-accel

**An actor-shaped face for compute acceleration.** NVIDIA CUDA ships
today through [`rakka-accel-cuda`](crates/rakka-accel-cuda); the
backend trait surface accommodates AMD ROCm, Apple Metal, Intel
oneAPI, and Vulkan compute when those crates land. Each backend
library ([cuBLAS][cublas], [cuDNN][cudnn], [cuFFT][cufft],
[cuRAND][curand], [cuSOLVER][cusolver], [cuSPARSE][cusparse],
[cuTENSOR][cutensor], [cuBLASLt][cublaslt], [NVRTC][nvrtc],
[NCCL][nccl]) becomes a typed [rakka](../rakka) actor with stable
supervision, generation-validated buffers, and a single async
surface. Drop GPU work into a Rust service without juggling streams,
contexts, or hand-rolled retry loops.

```toml
[dependencies]
rakka-accel = { version = "0.0", features = ["cuda"] }
```

```rust
use rakka_accel::cuda::prelude::*;   // active backend re-exported here
```

```rust
let device = system.actor_of(DeviceActor::props(DeviceConfig::new(0)), "gpu-0")?;
let a = ask_alloc::<f32>(&device, n * n).await?;
let b = ask_alloc::<f32>(&device, n * n).await?;
let c = ask_alloc::<f32>(&device, n * n).await?;

device.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
    a, b, c, m: n, n: n, k: n, alpha: 1.0, beta: 0.0, reply,
})));
// reply arrives once the kernel completes — no host blocking,
// no manual stream synchronization.
```

That's the whole shape. The same envelope wires up convolutions
([`cudnnConvolutionForward`][cudnn-conv]), tensor contractions
([`cutensorContract`][cutensor-contract]), JIT-compiled custom kernels
([`nvrtcCompileProgram`][nvrtc-compile]), and multi-GPU all-reduce
([`ncclAllReduce`][nccl-allreduce]).

## Why

Writing CUDA from Rust today means owning a long list of invariants
yourself:

| You'd otherwise hand-roll                                       | rakka-accel gives you             |
| --------------------------------------------------------------- | -------------------------------- |
| One [`CUcontext`][cuda-ctx] per device, restarted on poisoning  | `DeviceActor ↔ ContextActor` two-tier supervision |
| [Sticky-error][cuda-sticky] detection and graceful recovery     | `OneForOneStrategy` + `GpuError::ContextPoisoned` decider |
| Buffer staleness across context rebuilds                        | `GpuRef<T>` with generation tokens |
| Pinning library handles ([cuBLAS][cublas-handle], [cuDNN][cudnn-handle]) to a single OS thread | `GpuDispatcher` + per-actor handle |
| [Stream-event][cuda-events] choreography for kernel completion  | `HostFnCompletion` (sub-µs `cuLaunchHostFunc`) / `SyncCompletion` / `PolledCompletion` |
| [`cuMemcpyPeerAsync`][cuda-p2p] cross-stream synchronization     | `P2pTopology` with `last_write_stream` injection |
| [Page-locked][cuda-pinned] host buffer pooling                   | `PinnedBufferPool` actor |
| [CUDA Graph][cuda-graph] capture/replay                          | `GraphActor` (`Sgemm` / `Memcpy` / `RngFillUniform` / `FftR2C` record contracts) |
| Multi-GPU [communicator rebuild][nccl-comm] on context loss      | `NcclWorldActor` subscribes to `WatchGeneration`, tears down + rebuilds collectives |

Because every concern is an actor, you compose CUDA the same way you
compose any other Rust service: `tokio` runtime, structured
supervision, typed messages, async/await throughout.

## At a glance

```
   ┌──────────── ActorSystem ────────────┐
   │                                     │
   │   ┌─────────── DeviceActor ─────────┴─── stable address (ActorRef<DeviceMsg>)
   │   │   (queues work while context rebuilds)
   │   │
   │   │   ┌─── ContextActor ──── owns Arc<CudaContext> ── restartable
   │   │   │
   │   │   │   ├── BlasActor        ── cuBLAS handle  ── pinned to one stream
   │   │   │   ├── CudnnActor       ── cuDNN handle   ── pinned to one stream
   │   │   │   ├── FftActor         ── cuFFT plans    ── plan cache
   │   │   │   ├── RngActor         ── cuRAND gen     ── seedable
   │   │   │   ├── SolverActor      ── cuSOLVER       ── QR/LU/Chol/SVD/Syevd
   │   │   │   ├── SparseActor      ── cuSPARSE       ── CSR SpMv/SpMm
   │   │   │   ├── TensorActor      ── cuTENSOR       ── Einstein Contract
   │   │   │   ├── BlasLtActor      ── cuBLASLt       ── fused matmul + ReLU/GELU
   │   │   │   ├── NvrtcActor       ── NVRTC          ── JIT compile + launch
   │   │   │   └── CollectiveActor  ── NCCL comm      ── per-rank
   │   │   │
   │   │   └── PinnedBufferPool / ManagedAllocator / GraphActor / P2pTopology
   │
   └── PlacementActor / ReplayHarness / NcclWorldActor (top-level)
```

Each box is an actor. Messages are typed enums. Replies are
`oneshot::Sender` channels. Failures panic with a tagged string
(`"ContextPoisoned: …"` / `"OutOfMemory: …"` / `"Unrecoverable: …"`)
and the supervisor decides Restart / Resume / Stop / Escalate.

## Quick start

You need a sibling clone of the [rakka](../rakka) workspace:

```
your-workspace/
├── rakka/         # the rakka actor runtime (v0.2.x)
└── rakka-accel/   # this repo
```

cudarc loads CUDA dynamically, so the workspace **builds and
unit-tests on hosts without a GPU**. Real kernel paths are gated
behind `--features cuda-runtime-tests`.

```bash
# No GPU needed:
cargo check --workspace --no-default-features
cargo test  --workspace --no-default-features
cargo run   -p rakka-accel --example echo_no_gpu

# With GPU + CUDA toolkit:
cargo run   -p rakka-accel --example sgemm     --features cuda-runtime-tests
cargo run   -p rakka-accel --example fft_1d    --features cuda-runtime-tests,cufft
cargo run   -p rakka-accel --example jit_relu  --features cuda-runtime-tests,nvrtc
```

Read [`docs/getting-started.md`](docs/getting-started.md) for a
ten-minute tour, [`docs/concepts.md`](docs/concepts.md) for the
supervision / completion / generation model,
[`docs/architecture.md`](docs/architecture.md) for the full design,
[`docs/backends.md`](docs/backends.md) for the multi-backend trait
abstraction (and the ROCm / Metal / oneAPI roadmap),
[`docs/python-bridge.md`](docs/python-bridge.md) for the Python
bindings, and [`docs/features-matrix.md`](docs/features-matrix.md)
for the by-goal dependency picker.

If you're using an AI coding assistant (Claude Code, Cursor, etc.),
[`ai-skills/`](ai-skills/) ships seven `SKILL.md` files your tool
can pick up so the assistant gives you idiomatic rakka-accel
guidance instead of guessing.

## Library coverage

| Library                            | Actor              | NVIDIA reference                                  | Feature flag |
|------------------------------------|--------------------|---------------------------------------------------|--------------|
| [cuBLAS][cublas]                   | `BlasActor`        | [`cublasSgemm`][cublas-sgemm]                     | always-on    |
| [cuBLASLt][cublaslt]               | `BlasLtActor`      | [`cublasLtMatmul` + epilogue][cublaslt-matmul]    | `cublaslt`   |
| [cuDNN][cudnn]                     | `CudnnActor`       | [`cudnnConvolutionForward`][cudnn-conv]           | `cudnn`      |
| [cuFFT][cufft]                     | `FftActor`         | [`cufftPlan1d`][cufft-plan] / [`cufftExecR2C`][cufft-exec] | `cufft` |
| [cuRAND][curand]                   | `RngActor`         | [`curandGenerateUniform`][curand-uniform]         | `curand`     |
| [cuSOLVER][cusolver]               | `SolverActor`      | [`cusolverDnSgeqrf`][cusolver-qr] / `Sgetrf` / `Spotrf` / `Sgesvd` / `Ssyevd` | `cusolver` |
| [cuSPARSE][cusparse]               | `SparseActor`      | [`cusparseSpMV`][cusparse-spmv] / `SpMM` (CSR)    | `cusparse`   |
| [cuTENSOR][cutensor]               | `TensorActor`      | [`cutensorContract`][cutensor-contract]           | `cutensor`   |
| [NVRTC][nvrtc]                     | `NvrtcActor`       | [`nvrtcCompileProgram`][nvrtc-compile]            | `nvrtc`      |
| [NCCL][nccl]                       | `CollectiveActor` + `NcclWorldActor` | [`ncclAllReduce`][nccl-allreduce] | `nccl` |
| [Pinned host memory][cuda-pinned]  | `PinnedBufferPool` | [`cuMemHostAlloc`][cuda-pinned-api]               | always-on    |
| [Unified memory][cuda-um]          | `ManagedAllocatorActor` | [`cudaMallocManaged`][cuda-um-api]           | always-on    |
| [CUDA Graphs][cuda-graph]          | `GraphActor`       | [`cuGraphInstantiate` / `cuGraphLaunch`][cuda-graph-api] | always-on |
| [Peer-to-peer][cuda-p2p]           | `P2pTopology`      | [`cuMemcpyPeerAsync`][cuda-memcpy-peer]           | always-on    |

Aggregate features:

- `core-libs` = `cudnn` + `cufft` + `curand` + `cusparse`
- `training-libs` = `core-libs` + `cusolver` + `cublaslt` + `nvrtc` + `cutensor`
- `full-cuda` = `training-libs` + `nccl`

## rakka 0.2 integrations

rakka-accel is feature-gated for each rakka subsystem so you only pay
for what you use:

- `replay` — persists replay-journal entries through any
  [`rakka_persistence::Journal`](../rakka/crates/rakka-persistence)
  (in-memory, SQL, Redis, MongoDB, Cassandra, Dynamo). Build a deterministic
  replay harness with one constructor: `ReplayHarness::with_journal(journal, "pid")`.
- `cluster` — `placement::sharded::PlacementShardingAdapter` exposes
  a typed [`EntityRef<DeviceExtractor>`][rakka-sharding] over
  rakka-cluster-sharding, so device routing follows consistent-hash
  placement across a cluster.
- `streams` — `streams_pipeline::{source_from_unbounded, gpu_stage,
  run_collect}` build GPU pipelines with rakka-streams Source / Sink
  alongside the actor-based `pipeline::PipelineExecutor`.
- `telemetry` — `observability::install(system, "node-1")` wires up a
  `TelemetryExtension` plus GPU-specific probes (allocations, OOM
  count, generation, VRAM, in-flight kernels). Visualize live in
  [`rakka-dashboard`](../rakka/crates/rakka-dashboard).
- Typed supervision — `error::DeviceSupervisor` implements
  `SupervisorOf<C>` over `GpuError`. Pattern-match the error type
  instead of parsing panic strings.
- `#[derive(Actor)]` from `rakka-macros` — eliminates async-trait
  boilerplate. Used by `BlasActor`, `EmbeddingCache`, `GpuMockActor`,
  `GpuHashMapActor`; opt-in for the rest.

## Blueprint sub-crates

These ride on top of the foundation and demonstrate concrete patterns:

- **`rakka-accel-patterns`** — `DynamicBatchingServer`,
  `InferenceCascade`, `ModelReplicaPool`, `FairShareScheduler` (WFQ),
  `ModelHotSwapServer`, `SpeculativeDecoder`, `MoeRouter`, plus a CPU
  `GpuMockActor` for tests.
- **`rakka-accel-train`** — `DataParallelTrainer`,
  `PipelineParallelTrainer`, `TensorParallelTrainer`,
  `AsyncParameterServer`, optimizer + loss enums.
- **`rakka-accel-agents`** — `RagPipeline` (with `EmbeddingCache` LRU
  + `CpuVectorIndex`), `SharedGpuStateCoordinator`,
  `LangGraphGpuActor` (DAG executor with cycle detection).
- **`rakka-accel-py`** — Python bindings via PyO3. `pip install
  maturin && maturin develop` from `crates/rakka-accel-py/`. Exposes
  `rakka_accel.{System, Device, GpuBuffer}` plus typed exceptions; see
  [`docs/python-bridge.md`](docs/python-bridge.md).
- **`rakka-accel-cuda-realtime`** — `ImageFilterPipeline`,
  `ParticleSystemActor`, `ClothSimulationActor`,
  `FluidSimulationActor`, `SpatialIndexActor`,
  `GpuHashMapActor`, `GpuSparseStructureActor`,
  `MultiPassAnalysisActor`, `VideoEffectsGraph`. Real CUDA-C kernel
  sources for these actors live under
  `crates/rakka-accel-cuda-realtime/kernels/`.

Every pattern ships a `*_no_gpu` example you can run today:

```bash
cargo run -p rakka-accel-patterns --example batching_no_gpu
cargo run -p rakka-accel-patterns --example cascade_no_gpu
cargo run -p rakka-accel-patterns --example fair_share_no_gpu
cargo run -p rakka-accel-patterns --example moe_no_gpu
cargo run -p rakka-accel-patterns --example speculative_no_gpu
```

## What you don't have to think about

- **Stream allocation.** Three strategies (`PerActorAllocator`,
  `SingleStreamAllocator`, `PooledAllocator`) ship out of the box;
  inject one and forget about it.
- **Kernel completion.** `HostFnCompletion` registers a
  [`cuLaunchHostFunc`][cuda-launch-host] callback that wakes the reply
  future the moment the kernel finishes — no host syncs, no polling.
- **Cross-stream events.** `GpuRef<T>` records its
  `last_write_stream`; downstream readers automatically wait on the
  right [event][cuda-events] before launching.
- **Context loss.** `WatchGeneration` is a
  `tokio::sync::watch::Receiver<u64>` you can subscribe to from any
  observer; we use it internally to rebuild NCCL communicators and
  invalidate P2P caches.
- **OS-thread pinning.** `GpuDispatcher` keeps the cuBLAS/cuDNN
  handle on a stable OS thread for its lifetime — required by
  several library APIs and easy to get wrong in async Rust.

## Build matrix

```bash
# No-GPU dev box:
cargo check --workspace --no-default-features
cargo check --workspace --features rakka-accel-cuda/core-libs
cargo check --workspace --features rakka-accel-cuda/training-libs
cargo check --workspace --features rakka-accel-cuda/full-cuda

# rakka 0.2 subsystem integrations:
cargo check --workspace --features rakka-accel-cuda/replay
cargo check --workspace --features rakka-accel-cuda/cluster
cargo check --workspace --features rakka-accel-cuda/streams
cargo check --workspace --features rakka-accel-cuda/telemetry

cargo test  -p rakka-accel --features replay --test replay_persistence

# GPU host (requires CUDA toolkit):
cargo run   -p rakka-accel --example sgemm        --features cuda-runtime-tests
cargo run   -p rakka-accel --example rng_uniform  --features cuda-runtime-tests,curand
cargo run   -p rakka-accel --example fft_1d       --features cuda-runtime-tests,cufft
cargo run   -p rakka-accel --example jit_relu     --features cuda-runtime-tests,nvrtc

cargo bench -p rakka-accel --bench sgemm_overhead --features cuda-runtime-tests
cargo bench -p rakka-accel --bench rng_throughput --features cuda-runtime-tests,curand

cargo test  -p rakka-accel --test sgemm_e2e          --features cuda-runtime-tests
cargo test  -p rakka-accel --test pinned_memcpy_e2e  --features cuda-runtime-tests
cargo test  -p rakka-accel --test end_to_end_e2e     --features cuda-runtime-tests
cargo test  -p rakka-accel --test spmv_e2e           --features cuda-runtime-tests,cusparse
cargo test  -p rakka-accel --test contract_e2e       --features cuda-runtime-tests,cutensor
cargo test  -p rakka-accel --test svd_e2e            --features cuda-runtime-tests,cusolver
```

## Picking the right deps

Each sub-crate path-depends only on `rakka-accel-cuda` (the foundation) —
no implicit pulls of the other blueprints. Add what you need:

```toml
# Just batching:
rakka-accel          = "0.0"
rakka-accel-patterns = "0.0"

# Training pipeline with NCCL + replay journal:
rakka-accel       = { version = "0.0", features = ["full-cuda", "replay"] }
rakka-accel-train = "0.0"

# Realtime sims with JIT kernels:
rakka-accel          = { version = "0.0", features = ["nvrtc"] }
rakka-accel-cuda-realtime = { version = "0.0", features = ["nvrtc"] }
```

[`docs/features-matrix.md`](docs/features-matrix.md) shows the full
pick-by-goal table plus the transitive-dependency view of every
feature.

Every sub-crate ships a `prelude` module:

```rust
use rakka_accel_cuda::prelude::*;            // foundation
use rakka_accel_patterns::prelude::*;   // batching, cascade, …
use rakka_accel_train::prelude::*;      // trainers, optimizers
use rakka_accel_agents::prelude::*;     // RAG, embedding cache
use rakka_accel_cuda_realtime::prelude::*;   // particles, cloth, sparse
```

## Status

`F2 – F9 implemented + rakka 0.2 adoption complete.` The full feature
matrix builds clean; 60+ tests pass on a no-GPU CI; the GPU-runtime
suite covers SGEMM, FFT, RNG, pinned memcpy, SpMV, tensor contraction,
SVD, and the multi-actor end-to-end smoke.

## Releasing

`v*.*.*` git tags trigger two CI pipelines: `release-crates.yml`
publishes the five Rust crates to crates.io in topological order;
`release-pypi.yml` builds wheels (manylinux + macOS + Windows) and
uploads to PyPI. See [`RELEASING.md`](RELEASING.md) for the
end-to-end flow.

See [`docs/architecture.md`](docs/architecture.md) for the design
narrative and [`CHANGELOG.md`](CHANGELOG.md) (forthcoming) for
release-by-release surface changes.

## License

Apache-2.0.

---

[cuda-ctx]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
[cuda-sticky]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking
[cuda-events]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events
[cuda-pinned]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory
[cuda-pinned-api]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9
[cuda-um]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-memory-programming
[cuda-um-api]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b
[cuda-graph]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
[cuda-graph-api]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html
[cuda-p2p]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#peer-to-peer-memory-access
[cuda-memcpy-peer]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g0e6a92f5c0a8c9d8a1c3d9a7e72b7d6e
[cuda-launch-host]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g05841eaa5f90f27264c5d9eb96b16d2c
[cublas]: https://docs.nvidia.com/cuda/cublas/index.html
[cublas-handle]: https://docs.nvidia.com/cuda/cublas/index.html#cublas-context
[cublas-sgemm]: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
[cublaslt]: https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api
[cublaslt-matmul]: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
[cudnn]: https://docs.nvidia.com/deeplearning/cudnn/api/index.html
[cudnn-handle]: https://docs.nvidia.com/deeplearning/cudnn/api/cudnn-ops-library.html#cudnncreate
[cudnn-conv]: https://docs.nvidia.com/deeplearning/cudnn/api/cudnn-cnn-library.html#cudnnconvolutionforward
[cufft]: https://docs.nvidia.com/cuda/cufft/index.html
[cufft-plan]: https://docs.nvidia.com/cuda/cufft/index.html#function-cufftplan1d
[cufft-exec]: https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecr2c
[curand]: https://docs.nvidia.com/cuda/curand/index.html
[curand-uniform]: https://docs.nvidia.com/cuda/curand/host-api-overview.html#generation-functions
[cusolver]: https://docs.nvidia.com/cuda/cusolver/index.html
[cusolver-qr]: https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf
[cusparse]: https://docs.nvidia.com/cuda/cusparse/index.html
[cusparse-spmv]: https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv
[cutensor]: https://docs.nvidia.com/cuda/cutensor/latest/index.html
[cutensor-contract]: https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontract
[nvrtc]: https://docs.nvidia.com/cuda/nvrtc/index.html
[nvrtc-compile]: https://docs.nvidia.com/cuda/nvrtc/index.html#group__error_1ga0e0b48c4e6f7e69dbb5e1d8c6c58c1d8
[nccl]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
[nccl-comm]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
[nccl-allreduce]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
[rakka-sharding]: ../rakka/crates/rakka-cluster-sharding
