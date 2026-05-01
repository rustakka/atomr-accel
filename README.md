# rakka-cuda

GPU acceleration via the actor model — wraps NVIDIA CUDA libraries as
specialized actors on top of the [rakka](../rakka) actor runtime.

## Prerequisite: sibling rakka workspace

The crates in this workspace path-depend on `rakka` at `../rakka`.
Clone both side-by-side before building:

```
your-workspace/
├── rakka/         # the rakka actor runtime
└── rakka-cuda/    # this repo
```

A future revision will switch to a git/version dependency once
`rakka` is published.

## Status

**F2 – F8 implemented.** The workspace ships a working actor system
exposing the major CUDA libraries plus the four blueprint sub-crates
populated with concrete patterns and reference implementations.

### Foundation (`rakka-cuda` crate)

- Two-tier `DeviceActor` ↔ `ContextActor` supervision with
  panic-string-tagged restart (`ContextPoisoned`/`OutOfMemory`/
  `Unrecoverable`), 3 retries / 60 s circuit breaker.
- `GpuRef<T>` with generation tokens, `device_id`, `record_write`,
  and `last_write_stream` for cross-stream event injection.
- `GpuDispatcher` pinning execution to a single OS thread.
- Three completion strategies: `HostFnCompletion` (default,
  sub-microsecond `cuLaunchHostFunc`), `SyncCompletion`,
  `PolledCompletion` (real `cuEventQuery` polling with timeout).
- Three stream allocators: `PerActorAllocator` (Shared / Fresh modes),
  `SingleStreamAllocator`, `PooledAllocator` (real round-robin).
- `KernelEnvelope::run_kernel` factoring out the validate / enqueue /
  await / reply / drop-keep-alive sequence.
- Typed `DeviceMsg::AllocateF32 / F64 / I8 / I32 / I64 / U8 / U32 / U64`
  (and feature-gated `F16` / `Bf16`); per-dtype `CopyToHost*` /
  `CopyFromHost*` memcpy variants accepting either `Vec<T>` or
  `PinnedBuf<T>`.
- `DeviceMsg::SnapshotContext` / `SnapshotChildren` / `Stats` /
  `WatchGeneration` accessors.

### Per-library kernel actors

| Library   | Actor              | Feature flag |
|-----------|--------------------|--------------|
| cuBLAS    | `BlasActor`        | always-on    |
| cuDNN     | `CudnnActor`       | `cudnn`      |
| cuFFT     | `FftActor`         | `cufft`      |
| cuRAND    | `RngActor`         | `curand`     |
| cuSOLVER  | `SolverActor` (real QR / LU / LU-solve / Cholesky via sys::) | `cusolver` |
| cuBLASLt  | `BlasLtActor` (fused matmul + Relu/Gelu) | `cublaslt` |
| NVRTC     | `NvrtcActor` (Compile + Launch with generation-validated `KernelHandle`) | `nvrtc` |
| NCCL      | `CollectiveActor` + `NcclWorldActor` (multi-device with cross-validation + rebuild on `ContextLost`) | `nccl` |

Aggregate features: `core-libs` (cuDNN + cuFFT + cuRAND),
`training-libs` (`core-libs` + cuSOLVER + cuBLASLt + NVRTC),
`full-cuda` (`training-libs` + NCCL).

cuSPARSE / cuTENSOR are feature-gated empty placeholders pending
cudarc safe wrappers.

### Patterns layer (`rakka-cuda` core modules)

- `host::PinnedBufferPool` — page-locked host buffer pool.
- `pipeline::PipelineExecutor` (typed 2-stage) and
  `PipelineExecutorN` (heterogeneous N-stage with `Box<dyn Any>`
  type-erasure). `spawn_pipeline` returns
  `(PipelineSink<I>, PipelineSource<O>)` over bounded `mpsc` for
  backpressure.
- `graph::GraphActor` — CUDA Graph capture + replay. `RecordMode`
  contracts for `Sgemm`, `Memcpy`, `RngFillUniform`, `FftR2C`.
- `p2p::P2pTopology` — `cuDeviceCanAccessPeer` probe,
  `cuCtxEnablePeerAccess`, async `cuMemcpyPeerAsync` with
  cross-stream event injection from `last_write_stream`.
- `memory::ManagedAllocatorActor` + `ManagedRef<T>` — real
  `cudaMallocManaged` with host-slice access.
- `placement::PlacementActor` — `LeastLoadedPolicy` and
  `RoundRobinPolicy`; periodic `Stats` poll feedback loop.
- `replay::ReplayHarness` — Record / Replay / Off modes with
  `LoadJournal` + `replay_via_sink` for streaming a recorded
  journal through a typed sink actor.

### Blueprint sub-crates

- `rakka-cuda-patterns`: `DynamicBatchingServer`,
  `InferenceCascade`, `ModelReplicaPool`, `FairShareScheduler`
  (WFQ), `ModelHotSwapServer`, `SpeculativeDecoder`, `MoeRouter`,
  `GpuMockActor` (Sgemm / Conv / FftR2C / RngFill).
- `rakka-cuda-train`: `DataParallelTrainer`,
  `PipelineParallelTrainer`, `TensorParallelTrainer`,
  `AsyncParameterServer`, `OptimizerKind` (SGD / AdamW),
  `LossKind`.
- `rakka-cuda-agents`: `RagPipeline` (with EmbeddingCache + vector
  index), `EmbeddingCache` (LRU), `CpuVectorIndex`,
  `SharedGpuStateCoordinator` (FIFO write-token + ManagedRef
  Snapshot), `LangGraphGpuActor` (DAG executor with cycle
  detection).
- `rakka-cuda-realtime`: `ImageFilterPipeline` (CPU + cuDNN
  paths), `GpuHashMapActor`, `ParticleSystemActor`,
  `SpatialIndexActor`, `ReductionAnalysisActor`,
  `MultiPassAnalysisActor`, `VideoEffectsGraph`,
  `ClothSimulationActor`, `FluidSimulationActor`,
  `GpuSparseStructureActor`.

## Build

cudarc's default features include `fallback-dynamic-loading`, so
the workspace compiles on hosts without the CUDA SDK. Runtime GPU
paths are gated behind `cuda-runtime-tests`.

```bash
# Dev box (no GPU required):
cargo check --workspace --no-default-features
cargo test  --workspace --no-default-features
cargo run   -p rakka-cuda          --example echo_no_gpu
cargo run   -p rakka-cuda-patterns --example batching_no_gpu
cargo run   -p rakka-cuda-patterns --example cascade_no_gpu
cargo run   -p rakka-cuda-patterns --example fair_share_no_gpu
cargo run   -p rakka-cuda-patterns --example speculative_no_gpu
cargo run   -p rakka-cuda-patterns --example moe_no_gpu

# Aggregate feature builds:
cargo check --workspace --features rakka-cuda/core-libs
cargo check --workspace --features rakka-cuda/training-libs
cargo check --workspace --features rakka-cuda/full-cuda

# GPU host:
cargo run   -p rakka-cuda --example sgemm       --features cuda-runtime-tests
cargo run   -p rakka-cuda --example rng_uniform --features cuda-runtime-tests,curand
cargo run   -p rakka-cuda --example fft_1d      --features cuda-runtime-tests,cufft
cargo run   -p rakka-cuda --example jit_relu    --features cuda-runtime-tests,nvrtc

cargo bench -p rakka-cuda --bench sgemm_overhead --features cuda-runtime-tests
cargo bench -p rakka-cuda --bench rng_throughput --features cuda-runtime-tests,curand

cargo test  -p rakka-cuda --test sgemm_e2e       --features cuda-runtime-tests
cargo test  -p rakka-cuda --test rng_fill_e2e    --features cuda-runtime-tests,curand
cargo test  -p rakka-cuda --test pinned_memcpy_e2e --features cuda-runtime-tests
cargo test  -p rakka-cuda --test end_to_end_e2e  --features cuda-runtime-tests
```

## Deviations from the original architecture document

Four rakka API differences are baked into the implementation:

1. `ActorRef<DeviceActor>` is `ActorRef<DeviceMsg>` — rakka
   parameterises actor refs by message type.
2. `OneForOneStrategy::new().with_decider(|m| ...)` matches a panic
   message string instead of a typed error.
3. `async fn handle(...)` returns `()`; restarts are signalled by
   panicking with a recognisable message.
4. `ctx.spawn_child(...)` is `ctx.spawn(...)`.

## Outstanding upstream-blocked items

- cuSPARSE / cuTENSOR safe wrappers (need cudarc upstream).
- cuSOLVER SVD / eigen / sparse (cudarc only exposes handle
  management; full sys-level wrapping pending).
- Cluster-sharding placement integration (rakka-cluster-sharding
  glue).
- ReplayHarness `rakka-persistence` backend.
- Custom NVRTC kernel sources for `GpuHashMapActor` /
  `ParticleSystemActor` (CPU references currently shipped).
- Dashboard / observability layer.
