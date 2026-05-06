# atomr-accel

An **actor-shaped face for compute acceleration**, on top of the
[atomr](https://github.com/rustakka/atomr) actor runtime. NVIDIA CUDA
ships today through [`atomr-accel-cuda`](crates/atomr-accel-cuda); the
backend trait surface accommodates AMD ROCm, Apple Metal, Intel
oneAPI, and Vulkan compute when those crates land. Each backend
library ([cuBLAS][cublas], [cuDNN][cudnn], [cuFFT][cufft],
[cuRAND][curand], [cuSOLVER][cusolver], [cuSPARSE][cusparse],
[cuTENSOR][cutensor], [cuBLASLt][cublaslt], [NVRTC][nvrtc],
[NCCL][nccl]) becomes a typed atomr actor with stable supervision,
generation-validated buffers, and a single async surface. Drop GPU
work into a Rust service without juggling streams, contexts, or
hand-rolled retry loops.

```rust
use atomr_accel_cuda::prelude::*;

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

## Why an actor-shaped face for compute, in Rust, now

Modern workloads no longer live entirely on the CPU. Inference,
embedding, scoring, simulation — they want a GPU. Coordination,
control flow, I/O, persistence — they want a CPU. Today's stacks
force you to glue the two with ad-hoc batching layers, queues, and
serialization boundaries.

The actor model already encodes the right boundary: a message **is**
the dispatch unit. atomr-accel is built so that the same
`actor_ref.tell(msg)` can target a CPU mailbox today and a CUDA-backed
dispatcher tomorrow — with the same supervision, the same
backpressure, the same observability. The runtime is explicit about
*where* work runs without forcing the developer to write two programs.

Writing CUDA from Rust today otherwise means owning a long list of
invariants yourself:

| You'd otherwise hand-roll                                                                      | atomr-accel gives you                                                                            |
| ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| One [`CUcontext`][cuda-ctx] per device, restarted on poisoning                                 | `DeviceActor ↔ ContextActor` two-tier supervision                                               |
| [Sticky-error][cuda-sticky] detection and graceful recovery                                    | `OneForOneStrategy` + `GpuError::ContextPoisoned` decider                                       |
| Buffer staleness across context rebuilds                                                       | `GpuRef<T>` with generation tokens                                                              |
| Pinning library handles ([cuBLAS][cublas-handle], [cuDNN][cudnn-handle]) to a single OS thread | `GpuDispatcher` + per-actor handle                                                              |
| [Stream-event][cuda-events] choreography for kernel completion                                 | `HostFnCompletion` (sub-µs `cuLaunchHostFunc`) / `SyncCompletion` / `PolledCompletion`          |
| [`cuMemcpyPeerAsync`][cuda-p2p] cross-stream synchronization                                   | `P2pTopology` with `last_write_stream` injection                                                |
| [Page-locked][cuda-pinned] host buffer pooling                                                 | `PinnedBufferPool` actor                                                                        |
| [CUDA Graph][cuda-graph] capture/replay                                                        | `GraphActor` (`Sgemm` / `Memcpy` / `RngFillUniform` / `FftR2C` record contracts)                |
| Multi-GPU [communicator rebuild][nccl-comm] on context loss                                    | `NcclWorldActor` subscribes to `WatchGeneration`, tears down + rebuilds collectives             |

Because every concern is an actor, you compose CUDA the same way you
compose any other Rust service: `tokio` runtime, structured
supervision, typed messages, async/await throughout.

## What's in the box

| Crate                       | What it does                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `atomr-accel`               | Backend-agnostic core — `AccelBackend` trait, `AccelRef<T>`, `AccelDtype`/`DType`, `AccelError`, `CompletionStrategy`. Each backend crate (e.g. `atomr-accel-cuda`) depends on this for its trait surface. |
| `atomr-accel-cuda`          | NVIDIA CUDA implementation — `DeviceActor`/`ContextActor`, kernel actors for cuBLAS/cuBLASLt/cuDNN/cuFFT/cuRAND/cuSOLVER/cuSPARSE/cuTENSOR/NVRTC/NCCL, P2P topology, CUDA graphs, pinned pools |
| `atomr-accel-patterns`      | Universal blueprints — `DynamicBatchingServer`, `InferenceCascade`, `ModelReplicaPool`, `FairShareScheduler`, `ModelHotSwapServer`, `SpeculativeDecoder`, `MoeRouter`, plus a CPU `GpuMockActor` |
| `atomr-accel-train`         | Distributed-training blueprints — `DataParallelTrainer`, `PipelineParallelTrainer`, `TensorParallelTrainer`, `AsyncParameterServer`, optimizer + loss enums |
| `atomr-accel-agents`        | LLM blueprints — `RagPipeline` (with `EmbeddingCache` LRU + `CpuVectorIndex`), `SharedGpuStateCoordinator`, `LangGraphGpuActor` (DAG executor with cycle detection) |
| `atomr-accel-cuda-realtime` | NVRTC-backed realtime sims — `ImageFilterPipeline`, `ParticleSystemActor`, `ClothSimulationActor`, `FluidSimulationActor`, `SpatialIndexActor`, `GpuHashMapActor`, `GpuSparseStructureActor`, `MultiPassAnalysisActor`, `VideoEffectsGraph` |
| `atomr-accel-cub`           | CUB device-wide primitives — `CubActor` with reduce / scan / sort / histogram / select / partition / segmented-reduce dispatchers, NVRTC-templated per `(op, dtype, length-class)` |
| `atomr-accel-cutlass`       | CUTLASS kernel-template instantiation — `CutlassActor` for GEMM, grouped-GEMM, implicit-GEMM convolution, EVT (epilogue visitor tree), via NVRTC against vendored headers |
| `atomr-accel-flashattn`     | FlashAttention v2 + v3 kernels — `FlashAttnActor` with forward/backward, paged KV-cache, chunked prefill, varlen, ALiBi, sliding window, sink tokens, MQA/GQA, fp8 (fa3 only) |
| `atomr-accel-tensorrt`      | TensorRT engine builder + runtime — `TrtActor`, `IBuilderConfig` (fp32/fp16/bf16/int8/fp8/best), ONNX import, INT8 calibration, FP8 PTQ, `IPluginV3` Rust trampolines |
| `atomr-accel-telemetry`     | Observability backends — `NvtxKernelTrace` for kernel-range markers, `NvmlActor` for power/temp/ECC/clocks, `CuptiSession` for activity tracing |
| `atomr-accel-py`            | Python bindings via PyO3 — `atomr_accel.{System, Device, GpuBuffer}`, typed exceptions, GIL-released kernel paths    |

Plus a Python facade — `pip install atomr-accel` — that exposes the
same actor model with numpy float32 roundtrip and mock-mode for tests.

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

## Quick start (Rust)

The umbrella crate is published on crates.io as **`atomr-accel`**:

```toml
[dependencies]
atomr-accel = { version = "0.1", features = ["cuda"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Or pull in subsystem crates directly — `atomr-accel-cuda`,
`atomr-accel-patterns`, `atomr-accel-train`, `atomr-accel-agents`,
`atomr-accel-cuda-realtime` are all on crates.io.

```rust
use atomr_accel_cuda::prelude::*;
use atomr::prelude::*;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::create("gpu-app", Config::empty()).await?;

    // Real-mode device. Use `DeviceConfig::mock(0)` for no-GPU CI.
    let device = system.actor_of(
        DeviceActor::props(DeviceConfig::new(0)),
        "device-0",
    )?;

    // Allocate, copy, dispatch — see docs/getting-started.md.

    system.terminate().await;
    Ok(())
}
```

cudarc loads CUDA dynamically, so the workspace **builds and
unit-tests on hosts without a GPU**. Real kernel paths are gated
behind `--features cuda-runtime-tests`.

```bash
# No GPU needed:
cargo check --workspace --no-default-features
cargo test  --workspace --no-default-features
cargo run   -p atomr-accel-cuda --example echo_no_gpu

# With GPU + CUDA toolkit:
cargo run   -p atomr-accel-cuda --example sgemm     --features cuda-runtime-tests
cargo run   -p atomr-accel-cuda --example fft_1d    --features cuda-runtime-tests,cufft
cargo run   -p atomr-accel-cuda --example jit_relu  --features cuda-runtime-tests,nvrtc
```

## Quick start (Python)

```bash
python -m venv .venv && source .venv/bin/activate
pip install atomr-accel
```

```python
from atomr_accel import System, Device, GpuBuffer

system = System.create_blocking("gpu-app")
device = system.device(0)        # mock-mode if no CUDA driver
buf = device.alloc_f32(1024)
buf.copy_from_numpy(np_input)
# ...kernel dispatch...
buf.copy_to_numpy(np_output)
system.terminate_blocking()
```

See [`docs/python-bridge.md`](docs/python-bridge.md) for the full
binding surface — `System`, `Device`, `GpuBuffer`, typed exceptions,
the GIL-release contract, and mock-mode tests.

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
- `core-libs` = `cudnn` + `cufft` + `curand` + `cusparse` + `cutensor` + `cuda-managed`.
- `training-libs` = `core-libs` + `cusolver` + `cublaslt` + `nvrtc`.
- `full-cuda` = `training-libs` + `nccl` + `cuda-ipc` + `graphs-conditional`.
- `observability-full` = `telemetry` + `nvtx-trace` + `nvml` + `cupti`.

Sibling-crate gates (off by default; pull each in by enabling the
matching feature on `atomr-accel-cuda`):

- `cutlass` (+ `cutlass-evt`, `cutlass-grouped`, `cutlass-prebuilt`).
- `flashattn` (+ `flashattn-fp8`, `flashattn-paged`).
- `tensorrt` (+ `tensorrt-onnx`, `tensorrt-plugin`, `tensorrt-int8`, `tensorrt-fp8`).
- `nvtx-trace`, `nvml`, `cupti` — Phase 9 telemetry backends, layered on `telemetry`.

## atomr integrations

atomr-accel is feature-gated for each atomr subsystem so you only pay
for what you use:

- `replay` — persists replay-journal entries through any
  [`atomr_persistence::Journal`](https://docs.rs/atomr-persistence)
  (in-memory, SQL, Redis, MongoDB, Cassandra, Dynamo). Build a
  deterministic replay harness with one constructor:
  `ReplayHarness::with_journal(journal, "pid")`.
- `cluster` — `placement::sharded::PlacementShardingAdapter` exposes
  a typed `EntityRef<DeviceExtractor>` over
  [`atomr-cluster-sharding`](https://docs.rs/atomr-cluster-sharding),
  so device routing follows consistent-hash placement across a cluster.
- `streams` — `streams_pipeline::{source_from_unbounded, gpu_stage,
  run_collect}` build GPU pipelines with
  [`atomr-streams`](https://docs.rs/atomr-streams) Source / Sink
  alongside the actor-based `pipeline::PipelineExecutor`.
- `telemetry` — `observability::install(system, "node-1")` wires up a
  `TelemetryExtension` plus GPU-specific probes (allocations, OOM
  count, generation, VRAM, in-flight kernels). Visualize live in
  [`atomr-dashboard`](https://github.com/rustakka/atomr/tree/main/crates/atomr-dashboard).
- Typed supervision — `error::DeviceSupervisor` implements
  `SupervisorOf<C>` over `GpuError`. Pattern-match the error type
  instead of parsing panic strings.
- `#[derive(Actor)]` from
  [`atomr-macros`](https://docs.rs/atomr-macros) — eliminates
  async-trait boilerplate.

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

## Building from source

You need a sibling clone of the
[atomr](https://github.com/rustakka/atomr) workspace next to this
repo (the workspace.dependencies in `Cargo.toml` reference
`../atomr`):

```
your-workspace/
├── atomr/         # the atomr actor runtime
└── atomr-accel/   # this repo
```

```bash
# Rust
cargo build --workspace
cargo test  --workspace --no-default-features

# The full release-pipeline gate (fmt + clippy + test + multi-feature check + doc)
cargo xtask verify

# Python bindings (requires maturin + a Python dev toolchain)
cd crates/atomr-accel-py
maturin develop --release
pytest tests/ -v
```

GPU-host integration tests are **opt-in** and **not part of CI**. On a
CUDA-equipped workstation:

```bash
cargo xtask gpu-probe          # report local CUDA + library availability
cargo xtask gpu-test            # run all suites
cargo xtask gpu-test cublas     # run one suite
cargo xtask gpu-bench           # criterion perf-regression benches
```

Tests skip gracefully when the local driver / library / GPU isn't
present, so the same commands are safe on a no-GPU laptop. See
[`docs/gpu-testing.md`](docs/gpu-testing.md) for the full suite list,
the gating model (cargo feature + `#[ignore]` + runtime probe), and
the rationale for keeping these tests out of CI.

## Build matrix

```bash
# No-GPU dev box:
cargo check --workspace --no-default-features
cargo check --workspace --features atomr-accel-cuda/core-libs
cargo check --workspace --features atomr-accel-cuda/training-libs
cargo check --workspace --features atomr-accel-cuda/full-cuda

# atomr subsystem integrations:
cargo check --workspace --features atomr-accel-cuda/replay
cargo check --workspace --features atomr-accel-cuda/cluster
cargo check --workspace --features atomr-accel-cuda/streams
cargo check --workspace --features atomr-accel-cuda/telemetry

cargo test  -p atomr-accel-cuda --features replay --test replay_persistence

# GPU host (requires CUDA toolkit):
cargo run   -p atomr-accel-cuda --example sgemm        --features cuda-runtime-tests
cargo run   -p atomr-accel-cuda --example rng_uniform  --features cuda-runtime-tests,curand
cargo run   -p atomr-accel-cuda --example fft_1d       --features cuda-runtime-tests,cufft
cargo run   -p atomr-accel-cuda --example jit_relu     --features cuda-runtime-tests,nvrtc

cargo bench -p atomr-accel-cuda --bench sgemm_overhead --features cuda-runtime-tests
cargo bench -p atomr-accel-cuda --bench rng_throughput --features cuda-runtime-tests,curand
```

## Picking the right deps

Each sub-crate path-depends only on `atomr-accel-cuda` (the foundation) —
no implicit pulls of the other blueprints. Add what you need:

```toml
# Just batching:
atomr-accel          = "0.1"
atomr-accel-patterns = "0.1"

# Training pipeline with NCCL + replay journal:
atomr-accel       = { version = "0.1", features = ["full-cuda", "replay"] }
atomr-accel-train = "0.1"

# Realtime sims with JIT kernels:
atomr-accel               = { version = "0.1", features = ["nvrtc"] }
atomr-accel-cuda-realtime = { version = "0.1", features = ["nvrtc"] }
```

[`docs/features-matrix.md`](docs/features-matrix.md) shows the full
pick-by-goal table plus the transitive-dependency view of every
feature.

Every sub-crate ships a `prelude` module:

```rust
use atomr_accel_cuda::prelude::*;            // foundation
use atomr_accel_patterns::prelude::*;        // batching, cascade, …
use atomr_accel_train::prelude::*;           // trainers, optimizers
use atomr_accel_agents::prelude::*;          // RAG, embedding cache
use atomr_accel_cuda_realtime::prelude::*;   // particles, cloth, sparse
```

If you're using an AI coding assistant (Claude Code, Cursor, etc.),
[`ai-skills/`](ai-skills/) ships ten `SKILL.md` files your tool can
pick up so the assistant gives you idiomatic atomr-accel guidance
instead of guessing.

## Layout

```
crates/                       Rust workspace
crates/atomr-accel/           Backend-agnostic core (umbrella)
crates/atomr-accel-cuda/      NVIDIA CUDA implementation
crates/atomr-accel-patterns/  Universal blueprints (batching / cascade / scheduler / …)
crates/atomr-accel-train/     Distributed-training blueprints
crates/atomr-accel-agents/    LLM blueprints (RAG / DAG)
crates/atomr-accel-cuda-realtime/  NVRTC-backed realtime sims
crates/atomr-accel-cub/       CUB device-wide primitives (Phase 5)
crates/atomr-accel-cutlass/   CUTLASS templates via NVRTC (Phase 6)
crates/atomr-accel-flashattn/ FlashAttention v2 + v3 kernels (Phase 7)
crates/atomr-accel-tensorrt/  TensorRT engine builder + runtime (Phase 8)
crates/atomr-accel-telemetry/ NVTX / NVML / CUPTI observability (Phase 9)
crates/atomr-accel-py/        PyO3 bridge (Python module: atomr_accel)
ai-skills/                    Vendor-neutral SKILL.md files for AI assistants
docs/                         Architecture, getting-started, concepts, features-matrix, gpu-testing
xtask/                        Cargo xtask (bump, verify, gpu-probe, gpu-test, gpu-bench)
```

## Status

Phases 0 – 9 of the CUDA-coverage roadmap are merged. The workspace
ships **twelve library crates** spanning the foundation actor surface
(`atomr-accel`, `atomr-accel-cuda`), the blueprint sub-crates
(`atomr-accel-patterns`, `atomr-accel-train`, `atomr-accel-agents`,
`atomr-accel-cuda-realtime`, `atomr-accel-py`), Phase 1 – 4 library
expansions (full cuBLAS / cuBLASLt / cuFFT / cuRAND / cuSOLVER dtype
matrix, cuDNN frontend graph, NCCL collective set, cuTENSOR
contraction + reduce + permute, cuSPARSE generic API + cuSPARSELt
2:4), Phase 5 foundations (NVRTC v2 + Hopper/Blackwell +
`atomr-accel-cub`), and Phase 6 – 9 sibling crates
(`atomr-accel-cutlass`, `atomr-accel-flashattn`,
`atomr-accel-tensorrt`, `atomr-accel-telemetry`).

The full feature matrix builds clean on a no-GPU host. ≈ 175 unit
tests pass with the headline feature combo
(`f16,cudnn,curand,cufft,nvrtc,cusolver,cusparse,cusparse-generic,cutensor,cublaslt,nccl,nvtx,cuda-ipc,cuda-managed,graphs-conditional`).
The opt-in GPU integration suite — invoked via `cargo xtask gpu-test`
— covers SGEMM, FFT, RNG, pinned memcpy, SpMV, tensor contraction,
SVD, the dispatch tables for FlashAttention / CUTLASS / CUB, and
real NVML probes against installed devices. See
[`docs/gpu-testing.md`](docs/gpu-testing.md) for the suite catalog
and the rationale for keeping it out of CI.

## Releasing

`v*.*.*` git tags trigger a single `release.yml` pipeline that runs
the verify gate, builds Python wheels (manylinux x86_64, musllinux
x86_64, macOS universal2, Windows x86_64) + an sdist, creates a
GitHub Release, publishes the workspace crates to crates.io in
topological order, and uploads wheels + sdist to PyPI via trusted
publishing. See [`RELEASING.md`](RELEASING.md) for the end-to-end
flow.

## Learn more

- [`docs/getting-started.md`](docs/getting-started.md) — the
  ten-minute tour: wiring atomr-accel into a project, picking
  features, no-GPU vs GPU paths.
- [`docs/concepts.md`](docs/concepts.md) — the five mental models
  (supervision, generation tokens, completion, streams, watch).
- [`docs/architecture.md`](docs/architecture.md) — the full design
  narrative.
- [`docs/backends.md`](docs/backends.md) — the multi-backend trait
  abstraction (and the ROCm / Metal / oneAPI roadmap).
- [`docs/features-matrix.md`](docs/features-matrix.md) — pick the
  smallest dep footprint that fits your goal.
- [`docs/python-bridge.md`](docs/python-bridge.md) — Python bindings
  surface and GIL strategy.
- [`docs/gpu-testing.md`](docs/gpu-testing.md) — opt-in GPU
  integration suite, the three-layer gating model, and why the suite
  is intentionally not part of CI.
- [`ai-skills/README.md`](ai-skills/README.md) — install the skill
  bundle into Claude Code, Cursor, Codex CLI, Gemini CLI, or any
  harness that reads `SKILL.md`. Covers the foundation actors plus
  per-crate skills for FlashAttention, CUTLASS, and TensorRT.
- [`RELEASING.md`](RELEASING.md) — release pipeline, secrets,
  yanking, post-release verification.

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
