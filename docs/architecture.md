# Architecture

The full design narrative. If you're trying to decide whether
atomr-accel fits your workload, this is the document.

For the API tour, read [getting-started.md](getting-started.md).
For the conceptual model, read [concepts.md](concepts.md).

---

## Goal

Make the [NVIDIA CUDA Toolkit][cuda-toolkit] a first-class citizen in
a long-running Rust service. Every CUDA primitive — handles,
streams, contexts, events, graphs, communicators — has a lifetime
constraint that breaks down badly under naïve concurrency. We want
those constraints expressed once, in one place, and surfaced through
a typed message API that's hard to misuse.

The actor model is a remarkably good fit. Each CUDA library
([cuBLAS][cublas], [cuDNN][cudnn], [cuFFT][cufft],
[cuRAND][curand], [cuSOLVER][cusolver], [cuSPARSE][cusparse],
[cuTENSOR][cutensor]) wants:

- **One handle** that's logically owned by a single thread (the
  [cuBLAS context][cublas-handle] is documented as not thread-safe in
  the strict sense; the docs describe it as "thread-friendly" if each
  thread uses a distinct handle).
- **One stream** to serialize work against that handle.
- **One restart unit** when the underlying [`CUcontext`][cuda-ctx]
  poisons.

Those are exactly the invariants of a atomr actor: a single
`handle` method, one mailbox, one supervisor.

## Design principles

1. **Every library is an actor.** No exceptions. The same envelope
   wraps cuBLAS, cuDNN, cuFFT, cuSOLVER, cuSPARSE, cuTENSOR,
   cuBLASLt, NVRTC, and NCCL.
2. **GPU pointers are typed.** `GpuRef<T>` carries `T`,
   `device_id`, and a generation token. The compiler catches dtype
   mismatches; the runtime catches generation mismatches.
3. **Failures travel as panics with tagged messages.** Receive-side
   parsing maps to typed [`Directive`s][atomr-directive]
   (Restart / Resume / Stop / Escalate). The transport is a
   constraint of the atomr `Actor` trait, not a deliberate choice.
4. **Cudarc safe layer where it exists; `cudarc::*::sys` where it
   doesn't.** cuSPARSE / cuTENSOR / advanced cuSOLVER are wrapped at
   the `sys::` layer until cudarc upstream catches up. The actor
   surface is identical either way.
5. **Compose by feature flag.** Defaults stay tight (cuBLAS only).
   Aggregates (`core-libs`, `training-libs`, `full-cuda`) match
   common deployment shapes.
6. **Build on no-GPU hosts.** Cudarc loads CUDA dynamically; mock
   actor variants let unit tests run on CI without a driver.

## The two-tier device model

CUDA has a long-standing
[sticky-error][cuda-sticky] problem: once a kernel hits certain
errors (illegal memory access, asynchronous failure surfaced via
`cudaGetLastError`), the context is poisoned and every subsequent
call returns the same error. The only recovery is to tear the
context down and rebuild it.

```
┌─────────────────── DeviceActor ───────────────────┐
│ stable address: ActorRef<DeviceMsg>               │
│ pending: VecDeque<WorkRequest>                    │
│ state:   Arc<DeviceState>                         │
│                                                   │
│   ┌───────────── ContextActor ─────────────┐      │
│   │ owns: Arc<CudaContext>                 │      │
│   │ generation: bumped on every restart    │      │
│   │                                        │      │
│   │   ┌─ BlasActor    (cuBLAS handle)      │      │
│   │   ├─ CudnnActor   (cuDNN handle)       │      │
│   │   ├─ FftActor     (cuFFT plans)        │      │
│   │   ├─ RngActor     (cuRAND generator)   │      │
│   │   └─ ...                                │      │
│   └────────────────────────────────────────┘      │
└───────────────────────────────────────────────────┘
                       │
                       ▼
                 DeviceState
                 ┌─────────────────────────────┐
                 │ generation: AtomicU64       │
                 │ accepting_ops: AtomicBool   │
                 │ current_ctx: ArcSwapOption  │
                 │ generation_watch: tokio     │
                 │   ::sync::watch::Sender<u64>│
                 └─────────────────────────────┘
```

The supervision contract:

- `DeviceActor` spawns `ContextActor` in `pre_start`.
- `ContextActor` calls `cuInit` / [`cuCtxCreate`][cuda-ctxcreate],
  installs the result in `Arc<DeviceState>`, bumps the generation,
  spawns the per-library actors, and tells the parent `ContextReady`.
- `DeviceActor` drains its pending queue into the kernel actors.
- If a kernel panics with `"ContextPoisoned: …"`, the supervisor
  restarts the `ContextActor`. The new incarnation rebuilds the
  context from scratch.
- Three restarts inside a one-minute window opens the circuit and
  stops the device.

This is the [Erlang/OTP supervision tree][otp-sup] applied to CUDA.
The novel part is what survives the restart: `Arc<DeviceState>` and
every `GpuRef<T>` minted against it. The state's generation token is
how we know which `GpuRef`s went stale.

## The `KernelEnvelope`

Every library actor's `handle` is a thin shell. The shared body lives
in `kernel::envelope::run_kernel`:

```rust
pub fn run_kernel<O, KA, F>(
    lib_tag: &'static str,
    stream: &Arc<CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    output: O,
    reply: oneshot::Sender<Result<O, GpuError>>,
    enqueue: F,
)
where
    F: FnOnce() -> Result<KA, GpuError>,
    KA: Send + 'static,
{
    // 1. Run `enqueue` synchronously. It validates GpuRefs, calls the
    //    library entry point, and returns a "keep-alive" tuple.
    let keep_alive = match enqueue() { ... };
    // 2. Spawn a Tokio task that awaits the completion strategy.
    tokio::spawn(async move {
        let res = completion.await_completion(stream).await;
        let _ = reply.send(res.map(|_| output));
        drop(keep_alive); // only now do inputs become collectible
    });
}
```

This is the only place the workspace handles kernel completion.
Adding a new library means filling in the `enqueue` closure.

For `BlasActor::Sgemm` the closure calls
[`cublasSgemm_v2`][cublas-sgemm] via the cudarc safe wrapper. For
`SparseActor::SpMv` it calls
[`cusparseSpMV`][cusparse-spmv] via `cudarc::cusparse::sys`. Same
envelope, same supervision behaviour, same completion semantics.

## Memory: pinned, managed, and device-only

Three allocator stories, three actors:

- **Device-only:** `DeviceMsg::AllocateF32 { len, reply }` returns a
  `GpuRef<f32>`. The buffer lives on the GPU; copying to/from the
  host goes through `CopyToHostF32` / `CopyFromHostF32` with a
  `HostBuf` envelope.
- **Page-locked host (pinned):** `PinnedBufferPool` calls
  [`cuMemHostAlloc`][cuda-pinned-api] up to a configured cap. Pinned
  buffers participate in async H2D / D2H copies without a bounce
  buffer; `HostBuf::Pinned(buf)` is what you get back from the pool
  and what you pass into `CopyFromHost*`.
- **Unified memory (managed):** `ManagedAllocatorActor` calls
  [`cudaMallocManaged`][cuda-um-api]. The returned `ManagedRef<T>`
  exposes both a host slice and a device pointer; the driver
  migrates pages on demand. Useful for "shared state" patterns where
  CPU and GPU need to peek at the same data.

## Streams and events

Each kernel actor gets a stream from a `StreamAllocator`. Three
implementations:

- `PerActorAllocator` (`Shared` or `Fresh` mode) — the default.
- `SingleStreamAllocator` — every actor uses the same stream;
  serializes everything but eliminates cross-stream synchronization.
- `PooledAllocator` — round-robin across a fixed pool.

When two actors on different streams touch the same buffer, the
pipeline injects a [CUDA event][cuda-events] automatically:
`GpuRef::record_write(stream)` after the writer enqueues, and the
reader's enqueue path calls `dst_stream.wait(&event)` against the
recorded event. No host-side `cudaStreamSynchronize` is involved.

The default `CompletionStrategy` (`HostFnCompletion`) uses
[`cuLaunchHostFunc`][cuda-launch-host] to fire a callback the moment
the stream drains. The callback signals the Tokio reply oneshot.
End-to-end latency from "kernel finishes" to "reply future wakes" is
sub-microsecond on commodity hardware.

## Multi-GPU: `P2pTopology` and `NcclWorldActor`

`P2pTopology` probes
[`cuDeviceCanAccessPeer`][cuda-p2p-probe] for every device pair, calls
[`cuCtxEnablePeerAccess`][cuda-p2p-enable] in successful directions,
and exposes the resulting `P2pGraph`. `CopyF32 { src, dst, … }` uses
[`cuMemcpyPeerAsync`][cuda-memcpy-peer] on the destination stream
with cross-stream event injection from the source's
`last_write_stream`.

`NcclWorldActor` is the supervisor for an N-GPU NCCL world:

1. Spawn N `DeviceActor`s.
2. After each reports `ContextReady`, snapshot its
   `Arc<CudaContext>`.
3. Mint one stream per device.
4. Call [`ncclCommInitRank`][nccl-init] (via cudarc's
   `Comm::from_devices`).
5. Spawn one `CollectiveActor` per rank.
6. Subscribe to each device's `WatchGeneration`. If any device
   rebuilds, tear down the [communicator][nccl-comm] and rebuild
   from step 1.

Application code calls `world.tell(NcclWorldMsg::AllReduceF32 {
tensors, op, reply })` and the world fans out
[`ncclAllReduce`][nccl-allreduce] to every rank, joining the
per-rank replies into a single result.

## Replay: deterministic GPU pipelines

`ReplayHarness` records every "interesting" event (allocation, kernel
launch, RNG seed, batch size) into a journal. In `Replay` mode it
streams the journal back through a user-supplied
`ReplaySink<Msg>` so the system replays bit-for-bit. With the
`replay` feature on, `ReplayHarness::with_journal(journal,
"persistence-id")` round-trips every entry through any
[`atomr_persistence::Journal`][atomr-persistence] backend (in-memory,
SQL, Redis, MongoDB, Cassandra, Dynamo). Crash, restart, replay.

## Cluster-aware placement

`PlacementActor` polls each device's `Stats` (free VRAM, queue depth,
active streams) and picks one per request via a `PlacementPolicy`
(`RoundRobinPolicy` or `LeastLoadedPolicy`).

With the `cluster` feature, `placement::sharded::PlacementShardingAdapter`
adapts the placement layer to atomr's
[cluster-sharding][atomr-sharding]: every request is routed by a
typed [`EntityRef<DeviceExtractor>`][atomr-entity-ref] using a
consistent-hash on the entity id. Cross-node routing follows the
shard owner; intra-node routing hits the local device fleet.

## Observability

With the `telemetry` feature, `observability::install(system,
"node-1")` registers a `TelemetryExtension` plus four GPU-specific
probes:

- `allocations_total` / `oom_total` — counters.
- `max_generation_observed` — bumps on context restart.
- `kernels_in_flight` / `kernels_total` — gauge + counter.
- `vram_free_bytes` / `vram_total_bytes` — last poll snapshot.

The same `TelemetryExtension` ships actor-level, cluster-level,
sharding-level, persistence-level, and stream-level probes from
upstream atomr. The [`atomr-dashboard`][atomr-dashboard] SPA
visualizes the resulting `NodeSnapshot` over WebSocket — pointing it
at any atomr-accel host shows the GPU panels alongside the standard
ones.

## Crate layout

```
crates/
├── atomr-accel-cuda/         # foundation: actors, supervision, GpuRef, streams,
│                            # completion, kernel envelope, library actors,
│                            # P2pTopology, GraphActor, ReplayHarness, …
├── atomr-accel-patterns/     # universal blueprints
│   ├── batching.rs          # DynamicBatchingServer
│   ├── cascade.rs           # InferenceCascade
│   ├── replica_pool.rs      # ModelReplicaPool
│   ├── scheduler.rs         # FairShareScheduler (WFQ)
│   ├── hot_swap.rs          # ModelHotSwapServer
│   ├── speculative.rs       # SpeculativeDecoder
│   ├── moe.rs               # MoeRouter
│   └── mock.rs              # GpuMockActor (CPU stand-in)
├── atomr-accel-train/        # distributed training blueprints
│   ├── data_parallel.rs
│   ├── pipeline_parallel.rs
│   ├── tensor_parallel.rs
│   ├── parameter_server.rs
│   ├── optimizer.rs
│   └── loss.rs
├── atomr-accel-agents/       # agentic / LLM blueprints
│   ├── rag.rs
│   ├── embedding_cache.rs
│   ├── vector_index.rs
│   ├── shared_state.rs
│   └── langgraph_nodes.rs
└── atomr-accel-cuda-realtime/     # interactive-rate blueprints
    ├── kernels/             # CUDA-C kernel sources (NVRTC-compiled)
    │   ├── coo_spmv.cu
    │   ├── particle_step.cu
    │   ├── cloth_springs.cu
    │   └── hashmap_probe.cu
    └── src/
        ├── image_filter.rs
        ├── particle.rs
        ├── cloth.rs
        ├── fluid.rs
        ├── hashmap.rs
        ├── sparse.rs
        ├── spatial_index.rs
        ├── multi_pass.rs
        ├── reduction.rs
        └── video_effects.rs
```

## Trade-offs

- **Actor overhead.** Each request crosses the mailbox: an Arc clone,
  a channel send, a Tokio wakeup. For large kernels that's noise.
  For sub-100ns dispatch (e.g. tight RNG loops), measure first;
  consider `SingleStreamAllocator` to keep everything on one actor.
- **Panic-as-failure.** Restarts go through `panic!` because
  `Actor::handle` returns `()`. This is atomr's choice, not ours.
  Application code should never have to write or catch these
  panics; it interacts with `Result<_, GpuError>` exclusively.
- **`sys::`-level FFI for cuSPARSE / cuTENSOR / parts of cuSOLVER.**
  Until cudarc upstream ships safe wrappers, those library actors
  are several hundred lines of `unsafe` per op. The unsafety is
  bounded — every actor contains the FFI; the public surface is
  safe — but reviewing those files demands more attention than the
  cuBLAS / cuDNN ones.
- **No global allocator override.** Each actor owns its allocations.
  If you need a per-process arena, build one as another actor.

## Status & roadmap

`F2 – F9 implemented + atomr 0.2 adoption complete.` The project's
internal phase plan is "F"-numbered (foundation phases F1 through
F10):

- **F1** — `BlasActor` end-to-end (shipped).
- **F2** — generic kernel envelope, typed allocations (shipped).
- **F3** — `SolverActor`, `BlasLtActor`, `NvrtcActor` (shipped).
- **F4** — `CollectiveActor` + `NcclWorldActor` (shipped).
- **F5** — `PlacementActor` + `ManagedAllocatorActor` (shipped).
- **F6** — `ParticleSystemActor` and friends (shipped, CPU
  reference; GPU NVRTC sources bundled, full dispatch in F7+).
- **F7** — `ImageFilterPipeline` + `GraphActor` (shipped).
- **F8** — `ReplayHarness`, `ClothSimulationActor`,
  `GpuSparseStructureActor` (shipped).
- **F9** — `WatchGeneration` subscription, P2P cross-stream events,
  `SparseActor` (cuSPARSE), `TensorActor` (cuTENSOR), cuSOLVER
  SVD/Syevd (shipped).
- **F10** — host-fn completion replacing the last
  `dst_stream.synchronize()` in `P2pTopology::CopyF32`; per-actor
  NVRTC dispatch wiring for the remaining realtime actors;
  cuSOLVER sparse decompositions when cudarc adds them.

[cuda-toolkit]: https://developer.nvidia.com/cuda-toolkit
[cuda-ctx]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
[cuda-ctxcreate]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
[cuda-sticky]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking
[cuda-events]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events
[cuda-pinned-api]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g572ca4011bfcb25034888a14d4e035b9
[cuda-um-api]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
[cuda-launch-host]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g05841eaa5f90f27264c5d9eb96b16d2c
[cuda-p2p-probe]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g496bdaae1f632ebfb695b99d2c40f19e
[cuda-p2p-enable]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0ee2ee9d2c8ff4e7e3175b1d34e36f60
[cuda-memcpy-peer]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g0e6a92f5c0a8c9d8a1c3d9a7e72b7d6e
[cublas]: https://docs.nvidia.com/cuda/cublas/index.html
[cublas-handle]: https://docs.nvidia.com/cuda/cublas/index.html#cublas-context
[cublas-sgemm]: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
[cudnn]: https://docs.nvidia.com/deeplearning/cudnn/api/index.html
[cufft]: https://docs.nvidia.com/cuda/cufft/index.html
[curand]: https://docs.nvidia.com/cuda/curand/index.html
[cusolver]: https://docs.nvidia.com/cuda/cusolver/index.html
[cusparse]: https://docs.nvidia.com/cuda/cusparse/index.html
[cusparse-spmv]: https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv
[cutensor]: https://docs.nvidia.com/cuda/cutensor/latest/index.html
[nccl-init]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrank
[nccl-comm]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
[nccl-allreduce]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#ncclallreduce
[otp-sup]: https://www.erlang.org/doc/design_principles/sup_princ.html
[atomr-directive]: ../../atomr/crates/atomr-core/src/supervision.rs
[atomr-persistence]: ../../atomr/crates/atomr-persistence
[atomr-sharding]: ../../atomr/crates/atomr-cluster-sharding
[atomr-entity-ref]: ../../atomr/crates/atomr-cluster-sharding/src/entity_ref.rs
[atomr-dashboard]: ../../atomr/crates/atomr-dashboard
