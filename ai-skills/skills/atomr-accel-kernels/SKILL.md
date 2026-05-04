---
name: atomr-accel-kernels
description: Use when picking, wiring, or extending a per-library kernel actor — cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuTENSOR, NVRTC, NCCL — and the shared `KernelEnvelope::run_kernel` pattern. Triggers on adding a new kernel call, choosing between equivalent libraries (cuBLAS vs cuBLASLt, cuFFT vs MPS), JIT-compiling a custom kernel, or picking a `CompletionStrategy`.
---

# Per-library kernel actors

This skill covers the kernel-actor family that lives **under** the
`ContextActor`. For driving a kernel through `DeviceMsg::Sgemm`,
see [`atomr-accel-device`](../atomr-accel-device/SKILL.md).

## The actor family

| Library | Actor | Feature flag | Canonical op |
|---|---|---|---|
| cuBLAS | `BlasActor` | always-on | `Sgemm` |
| cuBLASLt | `BlasLtActor` | `cublaslt` | `MatmulRelu` / `MatmulGelu` (fused) |
| cuDNN | `CudnnActor` | `cudnn` | `ConvForward`, `Activation`, `Softmax` |
| cuFFT | `FftActor` | `cufft` | `Forward1dR2C`, `Inverse1dC2R`, `Forward2dC2C` |
| cuRAND | `RngActor` | `curand` | `FillUniformF32`, `FillNormalF32`, `Reseed` |
| cuSOLVER | `SolverActor` | `cusolver` | `QrFactorize`, `LuFactorize`, `LuSolve`, `Cholesky`, `Svd`, `Syevd` |
| cuSPARSE | `SparseActor` | `cusparse` | `SpMv`, `SpMm` (CSR layout) |
| cuTENSOR | `TensorActor` | `cutensor` | `Contract` (Einstein-summation) |
| NVRTC | `NvrtcActor` | `nvrtc` | `Compile`, `Launch` (custom CUDA-C kernels) |
| NCCL | `CollectiveActor` + `NcclWorldActor` | `nccl` | `AllReduceF32`, `BroadcastF32` |

Each actor owns a single library handle (e.g.
`cudarc::cublas::CudaBlas`) on a single stream. They're spawned
under `ContextActor` so they restart together when the device
context rebuilds.

## Getting a kernel-actor `ActorRef`

Kernel actors are children of the `ContextActor`, not of the
top-level `DeviceActor`. To get a typed ref, snapshot:

```rust
let children = device.ask_with(
    |tx| DeviceMsg::SnapshotChildren { reply: tx },
    Duration::from_secs(5),
).await??;

let blas: ActorRef<BlasMsg> = children.blas.clone();

#[cfg(feature = "cudnn")]
let cudnn: ActorRef<CudnnMsg> = children.cudnn.clone().expect("cudnn enabled");
```

`SnapshotChildren` returns `Option<KernelChildren>` — `None` until
`ContextReady` fires.

## Picking between cuBLAS and cuBLASLt

Use **cuBLAS** for plain SGEMM with no fused activation. Lower
latency for small matrices.

Use **cuBLASLt** when you want a fused matmul + ReLU/GELU
epilogue. The `BlasLtActor` packs `Linear → Activation` into one
launch — measurable win on transformer FFN blocks.

```rust
use atomr_accel_cuda::kernel::{BlasLtActor, BlasLtMsg, Activation};

blas_lt.tell(BlasLtMsg::MatmulRelu { a, b, c, m, n, k, reply });
// or:
blas_lt.tell(BlasLtMsg::MatmulGelu { a, b, c, m, n, k, reply });
```

## Custom kernels via NVRTC

```rust
let kernel: KernelHandle = nvrtc.ask_with(
    |tx| NvrtcMsg::Compile {
        src: r#"
            extern "C" __global__
            void scale(float* x, int n, float k) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) x[i] *= k;
            }
        "#.to_string(),
        kernel_name: "scale".to_string(),
        opts: NvrtcOpts::default(),
        reply: tx,
    },
    Duration::from_secs(30),
).await??;

nvrtc.ask_with(
    |tx| NvrtcMsg::Launch {
        kernel,
        args: vec![
            KernelArg::DevSliceF32(buffer),
            KernelArg::Usize(n),
            KernelArg::ScalarF32(2.0),
        ],
        cfg: cudarc::driver::LaunchConfig::for_num_elems(n as u32),
        reply: tx,
    },
    Duration::from_secs(5),
).await??;
```

`KernelHandle` carries a generation token. If the context rebuilds
between `Compile` and `Launch`, the launch fails fast with
`GpuError::GpuRefStale("nvrtc kernel from prior context
generation")`. Re-issue `Compile` against the new generation.

The `atomr-accel-cuda-realtime` crate ships pre-authored CUDA-C
sources (`coo_spmv.cu`, `particle_step.cu`, `cloth_springs.cu`,
`hashmap_probe.cu`) bundled via `include_str!` — see
`crates/atomr-accel-cuda-realtime/src/kernels.rs`. Pattern those
when authoring your own.

## The `KernelEnvelope::run_kernel` pattern

Every kernel actor's `handle` is a thin shell. The shared body
lives in `kernel::envelope::run_kernel`:

```rust
envelope::run_kernel(
    LIB_TAG,           // &'static str — populates GpuError::LibraryError.lib
    stream,            // &Arc<CudaStream>
    completion,        // &Arc<dyn CompletionStrategy>
    output,            // O: Send + 'static — what flows back on success
    reply,             // oneshot::Sender<Result<O, GpuError>>
    || -> Result<KA, GpuError> {
        // 1. Validate every input GpuRef via .access().
        // 2. Synchronously enqueue the kernel onto the stream.
        // 3. Return the keep-alive tuple (input Arcs, descriptor handles).
    },
);
```

The envelope spawns a Tokio task that awaits the completion
strategy, replies on the channel, and only then drops the
keep-alive — so the kernel can't outlive its inputs even though
`handle` returned long ago.

When authoring a new kernel actor, copy this pattern. Don't
hand-roll completion.

## Choosing a `CompletionStrategy`

| Strategy | Mechanism | Best for |
|---|---|---|
| `HostFnCompletion` (default) | `cuLaunchHostFunc` callback | Production. Sub-µs wakeup, no host blocking |
| `SyncCompletion` | `cudaStreamSynchronize` | Tests, quick experiments — easy to reason about |
| `PolledCompletion` | `cuEventQuery` loop with timeout | When you need a hard upper bound |

```rust
use atomr_accel_cuda::completion::HostFnCompletion;
let completion: Arc<dyn CompletionStrategy> = Arc::new(HostFnCompletion::new());
```

Pass it once at `ContextActor` construction; every kernel actor
shares the same instance.

## Stream allocators

Three policies, all `StreamAllocator`:

- `PerActorAllocator { Shared, Fresh }` — default. Each kernel
  actor gets its own stream (Fresh) or shares one across
  same-typed actors (Shared).
- `SingleStreamAllocator` — every kernel actor uses the same
  stream. Eliminates cross-stream synchronization.
- `PooledAllocator` — round-robin across a fixed pool. Predictable
  concurrency without proliferating streams.

Inject one at `ContextActor` construction; switch when measuring.

## Cross-stream synchronization is automatic

`GpuRef<T>` records its `last_write_stream`. When a downstream
reader is on a different stream, the kernel envelope injects
`cudaStreamWaitEvent` against the recorded event automatically.
You never call `cudaStreamSynchronize` from application code.

## Canonical references

- `crates/atomr-accel-cuda/src/kernel/mod.rs` — module map of
  every actor.
- `crates/atomr-accel-cuda/src/kernel/envelope.rs` —
  `run_kernel` definition + `access_all_2/3/4` helpers.
- `crates/atomr-accel-cuda/src/kernel/blas.rs` — minimal
  reference impl. Read this when authoring a new kernel actor.
- `crates/atomr-accel-cuda/src/kernel/nvrtc.rs` —
  `KernelHandle` lifecycle and module cache.
- [`docs/architecture.md`](../../../docs/architecture.md) §
  "The KernelEnvelope" — the full flow.
- `crates/atomr-accel-cuda/examples/jit_relu.rs` — NVRTC
  end-to-end smoke.

## Common pitfalls

- **Calling a kernel actor without first awaiting `ContextReady`.**
  `SnapshotChildren` returns `None` until the context is built.
  Loop with a small sleep, or have your bootstrap actor send its
  initial work via `DeviceMsg` (which queues in the parent's
  pending deque automatically).
- **Holding a `KernelHandle` past a context rebuild.** Same as
  `GpuRef`: regenerate after the watch tick.
- **Mixing completion strategies across kernel actors.** Pick one
  per `ContextActor` and stick with it. Mixing can race in the
  reply path on shared streams.
- **Ignoring the `lib` tag in `GpuError::LibraryError`.** When you
  want to discriminate cuBLAS errors from cuDNN errors, match on
  `LibraryError { lib, .. }` — the tag is `"cublas"`, `"cudnn"`,
  `"cufft"`, etc.
- **JITing the same NVRTC source on every request.** The
  `NvrtcActor` caches modules by source-hash. Compile once at
  startup; reuse the `KernelHandle` for every Launch.
