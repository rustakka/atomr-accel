# Backends

`atomr-accel` is the actor-shaped face of compute acceleration —
**generally**. NVIDIA CUDA is the first shipping implementation; the
abstraction layer is designed so AMD ROCm, Apple Metal, Intel oneAPI,
and Vulkan compute can plug into the same actor surface without
rewriting application code.

## The abstraction layer

The core crate (`atomr-accel`) defines five primitives that every
backend implements:

| Trait / type            | What it names                                       |
|-------------------------|-----------------------------------------------------|
| `AccelBackend`          | Marker for a backend, with associated `Device` / `Stream` / `Event` / `Error` types. |
| `AccelDevice`           | Device handle: stable id, generation counter, sticky-error recovery hook. |
| `AccelStream`           | Per-actor queue: `record_event` + `wait_event` for cross-stream synchronization. |
| `AccelRef<T, B>`        | Generation-validated typed pointer. `access()` fails fast if the device generation has moved. |
| `AccelError`            | `#[non_exhaustive]` typed enum: `ContextPoisoned` / `OutOfMemory` / `Unrecoverable` / `AccelRefStale` / `LibraryError { lib, msg }` / `Timeout`. |
| `CompletionStrategy<B>` | Async wakeup contract: host-fn callback / sync block / polled query. |
| `KernelOp`              | Marker for typed op envelopes (`GemmShape`, `FftKind`, …). |

The core ships **no concrete actors**. Each backend crate
(`atomr-accel-cuda` today; `atomr-accel-rocm`, `-metal`, `-oneapi`,
`-vulkan` future) provides its own `DeviceActor`, library-specific
kernel actors, and stream allocators. Each backend crate depends on
`atomr-accel` (the trait surface) and is imported directly:

```rust
use atomr_accel_cuda as cuda;       // CUDA backend
// use atomr_accel_rocm as rocm;     // future
// use atomr_accel_metal as metal;   // future
```

## What's *not* in the abstraction

The traits intentionally stop at the device-handle / stream / event
contract. They don't try to:

- **Compile one shader to many targets.** That's wgpu / SYCL
  territory. We're shipping mature vendor SDKs (cuBLAS, cuDNN, …)
  behind actor supervision, not a portable shader IR.
- **Hide library differences.** cuDNN's convolution surface is
  richer than MPS'; cuBLASLt's epilogue family doesn't have a
  perfect MPS equivalent. The trait surface is for *portable*
  code; backend-specific work uses the concrete crate directly:
  ```rust
  use atomr_accel_cuda::kernel::{CudnnActor, ConvForwardRequest};
  ```
- **Abstract over kernel launch shapes.** A CUDA kernel has
  blocks/threads; a Metal kernel has threadgroups; a Vulkan
  compute shader has workgroups. The launch geometry stays
  backend-specific. The op envelope (`KernelOp`) is shared; the
  launch is not.

## Why an abstraction at all, then?

Three reasons:

1. **Multi-vendor deployments.** A service that runs on NVIDIA in
   one region and AMD in another wants the same supervision +
   replay + telemetry stack. The actor tree, the typed messages,
   the fault model — all the same. Only the kernel actor crate
   changes.
2. **Code reuse for blueprints.** `atomr-accel-patterns`,
   `atomr-accel-train`, and `atomr-accel-agents` are
   backend-generic by design. Their `DynamicBatchingServer`,
   `DataParallelTrainer`, `RagPipeline`, etc. don't care which
   backend is underneath; they parameterize over the message
   surface.
3. **Future-proof error handling.** `AccelError` is
   `#[non_exhaustive]`. Adding `LibraryError { lib: "rocfft", … }`
   when the ROCm crate lands is a non-breaking minor bump.

## Roadmap

| Backend           | Status                                  | Tracking |
|-------------------|-----------------------------------------|----------|
| **CUDA (NVIDIA)** | **Shipping** in `atomr-accel-cuda`.     | F2–F9 implemented. |
| ROCm (AMD)        | Designed for. Crate skeleton not yet up. | hipBLAS / hipFFT / rocSPARSE / rocSOLVER all map cleanly to the existing kernel-actor pattern. |
| Metal (Apple)     | Designed for.                           | MPS + Metal command queues fit the `AccelStream` trait directly. |
| oneAPI (Intel)    | Designed for.                           | SYCL queue + oneMKL libraries. |
| Vulkan compute    | Speculative.                            | Useful for "any-GPU" batch jobs; harder to map cuBLAS-equivalent ops. |
| WebGPU            | Not planned.                            | Targets browsers and edge devices, doesn't share the actor / supervision concerns. |

## Adding a backend

1. New crate `atomr-accel-<name>` in the workspace.
2. `pub struct Backend;` + `impl AccelBackend for Backend { … }`.
3. Concrete `Device` / `Stream` / `Event` / `Error` types
   wrapping the vendor SDK's handles.
4. `pub type AccelRef<T> = MyConcreteRef<T>;` with the standard
   generation-token check.
5. Concrete `DeviceActor` + `ContextActor` + per-library kernel
   actors. Each kernel actor emits typed errors and panics with
   the standard `"ContextPoisoned: …"` / `"OutOfMemory: …"` /
   `"Unrecoverable: …"` tags so the supervisor decider routes
   directives correctly.
6. (Optional) feature-flag in the umbrella `atomr-accel/Cargo.toml`
   re-exporting the new crate at `atomr_accel::<name>`.

The blueprint sub-crates (`patterns`, `train`, `agents`) are
backend-agnostic; they don't need any change to support a new
backend.
