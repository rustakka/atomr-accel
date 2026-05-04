---
name: atomr-accel-backends
description: Use when choosing between portable (`atomr-accel` core trait surface) and vendor-specific (`atomr-accel-cuda`) APIs. Covers when to write generic code over `AccelBackend`, what's in `AccelRef<T, B>` / `AccelError` / `CompletionStrategy`, and the multi-vendor roadmap (ROCm, Metal, oneAPI, Vulkan compute). Triggers on "should I depend on `atomr-accel` or `atomr-accel-cuda`?", "is this code portable?", or design questions about the trait surface.
---

# Backends and the abstraction layer

This skill helps you decide where to draw the abstraction line in
your project — generic across backends, or CUDA-specific. For
the concrete CUDA API, see
[`atomr-accel-device`](../atomr-accel-device/SKILL.md).

## Which crate to depend on

| Goal | Depend on |
|---|---|
| Drive an NVIDIA GPU; you don't expect to support other vendors | `atomr-accel-cuda` directly |
| Write portable algorithms that should run on any backend atomr-accel ships | `atomr-accel` (with `cuda` feature for the active backend) |
| Both — public API portable, internals CUDA-specific | depend on both; expose the portable surface in your public API |

Most projects today should depend on `atomr-accel-cuda` directly
— the abstraction layer's value materializes once a second
backend ships, and your bytes-on-disk `Cargo.toml` is one line
either way.

## What's portable, what's not

The core crate (`atomr-accel`) defines:

| Type / trait | Portable? | Notes |
|---|---|---|
| `AccelBackend` | yes | Marker; identifies a backend with associated types |
| `AccelDevice` | yes | Device-handle contract: id + generation counter |
| `AccelStream` | yes | `record_event` / `wait_event` |
| `AccelRef<T, B>` | yes | Generation-validated pointer trait |
| `AccelError` | yes | `#[non_exhaustive]` enum; backends add `LibraryError` tags |
| `CompletionStrategy<B>` | yes | Async wakeup contract |
| `KernelOp` + `GemmShape` / `FftKind` | yes | Marker + a few canonical op shapes |

The CUDA crate (`atomr-accel-cuda`) **adds**:

| Type / actor | Portable? | Notes |
|---|---|---|
| `DeviceActor`, `ContextActor` | no | CUDA-specific concrete actors |
| `GpuRef<T>` | no | `Arc<CudaSlice<T>>` underneath |
| `BlasActor`, `CudnnActor`, `FftActor`, … | no | Wrap cudarc handles |
| `NvrtcActor` (custom kernel JIT) | no | NVRTC is CUDA-only |
| `NcclWorldActor` (multi-GPU) | partly | NCCL has a ROCm-ish equivalent (RCCL); not 1:1 |
| `pipeline::*`, `placement::*`, `replay::*` | partly | Use `GpuRef<T>` directly today; could be lifted to backend-generic |

The blueprint sub-crates (`atomr-accel-patterns`,
`atomr-accel-train`, `atomr-accel-agents`) are
**backend-agnostic by design** — they parameterize over message
types, not over CUDA primitives. Reuse them across backends as
they ship.

## When to write portable code

Three concrete cases:

### 1. You want the same code to run on multiple backends

```rust
use atomr_accel::prelude::*;

async fn observe<B: AccelBackend>(device: &B::Device) -> u64 {
    device.generation()
}
```

When `atomr-accel-rocm` lands, `observe::<RocmBackend>` works
without changes.

### 2. Library code reused by both backend-generic and CUDA-specific consumers

Define the public surface in terms of the `atomr-accel` trait:

```rust
// In your library:
use atomr_accel::AccelError;

pub fn validate_dimensions(m: i32, n: i32, k: i32) -> Result<(), AccelError> {
    if m * n * k > MAX_PROBLEM_SIZE {
        return Err(AccelError::Unrecoverable("problem too large".into()));
    }
    Ok(())
}
```

Both CUDA and (future) ROCm consumers can call this without
caring which backend is active.

### 3. Application-level error handling

`AccelError` is `#[non_exhaustive]`. Match on the variants you
care about and let new backends add `LibraryError { lib, msg }`
variants without breaking your `match`:

```rust
match err {
    AccelError::OutOfMemory(_) => shrink_batch(),
    AccelError::ContextPoisoned(_) => log_and_retry(),
    AccelError::LibraryError { lib: "cublas", .. } => /* cuBLAS-specific recovery */,
    AccelError::LibraryError { lib: "rocblas", .. } => /* ROCm equivalent */,
    _ => bail!(err),
}
```

## When to **not** abstract

The abstraction stops at the device-handle / stream / event level.
It deliberately does not:

- **Compile one shader to many targets.** That's wgpu / SYCL
  territory.
- **Hide library API differences.** cuDNN's convolution surface
  is richer than MPS'; cuBLASLt's epilogue family doesn't have a
  perfect MPS equivalent. The trait surface is for *portable*
  code; backend-specific work uses the concrete crate directly.
- **Abstract over kernel launch geometry.** A CUDA kernel has
  blocks/threads; a Metal kernel has threadgroups; a Vulkan
  compute shader has workgroups. The launch stays
  backend-specific.

If you find yourself writing a thin shim that only works on one
backend, just depend on `atomr-accel-cuda`. The portable layer
exists for code that genuinely benefits from it.

## Multi-vendor roadmap

| Backend | Status | Notes |
|---|---|---|
| **CUDA (NVIDIA)** | **Shipping** in `atomr-accel-cuda` | Full F2-F9 + atomr integration. |
| ROCm (AMD) | Designed for | hipBLAS / hipFFT / rocSPARSE / rocSOLVER map cleanly. RCCL ≈ NCCL. |
| Metal (Apple) | Designed for | MPS + Metal command queues fit `AccelStream` directly. No NVRTC equivalent (use Metal compute shaders compiled at build time). |
| oneAPI (Intel) | Designed for | SYCL + oneMKL libraries. |
| Vulkan compute | Speculative | Useful for "any-GPU" jobs; harder to map cuBLAS-equivalent ops. |
| WebGPU | Not planned | Targets browsers / edge devices; doesn't share atomr's actor / supervision concerns. |

A new backend crate adds a `pub struct Backend;` with
`impl AccelBackend`, concrete `Device`/`Stream`/`Event`/`Error`
types wrapping the vendor SDK, its own `DeviceActor` +
per-library kernel actors using the panic-tag protocol, and (via
a feature flag in the umbrella) a re-export at
`atomr_accel::<name>`.

The blueprint sub-crates require **no changes** to support a new
backend.

## Authoring a backend (high-level)

If you need to ship one (and you're not me):

1. New crate `atomr-accel-<name>` in the workspace.
2. `pub struct Backend;` + `impl AccelBackend for Backend { … }`.
3. Concrete `Device` / `Stream` / `Event` / `Error` types.
4. `pub type AccelRef<T> = MyConcreteRef<T>;` with the standard
   generation-token check.
5. Concrete `DeviceActor` + `ContextActor` + per-library kernel
   actors. Each emits `AccelError` variants and panics with
   `"ContextPoisoned: …"` / `"OutOfMemory: …"` /
   `"Unrecoverable: …"` tags so the supervisor decider routes
   directives correctly.
6. (Optional) feature-flag in `atomr-accel/Cargo.toml`
   re-exporting at `atomr_accel::<name>`.

See `crates/atomr-accel-cuda/` as the reference implementation.
[`docs/backends.md`](../../../docs/backends.md) has the full
recipe.

## Canonical references

- [`docs/backends.md`](../../../docs/backends.md) — the multi-vendor
  trait abstraction explainer + roadmap.
- `crates/atomr-accel/src/lib.rs` — what the core crate exports.
- `crates/atomr-accel/src/backend.rs` — `AccelBackend`,
  `AccelDevice`, `AccelStream` traits.
- `crates/atomr-accel/src/error.rs` — `AccelError` (the typed
  enum every backend wraps or re-exports).
- `crates/atomr-accel-cuda/src/lib.rs` — reference impl entry
  point.

## Common pitfalls

- **Premature abstraction.** Writing `fn foo<B: AccelBackend>(…)`
  before you have a second backend usually adds noise without
  enabling reuse. Build CUDA-specific first; lift to portable
  when a second backend is on the horizon.
- **Assuming `KernelOp` is exhaustive.** It's a marker trait —
  each backend's actor message set is much richer. The marker
  exists for *typed-op envelopes* in pipelines; concrete ops use
  the backend's full API.
- **Pulling `atomr-accel` without enabling `cuda`.** The umbrella
  ships only the trait surface by default. If you want the
  concrete actors, depend on `atomr-accel-cuda` directly or
  enable the umbrella's `cuda` feature.
- **Match on `AccelError` exhaustively.** It's
  `#[non_exhaustive]`; future variants will land in minor bumps.
  Always include a `_` arm.
