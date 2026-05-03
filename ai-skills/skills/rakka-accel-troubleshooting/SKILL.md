---
name: rakka-accel-troubleshooting
description: Use when diagnosing failures in a project that depends on rakka-accel. Covers compile-time errors (missing feature flags, wrong prelude), runtime issues (`GpuRefStale`, OOM loops, mailbox stalls, no-GPU host vs CI gating), Python wheel installation problems, and CUDA-driver-not-loaded misdiagnoses. Triggers on rakka-accel error messages, "missing symbol" errors, hangs, repeated restarts, or any "why isn't my GPU work running" question.
---

# Troubleshooting rakka-accel

A diagnostic checklist organized by symptom. For deep dives into a
subsystem, hand off to the matching skill
([`rakka-accel-device`](../rakka-accel-device/SKILL.md),
[`rakka-accel-supervision`](../rakka-accel-supervision/SKILL.md),
[`rakka-accel-python`](../rakka-accel-python/SKILL.md)).

## Compile-time

### "cannot find type `X` in crate `rakka_accel_cuda`"

You're probably missing a feature flag. Library-specific actors
are gated:

| Type / module | Feature |
|---|---|
| `BlasActor`, `BlasMsg`, `SgemmRequest` | (always-on) |
| `CudnnActor`, `ConvForwardRequest`, `Activation` | `cudnn` |
| `FftActor`, `FftKind`, `PlanKey` | `cufft` |
| `RngActor`, `RngMsg` | `curand` |
| `SolverActor`, `Uplo`, SVD/Syevd | `cusolver` |
| `SparseActor`, `CsrMatrix`, `SparseMsg` | `cusparse` |
| `TensorActor`, `TensorSpec` | `cutensor` |
| `BlasLtActor`, `Activation` (fused) | `cublaslt` |
| `NvrtcActor`, `KernelHandle`, `KernelArg` | `nvrtc` |
| `CollectiveActor`, `NcclWorldActor`, `ReduceOp` | `nccl` |
| `ReplayHarness::with_journal` | `replay` |
| `placement::sharded::PlacementShardingAdapter` | `cluster` |
| `streams_pipeline::*` | `streams` |
| `observability::install`, `GpuProbes` | `telemetry` |

```toml
rakka-accel-cuda = { version = "0.2", features = ["cudnn", "nvrtc", "replay"] }
```

For the by-goal picker with transitive dep impact, see
[`docs/features-matrix.md`](../../../docs/features-matrix.md).

### "trait `Actor` is not implemented"

You're using rakka 0.2 — the trait lives in `rakka_core::actor`.
The cleanest import is the rakka-accel-cuda prelude, which
re-exports the relevant traits:

```rust
use rakka_accel_cuda::prelude::*;
use rakka::prelude::*;     // Actor, ActorRef, Context, Props
```

`#[async_trait]` is required because `Actor::handle` is `async`.

### "the trait `Send` is not implemented for `Rc<…>`"

Your message owns something non-`Send`. Switch `Rc` → `Arc`,
`RefCell` → `Mutex` (`std` or `parking_lot`). Holding a
`MutexGuard` across an `.await` point also breaks `Send` — drop
the guard before awaiting.

### "feature `…` does not exist"

The library aggregate / pass-through name has to come from the
right crate. From a downstream user:

```toml
rakka-accel-cuda = { features = ["full-cuda"] }   # ✓
rakka-accel      = { features = ["cuda"] }        # ✓ (umbrella)
```

Don't mix them: feature names are **per crate**, so
`rakka-accel = { features = ["cudnn"] }` won't work — `cudnn`
is on `rakka-accel-cuda`.

## Runtime

### `Err(GpuError::GpuRefStale("…"))` returned from `access`

The context rebuilt between when the buffer was allocated and
now. This is **expected behavior**, not a bug. Recovery:

1. Reallocate against the new generation.
2. Re-upload host data if needed.
3. Resume the pipeline.

If you have a bootstrap actor that owns long-lived buffers,
subscribe to `WatchGeneration` and reallocate-on-tick. See
[`rakka-accel-supervision`](../rakka-accel-supervision/SKILL.md).

### `Err(GpuError::Unrecoverable("…not supported in mock mode"))`

The device was spawned with `DeviceConfig::mock(0)` instead of
`DeviceConfig::new(0)`. Mock mode runs the supervision plumbing
without touching cudarc — useful for no-GPU CI but not for real
work. Switch to `::new(0)` once you have a GPU host.

### `Err(GpuError::Unrecoverable("…driver not loadable"))`

cudarc loaded dynamically and couldn't find the CUDA driver
shared library. Possibilities:

- No NVIDIA GPU on the host.
- CUDA Toolkit not installed (or `LD_LIBRARY_PATH` doesn't
  include `/usr/local/cuda/lib64`).
- Running inside Docker without `--gpus all`.
- Linux kernel mismatch with the installed driver after a
  recent reboot.

Quick check: `nvidia-smi` from the same shell that runs your
binary. If `nvidia-smi` succeeds but rakka-accel fails, dlopen
isn't finding the library — usually `LD_LIBRARY_PATH`.

### Mailbox stall — `device.tell(...)` never replies

You're awaiting a kernel reply **inside another actor's
`handle`**. The calling actor's mailbox is parked while it waits
on cuda, but the reply has to come back through the actor system,
which is also parked. Classic deadlock.

Fix: don't `ask` from inside `handle`. Either:

```rust
// Option A: tell + reply-via-message.
device.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
    a, b, c, m, n, k, alpha, beta,
    reply: ctx.self_ref().reply_to(MyMsg::SgemmDone),  // pseudo-API
})));

// Option B: spawn the await on tokio, then post a self-message.
let self_ref = ctx.self_ref().clone();
let device_clone = device.clone();
tokio::spawn(async move {
    let r = device_clone.ask_with(|tx| DeviceMsg::Sgemm(Box::new(SgemmRequest { …, reply: tx })),
                                  Duration::from_secs(60)).await;
    self_ref.tell(MyMsg::SgemmDone(r));
});
```

### Repeated `ContextPoisoned` restarts (circuit-breaker open)

The supervisor allows 3 restarts inside 60 seconds; after that
the device stops permanently. Common causes:

- A bug in the kernel call that always faults (illegal memory
  access in a custom NVRTC kernel, dimension mismatch in cuBLAS
  call). Inspect the panic message — the lib tag (`"cublas"`,
  `"cudnn"`, etc.) tells you which call faulted.
- A dying GPU. Run `nvidia-smi -q | grep "ECC Errors"` and check
  dmesg for Xid messages.

If you genuinely want a different policy (more retries / longer
window), customize via `error::device_supervisor_strategy()` —
but most cases are bugs you should fix instead of hide.

### NCCL world won't rebuild after a context restart

`NcclWorldActor` subscribes to every device's `WatchGeneration`
and rebuilds the communicator on tick. If it doesn't rebuild:

- One of the device contexts didn't actually rebuild (check
  `device.ask_with(|tx| DeviceMsg::SnapshotContext { reply: tx },
  …)` returns `Some(_)`).
- The NCCL communicator's underlying connection (NVLink, IB) is
  still down — fix that first; the actor only handles cuda-side
  rebuild.

## No-GPU CI vs GPU-runtime gating

Tests that need a real CUDA driver should be feature-gated:

```rust
#![cfg(feature = "cuda-runtime-tests")]
```

CI runs `cargo test --workspace --no-default-features` to exercise
plumbing on no-GPU runners. GPU integration tests run on a
self-hosted runner with `--features cuda-runtime-tests`.

If your test mysteriously passes locally but fails in CI:
inspect whether it's gated. The convention: any test that calls
`DeviceConfig::new(0)` and expects success must be gated; tests
using `DeviceConfig::mock(0)` are safe everywhere.

## Python-side issues

### `ImportError: cannot import name 'RngGenerator' from 'rakka_accel'`

Wheel was built without `--features curand`. Either rebuild
with the feature on, or guard usage:

```python
if rakka_accel.RngGenerator is not None:
    rng = ...
else:
    raise RuntimeError("install rakka-accel with [curand] feature")
```

### `pip install rakka-accel` fails with "no matching distribution"

ARM Linux: there's no aarch64 wheel published — install from
sdist with a Rust toolchain available. Other platforms: check
your Python version is ≥ 3.10 (wheels are abi3-py310).

### Python-thread starvation during long kernels

Every blocking method releases the GIL via `py.allow_threads`.
If you still see Python threads stuck, the bottleneck is on
*your* side — likely you're holding the GIL by calling pure-Python
code in a tight loop. Profile with `py-spy`.

## Canonical references

- [`docs/getting-started.md`](../../../docs/getting-started.md) —
  the right-from-zero tour.
- [`docs/features-matrix.md`](../../../docs/features-matrix.md) —
  feature → transitive-deps mapping.
- [`docs/python-bridge.md`](../../../docs/python-bridge.md) — Python
  wheel + GIL strategy + exception mapping.
- `crates/rakka-accel-cuda/src/error.rs` — `GpuError` variants
  and their meanings.
- `crates/rakka-accel-cuda/tests/end_to_end_e2e.rs` — minimal
  multi-actor smoke when you want a known-good reference.
