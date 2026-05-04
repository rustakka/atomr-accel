---
name: atomr-accel-device
description: Use when driving a GPU through `DeviceActor` — typed allocations, host↔device memcpy, dispatching `Sgemm` or other kernel ops, choosing `tell` vs `ask_with`, and reasoning about `GpuRef<T>` lifetimes. Triggers on writing or editing code that talks to `DeviceMsg`, allocates `GpuRef<T>`, copies via `HostBuf`, or dispatches a kernel.
---

# Driving a `DeviceActor`

This skill helps you write idiomatic Rust against the CUDA
implementation's primary entry point. For the supervision and
recovery model, see
[`atomr-accel-supervision`](../atomr-accel-supervision/SKILL.md).
For per-library kernel actors (cuBLAS, cuDNN, …), see
[`atomr-accel-kernels`](../atomr-accel-kernels/SKILL.md).

## Mental model

A `DeviceActor` is the **stable public face of one GPU**. It owns
a child `ContextActor` that holds the `Arc<CudaContext>` plus the
per-library kernel actors. When the context is poisoned (sticky
error) the child restarts; the parent's `ActorRef<DeviceMsg>` does
not change. Application code can keep using the same handle across
unlimited restarts.

```
DeviceActor (stable; ActorRef<DeviceMsg>)
  └─ ContextActor (restartable; owns Arc<CudaContext>)
       ├─ BlasActor (cuBLAS handle, pinned to one stream)
       ├─ CudnnActor / FftActor / RngActor / …
       └─ ManagedAllocatorActor / GraphActor / …
```

Every public message that returns data carries a
`oneshot::Sender<Result<_, GpuError>>` so completion is
asynchronous and failures are typed.

## Spawning a `DeviceActor`

```rust
use atomr_accel_cuda::prelude::*;
use atomr::prelude::*;

let system = ActorSystem::create("my-app", Config::empty()).await?;
let device = system.actor_of(
    DeviceActor::props(DeviceConfig::new(0)),
    "device-0",
)?;
```

`DeviceConfig` knobs:

| Field | Default | What it does |
|---|---|---|
| `device_id` | required | CUDA ordinal (0, 1, 2, …) |
| `mock_mode` | false | Skip cudarc calls; run plumbing only (no-GPU CI) |
| `pending_queue_capacity` | 1024 | Bounded queue of work received before `ContextReady` |
| `enabled_libraries` | `BLAS` | Bitflags choosing which kernel actors `ContextActor` spawns |

```rust
let cfg = DeviceConfig::new(0)
    .with_libraries(EnabledLibraries::ALL);   // BLAS + CUDNN + CUFFT + CURAND + …
```

Compile-time feature flags still apply — `EnabledLibraries::CUDNN`
is a no-op if you didn't enable `atomr-accel-cuda/cudnn`.

## Typed allocations

Every dtype gets its own variant so `GpuRef<T>` keeps its `T` on
the receive side:

```rust
use tokio::sync::oneshot;

let (tx, rx) = oneshot::channel();
device.tell(DeviceMsg::AllocateF32 { len: 1024, reply: tx });
let buf: GpuRef<f32> = rx.await??;
```

Or via the typed-ask helper:

```rust
let buf = device.ask_with(
    |tx| DeviceMsg::AllocateF32 { len: 1024, reply: tx },
    Duration::from_secs(5),
).await??;
```

Available variants: `AllocateF32` / `F64` / `I8` / `I32` / `I64` /
`U8` / `U32` / `U64`, plus `AllocateF16` / `Bf16` under the `f16`
feature.

The legacy `DeviceMsg::Allocate { len, reply }` is a back-compat
alias for `AllocateF32`. New code should use the typed variant.

## Host ↔ device memcpy

```rust
// Upload from a host Vec.
let host_data = vec![1.0_f32; 1024];
device.ask_with(
    |tx| DeviceMsg::CopyFromHostF32 {
        src: HostBuf::Owned(host_data),
        dst: buf.clone(),
        reply: tx,
    },
    Duration::from_secs(5),
).await??;

// Download into a freshly allocated host Vec.
let result = device.ask_with(
    |tx| DeviceMsg::CopyToHostF32 {
        src: buf,
        dst: HostBuf::Owned(vec![0.0; 1024]),
        reply: tx,
    },
    Duration::from_secs(5),
).await??;
```

`HostBuf` is the H2D / D2H envelope:

- `HostBuf::Owned(Vec<T>)` — quick path; the actor moves the `Vec`
  in and back out to the reply channel.
- `HostBuf::Pinned(PinnedBuf<T>)` — page-locked; participates in
  async H2D / D2H copies without a bounce buffer. Source from
  `PinnedBufferPool`. Use this on hot paths.

`CopyToHost*` reply channels return the **same** `HostBuf`, so a
pinned buffer flows back into the pool without an extra
allocation:

```rust
let pinned = pool.acquire(1024).await;
let pinned = device.ask_with(
    |tx| DeviceMsg::CopyToHostF32 { src: buf, dst: HostBuf::Pinned(pinned), reply: tx },
    Duration::from_secs(5),
).await??;
pool.release(pinned).await;
```

## Dispatching a kernel via `DeviceMsg`

`DeviceMsg::Sgemm` is the canonical kernel dispatch on the device
itself (forwarded to the BLAS actor):

```rust
let (tx, rx) = oneshot::channel();
device.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
    a, b, c,
    m: 1024, n: 1024, k: 1024,
    alpha: 1.0, beta: 0.0,
    reply: tx,
})));
rx.await??;
```

For other kernels (Conv via cuDNN, FFT, RNG fill, NVRTC launch),
**don't go through DeviceMsg** — get a direct `ActorRef<...Msg>`
from `DeviceMsg::SnapshotChildren` and dispatch there. See
[`atomr-accel-kernels`](../atomr-accel-kernels/SKILL.md).

## `tell` vs `ask_with`

| Pattern | When |
|---|---|
| `device.tell(DeviceMsg::…)` | Fire-and-forget, or you'll await the reply rx separately |
| `device.ask_with(\|tx\| …, timeout)` | One-call typed ask. Returns the reply value or a timeout error |

Don't `ask_with` from inside another actor's `handle()` — it parks
the calling actor's mailbox. Use `tell` plus a follow-up message
to fold the reply back in (the `pattern::pipe_to` shape).

## Stats + observability

```rust
let load = device.ask_with(
    |tx| DeviceMsg::Stats { reply: tx },
    Duration::from_secs(2),
).await?;
println!("free: {} / total: {}", load.free_bytes, load.total_bytes);
```

`PlacementActor` polls `Stats` periodically to drive
load-balanced device selection.

## Canonical references

- [`docs/getting-started.md`](../../../docs/getting-started.md) §3 —
  end-to-end allocate → copy → roundtrip example.
- `crates/atomr-accel-cuda/src/device/device_actor.rs` —
  `DeviceMsg` definition and dispatch.
- `crates/atomr-accel-cuda/examples/sgemm.rs` — full SGEMM smoke
  test.
- [`atomr-accel-kernels`](../atomr-accel-kernels/SKILL.md) — for
  per-library kernels.
- [`atomr-accel-supervision`](../atomr-accel-supervision/SKILL.md)
  — for `GpuRef` generation tokens and recovery model.

## Common pitfalls

- **Holding a `GpuRef<T>` past a context restart.** The next
  `access()` returns `Err(GpuError::GpuRefStale)`. Re-allocate on
  the new generation. See the supervision skill.
- **Cloning a `GpuRef<T>` then expecting unique-write semantics
  on either copy.** `GpuRef` is `Arc`-shared; the actor's
  write-target unwrap (`Arc::try_unwrap`) fails if you've kept a
  clone. For write targets, mint and consume the ref once.
- **Awaiting a kernel reply inside another actor's `handle`.**
  Stalls the calling actor's mailbox while CUDA runs. Use
  `tell` + reply-via-message instead.
- **`HostBuf::Owned` on a hot loop.** Each call moves a `Vec` —
  fine, but allocates. Use `PinnedBufferPool` and `HostBuf::Pinned`
  for sustained throughput.
- **Sending pre-`ContextReady` work then expecting it served
  before the context builds.** It queues in the parent's
  `pending` deque (capacity per `DeviceConfig`). If the queue
  fills, the call fails with `Unrecoverable("device pending queue
  full")`. Either lift the capacity or wait on the
  `ContextReady` event before flooding work.
