---
name: atomr-accel-supervision
description: Use when reasoning about failure recovery — the two-tier `DeviceActor ↔ ContextActor` model, sticky-error context loss, the `ContextPoisoned` / `OutOfMemory` / `Unrecoverable` panic-tag protocol, `GpuRef` generation tokens, and the `WatchGeneration` subscription. Triggers on writing a custom decider, encountering `GpuRefStale`, designing a top-level observer that has to react to context rebuild, or debugging a restart loop.
---

# Supervision and recovery in atomr-accel

This skill helps you reason about failure handling. For the
device's public interface, see
[`atomr-accel-device`](../atomr-accel-device/SKILL.md). For
diagnosing common errors, see
[`atomr-accel-troubleshooting`](../atomr-accel-troubleshooting/SKILL.md).

## Why the design exists

CUDA has a long-standing **sticky-error** problem: once a kernel
hits certain failures (illegal memory access, async failure
surfaced via `cudaGetLastError`), the entire `CUcontext` is
poisoned and every subsequent call returns the same error. The
only recovery is to tear the context down and rebuild it.

atomr-accel mirrors this with two actors — a stable outer one
that owns the application-visible address, and a restartable
inner one that owns the `Arc<CudaContext>`.

```
DeviceActor               ← stable: ActorRef<DeviceMsg> never changes
  │ pending: VecDeque<WorkRequest>   (queues work while context rebuilds)
  │ state:   Arc<DeviceState>        (survives every restart)
  │
  └─ ContextActor         ← restartable: rebuild on poisoning
       │ owns Arc<CudaContext>
       │ generation: bumped every restart
       │
       ├─ BlasActor / CudnnActor / FftActor / RngActor / …
       └─ ManagedAllocatorActor / GraphActor / …
```

Application code holds an `ActorRef<DeviceMsg>` that survives
unlimited `ContextActor` restarts.

## The supervision contract

`DeviceActor::supervisor_strategy()` is a `OneForOneStrategy` with:

- **3 retries** inside a 60-second window (circuit-breaker).
- A decider that maps panic-message tags to directives.

After the third restart in 60s, the device stops permanently —
the parent of `DeviceActor` (often the `ActorSystem` root or your
top-level app actor) sees the failure and decides what to do.

## Failure transport: panic with a tag

`Actor::handle` returns `()`, so panics are how failures travel
up the tree. atomr-accel encodes the directive in the panic
message:

| Panic message contains | Directive | Meaning |
|---|---|---|
| `"ContextPoisoned: ..."` | `Restart` | Tear down + rebuild the `ContextActor`; bump generation |
| `"OutOfMemory: ..."` | `Resume` | Drop the failing message; keep state |
| `"Unrecoverable: ..."` | `Stop` | Hardware fault / past retry budget; stop the device |
| anything else | `Escalate` | Hand to grandparent |

```rust
// Inside a kernel actor that detected sticky-error context poisoning:
panic!("ContextPoisoned: cublasCreate returned {status:?}");
```

The application **never writes its own decider** unless it's
running its own non-CUDA actor next to the device. For
that, [`atomr-accel-cuda::error::DeviceSupervisor`](../../../crates/atomr-accel-cuda/src/error.rs)
implements atomr's typed `SupervisorOf<ContextActor>` trait
over `GpuError`, so you can pattern-match on the typed enum
directly:

```rust
use atomr_accel_cuda::error::DeviceSupervisor;
let directive = DeviceSupervisor::decide(&GpuError::ContextPoisoned("…".into()));
// → Directive::Restart
```

## `GpuRef<T>`: generation tokens

A device buffer is more than an opaque `*mut T`. The pointer is
only valid against the context that allocated it. After a context
rebuild, every previously-allocated pointer is **dangling**.

`GpuRef<T>` wraps `Arc<CudaSlice<T>>` plus a generation token
captured at allocation time. Reading goes through `GpuRef::access`,
which checks the token against `DeviceState::generation` and
returns `GpuError::GpuRefStale` if they diverge:

```rust
let a: GpuRef<f32> = /* allocated against generation N */;
// ContextActor restarts; generation bumps to N+1.
let _ = a.access()?;   // → Err(GpuError::GpuRefStale)
```

When you see `GpuRefStale`, the recovery is mechanical:

1. Reallocate against the new generation:
   `device.ask_with(|tx| DeviceMsg::AllocateF32 { len, reply: tx }, …)`.
2. Re-upload any host data that should have lived on-device.
3. Resume the pipeline.

If you have a bootstrap actor that owns all your `GpuRef`s, the
cleanest pattern is to subscribe it to the `WatchGeneration`
channel and re-allocate-on-tick.

## `WatchGeneration`: reacting to rebuild

Top-level observers (`P2pTopology`, `NcclWorldActor`,
`PlacementActor`, your own bootstrap) subscribe via:

```rust
let mut rx: tokio::sync::watch::Receiver<u64> = device.ask_with(
    |tx| DeviceMsg::WatchGeneration { reply: tx },
    Duration::from_secs(5),
).await??;

while rx.changed().await.is_ok() {
    let new_gen = *rx.borrow();
    tracing::warn!(new_gen, "device context rebuilt — reallocating buffers");
    // Reallocate / reseed RNG / rebuild caches keyed on the old context.
}
```

`NcclWorldActor` uses this internally to tear down + rebuild the
NCCL communicator when any participating device loses its
context. `P2pTopology` uses it to invalidate cached
`Arc<CudaContext>` snapshots.

## When to **not** customize supervision

Most callers don't need to. The default decider handles all three
tags correctly. Customize only if:

- You're embedding atomr-accel inside a larger actor system and
  want to apply additional policy (e.g. circuit-break harder).
- You're authoring a backend that needs different tags (e.g. a
  ROCm crate adds `"HipDriverFault: ..."`).

When you do customize, derive from
`error::device_supervisor_strategy()` rather than rolling from
scratch — that gives you the canonical 3-retry / 60-second
circuit breaker.

## Canonical references

- [`docs/concepts.md`](../../../docs/concepts.md) § "Two-tier
  supervision split" and § "Generation watch" — the full mental
  model.
- [`docs/architecture.md`](../../../docs/architecture.md) § "The
  two-tier device model" — the design narrative with NVIDIA links.
- `crates/atomr-accel-cuda/src/error.rs` — `GpuError`,
  `decider()`, `DeviceSupervisor`, the panic-tag constants.
- `crates/atomr-accel-cuda/src/device/state.rs` — `DeviceState`
  with the generation counter and watch channel.
- `crates/atomr-accel-cuda/tests/supervisor_decider.rs` — round
  trip: panic → decider → restart.
- `crates/atomr-accel-cuda/tests/watch_generation.rs` —
  subscribers receive ticks on rebuild.

## Common pitfalls

- **Catching `panic!` inside `handle`.** Don't. Let the supervisor
  see the panic-tagged message. Catching it strips the directive
  and turns a recoverable context-poisoning into a silent bug.
- **Treating `GpuRefStale` as fatal.** It's the explicit signal
  that recovery happened. Reallocate.
- **Subscribing to `WatchGeneration` from inside `handle`.** The
  `rx.changed().await` loop has to live in a `tokio::spawn` task
  that posts a follow-up message back to your actor. Don't park
  the calling actor's mailbox.
- **Assuming `pre_start` re-runs on restart.** atomr does **not**
  re-run `pre_start` on restart — it re-runs the `Props::create`
  factory, then calls `post_restart` on the fresh instance. Put
  resource setup in the factory or `post_restart`, not
  `pre_start`.
- **Replacing the panic-as-failure transport with custom
  Result-returning APIs.** `Actor::handle` returns `()` — the
  trait is fixed. Failures travel as panics. Application-visible
  results travel through `oneshot::Sender<Result<_, GpuError>>`
  on each message.
