# Concepts

The five ideas you need to be productive in rakka-accel. Each one
exists because it makes a CUDA invariant easier to live with.

## 1. The two-tier supervision split

CUDA has two lifetimes that don't match: a **device** (visible to the
operator, e.g. `gpu-0`) and a **context**
([`CUcontext`][cuda-ctx], the runtime state that drives that device).
Contexts can poison themselves on certain errors and have to be torn
down + rebuilt. Devices, from the user's point of view, just keep
working.

rakka-accel mirrors this with two actors:

- **`DeviceActor`** — stable address (`ActorRef<DeviceMsg>`). Lives
  for as long as the device is part of the system. Queues incoming
  work while the context is being built or rebuilt.
- **`ContextActor`** — child of `DeviceActor`. Owns the
  `Arc<CudaContext>` and the per-library actors that depend on it.
  Restartable.

The supervision strategy is `OneForOneStrategy::with_max_retries(3)
.with_within(60s)`. After three restarts inside a one-minute window
the circuit opens and the device stops — that's the
[sticky-error][cuda-sticky] convention exposed via supervision.

```
DeviceActor               (stable; queues work)
  └─ ContextActor         (restartable; owns CudaContext)
       ├─ BlasActor       (restarts with parent)
       ├─ CudnnActor
       └─ …
```

When a child panics with `"ContextPoisoned: cuInit failed: …"`, the
decider routes to `Directive::Restart`. `OutOfMemory` → `Resume`.
`Unrecoverable` → `Stop`. Anything else → `Escalate`.

The transport is intentionally a panic message: `Actor::handle`
returns `()` in rakka, so panics are how failures travel up the
tree. Receive-side parsing is either the closure-based
`error::decider()` or the typed `error::DeviceSupervisor` that
implements `SupervisorOf<C>` over `GpuError`.

## 2. `GpuRef<T>`: typed pointers with generation tokens

A device buffer is more than an opaque `*mut T`. CUDA semantics say
the pointer is only valid against the context that allocated it.
After a context rebuild, every previously-allocated pointer is
**dangling**.

`GpuRef<T>` is a `Clone` wrapper around `Arc<CudaSlice<T>>` plus a
generation token. Reading the buffer goes through `GpuRef::access`,
which checks the token against `DeviceState::generation` and returns
`GpuError::GpuRefStale` if they diverge. No silent data corruption.

```rust
let a: GpuRef<f32> = /* allocated against generation N */;
// ContextActor restarts; generation bumps to N+1.
let _ = a.access()?; // -> Err(GpuError::GpuRefStale)
```

`GpuRef` also remembers its `last_write_stream`, which the actor
pipeline uses to inject [`cudaStreamWaitEvent`][cuda-events]
automatically when a downstream reader is on a different stream. You
never call `cudaStreamSynchronize` from application code.

## 3. Completion strategies: how kernels signal "done"

A kernel launch is asynchronous. Three ways to find out when it
finishes, all behind the same `CompletionStrategy` trait:

| Strategy             | Mechanism                          | Best for                            |
| -------------------- | ---------------------------------- | ----------------------------------- |
| `HostFnCompletion`   | [`cuLaunchHostFunc`][cuda-launch-host] callback | The default. Sub-µs latency, no host blocking. |
| `SyncCompletion`     | `cudaStreamSynchronize`            | Tests, quick experiments.           |
| `PolledCompletion`   | `cuEventQuery` loop with timeout   | When you need a hard upper bound.   |

`HostFnCompletion` registers a callback that fires the moment the
stream drains, signaling a `oneshot` channel. The Tokio task waiting
on the reply wakes immediately. No host-side polling, no thread
parked on `cudaStreamSynchronize`.

The `kernel::envelope::run_kernel` helper wraps every library
actor's launch in:

1. Validate every input `GpuRef` (one shot for stale checks).
2. Synchronously enqueue the kernel.
3. Spawn a Tokio task that awaits the configured `CompletionStrategy`.
4. Send the reply.
5. Drop the keep-alive tuple (input `Arc`s, descriptor handles).

This is the same shape for cuBLAS, cuDNN, cuFFT, cuSOLVER, cuSPARSE,
cuTENSOR, cuBLASLt, NVRTC, and NCCL.

## 4. Stream allocators: who owns the stream?

Three policies, all `StreamAllocator`:

- **`PerActorAllocator`** — one stream per kernel actor. Two modes:
  - `Shared` — every actor of the same type shares a single stream
    (saves driver resources at the cost of intra-type serialization).
  - `Fresh` — every actor mints its own. Maximum concurrency.
- **`SingleStreamAllocator`** — every actor uses the *same* stream.
  Easiest to reason about; no cross-stream events ever needed.
- **`PooledAllocator`** — round-robin across a fixed-size pool.
  Predictable concurrency without proliferating streams.

Stream allocation is a pluggable detail. Inject one at
`ContextActor` construction and forget about it.

## 5. Generation watch: reacting to context loss

Top-level observers (`P2pTopology`, `NcclWorldActor`, your custom
`PlacementActor`) need to know when a `ContextActor` rebuilds. The
mechanism is a `tokio::sync::watch::Receiver<u64>` published by
`DeviceState::generation_watch()`. Send `DeviceMsg::WatchGeneration {
reply }`, get the receiver, subscribe.

```rust
let mut rx = device.ask_with(
    |tx| DeviceMsg::WatchGeneration { reply: tx },
    Duration::from_secs(5),
).await??;

while rx.changed().await.is_ok() {
    let new_gen = *rx.borrow();
    // Rebuild caches keyed on the old context.
}
```

`NcclWorldActor` uses this to tear down + rebuild the NCCL
[communicator][nccl-comm] when any participating device loses its
context. `P2pTopology` uses it to invalidate cached
`Arc<CudaContext>` snapshots and force a follow-up `EnableAll`.

---

## How these compose

Every public message follows the same pattern:

```rust
KernelMsg::DoSomething {
    /* typed inputs (often GpuRef<T>) */
    ...,
    /* required: a oneshot::Sender for the reply */
    reply: oneshot::Sender<Result<Output, GpuError>>,
}
```

Application code uses `actor_ref.tell(msg)` to fire-and-forget or
`actor_ref.ask_with(builder, timeout)` to await the reply. Inside the
actor, `kernel::envelope::run_kernel` handles validation, enqueue,
completion, and keep-alive. Errors travel as `GpuError`; stale
buffers bubble up before any kernel touches them; supervisor
directives recover the rest.

That uniformity is the point: once you've used one library actor,
you've used them all.

[cuda-ctx]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html
[cuda-sticky]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking
[cuda-events]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events
[cuda-launch-host]: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g05841eaa5f90f27264c5d9eb96b16d2c
[nccl-comm]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
