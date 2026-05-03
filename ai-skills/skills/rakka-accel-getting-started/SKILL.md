---
name: rakka-accel-getting-started
description: Use when wiring rakka-accel into a new Rust project — choosing crates, picking feature flags, bootstrapping an `ActorSystem` + `DeviceActor`, and running on a no-GPU dev box vs a real CUDA host. Triggers on first-time `Cargo.toml` setup, `cargo add rakka-accel`, picking which sub-crate to depend on, or "how do I start using this".
---

# Getting started with rakka-accel

This skill helps you wire rakka-accel into a Rust project for the
first time. For driving a device once it's wired, see
[`rakka-accel-device`](../rakka-accel-device/SKILL.md). For
choosing among kernel actors, see
[`rakka-accel-kernels`](../rakka-accel-kernels/SKILL.md).

## The crate family

```
                    ┌─────────────────┐
                    │  rakka-accel    │  ← backend-agnostic core (umbrella)
                    └────────┬────────┘
                             │ feature `cuda` re-exports below
                             ▼
                    ┌─────────────────┐
                    │ rakka-accel-cuda │  ← CUDA implementation
                    └────────┬────────┘
                             │ depended on by
       ┌────────────┬────────┴────────┬─────────────┬──────────┐
       ▼            ▼                 ▼             ▼          ▼
   patterns     train             agents       cuda-realtime  py
                (depends on
                 patterns too)
```

- `rakka-accel` — backend-agnostic traits + `AccelError` enum +
  optional `cuda` re-export. Pull this in when writing
  backend-portable code.
- `rakka-accel-cuda` — concrete CUDA implementation with
  `DeviceActor`, `GpuRef<T>`, all kernel actors. Most projects
  depend on this directly.
- `rakka-accel-patterns` / `-train` / `-agents` — universal,
  training, and LLM blueprint actors.
- `rakka-accel-cuda-realtime` — CUDA-specific NVRTC-backed
  realtime sims (particles, cloth, sparse SpMV, hashmap probe).

Sub-crates path-depend only on `rakka-accel-cuda` — no implicit
pulls of the other blueprints.

## Adding the dependencies

### Just cuBLAS

```toml
[dependencies]
rakka-accel-cuda = "0.2"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

cuBLAS is always-on. Every other CUDA library is gated behind a
feature flag — see the matrix below. The crate **builds without a
GPU**: cudarc loads the CUDA driver dynamically and falls back to a
no-op when it's missing.

### Common feature combinations

| Goal | Features |
|---|---|
| Training (cuDNN + NCCL + NVRTC + cuTENSOR + cuSOLVER + cuBLASLt + cuFFT + cuRAND + cuSPARSE) | `rakka-accel-cuda = { features = ["full-cuda"] }` |
| Inference + JIT custom kernels | `rakka-accel-cuda = { features = ["training-libs"] }` |
| Just core libraries (cuDNN + cuFFT + cuRAND + cuSPARSE) | `rakka-accel-cuda = { features = ["core-libs"] }` |
| Replay journal | `rakka-accel-cuda = { features = ["replay"] }` |
| Cluster-sharded placement | `rakka-accel-cuda = { features = ["cluster"] }` |
| Streams DSL on top | `rakka-accel-cuda = { features = ["streams"] }` |
| TelemetryExtension probes | `rakka-accel-cuda = { features = ["telemetry"] }` |

Aggregates compose: `core-libs` ⊂ `training-libs` ⊂ `full-cuda`.
The four rakka-0.2 integration features (`replay`, `cluster`,
`streams`, `telemetry`) are independent of the library aggregates.

For a goal-by-goal picker with explicit transitive-deps tables, see
[`docs/features-matrix.md`](../../../docs/features-matrix.md).

### Reaching for a blueprint sub-crate

```toml
[dependencies]
rakka-accel-cuda     = "0.2"
rakka-accel-patterns = "0.2"  # adds DynamicBatchingServer, etc.
```

Only adds the patterns crate; `train` / `agents` / `realtime`
stay out of your dep tree.

## Bootstrapping an `ActorSystem` + `DeviceActor`

```rust
use rakka_accel_cuda::prelude::*;
use rakka::prelude::*;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::create("my-app", Config::empty()).await?;

    // Real-mode device. Use `DeviceConfig::mock(0)` for no-GPU CI.
    let device = system.actor_of(
        DeviceActor::props(DeviceConfig::new(0)),
        "device-0",
    )?;

    // Now ask the device to allocate, copy, and run kernels.
    // See the `rakka-accel-device` skill.

    system.terminate().await;
    Ok(())
}
```

`Props::create` takes a factory closure so the supervisor can
re-instantiate the actor on restart. Don't capture mutable state in
the factory — use `Arc`-shared dependencies if you need to thread
configuration in.

## Mock mode for no-GPU CI

```rust
let device = system.actor_of(
    DeviceActor::props(DeviceConfig::mock(0)),  // ← was ::new(0)
    "device-0",
)?;
```

Every kernel call replies with `Err(GpuError::Unrecoverable("...
not supported in mock mode"))`. Use this to exercise the
supervision tree, message wiring, and `ContextReady` handshake on
hosts without a CUDA driver. The full feature matrix builds and
tests cleanly without a GPU.

GPU-only tests should be feature-gated behind `cuda-runtime-tests`
so they're skipped on no-GPU CI runners:

```rust
#![cfg(feature = "cuda-runtime-tests")]
// ...
```

## Running the no-GPU smoke

```bash
cargo run -p rakka-accel-cuda --example echo_no_gpu
cargo run -p rakka-accel-patterns --example batching_no_gpu
cargo run -p rakka-accel-patterns --example cascade_no_gpu
cargo run -p rakka-accel-patterns --example fair_share_no_gpu
cargo run -p rakka-accel-patterns --example moe_no_gpu
cargo run -p rakka-accel-patterns --example speculative_no_gpu
```

These are the canonical "does your dev box build and run the
plumbing" checks.

## Canonical references

- [`docs/getting-started.md`](../../../docs/getting-started.md) — the
  ten-minute tour.
- [`docs/features-matrix.md`](../../../docs/features-matrix.md) — pick
  the smallest dep footprint that fits your goal.
- [`docs/concepts.md`](../../../docs/concepts.md) — the five mental
  models (supervision, generation tokens, completion, streams,
  watch channel).
- `crates/rakka-accel-cuda/examples/echo_no_gpu.rs` — minimal
  end-to-end plumbing demo.

## Common pitfalls

- **Forgetting `tokio = { features = ["rt-multi-thread", ...] }`.**
  `DeviceActor::pre_start` spawns child actors and uses
  `tokio::spawn` internally — the runtime needs a multi-threaded
  scheduler.
- **Pinning to `rakka-accel = "0.2"` with no `features = ["cuda"]`.**
  The umbrella crate ships only the trait surface by default; you
  also need `cuda` to get the concrete `DeviceActor`. Most projects
  should depend on `rakka-accel-cuda` directly instead.
- **Building inside Docker without `--gpus all`.** The build
  succeeds because cudarc loads dynamically, but every kernel call
  surfaces `Unrecoverable("...driver not loadable")` at runtime. If
  you mean to run kernels, expose the GPU; if you don't, switch to
  mock mode.
