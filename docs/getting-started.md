# Getting started with atomr-accel

A ten-minute tour. By the end you'll have:

1. Built and tested the workspace with no GPU.
2. Spawned a `DeviceActor` and seen the supervision tree light up.
3. Issued a real cuBLAS SGEMM through the actor pipeline (if you have
   a GPU).
4. Wired in a custom NVRTC kernel.

## Prerequisites

- Rust **1.78+** (workspace toolchain in `rust-toolchain.toml`).
- A sibling clone of [atomr](../../atomr) at `../atomr` (path
  dependency, version 0.1.x).
- Optional but recommended: an NVIDIA GPU plus the
  [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (driver
  + libraries). Without these, the workspace still builds and unit
  tests pass — cudarc loads CUDA dynamically.

```
your-workspace/
├── atomr/         # the atomr actor runtime
└── atomr-accel/   # this repo
```

## 1. Build with no GPU

```bash
cargo check --workspace --no-default-features
cargo test  --workspace --no-default-features
```

You should see ~60 tests pass in under 30 seconds. None of them touch
the CUDA driver.

Run the smallest example:

```bash
cargo run -p atomr-accel --example echo_no_gpu
```

Expected output:

```
DeviceActor (mock) spawned. Sending Allocate request...
ContextActor (mock) ready
Got expected mock-mode error: alloc not supported in mock mode
Plumbing OK. Terminating system...
```

What just happened:

1. `ActorSystem::create` spun up atomr.
2. `DeviceActor::props(DeviceConfig::mock(0))` built a **mock-mode**
   device — supervision and message wiring run, but cudarc calls are
   stubbed.
3. The `pre_start` hook spawned a `ContextActor` child.
4. The `ContextActor` reported `ContextReady` back to the parent.
5. The pending `Allocate` request drained from the parent's queue
   into the (mock) `BlasActor`, which replied with a deliberate
   error.

Every real CUDA path takes the same shape — only the inner kernel
call changes.

## 2. Run on real hardware

If you have a GPU and the CUDA toolkit installed:

```bash
cargo run -p atomr-accel --example sgemm --features cuda-runtime-tests
```

This spawns a real `DeviceActor`, allocates three N×N f32 buffers
on-device, and issues an [`SGEMM`][cublas-sgemm] through the
`BlasActor`. The reply oneshot fires when the kernel completes.

Try the others:

```bash
cargo run -p atomr-accel --example rng_uniform --features cuda-runtime-tests,curand
cargo run -p atomr-accel --example fft_1d      --features cuda-runtime-tests,cufft
cargo run -p atomr-accel --example jit_relu    --features cuda-runtime-tests,nvrtc
```

The `jit_relu` example is the most instructive: it compiles a CUDA-C
ReLU kernel via [`nvrtcCompileProgram`][nvrtc-compile], loads it,
launches it on a freshly allocated buffer, and prints the result.
That's the full NVRTC roundtrip in ~80 lines of Rust.

## 3. Your first request from scratch

```rust
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use atomr_accel_cuda::prelude::*;
use std::time::Duration;
use tokio::sync::oneshot;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    let system = ActorSystem::create("my-gpu-app", Config::empty()).await?;

    // 1. Spawn a device. DeviceConfig::new is real-mode; ::mock(0) for tests.
    let device = system.actor_of(
        DeviceActor::props(DeviceConfig::new(0)),
        "device-0",
    )?;

    // 2. Allocate two buffers on-device. Each returns a typed GpuRef<T>.
    let len = 1024;
    let a: GpuRef<f32> = device
        .ask_with(|tx| DeviceMsg::AllocateF32 { len, reply: tx },
                  Duration::from_secs(5))
        .await??;
    let b: GpuRef<f32> = device
        .ask_with(|tx| DeviceMsg::AllocateF32 { len, reply: tx },
                  Duration::from_secs(5))
        .await??;

    // 3. Upload data from a host buffer.
    let host = vec![1.0f32; len];
    device.ask_with(
        |tx| DeviceMsg::CopyFromHostF32 {
            src: HostBuf::Vec(host),
            dst: a.clone(),
            reply: tx,
        },
        Duration::from_secs(5),
    ).await??;

    // 4. Read it back.
    let got = device.ask_with(
        |tx| DeviceMsg::CopyToHostF32 {
            src: a,
            dst: HostBuf::Vec(vec![0.0; len]),
            reply: tx,
        },
        Duration::from_secs(5),
    ).await??;

    println!("Roundtrip OK: {:?}", &match got { HostBuf::Vec(v) => v, _ => unreachable!() }[..4]);

    let _ = b; // device drops it cleanly via the supervisor
    system.terminate().await;
    Ok(())
}
```

A few things this demonstrates:

- `device.ask_with(builder, timeout)` is the typed ask: pass a
  closure that builds the message given a reply `oneshot::Sender`,
  get back the awaited value.
- `GpuRef<T>` is a typed device pointer with a generation token. If
  the underlying `CudaContext` is rebuilt while you hold one, the
  next call surfaces `GpuError::GpuRefStale` immediately — no silent
  data corruption.
- `HostBuf` is the H2D/D2H envelope: `HostBuf::Vec(v)` for owned
  `Vec<T>`, `HostBuf::Pinned(buf)` for [page-locked][cuda-pinned]
  buffers from `PinnedBufferPool`.

## 4. JIT-compile a kernel

```rust
let nvrtc = /* ActorRef<NvrtcMsg> from ContextActor::SnapshotChildren */;

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

The compiled `KernelHandle` is generation-validated: if the context
is rebuilt between `Compile` and `Launch`, the launch fails fast with
`GpuError::GpuRefStale`. Re-issue `Compile` on the new context.

## 5. Where to go next

- **[concepts.md](concepts.md)** — supervision, completion strategies,
  generation tokens, stream allocators. The "why" behind the API.
- **[architecture.md](architecture.md)** — the full design narrative
  with NVIDIA references for every primitive.
- **`crates/atomr-accel-patterns/examples/`** — concrete patterns
  (batching, cascade, MoE, fair-share, speculative decoding) that
  ship with no-GPU demo runners.
- **`crates/atomr-accel-cuda/tests/end_to_end_e2e.rs`** — multi-actor
  smoke that allocates, copies, runs SGEMM, and reads back.

[cublas-sgemm]: https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
[nvrtc-compile]: https://docs.nvidia.com/cuda/nvrtc/index.html#group__error_1ga0e0b48c4e6f7e69dbb5e1d8c6c58c1d8
[cuda-pinned]: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory
