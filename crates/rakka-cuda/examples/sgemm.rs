//! `sgemm` — F1 exit-criterion demo.
//!
//! Spawns a `DeviceActor` against the real CUDA runtime, allocates three
//! N×N f32 buffers, and issues a single SGEMM through the actor pipeline.
//! Verifies that the actor pipeline (DeviceActor → ContextActor →
//! BlasActor → HostFnCompletion → reply) round-trips end-to-end on a
//! real GPU.
//!
//! F1 does not yet ship a D2H copy actor message, so this example only
//! verifies that the kernel completes without error. Numeric correctness
//! (`max |C - C_ref| < 1e-3`) requires the F2 Memcpy plumbing.
//!
//! Run on a GPU host:
//!     cargo run -p rakka-cuda --example sgemm --features cuda-runtime-tests

use std::time::Duration;

use rakka_cuda::prelude::*;
use rakka_core::actor::ActorSystem;
use rakka_config::Config;
use tokio::sync::oneshot;

const N: i32 = 4096;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("rakka-cuda-sgemm", Config::empty()).await?;
    let device = system.actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")?;

    let a = ask_alloc(&device, (N * N) as usize).await?;
    let b = ask_alloc(&device, (N * N) as usize).await?;
    let c = ask_alloc(&device, (N * N) as usize).await?;

    let (reply_tx, reply_rx) = oneshot::channel();
    device.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
        a,
        b,
        c,
        m: N,
        n: N,
        k: N,
        alpha: 1.0,
        beta: 0.0,
        reply: reply_tx,
    })));

    let result = tokio::time::timeout(Duration::from_secs(60), reply_rx).await??;
    result?;
    println!(
        "SGEMM {N}×{N} completed via the actor pipeline. \
         (D2H read-back not implemented in F1; add a Memcpy message in F2 \
         to verify numeric correctness.)"
    );

    system.terminate().await;
    Ok(())
}

async fn ask_alloc(
    device: &rakka_core::actor::ActorRef<DeviceMsg>,
    len: usize,
) -> Result<GpuRef<f32>, Box<dyn std::error::Error>> {
    let r = device
        .ask_with::<Result<GpuRef<f32>, GpuError>, _>(
            move |tx| DeviceMsg::Allocate { len, reply: tx },
            Duration::from_secs(10),
        )
        .await??;
    Ok(r)
}
