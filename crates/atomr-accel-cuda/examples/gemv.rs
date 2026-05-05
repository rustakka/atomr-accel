//! `gemv` — matrix-vector multiply via `BlasMsg::L2(GemvRequest::<f32>)`.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example gemv \
//!         --features cuda-runtime-tests
use std::time::Duration;

use atomr_accel_cuda::kernel::{BlasMsg, GemvRequest};
use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use cudarc::cublas::sys::cublasOperation_t;
use tokio::sync::oneshot;

const M: i32 = 1024;
const N: i32 = 1024;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("atomr-accel-cuda-gemv", Config::empty()).await?;
    let device = system.actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")?;

    let a = ask_alloc_f32(&device, (M * N) as usize).await?;
    let x = ask_alloc_f32(&device, N as usize).await?;
    let y = ask_alloc_f32(&device, M as usize).await?;

    let children = device
        .ask_with::<Option<KernelChildren>, _>(
            |reply| DeviceMsg::SnapshotChildren { reply },
            Duration::from_secs(10),
        )
        .await?
        .ok_or("BlasActor not yet ready")?;

    let (reply_tx, reply_rx) = oneshot::channel();
    children.blas.tell(BlasMsg::L2(Box::new(GemvRequest::<f32> {
        trans: cublasOperation_t::CUBLAS_OP_N,
        m: M,
        n: N,
        alpha: 1.0,
        beta: 0.0,
        a,
        lda: M,
        x,
        incx: 1,
        y,
        incy: 1,
        reply: reply_tx,
    })));

    tokio::time::timeout(Duration::from_secs(30), reply_rx).await???;
    println!("GEMV {M}×{N} f32 completed via the typed dispatcher path.");

    system.terminate().await;
    Ok(())
}

async fn ask_alloc_f32(
    device: &atomr_core::actor::ActorRef<DeviceMsg>,
    len: usize,
) -> Result<GpuRef<f32>, Box<dyn std::error::Error>> {
    let r = device
        .ask_with::<Result<GpuRef<f32>, GpuError>, _>(
            move |tx| DeviceMsg::AllocateF32 { len, reply: tx },
            Duration::from_secs(10),
        )
        .await??;
    Ok(r)
}
