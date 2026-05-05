//! `axpy_dot` — pair of L1 ops (axpy + dot) via `BlasMsg::L1`.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example axpy_dot \
//!         --features cuda-runtime-tests
use std::time::Duration;

use atomr_accel_cuda::kernel::{AxpyRequest, BlasMsg, DotRequest};
use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use tokio::sync::oneshot;

const N: i32 = 1 << 20;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("atomr-accel-cuda-axpy-dot", Config::empty()).await?;
    let device = system.actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")?;

    let x = ask_alloc_f32(&device, N as usize).await?;
    let y = ask_alloc_f32(&device, N as usize).await?;

    let children = device
        .ask_with::<Option<KernelChildren>, _>(
            |reply| DeviceMsg::SnapshotChildren { reply },
            Duration::from_secs(10),
        )
        .await?
        .ok_or("BlasActor not yet ready")?;

    // y := 1.5*x + y
    let (axpy_tx, axpy_rx) = oneshot::channel();
    children.blas.tell(BlasMsg::L1(Box::new(AxpyRequest::<f32> {
        n: N,
        alpha: 1.5_f32,
        x: x.clone(),
        incx: 1,
        y,
        incy: 1,
        reply: axpy_tx,
    })));
    tokio::time::timeout(Duration::from_secs(15), axpy_rx).await???;
    println!("AXPY n={N} f32 completed.");

    // dot(x, x)
    let x2 = ask_alloc_f32(&device, N as usize).await?;
    let (dot_tx, dot_rx) = oneshot::channel();
    children.blas.tell(BlasMsg::L1(Box::new(DotRequest::<f32> {
        n: N,
        x: x.clone(),
        incx: 1,
        y: x2,
        incy: 1,
        reply: dot_tx,
    })));
    let result = tokio::time::timeout(Duration::from_secs(15), dot_rx).await???;
    println!("DOT result = {result}");

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
