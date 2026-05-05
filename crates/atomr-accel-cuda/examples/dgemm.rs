//! `dgemm` — f64 GEMM via the typed `BlasMsg::Gemm(GemmRequest::<f64>)`
//! path landed in Phase 1.
//!
//! Mirrors `examples/sgemm.rs` but exercises the dtype-generic
//! dispatcher path with f64. Verifies the kernel completes — D2H
//! readback for numeric correctness is left to integration tests.
//!
//! Run on a GPU host:
//!     cargo run -p atomr-accel-cuda --example dgemm \
//!         --features cuda-runtime-tests
use std::time::Duration;

use atomr_accel_cuda::kernel::{BlasMsg, GemmRequest};
use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use cudarc::cublas::sys::cublasOperation_t;
use tokio::sync::oneshot;

const N: i32 = 1024;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("atomr-accel-cuda-dgemm", Config::empty()).await?;
    let device = system.actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")?;

    let a = ask_alloc_f64(&device, (N * N) as usize).await?;
    let b = ask_alloc_f64(&device, (N * N) as usize).await?;
    let c = ask_alloc_f64(&device, (N * N) as usize).await?;

    // Snapshot the BlasActor address so we can talk to it directly.
    let children = device
        .ask_with::<Option<KernelChildren>, _>(
            |reply| DeviceMsg::SnapshotChildren { reply },
            Duration::from_secs(10),
        )
        .await?
        .ok_or("BlasActor not yet ready")?;

    let (reply_tx, reply_rx) = oneshot::channel();
    children.blas.tell(BlasMsg::gemm::<f64>(GemmRequest::<f64> {
        a,
        b,
        c,
        m: N,
        n: N,
        k: N,
        alpha: 1.0,
        beta: 0.0,
        trans_a: cublasOperation_t::CUBLAS_OP_N,
        trans_b: cublasOperation_t::CUBLAS_OP_N,
        lda: N,
        ldb: N,
        ldc: N,
        reply: reply_tx,
    }));

    tokio::time::timeout(Duration::from_secs(60), reply_rx).await???;
    println!("DGEMM {N}×{N} f64 completed via the typed dispatcher path.");

    system.terminate().await;
    Ok(())
}

async fn ask_alloc_f64(
    device: &atomr_core::actor::ActorRef<DeviceMsg>,
    len: usize,
) -> Result<GpuRef<f64>, Box<dyn std::error::Error>> {
    let r = device
        .ask_with::<Result<GpuRef<f64>, GpuError>, _>(
            move |tx| DeviceMsg::AllocateF64 { len, reply: tx },
            Duration::from_secs(10),
        )
        .await??;
    Ok(r)
}
