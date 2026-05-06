//! `echo_no_gpu` — F1 plumbing demo runnable on hosts without a GPU.
//!
//! Spins up an `ActorSystem`, starts a `DeviceActor` in mock mode, sends
//! it an `Allocate` request, and prints the resulting (synthetic) error.
//! Demonstrates that the supervision tree, message wiring, ContextReady
//! handshake, and pending-queue drain all work end-to-end without
//! touching cudarc's runtime.
//!
//! Build / run:
//!     cargo run -p atomr-accel-cuda --example echo_no_gpu
//!
//! Compare to `examples/sgemm.rs` (gated behind `--features cuda-runtime-tests`)
//! for the real GPU path.
#![allow(deprecated)]

use std::time::Duration;

use atomr_accel_cuda::prelude::*;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use tokio::sync::oneshot;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("atomr-accel-cuda-demo", Config::empty()).await?;
    let device = system.actor_of(DeviceActor::props(DeviceConfig::mock(0)), "device-0")?;

    println!("DeviceActor (mock) spawned. Sending Allocate request...");
    let (tx, rx) = oneshot::channel();
    device.tell(DeviceMsg::Allocate {
        len: 1024,
        reply: tx,
    });

    let reply = tokio::time::timeout(Duration::from_secs(5), rx).await??;
    match reply {
        Ok(buf) => {
            println!(
                "Allocated buffer (unexpected in mock mode): len={}",
                buf.len()
            );
        }
        Err(GpuError::Unrecoverable(msg)) => {
            println!("Got expected mock-mode error: {msg}");
        }
        Err(e) => {
            println!("Got error: {e}");
        }
    }

    println!("Plumbing OK. Terminating system...");
    system.terminate().await;
    Ok(())
}
