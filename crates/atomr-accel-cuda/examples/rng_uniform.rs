//! Demonstrates `RngActor` filling a device buffer with uniform
//! random f32 values via the public `SnapshotChildren` accessor.
//!
//! Build with: `cargo run -p atomr-accel-cuda --example rng_uniform \
//!     --features cuda-runtime-tests,curand`

use std::time::Duration;

use atomr_config::Config;
use atomr_core::actor::ActorSystem;

use atomr_accel_cuda::prelude::*;

const N: usize = 1024;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let sys = ActorSystem::create("rng-demo", Config::empty()).await?;
    let dev_cfg =
        DeviceConfig::new(0).with_libraries(EnabledLibraries::BLAS | EnabledLibraries::CURAND);
    let device = sys.actor_of(DeviceActor::props(dev_cfg), "device-0")?;

    // Wait briefly for context init.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Pull the kernel children so we can talk to RngActor directly.
    let snapshot: Option<KernelChildren> = device
        .ask_with(
            move |tx| DeviceMsg::SnapshotChildren { reply: tx },
            Duration::from_secs(5),
        )
        .await?;
    let children = snapshot.ok_or("device not yet ready")?;

    let rng = children
        .rng
        .ok_or("CURAND library not enabled in DeviceConfig")?;

    // Allocate target buffer.
    let buf: GpuRef<f32> = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len: N, reply: tx },
            Duration::from_secs(5),
        )
        .await??;

    // Fire the fill.
    let r: Result<(), _> = atomr_core::actor::ActorRef::ask_with(
        &rng,
        move |tx| RngMsg::FillUniformF32 {
            dst: buf,
            reply: tx,
        },
        Duration::from_secs(10),
    )
    .await?;
    r?;
    println!("Filled GpuRef<f32> len={N} with uniform random values");

    sys.terminate().await;
    Ok(())
}
