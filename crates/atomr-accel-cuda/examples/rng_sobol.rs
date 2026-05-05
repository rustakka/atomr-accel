//! Demonstrates `RngActor` switched to a Sobol32 quasi-random
//! generator and filling a device buffer with low-discrepancy
//! samples.
//!
//! Build with: `cargo run -p atomr-accel-cuda --example rng_sobol \
//!     --features cuda-runtime-tests,curand,curand-quasirandom`

use std::time::Duration;

use atomr_config::Config;
use atomr_core::actor::ActorSystem;

use atomr_accel_cuda::kernel::rng::RngGeneratorKind;
use atomr_accel_cuda::prelude::*;

const N: usize = 1024;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let sys = ActorSystem::create("rng-sobol", Config::empty()).await?;
    let dev_cfg =
        DeviceConfig::new(0).with_libraries(EnabledLibraries::BLAS | EnabledLibraries::CURAND);
    let device = sys.actor_of(DeviceActor::props(dev_cfg), "device-0")?;

    tokio::time::sleep(Duration::from_millis(500)).await;

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

    // Switch to a Sobol32 generator.
    let _: () = atomr_core::actor::ActorRef::ask_with(
        &rng,
        move |tx| RngMsg::SetGenerator {
            kind: RngGeneratorKind::Sobol32,
            reply: tx,
        },
        Duration::from_secs(5),
    )
    .await??;

    // Allocate target buffer.
    let buf: GpuRef<f32> = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len: N, reply: tx },
            Duration::from_secs(5),
        )
        .await??;

    // Fire the fill via the modern dispatch path.
    let (tx, rx) = tokio::sync::oneshot::channel();
    let req = FillRequest::<f32> {
        buf,
        dist: Distribution::Uniform { lo: 0.0, hi: 1.0 },
        reply: tx,
    };
    rng.tell(RngMsg::Fill(Box::new(req)));
    rx.await??;

    println!("Filled GpuRef<f32> len={N} with Sobol32 quasi-random samples");

    sys.terminate().await;
    Ok(())
}
