//! Demonstrates `FftActor::Forward1dR2C` — allocate real input,
//! complex output, run a 1024-point R2C FFT.
//!
//! Build with: `cargo run -p atomr-accel-cuda --example fft_1d \
//!     --features cuda-runtime-tests,cufft`

use std::time::Duration;

use atomr_config::Config;
use atomr_core::actor::ActorSystem;

use atomr_accel_cuda::prelude::*;

const N: i32 = 1024;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let sys = ActorSystem::create("fft-demo", Config::empty()).await?;
    let dev_cfg =
        DeviceConfig::new(0).with_libraries(EnabledLibraries::BLAS | EnabledLibraries::CUFFT);
    let device = sys.actor_of(DeviceActor::props(dev_cfg), "device-0")?;
    tokio::time::sleep(Duration::from_millis(500)).await;

    let snapshot: Option<KernelChildren> = device
        .ask_with(
            move |tx| DeviceMsg::SnapshotChildren { reply: tx },
            Duration::from_secs(5),
        )
        .await?;
    let children = snapshot.ok_or("device not yet ready")?;
    let fft = children.fft.ok_or("CUFFT not enabled")?;

    let real_buf: GpuRef<f32> = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 {
                len: N as usize,
                reply: tx,
            },
            Duration::from_secs(5),
        )
        .await??;
    // Complex output buffer would normally come from
    // `DeviceMsg::AllocateComplex32`; F2 doesn't ship that yet.
    // Document the wiring shape and stop here.
    println!(
        "Allocated real GpuRef<f32> len={}; FFT actor available at {:?}",
        real_buf.len(),
        fft.path()
    );
    sys.terminate().await;
    Ok(())
}
