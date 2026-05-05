//! Demonstrates the Phase-1 cuFFT surface: 3D R2C f32 + the typed
//! `FftRequest<T>` boxed-dispatch path through `FftMsg::Exec`.
//!
//! The example wires up a `DeviceActor` with `cufft` enabled,
//! constructs a 3D plan key for a 16×16×16 R2C transform, and shows
//! the request shape. Running an actual transform requires complex
//! buffer allocation (`AllocateComplex32`) which lands with Phase 1's
//! device-actor expansion; here we stop after structurally building
//! the request to keep the example dependency-light.
//!
//! Build with: `cargo run -p atomr-accel-cuda --example fft_3d \
//!     --features cuda-runtime-tests,cufft`

use std::time::Duration;

use atomr_config::Config;
use atomr_core::actor::ActorSystem;

use atomr_accel_cuda::prelude::*;

const NX: i32 = 16;
const NY: i32 = 16;
const NZ: i32 = 16;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().init();

    let sys = ActorSystem::create("fft-3d-demo", Config::empty()).await?;
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
                len: (NX * NY * NZ) as usize,
                reply: tx,
            },
            Duration::from_secs(5),
        )
        .await??;

    // Structural build of a 3D R2C plan key. `FftRequest<f32>` would
    // wrap byte-cast `GpuRef<u8>` inputs/outputs and route through
    // `FftMsg::Exec`. Complex output allocation isn't shipped yet
    // (lands with Phase 1's `AllocateComplex32`).
    let plan_key = PlanKey::plan_3d(NX, NY, NZ, FftKind::R2C);
    println!(
        "Allocated real GpuRef<f32> len={}; FFT actor available at {:?}",
        real_buf.len(),
        fft.path()
    );
    println!(
        "Built 3D R2C plan key: rank={} dims={:?} kind={:?} dtype={:?} batch={}",
        plan_key.rank, plan_key.dims, plan_key.kind, plan_key.dtype, plan_key.batch
    );

    sys.terminate().await;
    Ok(())
}
