//! Real-GPU integration test for `RngActor::FillUniformF32`.

#![cfg(all(feature = "cuda-runtime-tests", feature = "curand"))]

use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;

use rakka_accel_cuda::prelude::*;

const N: usize = 1024;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rng_uniform_fill_e2e() {
    let sys = ActorSystem::create("rng-e2e", Config::empty())
        .await
        .unwrap();
    let dev_cfg =
        DeviceConfig::new(0).with_libraries(EnabledLibraries::BLAS | EnabledLibraries::CURAND);
    let device = sys
        .actor_of(DeviceActor::props(dev_cfg), "device-0")
        .unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    let snap: Option<KernelChildren> = device
        .ask_with(
            move |tx| DeviceMsg::SnapshotChildren { reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap();
    let children = snap.expect("device ready");
    let rng = children.rng.expect("CURAND enabled");

    let buf: GpuRef<f32> = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len: N, reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();

    let r: Result<(), _> = rng
        .ask_with(
            move |tx| RngMsg::FillUniformF32 {
                dst: buf,
                reply: tx,
            },
            Duration::from_secs(10),
        )
        .await
        .unwrap();
    r.unwrap();

    sys.terminate().await;
}
