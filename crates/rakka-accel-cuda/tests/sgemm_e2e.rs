//! End-to-end real-GPU integration test for the SGEMM path.
//!
//! Gated on `cuda-runtime-tests` because it actually allocates a
//! `CudaContext`. Skipped on no-GPU CI.

#![cfg(feature = "cuda-runtime-tests")]

use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::sync::oneshot;

use rakka_accel_cuda::prelude::*;

const N: i32 = 64;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sgemm_64x64_via_actors() {
    let sys = ActorSystem::create("sgemm-e2e", Config::empty()).await.unwrap();
    let device = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")
        .unwrap();

    // Wait for context init.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let len = (N * N) as usize;
    let a = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len, reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();
    let b = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len, reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();
    let c = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len, reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();

    let (tx, rx) = oneshot::channel();
    device.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
        a, b, c, m: N, n: N, k: N,
        alpha: 1.0, beta: 0.0,
        reply: tx,
    })));
    let res = tokio::time::timeout(Duration::from_secs(30), rx).await.unwrap().unwrap();
    res.unwrap();

    sys.terminate().await;
}
