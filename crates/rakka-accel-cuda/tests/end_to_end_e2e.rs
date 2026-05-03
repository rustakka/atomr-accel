//! Cross-actor real-GPU smoke test: spins up a single device,
//! Placement actor, ManagedAllocator, ReplayHarness, runs a small
//! workload (allocate + fill + sgemm), and verifies the surface
//! plumbs through cleanly. Gated on `cuda-runtime-tests`.
//!
//! This is a smoke test for the F2-F9 surface end-to-end. Multi-GPU
//! tests live in `nccl_world_e2e.rs` (gated additionally on `nccl`
//! and runtime device count >= 2).

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::sync::oneshot;

use rakka_accel_cuda::prelude::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn end_to_end_smoke() {
    let sys = ActorSystem::create("e2e", Config::empty()).await.unwrap();

    let device = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")
        .unwrap();
    let _placement = sys
        .actor_of(
            PlacementActor::props(
                vec![(0, device.clone())],
                Arc::new(LeastLoadedPolicy),
            ),
            "placement",
        )
        .unwrap();
    let _managed = sys
        .actor_of(ManagedAllocatorActor::props(), "managed")
        .unwrap();
    let replay = sys
        .actor_of(ReplayHarness::props(ReplayMode::Record), "replay")
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    // Allocate three buffers via the device actor.
    let len = 64 * 64;
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

    // Run an SGEMM.
    let (tx, rx) = oneshot::channel();
    device.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
        a, b, c,
        m: 64, n: 64, k: 64,
        alpha: 1.0, beta: 0.0,
        reply: tx,
    })));
    let result = tokio::time::timeout(Duration::from_secs(30), rx).await.unwrap().unwrap();
    result.unwrap();

    // Record an entry.
    replay.tell(ReplayMsg::Record(JournalEntry::DeviceCmd {
        ts_micros: 0,
        name: "sgemm".into(),
        payload: "64x64x64".into(),
    }));
    tokio::time::sleep(Duration::from_millis(50)).await;
    let (tx, rx) = oneshot::channel();
    replay.tell(ReplayMsg::Snapshot { reply: tx });
    let entries = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap();
    assert_eq!(entries.len(), 1);

    sys.terminate().await;
}
