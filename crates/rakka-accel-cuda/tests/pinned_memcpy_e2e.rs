//! Round-trip a pinned host buffer through `CopyFromHost` →
//! `CopyToHost`. Exercises the full pinned-pool / device-actor
//! memcpy path. Gated behind `cuda-runtime-tests` because both
//! `cuMemHostAlloc` and `memcpy_htod_async` need a live CUDA
//! driver.

#![cfg(feature = "cuda-runtime-tests")]

use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::sync::oneshot;

use rakka_accel_cuda::prelude::*;

const N: usize = 1024;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinned_h2d_d2h_roundtrip() {
    let sys = ActorSystem::create("pinned-e2e", Config::empty())
        .await
        .unwrap();
    let device = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")
        .unwrap();
    let pool = sys
        .actor_of(
            PinnedBufferPool::props(PinnedBufferPoolConfig {
                initial_buffers: 0,
                max_buffers: 4,
                buffer_capacity_bytes: N * std::mem::size_of::<f32>(),
                allow_oversize: true,
            }),
            "pool",
        )
        .unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Allocate a pinned source buffer.
    let bytes = N * std::mem::size_of::<f32>();
    let handle = pool
        .ask_with(
            move |tx| PinnedPoolMsg::Acquire {
                len_bytes: bytes,
                reply: tx,
            },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();
    let mut src_pinned: PinnedBuf<f32> = handle.into_typed::<f32>(N).unwrap();
    for (i, slot) in src_pinned.as_mut_slice().iter_mut().enumerate() {
        *slot = i as f32;
    }

    // Allocate a destination GPU buffer.
    let gpu = device
        .ask_with(
            move |tx| DeviceMsg::AllocateF32 { len: N, reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();

    // H2D.
    let (tx, rx) = oneshot::channel();
    device.tell(DeviceMsg::CopyFromHostF32 {
        src: HostBuf::Pinned(src_pinned),
        dst: gpu.clone(),
        reply: tx,
    });
    let _ = tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .unwrap()
        .unwrap()
        .unwrap();

    // D2H into a fresh pinned buffer.
    let dst_handle = pool
        .ask_with(
            move |tx| PinnedPoolMsg::Acquire {
                len_bytes: bytes,
                reply: tx,
            },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();
    let dst_pinned: PinnedBuf<f32> = dst_handle.into_typed::<f32>(N).unwrap();
    let (tx, rx) = oneshot::channel();
    device.tell(DeviceMsg::CopyToHostF32 {
        src: gpu,
        dst: HostBuf::Pinned(dst_pinned),
        reply: tx,
    });
    let returned = tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let pinned = match returned {
        HostBuf::Pinned(p) => p,
        HostBuf::Owned(_) => panic!("expected pinned dst back"),
    };
    let read: Vec<f32> = pinned.as_slice().to_vec();
    for (i, v) in read.iter().enumerate() {
        assert!((*v - i as f32).abs() < 1e-5, "idx {i}: {v}");
    }

    sys.terminate().await;
}
