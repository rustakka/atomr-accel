//! End-to-end real-GPU integration test for the Phase 5.1 CubSort
//! single-tile dispatch (n ≤ 1024).

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::ActorSystem;
use atomr_core::prelude::Config;
use tokio::sync::oneshot;

use atomr_accel_cub::{cub_props, CubMsg, SortDirection, SortRequest};
use atomr_accel_cuda::completion::{CompletionStrategy, HostFnCompletion};
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use atomr_accel_cuda::gpu_ref::GpuRef;

const N: usize = 1024;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CUDA driver + NVRTC"]
async fn cub_sort_asc_u32_single_block() {
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    if !matches!(probe, Ok(Ok(_))) {
        eprintln!("[skip] CUDA driver unavailable / dlsym mismatch");
        return;
    }

    let sys = ActorSystem::create("cub-sort-e2e", Config::empty())
        .await
        .unwrap();
    let device = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")
        .unwrap();
    tokio::time::sleep(Duration::from_secs(1)).await;

    let children = device
        .ask_with(
            |tx| DeviceMsg::SnapshotChildren { reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();
    let nvrtc = children.nvrtc.expect("nvrtc actor");
    let stream = device
        .ask_with(
            |tx| DeviceMsg::SnapshotStream { reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .unwrap();

    let cub_state = Arc::new(DeviceState::new(0));
    let completion: Arc<dyn CompletionStrategy> = Arc::new(HostFnCompletion::new());
    let cub = sys
        .actor_of(
            cub_props(
                stream.clone(),
                completion,
                cub_state.clone(),
                stream.context().clone(),
                Some(Arc::new(nvrtc)),
            ),
            "cub-actor",
        )
        .unwrap();

    // Reverse-sorted input: sort should produce 1..=N.
    let mut host_in: Vec<u32> = (1..=N as u32).rev().collect();
    let mut keys_in_slice = stream.alloc_zeros::<u32>(N).expect("keys_in alloc");
    stream
        .memcpy_htod(&host_in, &mut keys_in_slice)
        .expect("htod");
    let keys_in = GpuRef::new(Arc::new(keys_in_slice), &cub_state);

    let keys_out_slice = stream.alloc_zeros::<u32>(N).expect("keys_out alloc");
    let keys_out = GpuRef::new(Arc::new(keys_out_slice), &cub_state);

    let (tx, rx) = oneshot::channel();
    cub.tell(CubMsg::Sort(Box::new(SortRequest::keys_only(
        SortDirection::Ascending,
        keys_in,
        keys_out.clone(),
        tx,
    ))));
    let res = tokio::time::timeout(Duration::from_secs(60), rx)
        .await
        .expect("sort timed out")
        .expect("reply dropped");
    res.expect("sort returned error");

    let host_out: Vec<u32> = stream
        .clone_dtoh(keys_out.access().unwrap().as_ref())
        .expect("dtoh");
    host_in.sort();
    assert_eq!(
        host_out, host_in,
        "single-block radix sort produced wrong order"
    );

    sys.terminate().await;
}
