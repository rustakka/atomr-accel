//! End-to-end real-GPU integration test for the Phase 5.1 CubSelect
//! single-tile flagged-select dispatch.

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::ActorSystem;
use atomr_core::prelude::Config;
use tokio::sync::oneshot;

use atomr_accel_cub::{cub_props, CubMsg, SelectMode, SelectRequest};
use atomr_accel_cuda::completion::{CompletionStrategy, HostFnCompletion};
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use atomr_accel_cuda::gpu_ref::GpuRef;

const N: usize = 1024;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CUDA driver + NVRTC"]
async fn cub_select_flagged_i32_keeps_evens() {
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    if !matches!(probe, Ok(Ok(_))) {
        eprintln!("[skip] CUDA driver unavailable / dlsym mismatch");
        return;
    }

    let sys = ActorSystem::create("cub-select-e2e", Config::empty())
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

    let host_in: Vec<i32> = (0..N as i32).collect();
    let host_flags: Vec<u8> = (0..N).map(|i| if i % 2 == 0 { 1 } else { 0 }).collect();

    let mut input_slice = stream.alloc_zeros::<i32>(N).expect("input");
    stream.memcpy_htod(&host_in, &mut input_slice).expect("htod input");
    let input = GpuRef::new(Arc::new(input_slice), &cub_state);

    let mut flags_slice = stream.alloc_zeros::<u8>(N).expect("flags");
    stream.memcpy_htod(&host_flags, &mut flags_slice).expect("htod flags");
    let flags = GpuRef::new(Arc::new(flags_slice), &cub_state);

    let output_slice = stream.alloc_zeros::<i32>(N).expect("output");
    let output = GpuRef::new(Arc::new(output_slice), &cub_state);

    let num_selected_slice = stream.alloc_zeros::<u32>(1).expect("num_selected");
    let num_selected = GpuRef::new(Arc::new(num_selected_slice), &cub_state);

    let (tx, rx) = oneshot::channel();
    cub.tell(CubMsg::Select(Box::new(
        SelectRequest::new(SelectMode::Flagged, input, output.clone(), num_selected.clone(), tx)
            .with_flags(flags),
    )));
    let res = tokio::time::timeout(Duration::from_secs(60), rx)
        .await
        .expect("select timed out")
        .expect("reply dropped");
    res.expect("select returned error");

    let host_out: Vec<i32> = stream
        .clone_dtoh(output.access().unwrap().as_ref())
        .expect("dtoh out");
    let host_count: Vec<u32> = stream
        .clone_dtoh(num_selected.access().unwrap().as_ref())
        .expect("dtoh count");
    assert_eq!(host_count[0], (N / 2) as u32, "selected count");
    let expected: Vec<i32> = (0..N as i32).filter(|i| i % 2 == 0).collect();
    assert_eq!(&host_out[..expected.len()], expected.as_slice());

    sys.terminate().await;
}
