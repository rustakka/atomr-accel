//! End-to-end real-GPU integration test for the Phase 5.1
//! CubSegmentedReduce one-CTA-per-segment dispatch.

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::ActorSystem;
use atomr_core::prelude::Config;
use tokio::sync::oneshot;

use atomr_accel_cub::{cub_props, CubMsg, ReductionOp, SegmentedReduceRequest};
use atomr_accel_cuda::completion::{CompletionStrategy, HostFnCompletion};
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use atomr_accel_cuda::gpu_ref::GpuRef;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CUDA driver + NVRTC"]
async fn cub_segmented_reduce_sum_f32_three_segments() {
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    if !matches!(probe, Ok(Ok(_))) {
        eprintln!("[skip] CUDA driver unavailable / dlsym mismatch");
        return;
    }

    let sys = ActorSystem::create("cub-seg-e2e", Config::empty())
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

    // 3 segments: lengths 5, 7, 4 -> input has 16 elements.
    // Segment 0: 1..=5  -> sum = 15
    // Segment 1: 6..=12 -> sum = 63
    // Segment 2: 13..=16 -> sum = 58
    let host_in: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let host_begin: Vec<i32> = vec![0, 5, 12];
    let host_end: Vec<i32> = vec![5, 12, 16];

    let mut input_slice = stream.alloc_zeros::<f32>(16).expect("input");
    stream
        .memcpy_htod(&host_in, &mut input_slice)
        .expect("htod input");
    let input = GpuRef::new(Arc::new(input_slice), &cub_state);

    let mut begin_slice = stream.alloc_zeros::<i32>(3).expect("begin");
    stream
        .memcpy_htod(&host_begin, &mut begin_slice)
        .expect("htod begin");
    let begin = GpuRef::new(Arc::new(begin_slice), &cub_state);

    let mut end_slice = stream.alloc_zeros::<i32>(3).expect("end");
    stream
        .memcpy_htod(&host_end, &mut end_slice)
        .expect("htod end");
    let end = GpuRef::new(Arc::new(end_slice), &cub_state);

    let output_slice = stream.alloc_zeros::<f32>(3).expect("output");
    let output = GpuRef::new(Arc::new(output_slice), &cub_state);

    let (tx, rx) = oneshot::channel();
    cub.tell(CubMsg::SegmentedReduce(Box::new(
        SegmentedReduceRequest::new(ReductionOp::Sum, input, output.clone(), begin, end, 3, tx),
    )));
    let res = tokio::time::timeout(Duration::from_secs(60), rx)
        .await
        .expect("seg-reduce timed out")
        .expect("reply dropped");
    res.expect("seg-reduce returned error");

    let host_out: Vec<f32> = stream
        .clone_dtoh(output.access().unwrap().as_ref())
        .expect("dtoh");
    assert!(
        (host_out[0] - 15.0).abs() < 1e-3,
        "seg 0 sum: {}",
        host_out[0]
    );
    assert!(
        (host_out[1] - 63.0).abs() < 1e-3,
        "seg 1 sum: {}",
        host_out[1]
    );
    assert!(
        (host_out[2] - 58.0).abs() < 1e-3,
        "seg 2 sum: {}",
        host_out[2]
    );

    sys.terminate().await;
}
