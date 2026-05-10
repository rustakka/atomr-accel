//! End-to-end real-GPU integration test for the Phase 5.1
//! CubHistogram dispatch (256-bin even-binned histogram of u8).

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::ActorSystem;
use atomr_core::prelude::Config;
use tokio::sync::oneshot;

use atomr_accel_cub::{cub_props, CubMsg, HistogramRequest};
use atomr_accel_cuda::completion::{CompletionStrategy, HostFnCompletion};
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use atomr_accel_cuda::gpu_ref::GpuRef;

const N: usize = 1 << 16;
const BINS: u32 = 256;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CUDA driver + NVRTC"]
async fn cub_histogram_u8_matches_host() {
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    if !matches!(probe, Ok(Ok(_))) {
        eprintln!("[skip] CUDA driver unavailable / dlsym mismatch");
        return;
    }

    let sys = ActorSystem::create("cub-hist-e2e", Config::empty())
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

    // Generate a deterministic byte stream cycling through every value.
    let host_in: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();
    let mut input_slice = stream.alloc_zeros::<u8>(N).expect("input alloc");
    stream
        .memcpy_htod(&host_in, &mut input_slice)
        .expect("htod");
    let input = GpuRef::new(Arc::new(input_slice), &cub_state);

    let bins_slice = stream
        .alloc_zeros::<u32>(BINS as usize)
        .expect("bins alloc");
    let bins = GpuRef::new(Arc::new(bins_slice), &cub_state);

    let (tx, rx) = oneshot::channel();
    cub.tell(CubMsg::Histogram(Box::new(HistogramRequest::new(
        input,
        bins.clone(),
        BINS,
        0.0,
        256.0,
        tx,
    ))));
    let res = tokio::time::timeout(Duration::from_secs(60), rx)
        .await
        .expect("histogram timed out")
        .expect("reply dropped");
    res.expect("histogram returned error");

    let host_bins: Vec<u32> = stream
        .clone_dtoh(bins.access().unwrap().as_ref())
        .expect("dtoh");
    let expected = N as u32 / BINS;
    for (b, c) in host_bins.iter().enumerate() {
        assert_eq!(*c, expected, "bin {b} count mismatch");
    }

    sys.terminate().await;
}
