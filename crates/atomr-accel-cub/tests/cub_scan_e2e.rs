//! End-to-end real-GPU integration test for the Phase 5.1 CubScan
//! dispatch path. Inclusive scan over `1..=10_000_i32` should yield
//! the triangular-number sequence `n*(n+1)/2`.

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::ActorSystem;
use atomr_core::prelude::Config;
use tokio::sync::oneshot;

use atomr_accel_cub::{cub_props, CubMsg, ScanKind, ScanRequest};
use atomr_accel_cuda::completion::{CompletionStrategy, HostFnCompletion};
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use atomr_accel_cuda::gpu_ref::GpuRef;

const N: usize = 10_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CUDA driver + NVRTC"]
async fn cub_scan_inclusive_i32_matches_host() {
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    match probe {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            eprintln!("[skip] CUDA driver init failed: {e}");
            return;
        }
        Err(_) => {
            eprintln!("[skip] cudarc panicked on dlsym (driver older than its bindings)");
            return;
        }
    }

    let sys = ActorSystem::create("cub-scan-e2e", Config::empty())
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

    let host_in: Vec<i32> = (1..=N as i32).collect();
    let mut input_slice = stream.alloc_zeros::<i32>(N).expect("input alloc");
    stream
        .memcpy_htod(&host_in, &mut input_slice)
        .expect("htod");
    let input = GpuRef::new(Arc::new(input_slice), &cub_state);

    let output_slice = stream.alloc_zeros::<i32>(N).expect("output alloc");
    let output = GpuRef::new(Arc::new(output_slice), &cub_state);

    let (tx, rx) = oneshot::channel();
    cub.tell(CubMsg::Scan(Box::new(ScanRequest::new(
        ScanKind::Inclusive,
        input,
        output.clone(),
        tx,
    ))));
    let res = tokio::time::timeout(Duration::from_secs(60), rx)
        .await
        .expect("scan timed out")
        .expect("scan reply dropped");
    res.expect("scan returned error");

    let host_out: Vec<i32> = stream
        .clone_dtoh(output.access().unwrap().as_ref())
        .expect("dtoh");

    // Triangular numbers: prefix[k] = (k+1)*(k+2)/2 for input 1..=N.
    for (k, observed) in host_out.iter().enumerate() {
        let n = (k + 1) as i64;
        let expected = (n * (n + 1) / 2) as i32;
        assert_eq!(*observed, expected, "scan mismatch at index {k}");
    }

    sys.terminate().await;
}
