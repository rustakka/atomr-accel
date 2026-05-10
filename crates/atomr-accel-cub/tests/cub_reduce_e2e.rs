//! End-to-end real-GPU integration test for the Phase 5.1 CubReduce
//! dispatch path. Boots an `ActorSystem` + `DeviceActor`, snapshots
//! the parent's `NvrtcActor` ref out of `KernelChildren`, spawns a
//! `CubActor` sibling, and submits a `ReduceRequest::<f32>` over a
//! 16K-element constant-fill buffer. Verifies the device-side sum
//! matches the host expectation within fp32 rounding tolerance.
//!
//! Skipped when the `cuda-runtime-tests` cargo feature is off (the
//! default). Run with:
//!
//! ```text
//! cargo xtask gpu-test cub
//! # or directly:
//! cargo test -p atomr-accel-cub --features cuda-runtime-tests \
//!     -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;
use std::time::Duration;

use atomr_core::actor::ActorSystem;
use atomr_core::prelude::Config;
use tokio::sync::oneshot;

use atomr_accel_cub::{cub_props, CubMsg, ReduceRequest, ReductionOp};
use atomr_accel_cuda::completion::{CompletionStrategy, HostFnCompletion};
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use atomr_accel_cuda::gpu_ref::GpuRef;

const N: usize = 16 * 1024;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CUDA driver + NVRTC"]
async fn cub_reduce_sum_f32_matches_host() {
    // Skip cleanly when no CUDA driver is available so the test
    // stays useful as a smoke probe on no-GPU CI. cudarc 0.19's
    // bindings reference `cuCoredumpDeregister*` symbols that older
    // drivers don't ship; the dlsym lookup panics rather than
    // returning Err, so we wrap the probe in `catch_unwind`.
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

    let sys = ActorSystem::create("cub-reduce-e2e", Config::empty())
        .await
        .unwrap();
    let device = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")
        .unwrap();

    // Wait for the ContextActor to finish initialising and stash its
    // KernelChildren on the device.
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Snapshot the live NvrtcActor + stream from the parent device.
    let children = device
        .ask_with(
            |tx| DeviceMsg::SnapshotChildren { reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .expect("ContextReady never fired — driver init failed");
    let nvrtc = children
        .nvrtc
        .expect("NvrtcActor not spawned (is the `nvrtc` cargo feature on?)");

    let stream = device
        .ask_with(
            |tx| DeviceMsg::SnapshotStream { reply: tx },
            Duration::from_secs(5),
        )
        .await
        .unwrap()
        .expect("device stream not available");

    // Spawn a CubActor as a sibling of the device, wired to the
    // device's NvrtcActor. We mint a fresh DeviceState bound to
    // device 0 — the kernel cache + GpuRef generation checks only
    // see this state, so the input/output we allocate below are
    // generation-consistent with the dispatcher's view.
    let cub_state = Arc::new(DeviceState::new(0));
    let completion: Arc<dyn CompletionStrategy> = Arc::new(HostFnCompletion::new());
    let cuda_ctx = stream.context().clone();
    let cub = sys
        .actor_of(
            cub_props(
                stream.clone(),
                completion,
                cub_state.clone(),
                cuda_ctx,
                Some(Arc::new(nvrtc)),
            ),
            "cub-actor",
        )
        .unwrap();

    // Allocate input + output buffers directly on the actor's stream
    // and wrap them in `GpuRef`s minted against `cub_state`.
    let mut input_slice = stream.alloc_zeros::<f32>(N).expect("input alloc");
    let host: Vec<f32> = vec![1.0_f32; N];
    stream.memcpy_htod(&host, &mut input_slice).expect("htod");
    let input = GpuRef::new(Arc::new(input_slice), &cub_state);

    let output_slice = stream.alloc_zeros::<f32>(1).expect("output alloc");
    let output = GpuRef::new(Arc::new(output_slice), &cub_state);

    let (tx, rx) = oneshot::channel();
    cub.tell(CubMsg::Reduce(Box::new(ReduceRequest::new(
        ReductionOp::Sum,
        input,
        output.clone(),
        tx,
    ))));
    let res = tokio::time::timeout(Duration::from_secs(60), rx)
        .await
        .expect("reduce timed out")
        .expect("reduce reply dropped");
    res.expect("reduce returned error");

    // Read back the single-element output.
    let out_slice = output.access().unwrap();
    let host_out: Vec<f32> = stream.clone_dtoh(out_slice.as_ref()).expect("dtoh");
    let observed = host_out[0];
    let expected = N as f32;
    let rel_err = (observed - expected).abs() / expected.max(1.0);
    assert!(
        rel_err < 1e-3,
        "cub reduce sum mismatch: observed={observed}, expected={expected}, rel_err={rel_err}"
    );

    sys.terminate().await;
}
