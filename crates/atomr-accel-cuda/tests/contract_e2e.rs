//! End-to-end test for `TensorActor::Contract` against a real cuTENSOR
//! handle. Gated on `cuda-runtime-tests,cutensor` so it only runs on
//! hosts with a CUDA driver and libcutensor.so available.
//!
//! Computes a simple matrix-matrix multiply expressed as a contraction
//! "ij,jk->ik" with 2×2 inputs.

#![cfg(all(feature = "cuda-runtime-tests", feature = "cutensor"))]

use std::sync::Arc;
use std::time::Duration;

use atomr_accel_cuda::completion::HostFnCompletion;
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState, HostBuf};
use atomr_accel_cuda::kernel::{TensorActor, TensorMsg, TensorSpec};
use atomr_accel_cuda::stream::SingleStreamAllocator;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cutensor_matmul_via_contraction() {
    let sys = ActorSystem::create("contract-e2e", Config::empty())
        .await
        .unwrap();
    let dev = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "dev0")
        .unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::SnapshotContext { reply: tx });
    let ctx = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .expect("snapshot")
        .expect("oneshot")
        .expect("context not yet built");
    let stream = Arc::new(ctx.new_stream().expect("new_stream"));
    let allocator = Arc::new(SingleStreamAllocator::new(stream.clone()));
    let completion = Arc::new(HostFnCompletion::new());
    let state = Arc::new(DeviceState::new(0));

    let tensor = sys
        .actor_of(
            TensorActor::props(stream, allocator, completion, state),
            "tensor",
        )
        .unwrap();

    // 2×2 matmul: A=[[1,2],[3,4]] (col-major: [1,3,2,4]),
    //             B=[[5,6],[7,8]] (col-major: [5,7,6,8]),
    //             C=A·B=[[19,22],[43,50]] (col-major: [19,43,22,50]).
    let a_h = vec![1.0f32, 3.0, 2.0, 4.0];
    let b_h = vec![5.0f32, 7.0, 6.0, 8.0];
    let c_h = vec![0.0f32; 4];

    let a_buf = alloc_and_copy(&dev, &a_h).await;
    let b_buf = alloc_and_copy(&dev, &b_h).await;
    let c_buf = alloc_and_copy(&dev, &c_h).await;

    let (tx, rx) = oneshot::channel();
    tensor.tell(TensorMsg::Contract {
        a: TensorSpec {
            buf: a_buf,
            extent: vec![2, 2],
            stride: vec![],
            modes: vec![1, 2],
        },
        b: TensorSpec {
            buf: b_buf,
            extent: vec![2, 2],
            stride: vec![],
            modes: vec![2, 3],
        },
        c: TensorSpec {
            buf: c_buf.clone(),
            extent: vec![2, 2],
            stride: vec![],
            modes: vec![1, 3],
        },
        alpha: 1.0,
        beta: 0.0,
        reply: tx,
    });
    tokio::time::timeout(Duration::from_secs(15), rx)
        .await
        .expect("Contract timeout")
        .expect("oneshot dropped")
        .expect("Contract returned error");

    let c_out = copy_to_host(&dev, &c_buf, 4).await;
    // Column-major layout: c[0,0]=19, c[1,0]=43, c[0,1]=22, c[1,1]=50.
    assert!((c_out[0] - 19.0).abs() < 1e-3, "c[0,0]={}", c_out[0]);
    assert!((c_out[1] - 43.0).abs() < 1e-3, "c[1,0]={}", c_out[1]);
    assert!((c_out[2] - 22.0).abs() < 1e-3, "c[0,1]={}", c_out[2]);
    assert!((c_out[3] - 50.0).abs() < 1e-3, "c[1,1]={}", c_out[3]);

    sys.terminate().await;
}

async fn alloc_and_copy(
    dev: &atomr_core::actor::ActorRef<DeviceMsg>,
    host: &[f32],
) -> atomr_accel_cuda::gpu_ref::GpuRef<f32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 {
        len: host.len(),
        reply: tx,
    });
    let g = rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyFromHostF32 {
        src: HostBuf::Vec(host.to_vec()),
        dst: g.clone(),
        reply: tx,
    });
    let _ = rx.await.unwrap().unwrap();
    g
}

async fn copy_to_host(
    dev: &atomr_core::actor::ActorRef<DeviceMsg>,
    g: &atomr_accel_cuda::gpu_ref::GpuRef<f32>,
    len: usize,
) -> Vec<f32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyToHostF32 {
        src: g.clone(),
        dst: HostBuf::Vec(vec![0.0; len]),
        reply: tx,
    });
    match rx.await.unwrap().unwrap() {
        HostBuf::Vec(v) => v,
        HostBuf::Pinned(_) => panic!("expected Vec"),
    }
}
