//! End-to-end test for `SparseActor::SpMv` against a real cuSPARSE
//! handle. Gated on `cuda-runtime-tests,cusparse` so it only runs on
//! hosts with a CUDA driver and the cuSPARSE shared library
//! available.
//!
//! The matrix is the 3×3 identity in CSR; SpMv on `[10,20,30]` should
//! return `[10,20,30]`.

#![cfg(all(feature = "cuda-runtime-tests", feature = "cusparse"))]

use std::sync::Arc;
use std::time::Duration;

use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use rakka_accel_cuda::completion::HostFnCompletion;
use rakka_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState};
use rakka_accel_cuda::kernel::{CsrMatrix, SparseActor, SparseMsg};
use rakka_accel_cuda::stream::SingleStreamAllocator;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cusparse_spmv_identity_matrix() {
    let sys = ActorSystem::create("spmv-e2e", Config::empty()).await.unwrap();
    let dev = sys
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "dev0")
        .unwrap();

    // Wait for ContextActor::Init to complete.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Pull device context to mint a stream for the SparseActor.
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

    let sparse = sys
        .actor_of(
            SparseActor::props(stream.clone(), allocator, completion, state),
            "sparse",
        )
        .unwrap();

    // Identity 3×3 in CSR: row_offsets=[0,1,2,3], col_indices=[0,1,2], values=[1,1,1].
    let row_off_h = vec![0i32, 1, 2, 3];
    let col_idx_h = vec![0i32, 1, 2];
    let vals_h = vec![1.0f32, 1.0, 1.0];
    let x_h = vec![10.0f32, 20.0, 30.0];

    // Allocate and upload via DeviceActor.
    let row_off = alloc_and_copy_i32(&dev, &row_off_h).await;
    let col_idx = alloc_and_copy_i32(&dev, &col_idx_h).await;
    let vals = alloc_and_copy_f32(&dev, &vals_h).await;
    let x = alloc_and_copy_f32(&dev, &x_h).await;
    let y = alloc_zeros_f32(&dev, 3).await;

    let csr = CsrMatrix {
        row_offsets: row_off,
        col_indices: col_idx,
        values: vals,
        rows: 3,
        cols: 3,
        nnz: 3,
    };

    let (tx, rx) = oneshot::channel();
    sparse.tell(SparseMsg::SpMv {
        csr,
        x,
        y: y.clone(),
        alpha: 1.0,
        beta: 0.0,
        reply: tx,
    });
    tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .expect("SpMv timeout")
        .expect("oneshot dropped")
        .expect("SpMv returned error");

    // Pull y back and assert.
    let y_h = copy_to_host_f32(&dev, &y, 3).await;
    assert!((y_h[0] - 10.0).abs() < 1e-5);
    assert!((y_h[1] - 20.0).abs() < 1e-5);
    assert!((y_h[2] - 30.0).abs() < 1e-5);

    sys.terminate().await;
}

async fn alloc_and_copy_f32(dev: &rakka_core::actor::ActorRef<DeviceMsg>, host: &[f32]) -> rakka_accel_cuda::gpu_ref::GpuRef<f32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 { len: host.len(), reply: tx });
    let g = rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyFromHostF32 {
        src: rakka_accel_cuda::device::HostBuf::Vec(host.to_vec()),
        dst: g.clone(),
        reply: tx,
    });
    let _ = rx.await.unwrap().unwrap();
    g
}

async fn alloc_and_copy_i32(dev: &rakka_core::actor::ActorRef<DeviceMsg>, host: &[i32]) -> rakka_accel_cuda::gpu_ref::GpuRef<i32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateI32 { len: host.len(), reply: tx });
    let g = rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyFromHostI32 {
        src: rakka_accel_cuda::device::HostBuf::Vec(host.to_vec()),
        dst: g.clone(),
        reply: tx,
    });
    let _ = rx.await.unwrap().unwrap();
    g
}

async fn alloc_zeros_f32(dev: &rakka_core::actor::ActorRef<DeviceMsg>, len: usize) -> rakka_accel_cuda::gpu_ref::GpuRef<f32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 { len, reply: tx });
    rx.await.unwrap().unwrap()
}

async fn copy_to_host_f32(
    dev: &rakka_core::actor::ActorRef<DeviceMsg>,
    g: &rakka_accel_cuda::gpu_ref::GpuRef<f32>,
    len: usize,
) -> Vec<f32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyToHostF32 {
        src: g.clone(),
        dst: rakka_accel_cuda::device::HostBuf::Vec(vec![0.0; len]),
        reply: tx,
    });
    match rx.await.unwrap().unwrap() {
        rakka_accel_cuda::device::HostBuf::Vec(v) => v,
        rakka_accel_cuda::device::HostBuf::Pinned(_) => panic!("expected Vec"),
    }
}
