//! End-to-end test for `SolverActor::Svd` and `SolverActor::Syevd`
//! against a real cuSOLVER handle. Gated on
//! `cuda-runtime-tests,cusolver`.
//!
//! SVD on a 2×2 diagonal matrix [[3,0],[0,5]] should produce singular
//! values [5, 3] (sorted descending).
//!
//! Syevd on the symmetric matrix [[2,1],[1,2]] should produce
//! eigenvalues [1, 3] (ascending) with `compute_vectors=false`.

#![cfg(all(feature = "cuda-runtime-tests", feature = "cusolver"))]

use std::sync::Arc;
use std::time::Duration;

use atomr_accel_cuda::completion::HostFnCompletion;
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState, HostBuf};
use atomr_accel_cuda::kernel::{SolverActor, SolverMsg, Uplo};
use atomr_accel_cuda::stream::SingleStreamAllocator;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cusolver_svd_diagonal_singular_values() {
    let sys = ActorSystem::create("svd-e2e", Config::empty())
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
        .expect("context");
    let stream = Arc::new(ctx.new_stream().expect("new_stream"));
    let allocator = Arc::new(SingleStreamAllocator::new(stream.clone()));
    let completion = Arc::new(HostFnCompletion::new());
    let state = Arc::new(DeviceState::new(0));

    let solver = sys
        .actor_of(
            SolverActor::props(stream, allocator, completion, state),
            "solver",
        )
        .unwrap();

    // Diagonal 2×2: A=[[3,0],[0,5]] (column-major: [3,0,0,5]).
    let a_h = vec![3.0f32, 0.0, 0.0, 5.0];
    let a = alloc_and_copy(&dev, &a_h).await;
    let s = alloc_zeros(&dev, 2).await;

    let (tx, rx) = oneshot::channel();
    solver.tell(SolverMsg::Svd {
        a,
        m: 2,
        n: 2,
        s: s.clone(),
        u: None,
        vt: None,
        reply: tx,
    });
    tokio::time::timeout(Duration::from_secs(15), rx)
        .await
        .expect("Svd timeout")
        .expect("oneshot")
        .expect("Svd error");

    let s_h = copy_to_host(&dev, &s, 2).await;
    // cuSOLVER returns singular values in descending order.
    assert!((s_h[0] - 5.0).abs() < 1e-3, "s[0]={}", s_h[0]);
    assert!((s_h[1] - 3.0).abs() < 1e-3, "s[1]={}", s_h[1]);

    sys.terminate().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cusolver_syevd_symmetric_eigenvalues() {
    let sys = ActorSystem::create("syevd-e2e", Config::empty())
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
        .expect("context");
    let stream = Arc::new(ctx.new_stream().expect("new_stream"));
    let allocator = Arc::new(SingleStreamAllocator::new(stream.clone()));
    let completion = Arc::new(HostFnCompletion::new());
    let state = Arc::new(DeviceState::new(0));

    let solver = sys
        .actor_of(
            SolverActor::props(stream, allocator, completion, state),
            "solver",
        )
        .unwrap();

    // Symmetric A=[[2,1],[1,2]] (column-major: [2,1,1,2]).
    let a_h = vec![2.0f32, 1.0, 1.0, 2.0];
    let a = alloc_and_copy(&dev, &a_h).await;
    let w = alloc_zeros(&dev, 2).await;

    let (tx, rx) = oneshot::channel();
    solver.tell(SolverMsg::Syevd {
        a,
        n: 2,
        uplo: Uplo::Upper,
        w: w.clone(),
        compute_vectors: false,
        reply: tx,
    });
    tokio::time::timeout(Duration::from_secs(15), rx)
        .await
        .expect("Syevd timeout")
        .expect("oneshot")
        .expect("Syevd error");

    let w_h = copy_to_host(&dev, &w, 2).await;
    // Eigenvalues of [[2,1],[1,2]] are 1 and 3, ascending.
    assert!((w_h[0] - 1.0).abs() < 1e-3, "w[0]={}", w_h[0]);
    assert!((w_h[1] - 3.0).abs() < 1e-3, "w[1]={}", w_h[1]);

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

async fn alloc_zeros(
    dev: &atomr_core::actor::ActorRef<DeviceMsg>,
    len: usize,
) -> atomr_accel_cuda::gpu_ref::GpuRef<f32> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 { len, reply: tx });
    rx.await.unwrap().unwrap()
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
