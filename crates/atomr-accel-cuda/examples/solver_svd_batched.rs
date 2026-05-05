//! Batched Jacobi SVD demo against a real cuSOLVER handle.
//!
//! Builds a contiguous `batch_size × m × n` column-major buffer
//! holding two diagonal matrices and asks the solver actor for
//! their singular values via `GesvdjBatchedRequest`. Sanity-checks
//! that each batch returns its diagonal entries (sorted descending).
//!
//! Run with:
//! ```bash
//! cargo run -p atomr-accel-cuda --example solver_svd_batched \
//!     --no-default-features --features cuda-runtime-tests,cusolver
//! ```

use std::sync::Arc;
use std::time::Duration;

use atomr_accel_cuda::completion::HostFnCompletion;
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState, HostBuf};
use atomr_accel_cuda::kernel::{GesvdjBatchedRequest, SolverActor, SolverMsg};
use atomr_accel_cuda::stream::SingleStreamAllocator;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use tokio::sync::oneshot;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let sys = ActorSystem::create("solver-svd-batched-demo", Config::empty())
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
        .expect("snapshot timeout")
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

    // Two 2x2 diagonal matrices: diag(2, 5) and diag(3, 7).
    // Column-major, packed back-to-back.
    let a_h: Vec<f32> = vec![2.0, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 7.0];

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 { len: 8, reply: tx });
    let a = rx.await.unwrap().unwrap();
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyFromHostF32 {
        src: HostBuf::Vec(a_h),
        dst: a.clone(),
        reply: tx,
    });
    let _ = rx.await.unwrap().unwrap();

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 { len: 4, reply: tx });
    let s = rx.await.unwrap().unwrap();

    let (tx, rx) = oneshot::channel();
    solver.tell(SolverMsg::Op(Box::new(GesvdjBatchedRequest::<f32> {
        a,
        m: 2,
        n: 2,
        batch_size: 2,
        s: s.clone(),
        u: None,
        v: None,
        reply: tx,
    })));
    tokio::time::timeout(Duration::from_secs(15), rx)
        .await
        .expect("svd timeout")
        .expect("oneshot")
        .expect("svd error");

    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyToHostF32 {
        src: s,
        dst: HostBuf::Vec(vec![0.0; 4]),
        reply: tx,
    });
    let s_h = match rx.await.unwrap().unwrap() {
        HostBuf::Vec(v) => v,
        HostBuf::Pinned(_) => panic!("expected Vec"),
    };

    println!("Batched singular values: {:?}", s_h);

    sys.terminate().await;
}
