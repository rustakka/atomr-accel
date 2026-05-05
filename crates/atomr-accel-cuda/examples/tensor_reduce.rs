//! `tensor_reduce` — Phase 2 cuTENSOR reduction demo.
//!
//! Spawns a `TensorActor`, builds a 4×8 f32 tensor on the device, and
//! reduces along axis 1 to a length-4 vector via
//! `cutensorReduce(op = ADD)`. The example exercises:
//!
//! * the new dtype-generic `ReductionRequest<T>` dispatch path,
//! * the LRU plan cache (the second call hits cache),
//! * the bucketed `WorkspacePool`.
//!
//! Run on a GPU host with cuTENSOR installed:
//!     cargo run -p atomr-accel-cuda --example tensor_reduce \
//!         --features "cuda-runtime-tests cutensor f16"

use std::sync::Arc;
use std::time::Duration;

use atomr_accel_cuda::completion::HostFnCompletion;
use atomr_accel_cuda::device::{DeviceActor, DeviceConfig, DeviceMsg, DeviceState, HostBuf};
use atomr_accel_cuda::kernel::tensor::{
    ContractRequest, OperandSpec, PermutationRequest, ReductionRequest, TensorActor, TensorMsg,
};
use atomr_accel_cuda::stream::SingleStreamAllocator;
use atomr_config::Config;
use atomr_core::actor::ActorSystem;
use cudarc::cutensor::sys as ct_sys;
use tokio::sync::oneshot;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let system = ActorSystem::create("atomr-accel-cuda-tensor-reduce", Config::empty()).await?;
    let device = system.actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")?;
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Snapshot the CUDA context to mint a stream + allocator.
    let (tx, rx) = oneshot::channel();
    device.tell(DeviceMsg::SnapshotContext { reply: tx });
    let ctx = tokio::time::timeout(Duration::from_secs(5), rx)
        .await??
        .ok_or("SnapshotContext returned None — context not yet built")?;
    let stream = ctx.new_stream()?;
    let allocator = Arc::new(SingleStreamAllocator::new(stream.clone()));
    let completion = Arc::new(HostFnCompletion::new());
    let state = Arc::new(DeviceState::new(0));

    let tensor = system.actor_of(
        TensorActor::props(stream, allocator, completion, state),
        "tensor",
    )?;

    // 4×8 input — values 0..31 stored row-major (col-major in cuTENSOR
    // terms, contiguous along mode 0).
    let m: usize = 4;
    let n: usize = 8;
    let mut a_h = Vec::with_capacity(m * n);
    for i in 0..m * n {
        a_h.push(i as f32);
    }
    let zero_h = vec![0.0f32; m];

    let a_buf = alloc_and_copy(&device, &a_h).await?;
    let c_buf = alloc_and_copy(&device, &zero_h).await?;

    // Reduce along mode 1 — output keeps mode 0, length m=4.
    // cuTENSOR's reduction semantics: D = alpha * reduce(A) + beta * C.
    // alpha=1, beta=0 → C = sum_axis1(A).
    let (tx, rx) = oneshot::channel();
    let req = ReductionRequest::<f32>::new(
        OperandSpec::<f32> {
            buf: a_buf.clone(),
            extent: vec![m as i64, n as i64],
            stride: vec![],
            modes: vec![1, 2],
        },
        OperandSpec::<f32> {
            buf: c_buf.clone(),
            extent: vec![m as i64],
            stride: vec![],
            modes: vec![1],
        },
        1.0,
        0.0,
        ct_sys::cutensorOperator_t::CUTENSOR_OP_ADD,
        tx,
    );
    tensor.tell(TensorMsg::Op(Box::new(req)));
    tokio::time::timeout(Duration::from_secs(30), rx).await???;

    let c_out = copy_to_host(&device, &c_buf, m).await?;
    println!("reduction result: {c_out:?}");

    // Sanity bounds: each row sum is 8*i + (0+1+...+7) = 8i + 28.
    for (i, v) in c_out.iter().enumerate() {
        let want = 8.0 * i as f32 + 28.0;
        let diff = (v - want).abs();
        assert!(diff < 1e-3, "row {i}: got {v}, want {want}");
    }

    // Demonstrate that the same actor mailbox carries other op kinds.
    // Permutation: B(i, j) = A(j, i).
    let mut perm_in = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            // Column-major-friendly source: a[i + j*m].
            perm_in.push((i + j * m) as f32);
        }
    }
    let perm_in_buf = alloc_and_copy(&device, &perm_in).await?;
    let perm_out_buf = alloc_and_copy(&device, &vec![0.0f32; m * n]).await?;
    let (tx, rx) = oneshot::channel();
    let req = PermutationRequest::<f32>::new(
        OperandSpec::<f32> {
            buf: perm_in_buf,
            extent: vec![m as i64, n as i64],
            stride: vec![],
            modes: vec![1, 2],
        },
        OperandSpec::<f32> {
            buf: perm_out_buf.clone(),
            extent: vec![n as i64, m as i64],
            stride: vec![],
            modes: vec![2, 1],
        },
        1.0,
        tx,
    );
    tensor.tell(TensorMsg::Op(Box::new(req)));
    tokio::time::timeout(Duration::from_secs(30), rx).await???;

    // Show that contraction still works on the same actor by issuing a
    // small 2x2 matmul through ContractRequest::<f32>.
    let m_h = vec![1.0f32, 3.0, 2.0, 4.0];
    let n_h = vec![5.0f32, 7.0, 6.0, 8.0];
    let z_h = vec![0.0f32; 4];
    let m_buf = alloc_and_copy(&device, &m_h).await?;
    let n_buf = alloc_and_copy(&device, &n_h).await?;
    let z_buf = alloc_and_copy(&device, &z_h).await?;
    let (tx, rx) = oneshot::channel();
    let req = ContractRequest::<f32>::new(
        OperandSpec::<f32> {
            buf: m_buf,
            extent: vec![2, 2],
            stride: vec![],
            modes: vec![1, 2],
        },
        OperandSpec::<f32> {
            buf: n_buf,
            extent: vec![2, 2],
            stride: vec![],
            modes: vec![2, 3],
        },
        OperandSpec::<f32> {
            buf: z_buf,
            extent: vec![2, 2],
            stride: vec![],
            modes: vec![1, 3],
        },
        1.0,
        0.0,
        tx,
    );
    tensor.tell(TensorMsg::Op(Box::new(req)));
    tokio::time::timeout(Duration::from_secs(30), rx).await???;

    println!("all cuTENSOR ops completed");
    system.terminate().await;
    Ok(())
}

async fn alloc_and_copy(
    dev: &atomr_core::actor::ActorRef<DeviceMsg>,
    host: &[f32],
) -> Result<atomr_accel_cuda::gpu_ref::GpuRef<f32>, Box<dyn std::error::Error>> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::AllocateF32 {
        len: host.len(),
        reply: tx,
    });
    let g = rx.await??;
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyFromHostF32 {
        src: HostBuf::Owned(host.to_vec()),
        dst: g.clone(),
        reply: tx,
    });
    let _ = rx.await??;
    Ok(g)
}

async fn copy_to_host(
    dev: &atomr_core::actor::ActorRef<DeviceMsg>,
    g: &atomr_accel_cuda::gpu_ref::GpuRef<f32>,
    len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (tx, rx) = oneshot::channel();
    dev.tell(DeviceMsg::CopyToHostF32 {
        src: g.clone(),
        dst: HostBuf::Owned(vec![0.0; len]),
        reply: tx,
    });
    Ok(match rx.await?? {
        HostBuf::Owned(v) => v,
        HostBuf::Pinned(_) => unreachable!("Owned back"),
    })
}
