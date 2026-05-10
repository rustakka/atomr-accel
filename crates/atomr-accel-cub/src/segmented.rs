//! `cub::DeviceSegmentedReduce` — segment-wise reductions driven by a
//! pair of offset arrays.
//!
//! Phase 5.1 launches one CTA per segment; each block uses
//! `cub::BlockReduce` to reduce its slice of the input.

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::{AccelDtype, CudaDtype};
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::nvrtc::SmArch;
use atomr_accel_cuda::kernel::{KernelArg, NvrtcMsg};
use atomr_core::actor::ActorRef;
use cudarc::driver::LaunchConfig;

use crate::dispatch::{compile_or_get_handle, launch};
use crate::kernels::{emit_segmented_reduce_source, BLOCK_THREADS};
use crate::reduce::ReductionOp;
use crate::{reply_err, CubDispatchBase, CubDispatchCtx, KernelSourceCache};

pub struct SegmentedReduceRequest<T: CudaDtype> {
    pub op: ReductionOp,
    pub input: GpuRef<T>,
    pub output: GpuRef<T>,
    pub begin_offsets: GpuRef<i32>,
    pub end_offsets: GpuRef<i32>,
    pub num_segments: u32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> SegmentedReduceRequest<T> {
    pub fn new(
        op: ReductionOp,
        input: GpuRef<T>,
        output: GpuRef<T>,
        begin_offsets: GpuRef<i32>,
        end_offsets: GpuRef<i32>,
        num_segments: u32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            op,
            input,
            output,
            begin_offsets,
            end_offsets,
            num_segments,
            reply,
            _phantom: PhantomData,
        }
    }
}

pub trait CubSegmentedReduceDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for SegmentedReduceRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        match self.op {
            ReductionOp::Sum => "segmented_reduce_sum",
            ReductionOp::Max => "segmented_reduce_max",
            ReductionOp::Min => "segmented_reduce_min",
            ReductionOp::ArgMax => "segmented_reduce_argmax",
            ReductionOp::ArgMin => "segmented_reduce_argmin",
            ReductionOp::Product => "segmented_reduce_product",
        }
    }
    fn dtype_name(&self) -> &'static str {
        <T as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubSegmentedReduceDispatch for SegmentedReduceRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>) {
        let nvrtc = match ctx.nvrtc {
            Some(n) => n.clone(),
            None => {
                reply_err(
                    self.reply,
                    GpuError::Unrecoverable(
                        "atomr-accel-cub::CubSegmentedReduce: NvrtcActor not wired".into(),
                    ),
                );
                return;
            }
        };
        let cache = ctx.kernel_cache.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(run_segmented_reduce::<T>(me, nvrtc, cache, arch));
    }
}

async fn run_segmented_reduce<T: CudaDtype>(
    req: SegmentedReduceRequest<T>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) {
    let SegmentedReduceRequest {
        op,
        input,
        output,
        begin_offsets,
        end_offsets,
        num_segments,
        reply,
        ..
    } = req;
    let result = compile_and_launch::<T>(
        op,
        input,
        output,
        begin_offsets,
        end_offsets,
        num_segments,
        nvrtc,
        cache,
        arch,
    )
    .await;
    let _ = reply.send(result);
}

#[allow(clippy::too_many_arguments)]
async fn compile_and_launch<T: CudaDtype>(
    op: ReductionOp,
    input: GpuRef<T>,
    output: GpuRef<T>,
    begin_offsets: GpuRef<i32>,
    end_offsets: GpuRef<i32>,
    num_segments: u32,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let dtype = <T as AccelDtype>::NAME.to_string();
    let op_name = match op {
        ReductionOp::Sum => "segmented_reduce_sum",
        ReductionOp::Max => "segmented_reduce_max",
        ReductionOp::Min => "segmented_reduce_min",
        ReductionOp::ArgMax => "segmented_reduce_argmax",
        ReductionOp::ArgMin => "segmented_reduce_argmin",
        ReductionOp::Product => "segmented_reduce_product",
    };
    let (src, kname) = emit_segmented_reduce_source::<T>(op);
    let handle = compile_or_get_handle(
        nvrtc.clone(),
        cache,
        op_name.into(),
        dtype,
        src,
        kname,
        arch,
    )
    .await?;

    let args = vec![
        KernelArg::DevSlice(Box::new(input)),
        KernelArg::DevSlice(Box::new(output)),
        KernelArg::DevSlice(Box::new(begin_offsets)),
        KernelArg::DevSlice(Box::new(end_offsets)),
        KernelArg::Scalar(Box::new(num_segments)),
    ];
    let cfg = LaunchConfig {
        grid_dim: (num_segments.max(1), 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    launch(&nvrtc, handle, args, cfg).await
}
