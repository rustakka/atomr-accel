//! `cub::DeviceSegmentedReduce` — segment-wise reductions driven by a
//! pair of offset arrays.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;

use crate::reduce::ReductionOp;
use crate::{reply_err, CubDispatchBase, CubDispatchCtx};

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
        <T as atomr_accel_cuda::dtype::AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubSegmentedReduceDispatch for SegmentedReduceRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        let op = self.op_name();
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubSegmentedReduce::{}<{}> — kernel compile path lands in Phase 5.1",
                op,
                <T as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
}
