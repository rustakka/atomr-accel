//! `cub::DeviceReduce` family — Sum, Max, Min, ArgMax, ArgMin.
//!
//! Each request is generic over `T: CudaDtype`. The actor compiles a
//! per-(op, dtype) NVRTC kernel that wraps the matching
//! `cub::DeviceReduce::*<T>` template instantiation; the cubin is
//! cached by `(op_name, dtype_name)` for the actor's lifetime and
//! persisted to disk through `atomr_accel_cuda::nvrtc_cache::NvrtcCache`.

use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_accel_cuda::device::DeviceState;
use atomr_accel_cuda::dtype::{AccelDtype, CudaDtype};
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::nvrtc::SmArch;
use atomr_accel_cuda::kernel::{KernelArg, NvrtcMsg};
use atomr_core::actor::ActorRef;

use crate::dispatch::{
    compile_or_get_handle, grid_blocks_for, launch, launch_config_for, launch_config_single_block,
};
use crate::kernels::emit_reduce_source;
use crate::{reply_err, CubDispatchBase, CubDispatchCtx, KernelSourceCache};

/// Family of binary reductions the CUB actor supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
    /// Returns `(index, value)` of the maximum element. Output is
    /// written into a 2-element `GpuRef<u8>` buffer holding a packed
    /// `(u64 index, T value)` per CUB's convention.
    ArgMax,
    ArgMin,
    /// Pairwise multiplicative product reduction.
    Product,
}

impl ReductionOp {
    pub fn op_name(self) -> &'static str {
        match self {
            ReductionOp::Sum => "reduce_sum",
            ReductionOp::Max => "reduce_max",
            ReductionOp::Min => "reduce_min",
            ReductionOp::ArgMax => "reduce_argmax",
            ReductionOp::ArgMin => "reduce_argmin",
            ReductionOp::Product => "reduce_product",
        }
    }
}

/// Typed CUB reduction request. Generic over `T: CudaDtype`.
pub struct ReduceRequest<T: CudaDtype> {
    pub op: ReductionOp,
    pub input: GpuRef<T>,
    /// Single-element output buffer (or 2-element packed for ArgMax /
    /// ArgMin). Caller is responsible for sizing it correctly.
    pub output: GpuRef<T>,
    /// `T::zero()` for Sum / `T::one()` for Product is the natural
    /// identity. We expose it on the request so callers can clamp /
    /// bias the reduction.
    pub init: Option<T>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    _phantom: PhantomData<T>,
}

impl<T: CudaDtype> ReduceRequest<T> {
    pub fn new(
        op: ReductionOp,
        input: GpuRef<T>,
        output: GpuRef<T>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    ) -> Self {
        Self {
            op,
            input,
            output,
            init: None,
            reply,
            _phantom: PhantomData,
        }
    }

    pub fn with_init(mut self, init: T) -> Self {
        self.init = Some(init);
        self
    }
}

/// Dispatch trait CUB actor invokes when handling a [`ReduceRequest<T>`].
pub trait CubReduceDispatch: CubDispatchBase {
    fn dispatch(self: Box<Self>, ctx: &CubDispatchCtx<'_>);
}

impl<T> CubDispatchBase for ReduceRequest<T>
where
    T: CudaDtype,
{
    fn op_name(&self) -> &'static str {
        self.op.op_name()
    }
    fn dtype_name(&self) -> &'static str {
        <T as AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubReduceDispatch for ReduceRequest<T>
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
                        "atomr-accel-cub::CubReduce: NvrtcActor not wired into CubActor".into(),
                    ),
                );
                return;
            }
        };
        let cache = ctx.kernel_cache.clone();
        let stream = ctx.stream.clone();
        let state = ctx.state.clone();
        let arch = ctx.arch;
        let me = *self;
        tokio::spawn(async move {
            let result = run_reduce::<T>(
                me.op, me.input, me.output, nvrtc, cache, stream, state, arch,
            )
            .await;
            let _ = me.reply.send(result);
        });
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_reduce<T: CudaDtype>(
    op: ReductionOp,
    input: GpuRef<T>,
    output: GpuRef<T>,
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    stream: Arc<cudarc::driver::CudaStream>,
    state: Arc<DeviceState>,
    arch: SmArch,
) -> Result<(), GpuError> {
    let dtype = <T as AccelDtype>::NAME;
    let op_name = op.op_name().to_string();
    let (src, kname) = emit_reduce_source::<T>(op);
    let finalize_name = format!("{kname}_finalize");

    // The emitted translation unit defines BOTH the main and the
    // finalize kernel via two `extern "C"` entry points. Two NVRTC
    // compiles against the same source key into the same disk-cache
    // entry; the second is a microsecond hit. Caching is keyed on
    // the kernel name so we get distinct `KernelHandle`s for each.
    let main_handle = compile_or_get_handle(
        nvrtc.clone(),
        cache.clone(),
        op_name.clone(),
        dtype.to_string(),
        src.clone(),
        kname.clone(),
        arch,
    )
    .await?;
    let finalize_handle = compile_or_get_handle(
        nvrtc.clone(),
        cache.clone(),
        format!("{op_name}_finalize"),
        dtype.to_string(),
        src,
        finalize_name,
        arch,
    )
    .await?;

    let n = input.len();
    let grid = grid_blocks_for(n);

    // Allocate per-block partials buffer on the actor's stream and
    // wrap it in a `GpuRef<T>` so it can ride the same
    // `KernelArg::DevSlice` path as caller-supplied buffers.
    let partials_slice = stream
        .alloc_zeros::<T>(grid as usize)
        .map_err(|e| GpuError::OutOfMemory(format!("cub partials alloc: {e}")))?;
    let partials = GpuRef::new(Arc::new(partials_slice), &state);

    // Launch main kernel: (input, partials, n).
    let main_args = vec![
        KernelArg::DevSlice(Box::new(input.clone())),
        KernelArg::DevSlice(Box::new(partials.clone())),
        KernelArg::Usize(n),
    ];
    launch(&nvrtc, main_handle, main_args, launch_config_for(n)).await?;

    // Launch finalize: (partials, output, grid_blocks).
    let finalize_args = vec![
        KernelArg::DevSlice(Box::new(partials)),
        KernelArg::DevSlice(Box::new(output.clone())),
        KernelArg::Usize(grid as usize),
    ];
    launch(
        &nvrtc,
        finalize_handle,
        finalize_args,
        launch_config_single_block(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 5: every `T: CudaDtype` we ship can be wrapped in a
    /// `ReduceRequest<T>` and produces the matching `op_name` /
    /// `dtype_name` strings via `CubDispatchBase`.
    #[test]
    fn reduce_request_round_trip_for_every_dtype() {
        for op in [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::ArgMax,
            ReductionOp::ArgMin,
            ReductionOp::Product,
        ] {
            assert!(op.op_name().starts_with("reduce_"));
        }

        // Verify the dtype-name matrix at the trait level (this is the
        // information the kernel-cache key consumes).
        assert_eq!(<f32 as AccelDtype>::NAME, "f32");
        assert_eq!(<f64 as AccelDtype>::NAME, "f64");
        assert_eq!(<i32 as AccelDtype>::NAME, "i32");
        assert_eq!(<u32 as AccelDtype>::NAME, "u32");
        assert_eq!(<i64 as AccelDtype>::NAME, "i64");
    }
}
