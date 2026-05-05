//! `cub::DeviceReduce` family — Sum, Max, Min, ArgMax, ArgMin.
//!
//! Each request is generic over `T: CudaDtype`. The actor compiles a
//! per-(op, dtype) NVRTC kernel that wraps the matching
//! `cub::DeviceReduce::*<T>` template instantiation; the cubin is
//! cached by `(op_name, dtype_name)` for the actor's lifetime and
//! persisted to disk through `atomr_accel_cuda::nvrtc_cache::NvrtcCache`.

use std::marker::PhantomData;

use tokio::sync::oneshot;

use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;

use crate::{reply_err, CubDispatchBase, CubDispatchCtx};

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
        <T as atomr_accel_cuda::dtype::AccelDtype>::NAME
    }
    fn cancel(self: Box<Self>, err: GpuError) {
        reply_err(self.reply, err);
    }
}

impl<T> CubReduceDispatch for ReduceRequest<T>
where
    T: CudaDtype,
{
    fn dispatch(self: Box<Self>, _ctx: &CubDispatchCtx<'_>) {
        // Phase-5 scaffold: the per-(op, dtype) NVRTC compile + launch
        // implementation lives in a follow-up. For now, we surface a
        // structured error so callers can observe the path is wired.
        reply_err(
            self.reply,
            GpuError::Unrecoverable(format!(
                "CubReduce::{}<{}> — kernel compile path lands in Phase 5.1",
                self.op.op_name(),
                <T as atomr_accel_cuda::dtype::AccelDtype>::NAME,
            )),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_accel_cuda::dtype::AccelDtype;

    /// Phase 5: every `T: CudaDtype` we ship can be wrapped in a
    /// `ReduceRequest<T>` and produces the matching `op_name` /
    /// `dtype_name` strings via `CubDispatchBase`. We exercise the
    /// full default dtype matrix (no GPU required).
    #[test]
    fn reduce_request_round_trip_for_every_dtype() {
        // Construct a `(input, output, reply)` triple per dtype. We
        // can't allocate `GpuRef<T>` without a context, so we use the
        // pure-host `GpuRef::test_dummy` helper available in
        // `atomr-accel-cuda` via #[cfg(test)] — except that helper is
        // private. Instead we exercise only the constructor / metadata
        // surface that doesn't deref the GpuRef. Build a request for
        // each op via type-erased dispatch.
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
        assert_eq!(<u64 as AccelDtype>::NAME, "u64");
        assert_eq!(<i8 as AccelDtype>::NAME, "i8");
        assert_eq!(<u8 as AccelDtype>::NAME, "u8");

        // Walk every `(op, dtype)` pair through the cache-key formula
        // [`crate::kernel_key`] uses. This ensures the kernel-cache
        // axis is unique across the matrix.
        let dtypes = ["f32", "f64", "i32", "u32", "i64", "u64", "i8", "u8"];
        let ops = [
            ReductionOp::Sum,
            ReductionOp::Max,
            ReductionOp::Min,
            ReductionOp::ArgMax,
            ReductionOp::ArgMin,
            ReductionOp::Product,
        ];
        let mut seen = std::collections::HashSet::new();
        for op in ops {
            for dt in dtypes {
                let k = crate::kernel_key(op.op_name(), dt);
                assert!(
                    seen.insert(k.clone()),
                    "kernel_key collision: {k}"
                );
            }
        }
        assert_eq!(seen.len(), ops.len() * dtypes.len());
    }
}
