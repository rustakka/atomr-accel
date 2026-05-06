//! Typed `GemmStridedBatchedRequest<T>` + `GemmStridedBatchedDispatch`
//! impls.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use tokio::sync::oneshot;

use crate::dtype::GemmSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{BlasDispatchCtx, GemmStridedBatchedDispatch};
use crate::kernel::envelope;

const LIB: &str = "cublas";

/// Typed strided-batched gemm request. Per-batch strides describe the
/// element offset between consecutive batch entries inside a single
/// allocation.
pub struct GemmStridedBatchedRequest<T: GemmSupported> {
    pub a: GpuRef<T>,
    pub b: GpuRef<T>,
    pub c: GpuRef<T>,
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub alpha: T,
    pub beta: T,
    pub trans_a: cublasOperation_t,
    pub trans_b: cublasOperation_t,
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub stride_a: i64,
    pub stride_b: i64,
    pub stride_c: i64,
    pub batch_size: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_strided_batched<T>(req: GemmStridedBatchedRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: GemmSupported + Copy,
    cudarc::cublas::CudaBlas: Gemm<T>,
{
    let GemmStridedBatchedRequest {
        a,
        b,
        c,
        m,
        n,
        k,
        alpha,
        beta,
        trans_a,
        trans_b,
        lda,
        ldb,
        ldc,
        stride_a,
        stride_b,
        stride_c,
        batch_size,
        reply,
    } = req;

    let (a_slice, b_slice, c_slice) = match envelope::access_all_3(&a, &b, &c) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    let cfg = StridedBatchedConfig::<T> {
        gemm: GemmConfig::<T> {
            transa: trans_a,
            transb: trans_b,
            m,
            n,
            k,
            alpha,
            lda,
            ldb,
            beta,
            ldc,
        },
        batch_size,
        stride_a,
        stride_b,
        stride_c,
    };

    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_arc) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GEMM strided-batched target buffer C has more than one live reference; \
                 caller must hold the unique GpuRef to write to it"
                    .into(),
            )));
            return;
        }
    };

    c.record_write(ctx.stream);

    let cublas = ctx.cublas.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = unsafe { cublas.gemm_strided_batched(cfg, &*a_slice, &*b_slice, &mut c_owned) };
        match res {
            Ok(()) => Ok((cublas, a_slice, b_slice, c_owned)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("gemm_strided_batched enqueue: {e}"),
            }),
        }
    });
}

impl GemmStridedBatchedDispatch for GemmStridedBatchedRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm_strided_batched"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_strided_batched::<f32>(*self, ctx);
    }
}

impl GemmStridedBatchedDispatch for GemmStridedBatchedRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm_strided_batched"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_strided_batched::<f64>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl GemmStridedBatchedDispatch for GemmStridedBatchedRequest<half::f16> {
    fn dtype_name(&self) -> &'static str {
        <half::f16 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm_strided_batched"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_strided_batched::<half::f16>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl GemmStridedBatchedDispatch for GemmStridedBatchedRequest<half::bf16> {
    fn dtype_name(&self) -> &'static str {
        <half::bf16 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm_strided_batched"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_strided_batched::<half::bf16>(*self, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    #[test]
    fn strided_batched_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = GemmStridedBatchedRequest::<f32> {
            a: super::super::gemm::tests_helpers::gpu_ref_stub::<f32>(),
            b: super::super::gemm::tests_helpers::gpu_ref_stub::<f32>(),
            c: super::super::gemm::tests_helpers::gpu_ref_stub::<f32>(),
            m: 1,
            n: 1,
            k: 1,
            alpha: 1.0,
            beta: 0.0,
            trans_a: cublasOperation_t::CUBLAS_OP_N,
            trans_b: cublasOperation_t::CUBLAS_OP_N,
            lda: 1,
            ldb: 1,
            ldc: 1,
            stride_a: 1,
            stride_b: 1,
            stride_c: 1,
            batch_size: 4,
            reply: tx,
        };
        let boxed: Box<dyn GemmStridedBatchedDispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "gemm_strided_batched");
        assert_eq!(boxed.dtype_name(), "f32");
        Box::leak(boxed);
    }
}
