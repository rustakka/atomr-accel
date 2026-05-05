//! Typed `GemmRequest<T>` + `GemmDispatch` impls.
//!
//! cudarc 0.19 exposes `cudarc::cublas::Gemm<T>` for f32, f64, and
//! (under feature `f16`) `half::f16` and `half::bf16`. The dispatcher
//! re-uses that safe trait so we don't have to touch
//! `cublasGemmEx` directly for the common dtypes — fp8 is the future
//! follow-up that lights up `crate::sys::cublas::gemm_ex` once the
//! `cublas-fp8` feature is wired (see [`super::scaling`]).

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemm, GemmConfig};
use tokio::sync::oneshot;

use crate::dtype::{CudaDtype, GemmSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{BlasDispatchCtx, GemmDispatch};
use crate::kernel::envelope;

const LIB: &str = "cublas";

/// Typed cuBLAS gemm request: `C = α·op(A)·op(B) + β·C`.
///
/// `lda`/`ldb`/`ldc` follow cuBLAS's column-major convention (see
/// cuBLAS docs). For the no-transpose case, `lda = m`, `ldb = k`,
/// `ldc = m`.
///
/// # Capability marker compile-fail
///
/// `T: GemmSupported` gates the dtype matrix. cuBLAS does **not**
/// support i64 gemm, so building a `GemmRequest::<i64>` is rejected
/// at compile time:
///
/// ```compile_fail
/// # use atomr_accel_cuda::kernel::GemmRequest;
/// # use atomr_accel_cuda::gpu_ref::GpuRef;
/// # use cudarc::cublas::sys::cublasOperation_t;
/// # let (tx, _rx) = tokio::sync::oneshot::channel();
/// # let a: GpuRef<i64> = unimplemented!();
/// # let b: GpuRef<i64> = unimplemented!();
/// # let c: GpuRef<i64> = unimplemented!();
/// // Fails: i64 does not implement `GemmSupported`.
/// let _req = GemmRequest::<i64> {
///     a, b, c,
///     m: 1, n: 1, k: 1,
///     alpha: 1, beta: 0,
///     trans_a: cublasOperation_t::CUBLAS_OP_N,
///     trans_b: cublasOperation_t::CUBLAS_OP_N,
///     lda: 1, ldb: 1, ldc: 1,
///     reply: tx,
/// };
/// ```
pub struct GemmRequest<T: GemmSupported> {
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
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> GemmRequest<T>
where
    T: GemmSupported,
    GemmRequest<T>: GemmDispatch,
{
    /// Box-and-wrap into a [`crate::kernel::BlasMsg::Gemm`] variant.
    pub fn into_msg(self) -> crate::kernel::BlasMsg {
        crate::kernel::BlasMsg::Gemm(Box::new(self))
    }
}

/// Generic dispatch body shared across every `Gemm<T>` cudarc impl.
///
/// We split it into a function so the trait impl stays tiny. The
/// `T: Gemm<...> for CudaBlas` bound forces every call site to pick a
/// dtype cudarc actually implements; calling `gemm::<i64>(...)` would
/// fail to compile.
fn dispatch_gemm<T>(req: GemmRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: GemmSupported + Copy,
    cudarc::cublas::CudaBlas: Gemm<T>,
{
    let GemmRequest {
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
        reply,
    } = req;

    let (a_slice, b_slice, c_slice) = match envelope::access_all_3(&a, &b, &c) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    let cfg = GemmConfig::<T> {
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
    };

    // cudarc's `gemm` requires `&mut C: DevicePtrMut<T>`. An
    // `Arc<CudaSlice<T>>` doesn't satisfy that: we have to unwrap the
    // Arc. The caller must hold the unique `GpuRef` to the output
    // buffer or the unwrap fails — single-writer enforcement.
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_arc) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GEMM target buffer C has more than one live reference; \
                 caller must hold the unique GpuRef to write to it"
                    .into(),
            )));
            return;
        }
    };

    c.record_write(ctx.stream);

    let cublas = ctx.cublas.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        // SAFETY: cudarc's `gemm` is unsafe because invalid
        // m/n/k/lda/ldb/ldc can read out of bounds. The caller is
        // responsible for valid dims.
        let res = unsafe { cublas.gemm(cfg, &*a_slice, &*b_slice, &mut c_owned) };
        match res {
            Ok(()) => Ok((cublas, a_slice, b_slice, c_owned)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("gemm enqueue: {e}"),
            }),
        }
    });
}

// ─────────────── concrete `GemmDispatch` impls ───────────────

impl GemmDispatch for GemmRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_gemm::<f32>(*self, ctx);
    }
}

impl GemmDispatch for GemmRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_gemm::<f64>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl GemmDispatch for GemmRequest<half::f16> {
    fn dtype_name(&self) -> &'static str {
        <half::f16 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_gemm::<half::f16>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl GemmDispatch for GemmRequest<half::bf16> {
    fn dtype_name(&self) -> &'static str {
        <half::bf16 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemm"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_gemm::<half::bf16>(*self, ctx);
    }
}

#[cfg(test)]
pub(crate) mod tests_helpers {
    use crate::gpu_ref::GpuRef;

    /// Fabricate a `GpuRef<T>` for op-name / dtype-name unit tests
    /// that never dispatch the request. The returned `GpuRef` is
    /// **leaked** by the caller (via `Box::leak` on the surrounding
    /// boxed request) so cudarc's `Drop for CudaSlice<T>` never runs.
    ///
    /// SAFETY: the underlying `CudaSlice` is uninitialized. Reading
    /// from it or dispatching the request is undefined behaviour.
    /// Tests that use this helper only inspect the boxed
    /// dispatcher's `op_name` / `dtype_name`, which don't touch
    /// the slice.
    pub fn gpu_ref_stub<T>() -> GpuRef<T> {
        GpuRef::<T>::for_test_no_gpu_leaked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    fn _assert_send<T: Send>() {}

    #[test]
    fn gemm_request_dispatches_for_f32_f64_f16_bf16() {
        // Compile-time assertion: every `GemmRequest<T>` for the
        // dtypes we ship is `Send + 'static` so it can travel through
        // the boxed-dispatcher mailbox.
        _assert_send::<GemmRequest<f32>>();
        _assert_send::<GemmRequest<f64>>();
        #[cfg(feature = "f16")]
        {
            _assert_send::<GemmRequest<half::f16>>();
            _assert_send::<GemmRequest<half::bf16>>();
        }

        // Runtime assertion: each dtype's boxed dispatcher reports
        // its op + dtype correctly. We fabricate a `GemmRequest` and
        // immediately box-leak it so cudarc's `Drop` for the
        // fabricated slice never runs.
        let req = stub_request::<f32>();
        let boxed: Box<dyn GemmDispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "gemm");
        assert_eq!(boxed.dtype_name(), "f32");
        // Leak — see helper docs.
        Box::leak(boxed);

        let req = stub_request::<f64>();
        let boxed: Box<dyn GemmDispatch> = Box::new(req);
        assert_eq!(boxed.dtype_name(), "f64");
        Box::leak(boxed);

        #[cfg(feature = "f16")]
        {
            let req = stub_request::<half::f16>();
            let boxed: Box<dyn GemmDispatch> = Box::new(req);
            assert_eq!(boxed.dtype_name(), "f16");
            Box::leak(boxed);

            let req = stub_request::<half::bf16>();
            let boxed: Box<dyn GemmDispatch> = Box::new(req);
            assert_eq!(boxed.dtype_name(), "bf16");
            Box::leak(boxed);
        }
    }

    #[test]
    fn deprecated_sgemm_alias_still_constructs() {
        // Build a legacy `BlasMsg::Sgemm(Box<SgemmRequest>)`. The
        // explicit `#[allow(deprecated)]` exercises the back-compat
        // path that constructs the variant; routing through
        // `Gemm<f32>` is exercised at the actor handler level (live
        // GPU integration tests).
        #[allow(deprecated)]
        {
            let (tx, _rx) = oneshot::channel();
            let req = crate::device::SgemmRequest {
                a: gpu_ref_stub::<f32>(),
                b: gpu_ref_stub::<f32>(),
                c: gpu_ref_stub::<f32>(),
                m: 1,
                n: 1,
                k: 1,
                alpha: 1.0,
                beta: 0.0,
                reply: tx,
            };
            let msg = crate::kernel::BlasMsg::Sgemm(Box::new(req));
            // Leak: same rationale as `gpu_ref_stub` — the
            // fabricated CudaSlice's Drop must not run.
            Box::leak(Box::new(msg));
        }
    }

    /// Build a fully-populated [`GemmRequest<T>`] backed by the
    /// fabricated [`gpu_ref_stub`] buffers. Used for op-name /
    /// dtype-name assertions only — never dispatched.
    fn stub_request<T>() -> GemmRequest<T>
    where
        T: GemmSupported + num_one_zero::NumOneZero,
        GemmRequest<T>: GemmDispatch,
    {
        let (tx, _rx) = oneshot::channel();
        // The Receiver drops at function exit; the Sender stays
        // inside the request and is leaked along with it via
        // `Box::leak` at the call site.
        GemmRequest::<T> {
            a: gpu_ref_stub::<T>(),
            b: gpu_ref_stub::<T>(),
            c: gpu_ref_stub::<T>(),
            m: 1,
            n: 1,
            k: 1,
            alpha: <T as num_one_zero::NumOneZero>::one(),
            beta: <T as num_one_zero::NumOneZero>::zero(),
            trans_a: cublasOperation_t::CUBLAS_OP_N,
            trans_b: cublasOperation_t::CUBLAS_OP_N,
            lda: 1,
            ldb: 1,
            ldc: 1,
            reply: tx,
        }
    }

    /// Local "one"/"zero" trait so the test stub can build alpha/beta
    /// without depending on `num-traits`. f32/f64 use literals;
    /// half-precision uses `half::f16::ZERO`/`ONE`.
    mod num_one_zero {
        pub trait NumOneZero: Copy {
            fn one() -> Self;
            fn zero() -> Self;
        }
        impl NumOneZero for f32 {
            fn one() -> Self {
                1.0
            }
            fn zero() -> Self {
                0.0
            }
        }
        impl NumOneZero for f64 {
            fn one() -> Self {
                1.0
            }
            fn zero() -> Self {
                0.0
            }
        }
        #[cfg(feature = "f16")]
        impl NumOneZero for half::f16 {
            fn one() -> Self {
                half::f16::ONE
            }
            fn zero() -> Self {
                half::f16::ZERO
            }
        }
        #[cfg(feature = "f16")]
        impl NumOneZero for half::bf16 {
            fn one() -> Self {
                half::bf16::ONE
            }
            fn zero() -> Self {
                half::bf16::ZERO
            }
        }
    }

    use super::tests_helpers::gpu_ref_stub;
}
