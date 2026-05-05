//! Typed L3 ops other than gemm: geam (matrix add/scale), syrk
//! (symmetric rank-k update), trsm (triangular solve).
//!
//! Each of these drops to the local sys-level wrappers in
//! [`crate::sys::cublas`] because cudarc 0.19 has no safe trait for
//! them.

use std::sync::Arc;

use cudarc::cublas::sys::{cublasDiagType_t, cublasFillMode_t, cublasOperation_t, cublasSideMode_t};
use cudarc::driver::{sys::CUdeviceptr, DevicePtr, DevicePtrMut};
use tokio::sync::oneshot;

use crate::dtype::{CudaDtype, GeamSupported, SyrkSupported, TrsmSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{BlasDispatchCtx, BlasL3Dispatch};
use crate::kernel::envelope;
use crate::sys::cublas as syscublas;

const LIB: &str = "cublas";

// ─────────────────────── GEAM ───────────────────────

/// `C = α·op(A) + β·op(B)`.
pub struct GeamRequest<T: GeamSupported> {
    pub trans_a: cublasOperation_t,
    pub trans_b: cublasOperation_t,
    pub m: i32,
    pub n: i32,
    pub alpha: T,
    pub a: GpuRef<T>,
    pub lda: i32,
    pub beta: T,
    pub b: GpuRef<T>,
    pub ldb: i32,
    pub c: GpuRef<T>,
    pub ldc: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

trait GeamCall: GeamSupported {
    /// # Safety
    /// All pointers must be valid for the encoded sizes.
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        beta: *const Self,
        b: CUdeviceptr,
        ldb: i32,
        c: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError>;
}

impl GeamCall for f32 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        beta: *const Self,
        b: CUdeviceptr,
        ldb: i32,
        c: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        syscublas::sgeam(
            handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c, ldc,
        )
    }
}

impl GeamCall for f64 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        beta: *const Self,
        b: CUdeviceptr,
        ldb: i32,
        c: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        syscublas::dgeam(
            handle, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c, ldc,
        )
    }
}

fn dispatch_geam<T>(req: GeamRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: GeamSupported + GeamCall + Copy,
{
    let GeamRequest {
        trans_a,
        trans_b,
        m,
        n,
        alpha,
        a,
        lda,
        beta,
        b,
        ldb,
        c,
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
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GEAM target buffer C has more than one live reference".into(),
            )));
            return;
        }
    };
    c.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (a_ptr, _a_rec) = (&*a_slice).device_ptr(&stream);
            let (b_ptr, _b_rec) = (&*b_slice).device_ptr(&stream);
            let (c_ptr, _c_rec) = c_owned.device_ptr_mut(&stream);
            unsafe {
                T::call(
                    *cublas.handle(),
                    trans_a,
                    trans_b,
                    m,
                    n,
                    (&alpha) as *const T,
                    a_ptr,
                    lda,
                    (&beta) as *const T,
                    b_ptr,
                    ldb,
                    c_ptr,
                    ldc,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, a_slice, b_slice, c_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL3Dispatch for GeamRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "geam"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_geam::<f32>(*self, ctx);
    }
}

impl BlasL3Dispatch for GeamRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "geam"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_geam::<f64>(*self, ctx);
    }
}

// ─────────────────────── SYRK ───────────────────────

/// `C := α·op(A)·op(A)^T + β·C`. `op` is `N` or `T`. Updates either
/// the upper or lower triangle of `C`.
pub struct SyrkRequest<T: SyrkSupported> {
    pub uplo: cublasFillMode_t,
    pub trans: cublasOperation_t,
    pub n: i32,
    pub k: i32,
    pub alpha: T,
    pub a: GpuRef<T>,
    pub lda: i32,
    pub beta: T,
    pub c: GpuRef<T>,
    pub ldc: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

trait SyrkCall: SyrkSupported {
    /// # Safety
    /// All pointers must be valid for the encoded sizes.
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        beta: *const Self,
        c: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError>;
}

impl SyrkCall for f32 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        beta: *const Self,
        c: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        syscublas::ssyrk(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    }
}

impl SyrkCall for f64 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: i32,
        k: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        beta: *const Self,
        c: CUdeviceptr,
        ldc: i32,
    ) -> Result<(), GpuError> {
        syscublas::dsyrk(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    }
}

fn dispatch_syrk<T>(req: SyrkRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: SyrkSupported + SyrkCall + Copy,
{
    let SyrkRequest {
        uplo,
        trans,
        n,
        k,
        alpha,
        a,
        lda,
        beta,
        c,
        ldc,
        reply,
    } = req;
    let (a_slice, c_slice) = match envelope::access_all_2(&a, &c) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SYRK target buffer C has more than one live reference".into(),
            )));
            return;
        }
    };
    c.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (a_ptr, _a_rec) = (&*a_slice).device_ptr(&stream);
            let (c_ptr, _c_rec) = c_owned.device_ptr_mut(&stream);
            unsafe {
                T::call(
                    *cublas.handle(),
                    uplo,
                    trans,
                    n,
                    k,
                    (&alpha) as *const T,
                    a_ptr,
                    lda,
                    (&beta) as *const T,
                    c_ptr,
                    ldc,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, a_slice, c_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL3Dispatch for SyrkRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "syrk"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_syrk::<f32>(*self, ctx);
    }
}

impl BlasL3Dispatch for SyrkRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "syrk"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_syrk::<f64>(*self, ctx);
    }
}

// ─────────────────────── TRSM ───────────────────────

/// Triangular solve: `op(A) · X = α·B` (or `X · op(A) = α·B`).
/// Solution is written in-place over `B`.
pub struct TrsmRequest<T: TrsmSupported> {
    pub side: cublasSideMode_t,
    pub uplo: cublasFillMode_t,
    pub trans: cublasOperation_t,
    pub diag: cublasDiagType_t,
    pub m: i32,
    pub n: i32,
    pub alpha: T,
    pub a: GpuRef<T>,
    pub lda: i32,
    pub b: GpuRef<T>,
    pub ldb: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

trait TrsmCall: TrsmSupported {
    /// # Safety
    /// All pointers must be valid for the encoded sizes.
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        b: CUdeviceptr,
        ldb: i32,
    ) -> Result<(), GpuError>;
}

impl TrsmCall for f32 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        b: CUdeviceptr,
        ldb: i32,
    ) -> Result<(), GpuError> {
        syscublas::strsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
    }
}

impl TrsmCall for f64 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        a: CUdeviceptr,
        lda: i32,
        b: CUdeviceptr,
        ldb: i32,
    ) -> Result<(), GpuError> {
        syscublas::dtrsm(handle, side, uplo, trans, diag, m, n, alpha, a, lda, b, ldb)
    }
}

fn dispatch_trsm<T>(req: TrsmRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: TrsmSupported + TrsmCall + Copy,
{
    let TrsmRequest {
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        a,
        lda,
        b,
        ldb,
        reply,
    } = req;
    let (a_slice, b_slice) = match envelope::access_all_2(&a, &b) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut b_owned = match Arc::try_unwrap(b_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "TRSM target buffer B has more than one live reference".into(),
            )));
            return;
        }
    };
    b.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (a_ptr, _a_rec) = (&*a_slice).device_ptr(&stream);
            let (b_ptr, _b_rec) = b_owned.device_ptr_mut(&stream);
            unsafe {
                T::call(
                    *cublas.handle(),
                    side,
                    uplo,
                    trans,
                    diag,
                    m,
                    n,
                    (&alpha) as *const T,
                    a_ptr,
                    lda,
                    b_ptr,
                    ldb,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, a_slice, b_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL3Dispatch for TrsmRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "trsm"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_trsm::<f32>(*self, ctx);
    }
}

impl BlasL3Dispatch for TrsmRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "trsm"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_trsm::<f64>(*self, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gemm::tests_helpers::gpu_ref_stub;
    use tokio::sync::oneshot;

    #[test]
    fn geam_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = GeamRequest::<f32> {
            trans_a: cublasOperation_t::CUBLAS_OP_N,
            trans_b: cublasOperation_t::CUBLAS_OP_N,
            m: 4,
            n: 4,
            alpha: 1.0,
            a: gpu_ref_stub::<f32>(),
            lda: 4,
            beta: 1.0,
            b: gpu_ref_stub::<f32>(),
            ldb: 4,
            c: gpu_ref_stub::<f32>(),
            ldc: 4,
            reply: tx,
        };
        let boxed: Box<dyn BlasL3Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "geam");
        assert_eq!(boxed.dtype_name(), "f32");
        Box::leak(boxed);

        let (tx, _rx) = oneshot::channel();
        let req = GeamRequest::<f64> {
            trans_a: cublasOperation_t::CUBLAS_OP_N,
            trans_b: cublasOperation_t::CUBLAS_OP_N,
            m: 4,
            n: 4,
            alpha: 1.0,
            a: gpu_ref_stub::<f64>(),
            lda: 4,
            beta: 1.0,
            b: gpu_ref_stub::<f64>(),
            ldb: 4,
            c: gpu_ref_stub::<f64>(),
            ldc: 4,
            reply: tx,
        };
        let boxed: Box<dyn BlasL3Dispatch> = Box::new(req);
        assert_eq!(boxed.dtype_name(), "f64");
        Box::leak(boxed);
    }

    #[test]
    fn syrk_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = SyrkRequest::<f32> {
            uplo: cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            trans: cublasOperation_t::CUBLAS_OP_N,
            n: 4,
            k: 4,
            alpha: 1.0,
            a: gpu_ref_stub::<f32>(),
            lda: 4,
            beta: 0.0,
            c: gpu_ref_stub::<f32>(),
            ldc: 4,
            reply: tx,
        };
        let boxed: Box<dyn BlasL3Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "syrk");
        Box::leak(boxed);
    }

    #[test]
    fn trsm_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = TrsmRequest::<f32> {
            side: cublasSideMode_t::CUBLAS_SIDE_LEFT,
            uplo: cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            trans: cublasOperation_t::CUBLAS_OP_N,
            diag: cublasDiagType_t::CUBLAS_DIAG_NON_UNIT,
            m: 4,
            n: 4,
            alpha: 1.0,
            a: gpu_ref_stub::<f32>(),
            lda: 4,
            b: gpu_ref_stub::<f32>(),
            ldb: 4,
            reply: tx,
        };
        let boxed: Box<dyn BlasL3Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "trsm");
        Box::leak(boxed);
    }
}
