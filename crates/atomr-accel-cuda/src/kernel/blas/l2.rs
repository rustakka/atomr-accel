//! Typed L2 ops: gemv, ger.
//!
//! cudarc 0.19's safe `Gemv<T>` covers f32/f64. For `ger` we drop to
//! the local sys-level [`crate::sys::cublas::sger`] /
//! [`crate::sys::cublas::dger`] wrappers since cudarc has no safe
//! `Ger<T>` trait.

use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{Gemv, GemvConfig};
use cudarc::driver::{DevicePtr, DevicePtrMut};
use tokio::sync::oneshot;

use crate::dtype::{GemvSupported, GerSupported};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{BlasDispatchCtx, BlasL2Dispatch};
use crate::kernel::envelope;
use crate::sys::cublas as syscublas;

const LIB: &str = "cublas";

// ─────────────────────── GEMV ───────────────────────

pub struct GemvRequest<T: GemvSupported> {
    pub trans: cublasOperation_t,
    pub m: i32,
    pub n: i32,
    pub alpha: T,
    pub beta: T,
    pub a: GpuRef<T>,
    pub lda: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_gemv<T>(req: GemvRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: GemvSupported + Copy,
    cudarc::cublas::CudaBlas: Gemv<T>,
{
    let GemvRequest {
        trans,
        m,
        n,
        alpha,
        beta,
        a,
        lda,
        x,
        incx,
        y,
        incy,
        reply,
    } = req;

    let (a_slice, x_slice, y_slice) = match envelope::access_all_3(&a, &x, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GEMV target buffer Y has more than one live reference".into(),
            )));
            return;
        }
    };

    y.record_write(ctx.stream);

    let cfg = GemvConfig::<T> {
        trans,
        m,
        n,
        alpha,
        lda,
        incx,
        beta,
        incy,
    };

    let cublas = ctx.cublas.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = unsafe { cublas.gemv(cfg, &*a_slice, &*x_slice, &mut y_owned) };
        match res {
            Ok(()) => Ok((cublas, a_slice, x_slice, y_owned)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("gemv enqueue: {e}"),
            }),
        }
    });
}

impl BlasL2Dispatch for GemvRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemv"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_gemv::<f32>(*self, ctx);
    }
}

impl BlasL2Dispatch for GemvRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "gemv"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_gemv::<f64>(*self, ctx);
    }
}

// ─────────────────────── GER ───────────────────────

/// Rank-1 update: `A := α·x·y^T + A`.
pub struct GerRequest<T: GerSupported> {
    pub m: i32,
    pub n: i32,
    pub alpha: T,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub a: GpuRef<T>,
    pub lda: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

trait GerCall: GerSupported {
    /// Call the right `cublasSger_v2` / `cublasDger_v2` wrapper.
    ///
    /// # Safety
    /// All pointers must be valid for the encoded sizes.
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        x: cudarc::driver::sys::CUdeviceptr,
        incx: i32,
        y: cudarc::driver::sys::CUdeviceptr,
        incy: i32,
        a: cudarc::driver::sys::CUdeviceptr,
        lda: i32,
    ) -> Result<(), GpuError>;
}

impl GerCall for f32 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        x: cudarc::driver::sys::CUdeviceptr,
        incx: i32,
        y: cudarc::driver::sys::CUdeviceptr,
        incy: i32,
        a: cudarc::driver::sys::CUdeviceptr,
        lda: i32,
    ) -> Result<(), GpuError> {
        syscublas::sger(handle, m, n, alpha, x, incx, y, incy, a, lda)
    }
}

impl GerCall for f64 {
    unsafe fn call(
        handle: cudarc::cublas::sys::cublasHandle_t,
        m: i32,
        n: i32,
        alpha: *const Self,
        x: cudarc::driver::sys::CUdeviceptr,
        incx: i32,
        y: cudarc::driver::sys::CUdeviceptr,
        incy: i32,
        a: cudarc::driver::sys::CUdeviceptr,
        lda: i32,
    ) -> Result<(), GpuError> {
        syscublas::dger(handle, m, n, alpha, x, incx, y, incy, a, lda)
    }
}

fn dispatch_ger<T>(req: GerRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: GerSupported + GerCall + Copy,
{
    let GerRequest {
        m,
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        a,
        lda,
        reply,
    } = req;
    let (a_slice, x_slice, y_slice) = match envelope::access_all_3(&a, &x, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut a_owned = match Arc::try_unwrap(a_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GER target matrix A has more than one live reference".into(),
            )));
            return;
        }
    };
    a.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (x_ptr, _x_rec) = (*x_slice).device_ptr(&stream);
            let (y_ptr, _y_rec) = (*y_slice).device_ptr(&stream);
            let (a_ptr, _a_rec) = a_owned.device_ptr_mut(&stream);
            unsafe {
                T::call(
                    *cublas.handle(),
                    m,
                    n,
                    (&alpha) as *const T,
                    x_ptr,
                    incx,
                    y_ptr,
                    incy,
                    a_ptr,
                    lda,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, x_slice, y_slice, a_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL2Dispatch for GerRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        <f32 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "ger"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_ger::<f32>(*self, ctx);
    }
}

impl BlasL2Dispatch for GerRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        <f64 as atomr_accel::AccelDtype>::NAME
    }
    fn op_name(&self) -> &'static str {
        "ger"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_ger::<f64>(*self, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::super::gemm::tests_helpers::gpu_ref_stub;
    use super::*;
    use tokio::sync::oneshot;

    #[test]
    fn gemv_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = GemvRequest::<f32> {
            trans: cublasOperation_t::CUBLAS_OP_N,
            m: 4,
            n: 4,
            alpha: 1.0,
            beta: 0.0,
            a: gpu_ref_stub::<f32>(),
            lda: 4,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            y: gpu_ref_stub::<f32>(),
            incy: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL2Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "gemv");
        assert_eq!(boxed.dtype_name(), "f32");
        Box::leak(boxed);

        let (tx, _rx) = oneshot::channel();
        let req = GemvRequest::<f64> {
            trans: cublasOperation_t::CUBLAS_OP_N,
            m: 4,
            n: 4,
            alpha: 1.0,
            beta: 0.0,
            a: gpu_ref_stub::<f64>(),
            lda: 4,
            x: gpu_ref_stub::<f64>(),
            incx: 1,
            y: gpu_ref_stub::<f64>(),
            incy: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL2Dispatch> = Box::new(req);
        assert_eq!(boxed.dtype_name(), "f64");
        Box::leak(boxed);
    }

    #[test]
    fn ger_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = GerRequest::<f32> {
            m: 4,
            n: 4,
            alpha: 1.0,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            y: gpu_ref_stub::<f32>(),
            incy: 1,
            a: gpu_ref_stub::<f32>(),
            lda: 4,
            reply: tx,
        };
        let boxed: Box<dyn BlasL2Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "ger");
        Box::leak(boxed);
    }
}
