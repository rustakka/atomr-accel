//! Typed L1 ops: axpy, dot, nrm2, scal, asum, iamax, iamin, copy,
//! swap, rot.
//!
//! Each op routes through the cuBLAS *Ex entry point so we can ship
//! the same code for f32, f64, f16, and bf16 (and later fp8) without
//! the per-dtype `cublasS*`/`cublasD*` ladder. The wrappers live in
//! [`crate::sys::cublas`].

use std::sync::Arc;

use cudarc::driver::{DevicePtr, DevicePtrMut};
use tokio::sync::oneshot;

use crate::dtype::{AxpyDotNrm2Supported, CudaDtype};
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{BlasDispatchCtx, BlasL1Dispatch};
use crate::kernel::envelope;
use crate::sys::cublas as syscublas;

const LIB: &str = "cublas";

// ─────────────────────── AXPY ───────────────────────

pub struct AxpyRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub alpha: T::Scalar,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_axpy<T>(req: AxpyRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
{
    let AxpyRequest {
        n,
        alpha,
        x,
        incx,
        y,
        incy,
        reply,
    } = req;

    let (x_slice, y_slice) = match envelope::access_all_2(&x, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_arc) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "AXPY target buffer Y has more than one live reference".into(),
            )));
            return;
        }
    };

    y.record_write(ctx.stream);

    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            // Scope the SyncOnDrop guards so they release the
            // borrow on x_slice/y_owned before we move them into
            // the keep-alive tuple.
            let (x_ptr, _x_rec) = (&*x_slice).device_ptr(&stream);
            let (y_ptr, _y_rec) = y_owned.device_ptr_mut(&stream);
            // SAFETY: handle valid; pointers/length come from a
            // generation-checked GpuRef.
            unsafe {
                syscublas::axpy_ex(
                    *cublas.handle(),
                    n,
                    (&alpha) as *const T::Scalar as *const _,
                    scalar_data_type::<T>(),
                    x_ptr,
                    T::cuda_data_type(),
                    incx,
                    y_ptr,
                    T::cuda_data_type(),
                    incy,
                    scalar_data_type::<T>(),
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, x_slice, y_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL1Dispatch for AxpyRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "axpy"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_axpy::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for AxpyRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "axpy"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_axpy::<f64>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl BlasL1Dispatch for AxpyRequest<half::f16> {
    fn dtype_name(&self) -> &'static str {
        <half::f16 as CudaDtype>::name()
    }
    fn op_name(&self) -> &'static str {
        "axpy"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_axpy::<half::f16>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl BlasL1Dispatch for AxpyRequest<half::bf16> {
    fn dtype_name(&self) -> &'static str {
        <half::bf16 as CudaDtype>::name()
    }
    fn op_name(&self) -> &'static str {
        "axpy"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_axpy::<half::bf16>(*self, ctx);
    }
}

// ─────────────────────── SCAL ───────────────────────

pub struct ScalRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub alpha: T::Scalar,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_scal<T>(req: ScalRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
{
    let ScalRequest {
        n,
        alpha,
        x,
        incx,
        reply,
    } = req;
    let x_slice = match x.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut x_owned = match Arc::try_unwrap(x_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SCAL target buffer X has more than one live reference".into(),
            )));
            return;
        }
    };
    x.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (x_ptr, _x_rec) = x_owned.device_ptr_mut(&stream);
            unsafe {
                syscublas::scal_ex(
                    *cublas.handle(),
                    n,
                    (&alpha) as *const T::Scalar as *const _,
                    scalar_data_type::<T>(),
                    x_ptr,
                    T::cuda_data_type(),
                    incx,
                    scalar_data_type::<T>(),
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, x_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL1Dispatch for ScalRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "scal"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_scal::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for ScalRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "scal"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_scal::<f64>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl BlasL1Dispatch for ScalRequest<half::f16> {
    fn dtype_name(&self) -> &'static str {
        <half::f16 as CudaDtype>::name()
    }
    fn op_name(&self) -> &'static str {
        "scal"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_scal::<half::f16>(*self, ctx);
    }
}

#[cfg(feature = "f16")]
impl BlasL1Dispatch for ScalRequest<half::bf16> {
    fn dtype_name(&self) -> &'static str {
        <half::bf16 as CudaDtype>::name()
    }
    fn op_name(&self) -> &'static str {
        "scal"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_scal::<half::bf16>(*self, ctx);
    }
}

// ─────────────────────── NRM2 ───────────────────────

/// Compute `||x||_2`. The result is written to a host-side
/// `Box<MaybeUninit<T::Scalar>>` and forwarded back through `reply`.
pub struct Nrm2Request<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub reply: oneshot::Sender<Result<T::Scalar, GpuError>>,
}

fn dispatch_nrm2<T>(req: Nrm2Request<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
    T::Scalar: Default,
{
    let Nrm2Request { n, x, incx, reply } = req;
    let x_slice = match x.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    let stream_for_kernel = ctx.stream.clone();
    let completion = ctx.completion.clone();
    // We need a host-side scalar. nrm2 with `CUBLAS_POINTER_MODE_HOST`
    // (the default cuBLAS state) writes to host memory but blocks
    // until the stream finishes — that defeats the actor's
    // never-block contract. So we allocate the result on the host
    // and let cuBLAS sync inline; the caller doesn't await the
    // completion future, just the reply.
    //
    // For Phase 1 we keep the simple path: enqueue + completion.
    // The kernel writes to a host scalar held in a Box that we
    // keep alive past completion.
    let mut result_box = Box::new(T::Scalar::default());
    let result_ptr = (&mut *result_box) as *mut T::Scalar as *mut core::ffi::c_void;

    let scalar_dt = scalar_data_type::<T>();
    let exec_dt = T::cuda_data_type();

    let final_reply = reply;
    // We drive a manual variant of run_kernel so the success arm can
    // forward the host-side scalar back to the caller.
    let (inner_tx, inner_rx) = oneshot::channel::<Result<(), GpuError>>();
    envelope::run_kernel(
        LIB,
        ctx.stream,
        ctx.completion,
        (),
        inner_tx,
        move || {
            let res = {
                let (x_ptr, _x_rec) = (&*x_slice).device_ptr(&stream);
                // SAFETY: result_ptr is a valid host pointer to
                // T::Scalar; the stream callback fires after the
                // kernel has populated it.
                unsafe {
                    syscublas::nrm2_ex(
                        *cublas.handle(),
                        n,
                        x_ptr,
                        T::cuda_data_type(),
                        incx,
                        result_ptr,
                        scalar_dt,
                        exec_dt,
                    )
                }
            };
            match res {
                Ok(()) => Ok((cublas, x_slice)),
                Err(e) => Err(e),
            }
        },
    );
    let _ = stream_for_kernel; // silence unused while the pattern matches the other ops
    let _ = completion;
    tokio::spawn(async move {
        match inner_rx.await {
            Ok(Ok(())) => {
                let _ = final_reply.send(Ok(*result_box));
            }
            Ok(Err(e)) => {
                let _ = final_reply.send(Err(e));
            }
            Err(_) => {
                let _ = final_reply.send(Err(GpuError::Timeout));
            }
        }
    });
}

impl BlasL1Dispatch for Nrm2Request<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "nrm2"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_nrm2::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for Nrm2Request<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "nrm2"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_nrm2::<f64>(*self, ctx);
    }
}

// ─────────────────────── DOT ───────────────────────

pub struct DotRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub reply: oneshot::Sender<Result<T::Scalar, GpuError>>,
}

fn dispatch_dot<T>(req: DotRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
    T::Scalar: Default,
{
    let DotRequest {
        n,
        x,
        incx,
        y,
        incy,
        reply,
    } = req;
    let (x_slice, y_slice) = match envelope::access_all_2(&x, &y) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    let mut result_box = Box::new(T::Scalar::default());
    let result_ptr = (&mut *result_box) as *mut T::Scalar as *mut core::ffi::c_void;
    let scalar_dt = scalar_data_type::<T>();
    let exec_dt = T::cuda_data_type();

    let final_reply = reply;
    let (inner_tx, inner_rx) = oneshot::channel::<Result<(), GpuError>>();
    envelope::run_kernel(
        LIB,
        ctx.stream,
        ctx.completion,
        (),
        inner_tx,
        move || {
            let res = {
                let (x_ptr, _x_rec) = (&*x_slice).device_ptr(&stream);
                let (y_ptr, _y_rec) = (&*y_slice).device_ptr(&stream);
                unsafe {
                    syscublas::dot_ex(
                        *cublas.handle(),
                        n,
                        x_ptr,
                        T::cuda_data_type(),
                        incx,
                        y_ptr,
                        T::cuda_data_type(),
                        incy,
                        result_ptr,
                        scalar_dt,
                        exec_dt,
                    )
                }
            };
            match res {
                Ok(()) => Ok((cublas, x_slice, y_slice)),
                Err(e) => Err(e),
            }
        },
    );
    tokio::spawn(async move {
        match inner_rx.await {
            Ok(Ok(())) => {
                let _ = final_reply.send(Ok(*result_box));
            }
            Ok(Err(e)) => {
                let _ = final_reply.send(Err(e));
            }
            Err(_) => {
                let _ = final_reply.send(Err(GpuError::Timeout));
            }
        }
    });
}

impl BlasL1Dispatch for DotRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "dot"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_dot::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for DotRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "dot"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_dot::<f64>(*self, ctx);
    }
}

// ─────────────────────── ASUM ───────────────────────

pub struct AsumRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub reply: oneshot::Sender<Result<T::Scalar, GpuError>>,
}

fn dispatch_asum<T>(req: AsumRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
    T::Scalar: Default,
{
    let AsumRequest { n, x, incx, reply } = req;
    let x_slice = match x.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    let mut result_box = Box::new(T::Scalar::default());
    let result_ptr = (&mut *result_box) as *mut T::Scalar as *mut core::ffi::c_void;
    let scalar_dt = scalar_data_type::<T>();
    let exec_dt = T::cuda_data_type();
    let final_reply = reply;
    let (inner_tx, inner_rx) = oneshot::channel::<Result<(), GpuError>>();
    envelope::run_kernel(
        LIB,
        ctx.stream,
        ctx.completion,
        (),
        inner_tx,
        move || {
            let res = {
                let (x_ptr, _x_rec) = (&*x_slice).device_ptr(&stream);
                unsafe {
                    syscublas::asum_ex(
                        *cublas.handle(),
                        n,
                        x_ptr,
                        T::cuda_data_type(),
                        incx,
                        result_ptr,
                        scalar_dt,
                        exec_dt,
                    )
                }
            };
            match res {
                Ok(()) => Ok((cublas, x_slice)),
                Err(e) => Err(e),
            }
        },
    );
    tokio::spawn(async move {
        match inner_rx.await {
            Ok(Ok(())) => {
                let _ = final_reply.send(Ok(*result_box));
            }
            Ok(Err(e)) => {
                let _ = final_reply.send(Err(e));
            }
            Err(_) => {
                let _ = final_reply.send(Err(GpuError::Timeout));
            }
        }
    });
}

impl BlasL1Dispatch for AsumRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "asum"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_asum::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for AsumRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "asum"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_asum::<f64>(*self, ctx);
    }
}

// ─────────────────────── IAMAX / IAMIN ───────────────────────

pub struct IamaxRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub reply: oneshot::Sender<Result<i32, GpuError>>,
}

pub struct IaminRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub reply: oneshot::Sender<Result<i32, GpuError>>,
}

fn dispatch_iamax_impl<T>(req: IamaxRequest<T>, ctx: &BlasDispatchCtx<'_>, find_min: bool)
where
    T: AxpyDotNrm2Supported,
{
    let IamaxRequest { n, x, incx, reply } = req;
    let x_slice = match x.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    let mut result_box = Box::new(0i32);
    let result_ptr = (&mut *result_box) as *mut i32;
    let final_reply = reply;
    let (inner_tx, inner_rx) = oneshot::channel::<Result<(), GpuError>>();
    envelope::run_kernel(
        LIB,
        ctx.stream,
        ctx.completion,
        (),
        inner_tx,
        move || {
            let res = {
                let (x_ptr, _x_rec) = (&*x_slice).device_ptr(&stream);
                if find_min {
                    unsafe {
                        syscublas::iamin_ex(
                            *cublas.handle(),
                            n,
                            x_ptr,
                            T::cuda_data_type(),
                            incx,
                            result_ptr,
                        )
                    }
                } else {
                    unsafe {
                        syscublas::iamax_ex(
                            *cublas.handle(),
                            n,
                            x_ptr,
                            T::cuda_data_type(),
                            incx,
                            result_ptr,
                        )
                    }
                }
            };
            match res {
                Ok(()) => Ok((cublas, x_slice)),
                Err(e) => Err(e),
            }
        },
    );
    tokio::spawn(async move {
        match inner_rx.await {
            Ok(Ok(())) => {
                let _ = final_reply.send(Ok(*result_box));
            }
            Ok(Err(e)) => {
                let _ = final_reply.send(Err(e));
            }
            Err(_) => {
                let _ = final_reply.send(Err(GpuError::Timeout));
            }
        }
    });
}

impl BlasL1Dispatch for IamaxRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "iamax"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_iamax_impl::<f32>(*self, ctx, false);
    }
}

impl BlasL1Dispatch for IamaxRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "iamax"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_iamax_impl::<f64>(*self, ctx, false);
    }
}

impl BlasL1Dispatch for IaminRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "iamin"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        // IaminRequest<T> has the same field shape as IamaxRequest<T>;
        // collapse via a helper.
        let IaminRequest { n, x, incx, reply } = *self;
        let req = IamaxRequest::<f32> {
            n,
            x,
            incx,
            reply,
        };
        dispatch_iamax_impl::<f32>(req, ctx, true);
    }
}

impl BlasL1Dispatch for IaminRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "iamin"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        let IaminRequest { n, x, incx, reply } = *self;
        let req = IamaxRequest::<f64> {
            n,
            x,
            incx,
            reply,
        };
        dispatch_iamax_impl::<f64>(req, ctx, true);
    }
}

// ─────────────────────── COPY ───────────────────────

pub struct CopyRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_copy<T>(req: CopyRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
{
    let CopyRequest {
        n,
        x,
        incx,
        y,
        incy,
        reply,
    } = req;
    let (x_slice, y_slice) = match envelope::access_all_2(&x, &y) {
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
                "COPY target buffer Y has more than one live reference".into(),
            )));
            return;
        }
    };
    y.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (x_ptr, _x_rec) = (&*x_slice).device_ptr(&stream);
            let (y_ptr, _y_rec) = y_owned.device_ptr_mut(&stream);
            unsafe {
                syscublas::copy_ex(
                    *cublas.handle(),
                    n,
                    x_ptr,
                    T::cuda_data_type(),
                    incx,
                    y_ptr,
                    T::cuda_data_type(),
                    incy,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, x_slice, y_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL1Dispatch for CopyRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "copy"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_copy::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for CopyRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "copy"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_copy::<f64>(*self, ctx);
    }
}

// ─────────────────────── SWAP ───────────────────────

pub struct SwapRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_swap<T>(req: SwapRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
{
    let SwapRequest {
        n,
        x,
        incx,
        y,
        incy,
        reply,
    } = req;
    let x_slice = match x.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let y_slice = match y.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut x_owned = match Arc::try_unwrap(x_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SWAP buffer X has more than one live reference".into(),
            )));
            return;
        }
    };
    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SWAP buffer Y has more than one live reference".into(),
            )));
            return;
        }
    };
    x.record_write(ctx.stream);
    y.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (x_ptr, _x_rec) = x_owned.device_ptr_mut(&stream);
            let (y_ptr, _y_rec) = y_owned.device_ptr_mut(&stream);
            unsafe {
                syscublas::swap_ex(
                    *cublas.handle(),
                    n,
                    x_ptr,
                    T::cuda_data_type(),
                    incx,
                    y_ptr,
                    T::cuda_data_type(),
                    incy,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, x_owned, y_owned)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL1Dispatch for SwapRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "swap"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_swap::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for SwapRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "swap"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_swap::<f64>(*self, ctx);
    }
}

// ─────────────────────── ROT ───────────────────────

/// Givens rotation: applies a 2D rotation `(x_i, y_i) := (c·x_i +
/// s·y_i, -s·x_i + c·y_i)` in-place across two vectors.
///
/// `c` and `s` are passed through `T::Scalar` to match the cuBLAS-Ex
/// pointer-mode-host convention.
pub struct RotRequest<T: AxpyDotNrm2Supported> {
    pub n: i32,
    pub x: GpuRef<T>,
    pub incx: i32,
    pub y: GpuRef<T>,
    pub incy: i32,
    pub c: T::Scalar,
    pub s: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

fn dispatch_rot<T>(req: RotRequest<T>, ctx: &BlasDispatchCtx<'_>)
where
    T: AxpyDotNrm2Supported,
{
    let RotRequest {
        n,
        x,
        incx,
        y,
        incy,
        c,
        s,
        reply,
    } = req;
    let x_slice = match x.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let y_slice = match y.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut x_owned = match Arc::try_unwrap(x_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "ROT buffer X has more than one live reference".into(),
            )));
            return;
        }
    };
    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "ROT buffer Y has more than one live reference".into(),
            )));
            return;
        }
    };
    x.record_write(ctx.stream);
    y.record_write(ctx.stream);
    let cublas = ctx.cublas.clone();
    let stream = ctx.stream.clone();
    let scalar_dt = scalar_data_type::<T>();
    let exec_dt = T::cuda_data_type();
    envelope::run_kernel(LIB, ctx.stream, ctx.completion, (), reply, move || {
        let res = {
            let (x_ptr, _x_rec) = x_owned.device_ptr_mut(&stream);
            let (y_ptr, _y_rec) = y_owned.device_ptr_mut(&stream);
            unsafe {
                syscublas::rot_ex(
                    *cublas.handle(),
                    n,
                    x_ptr,
                    T::cuda_data_type(),
                    incx,
                    y_ptr,
                    T::cuda_data_type(),
                    incy,
                    (&c) as *const T::Scalar as *const _,
                    (&s) as *const T::Scalar as *const _,
                    scalar_dt,
                    exec_dt,
                )
            }
        };
        match res {
            Ok(()) => Ok((cublas, x_owned, y_owned, c, s)),
            Err(e) => Err(e),
        }
    });
}

impl BlasL1Dispatch for RotRequest<f32> {
    fn dtype_name(&self) -> &'static str {
        f32::name()
    }
    fn op_name(&self) -> &'static str {
        "rot"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_rot::<f32>(*self, ctx);
    }
}

impl BlasL1Dispatch for RotRequest<f64> {
    fn dtype_name(&self) -> &'static str {
        f64::name()
    }
    fn op_name(&self) -> &'static str {
        "rot"
    }
    fn dispatch(self: Box<Self>, ctx: &BlasDispatchCtx<'_>) {
        dispatch_rot::<f64>(*self, ctx);
    }
}

// ─────────────────────── helpers ───────────────────────

/// Map `T::Scalar` to its `cudaDataType_t`. f32→`CUDA_R_32F`,
/// f64→`CUDA_R_64F`. Used as the alpha/result-precision argument to
/// nrm2/dot/asum/axpy/scal-Ex.
fn scalar_data_type<T: CudaDtype>() -> cudarc::cublas::sys::cudaDataType_t {
    use core::any::TypeId;
    if TypeId::of::<T::Scalar>() == TypeId::of::<f32>() {
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F
    } else if TypeId::of::<T::Scalar>() == TypeId::of::<f64>() {
        cudarc::cublas::sys::cudaDataType_t::CUDA_R_64F
    } else {
        // Should be unreachable: every `T::Scalar` we ship is
        // f32 or f64.
        panic!("Unrecoverable: scalar type for {} is not f32/f64", T::name());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::gemm::tests_helpers::gpu_ref_stub;
    use tokio::sync::oneshot;

    #[test]
    fn axpy_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = AxpyRequest::<f32> {
            n: 8,
            alpha: 1.0,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            y: gpu_ref_stub::<f32>(),
            incy: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "axpy");
        assert_eq!(boxed.dtype_name(), "f32");
        Box::leak(boxed);

        let (tx, _rx) = oneshot::channel();
        let req = AxpyRequest::<f64> {
            n: 8,
            alpha: 1.0,
            x: gpu_ref_stub::<f64>(),
            incx: 1,
            y: gpu_ref_stub::<f64>(),
            incy: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.dtype_name(), "f64");
        Box::leak(boxed);
    }

    #[test]
    fn scal_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = ScalRequest::<f32> {
            n: 4,
            alpha: 2.0,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "scal");
        Box::leak(boxed);
    }

    #[test]
    fn dot_nrm2_asum_iamax_request_round_trip() {
        let (tx, _rx) = oneshot::channel();
        let req = DotRequest::<f32> {
            n: 4,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            y: gpu_ref_stub::<f32>(),
            incy: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "dot");
        Box::leak(boxed);

        let (tx, _rx) = oneshot::channel();
        let req = Nrm2Request::<f32> {
            n: 4,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "nrm2");
        Box::leak(boxed);

        let (tx, _rx) = oneshot::channel();
        let req = IamaxRequest::<f32> {
            n: 4,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "iamax");
        Box::leak(boxed);

        let (tx, _rx) = oneshot::channel();
        let req = IaminRequest::<f32> {
            n: 4,
            x: gpu_ref_stub::<f32>(),
            incx: 1,
            reply: tx,
        };
        let boxed: Box<dyn BlasL1Dispatch> = Box::new(req);
        assert_eq!(boxed.op_name(), "iamin");
        Box::leak(boxed);
    }
}
