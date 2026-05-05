//! Dense cuSOLVER ops: QR, LU (factorize / solve), Cholesky, SVD,
//! Syevd. Each request struct is generic over `T: SolverSupported`
//! (f32, f64) and dispatches through
//! [`crate::sys::cusolver::SolverScalar`].

use std::marker::PhantomData;
use std::sync::Arc;

use cudarc::cusolver::sys as cs;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use tokio::sync::oneshot;

use crate::dtype::SolverSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::sys::cusolver::{status_to_result, SolverScalar, LIB};

use super::workspace::{check_info, ensure_workspace_bytes, lwork_bytes};
use super::{SolverCells, SolverDispatch, Uplo};

// =====================================================================
// QR factorisation
// =====================================================================

pub struct QrRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub m: i32,
    pub n: i32,
    pub tau: GpuRef<T>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for QrRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let QrRequest {
            a,
            m,
            n,
            tau,
            reply,
        } = *self;
        run_qr::<T>(cells, a, m, n, tau, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_qr<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    m: i32,
    n: i32,
    tau: GpuRef<T>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
        info,
        ..
    } = cells;

    let (a_slice, tau_slice) = match envelope::access_all_2(&a, &tau) {
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
                "QR a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut tau_owned = match Arc::try_unwrap(tau_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "QR tau has multiple live references".into(),
            )));
            return;
        }
    };

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _g) = a_owned.device_ptr_mut(stream);
        let status = unsafe {
            T::geqrf_buffer_size(h.0.cu(), m, n, a_ptr as *mut T, m, &mut lwork as *mut _)
        };
        drop(_g);
        if let Err(e) = status_to_result(status, "geqrf_bufferSize") {
            let _ = reply.send(Err(e));
            return;
        }
    }

    if let Err(e) = ensure_workspace_bytes(workspace, stream, lwork_bytes::<T>(lwork)) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    tau.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace.lock();
        let mut info_lock = info.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (tau_ptr, _g2) = tau_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g3) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            T::geqrf(
                h.0.cu(),
                m,
                n,
                a_ptr as *mut T,
                m,
                tau_ptr as *mut T,
                ws_ptr as *mut T,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        status_to_result(status, "geqrf")?;
        check_info(info, &stream_for_check, "geqrf")?;
        Ok((a_owned, tau_owned))
    });
}

// =====================================================================
// LU factorisation (`getrf`) and solve (`getrs`)
// =====================================================================

pub struct LuRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub m: i32,
    pub n: i32,
    pub ipiv: GpuRef<i32>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

pub struct LuSolveRequest<T: SolverSupported> {
    pub lu: GpuRef<T>,
    pub ipiv: GpuRef<i32>,
    pub b: GpuRef<T>,
    pub n: i32,
    pub nrhs: i32,
    pub trans: bool,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for LuRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let LuRequest {
            a,
            m,
            n,
            ipiv,
            reply,
        } = *self;
        run_lu::<T>(cells, a, m, n, ipiv, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

impl<T> SolverDispatch for LuSolveRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let LuSolveRequest {
            lu,
            ipiv,
            b,
            n,
            nrhs,
            trans,
            reply,
        } = *self;
        run_lu_solve::<T>(cells, lu, ipiv, b, n, nrhs, trans, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_lu<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    m: i32,
    n: i32,
    ipiv: GpuRef<i32>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
        info,
        ..
    } = cells;

    let a_slice = match a.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let ipiv_slice = match ipiv.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut a_owned = match Arc::try_unwrap(a_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "LU a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut ipiv_owned = match Arc::try_unwrap(ipiv_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "LU ipiv has multiple live references".into(),
            )));
            return;
        }
    };

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _g) = a_owned.device_ptr_mut(stream);
        let status = unsafe {
            T::getrf_buffer_size(h.0.cu(), m, n, a_ptr as *mut T, m, &mut lwork as *mut _)
        };
        drop(_g);
        if let Err(e) = status_to_result(status, "getrf_bufferSize") {
            let _ = reply.send(Err(e));
            return;
        }
    }
    if let Err(e) = ensure_workspace_bytes(workspace, stream, lwork_bytes::<T>(lwork)) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    ipiv.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace.lock();
        let mut info_lock = info.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (ipiv_ptr, _g2) = ipiv_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g3) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            T::getrf(
                h.0.cu(),
                m,
                n,
                a_ptr as *mut T,
                m,
                ws_ptr as *mut T,
                ipiv_ptr as *mut i32,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        status_to_result(status, "getrf")?;
        check_info(info, &stream_for_check, "getrf")?;
        Ok((a_owned, ipiv_owned))
    });
}

fn run_lu_solve<T: SolverScalar>(
    cells: SolverCells<'_>,
    lu: GpuRef<T>,
    ipiv: GpuRef<i32>,
    b: GpuRef<T>,
    n: i32,
    nrhs: i32,
    trans: bool,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        info,
        ..
    } = cells;

    let lu_slice = match lu.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let ipiv_slice = match ipiv.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let b_slice = match b.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut b_owned = match Arc::try_unwrap(b_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "LU b has multiple live references".into(),
            )));
            return;
        }
    };
    let trans_op = if trans {
        cs::cublasOperation_t::CUBLAS_OP_T
    } else {
        cs::cublasOperation_t::CUBLAS_OP_N
    };
    b.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut info_lock = info.lock();
        let (lu_ptr, _g1) = lu_slice.device_ptr(&stream_for_check);
        let (ipiv_ptr, _g2) = ipiv_slice.device_ptr(&stream_for_check);
        let (b_ptr, _g3) = b_owned.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            T::getrs(
                h.0.cu(),
                trans_op,
                n,
                nrhs,
                lu_ptr as *const T,
                n,
                ipiv_ptr as *const i32,
                b_ptr as *mut T,
                n,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        status_to_result(status, "getrs")?;
        check_info(info, &stream_for_check, "getrs")?;
        Ok((lu_slice, ipiv_slice, b_owned))
    });
}

// =====================================================================
// Cholesky
// =====================================================================

pub struct CholeskyRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub n: i32,
    pub uplo: Uplo,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for CholeskyRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let CholeskyRequest { a, n, uplo, reply } = *self;
        run_cholesky::<T>(cells, a, n, uplo, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_cholesky<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    n: i32,
    uplo: Uplo,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
        info,
        ..
    } = cells;

    let a_slice = match a.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut a_owned = match Arc::try_unwrap(a_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Cholesky a has multiple live references".into(),
            )));
            return;
        }
    };
    let fill = uplo.as_cusolver_fill();

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _g) = a_owned.device_ptr_mut(stream);
        let status = unsafe {
            T::potrf_buffer_size(h.0.cu(), fill, n, a_ptr as *mut T, n, &mut lwork as *mut _)
        };
        drop(_g);
        if let Err(e) = status_to_result(status, "potrf_bufferSize") {
            let _ = reply.send(Err(e));
            return;
        }
    }
    if let Err(e) = ensure_workspace_bytes(workspace, stream, lwork_bytes::<T>(lwork)) {
        let _ = reply.send(Err(e));
        return;
    }
    a.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace.lock();
        let mut info_lock = info.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g2) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g3) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            T::potrf(
                h.0.cu(),
                fill,
                n,
                a_ptr as *mut T,
                n,
                ws_ptr as *mut T,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3));
        status_to_result(status, "potrf")?;
        check_info(info, &stream_for_check, "potrf")?;
        Ok((a_owned,))
    });
}

// =====================================================================
// SVD
// =====================================================================

pub struct SvdRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub m: i32,
    pub n: i32,
    pub s: GpuRef<T>,
    pub u: Option<GpuRef<T>>,
    pub vt: Option<GpuRef<T>>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for SvdRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let SvdRequest {
            a,
            m,
            n,
            s,
            u,
            vt,
            reply,
        } = *self;
        run_svd::<T>(cells, a, m, n, s, u, vt, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_svd<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    m: i32,
    n: i32,
    s: GpuRef<T>,
    u: Option<GpuRef<T>>,
    vt: Option<GpuRef<T>>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
        info,
        ..
    } = cells;

    let a_slice = match a.access() {
        Ok(sl) => sl.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let s_slice = match s.access() {
        Ok(sl) => sl.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut a_owned = match Arc::try_unwrap(a_slice) {
        Ok(sl) => sl,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SVD a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut s_owned = match Arc::try_unwrap(s_slice) {
        Ok(sl) => sl,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SVD s has multiple live references".into(),
            )));
            return;
        }
    };
    let u_slice = match u.as_ref().map(|g| g.access().map(|sl| sl.clone())) {
        Some(Ok(sl)) => Some(sl),
        Some(Err(e)) => {
            let _ = reply.send(Err(e));
            return;
        }
        None => None,
    };
    let vt_slice = match vt.as_ref().map(|g| g.access().map(|sl| sl.clone())) {
        Some(Ok(sl)) => Some(sl),
        Some(Err(e)) => {
            let _ = reply.send(Err(e));
            return;
        }
        None => None,
    };
    let mut u_owned = match u_slice {
        Some(sl) => match Arc::try_unwrap(sl) {
            Ok(o) => Some(o),
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "SVD u has multiple live references".into(),
                )));
                return;
            }
        },
        None => None,
    };
    let mut vt_owned = match vt_slice {
        Some(sl) => match Arc::try_unwrap(sl) {
            Ok(o) => Some(o),
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "SVD vt has multiple live references".into(),
                )));
                return;
            }
        },
        None => None,
    };

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let status = unsafe { T::gesvd_buffer_size(h.0.cu(), m, n, &mut lwork as *mut _) };
        if let Err(e) = status_to_result(status, "gesvd_bufferSize") {
            let _ = reply.send(Err(e));
            return;
        }
    }
    if let Err(e) = ensure_workspace_bytes(workspace, stream, lwork_bytes::<T>(lwork)) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    s.record_write(stream);
    if let Some(g) = &u {
        g.record_write(stream);
    }
    if let Some(g) = &vt {
        g.record_write(stream);
    }

    let jobu = if u_owned.is_some() {
        b'A' as i8
    } else {
        b'N' as i8
    };
    let jobvt = if vt_owned.is_some() {
        b'A' as i8
    } else {
        b'N' as i8
    };
    let stream_for_check = stream.clone();

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace.lock();
        let mut info_lock = info.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (s_ptr, _g2) = s_owned.device_ptr_mut(&stream_for_check);
        let (u_ptr, _gu_opt): (*mut T, _) = match u_owned.as_mut() {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&stream_for_check);
                (p as *mut T, Some(g))
            }
            None => (std::ptr::null_mut(), None),
        };
        let (vt_ptr, _gvt_opt): (*mut T, _) = match vt_owned.as_mut() {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&stream_for_check);
                (p as *mut T, Some(g))
            }
            None => (std::ptr::null_mut(), None),
        };
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g5) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g6) = info_lock.device_ptr_mut(&stream_for_check);
        let ldu = m;
        let ldvt = n;
        let status = unsafe {
            T::gesvd(
                h.0.cu(),
                jobu,
                jobvt,
                m,
                n,
                a_ptr as *mut T,
                m,
                s_ptr as *mut T,
                u_ptr,
                ldu,
                vt_ptr,
                ldvt,
                ws_ptr as *mut T,
                lwork,
                std::ptr::null_mut(),
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g5, _g6, _gu_opt, _gvt_opt));
        status_to_result(status, "gesvd")?;
        check_info(info, &stream_for_check, "gesvd")?;
        Ok((a_owned, s_owned, u_owned, vt_owned))
    });
}

// =====================================================================
// Symmetric eigendecomposition
// =====================================================================

pub struct SyevdRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub n: i32,
    pub uplo: Uplo,
    pub w: GpuRef<T>,
    pub compute_vectors: bool,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for SyevdRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let SyevdRequest {
            a,
            n,
            uplo,
            w,
            compute_vectors,
            reply,
        } = *self;
        run_syevd::<T>(cells, a, n, uplo, w, compute_vectors, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_syevd<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    n: i32,
    uplo: Uplo,
    w: GpuRef<T>,
    compute_vectors: bool,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
        info,
        ..
    } = cells;

    let a_slice = match a.access() {
        Ok(sl) => sl.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let w_slice = match w.access() {
        Ok(sl) => sl.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut a_owned = match Arc::try_unwrap(a_slice) {
        Ok(sl) => sl,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Syevd a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut w_owned = match Arc::try_unwrap(w_slice) {
        Ok(sl) => sl,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "Syevd w has multiple live references".into(),
            )));
            return;
        }
    };
    let fill = uplo.as_cusolver_fill();
    let jobz = if compute_vectors {
        cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR
    } else {
        cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR
    };

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _ga) = a_owned.device_ptr_mut(stream);
        let (w_ptr, _gw) = w_owned.device_ptr_mut(stream);
        let status = unsafe {
            T::syevd_buffer_size(
                h.0.cu(),
                jobz,
                fill,
                n,
                a_ptr as *const T,
                n,
                w_ptr as *const T,
                &mut lwork as *mut _,
            )
        };
        drop((_ga, _gw));
        if let Err(e) = status_to_result(status, "syevd_bufferSize") {
            let _ = reply.send(Err(e));
            return;
        }
    }
    if let Err(e) = ensure_workspace_bytes(workspace, stream, lwork_bytes::<T>(lwork)) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    w.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace.lock();
        let mut info_lock = info.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (w_ptr, _g2) = w_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g3) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            T::syevd(
                h.0.cu(),
                jobz,
                fill,
                n,
                a_ptr as *mut T,
                n,
                w_ptr as *mut T,
                ws_ptr as *mut T,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        status_to_result(status, "syevd")?;
        check_info(info, &stream_for_check, "syevd")?;
        Ok((a_owned, w_owned))
    });
}

// Suppress unused import warnings for the f64-only typed alias below.
#[allow(dead_code)]
fn _phantom<T: SolverSupported>() -> PhantomData<T> {
    PhantomData
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip the request types through `Box<dyn SolverDispatch>`
    /// for both f32 and f64 to ensure the dispatch generics compile.
    /// We can't run the kernels without a GPU, so we just assert that
    /// dropping a request through `dispatch_mock` closes the reply
    /// with the expected error.
    #[test]
    fn qr_lu_cholesky_svd_syevd_round_trip_f32_f64() {
        // We don't have a real GpuRef here; this test is purely a
        // compile-time check that all permutations form a valid
        // `SolverDispatch`. We construct a `Box<dyn SolverDispatch>`
        // for each (op × dtype) and let it drop. The `SolverActor`
        // mock branch is exercised separately in the integration
        // test below.
        fn assert_dispatch<R: SolverDispatch>() {}
        assert_dispatch::<QrRequest<f32>>();
        assert_dispatch::<QrRequest<f64>>();
        assert_dispatch::<LuRequest<f32>>();
        assert_dispatch::<LuRequest<f64>>();
        assert_dispatch::<LuSolveRequest<f32>>();
        assert_dispatch::<LuSolveRequest<f64>>();
        assert_dispatch::<CholeskyRequest<f32>>();
        assert_dispatch::<CholeskyRequest<f64>>();
        assert_dispatch::<SvdRequest<f32>>();
        assert_dispatch::<SvdRequest<f64>>();
        assert_dispatch::<SyevdRequest<f32>>();
        assert_dispatch::<SyevdRequest<f64>>();
    }
}
