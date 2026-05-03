//! `SolverActor` — wraps [`cudarc::cusolver::DnHandle`] for dense
//! linear algebra (QR, LU, Cholesky).
//!
//! Implementation notes:
//! - cudarc 0.19's safe layer only exposes handle management; the
//!   factorization functions live in `cusolver::sys::lib::*` and
//!   require unsafe FFI calls. We wrap them carefully here.
//! - Each op queries the cuSOLVER workspace size, grows our
//!   on-demand `CudaSlice<f32>` workspace, then dispatches the
//!   factorization. The 1-element `info` buffer is read back to
//!   detect failures (singular matrix, illegal arg, etc.).
//! - Sparse cuSOLVER (`SpHandle`) and SVD/eigen are F5.x.

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::cusolver::sys as cusolver_sys;
use cudarc::cusolver::DnHandle;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "cusolver";

#[derive(Debug, Clone, Copy)]
pub enum Uplo {
    Upper,
    Lower,
}

impl Uplo {
    fn as_cusolver_fill(self) -> cusolver_sys::cublasFillMode_t {
        match self {
            Uplo::Upper => cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            Uplo::Lower => cusolver_sys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
        }
    }
}

pub enum SolverMsg {
    /// In-place QR factorization of an `m × n` matrix `a` (column-major).
    /// `tau` must be at least `min(m, n)` long. The lower-triangular
    /// part of `a` plus `tau` encodes Q via Householder reflections;
    /// the upper-triangular part is R.
    QrFactorize {
        a: GpuRef<f32>,
        m: i32,
        n: i32,
        tau: GpuRef<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// In-place LU factorization (with partial pivoting) of an `m × n`
    /// matrix `a`. `ipiv` is an `i32` buffer of length `min(m, n)`
    /// receiving the pivot indices.
    LuFactorize {
        a: GpuRef<f32>,
        m: i32,
        n: i32,
        ipiv: GpuRef<i32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// LU-solve: given a previously factored `lu` + `ipiv` from
    /// `LuFactorize` on a square `n × n` matrix, solve `op(A) X = B`
    /// in place on `b` (overwritten with the solution `X`).
    LuSolve {
        lu: GpuRef<f32>,
        ipiv: GpuRef<i32>,
        b: GpuRef<f32>,
        n: i32,
        nrhs: i32,
        trans: bool,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// In-place Cholesky factorization of an `n × n` SPD matrix `a`.
    Cholesky {
        a: GpuRef<f32>,
        n: i32,
        uplo: Uplo,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// SVD: `a` (m×n, column-major) is overwritten with intermediate
    /// state. `s` receives min(m,n) singular values. `u` and `vt` are
    /// optional output matrices for left/right singular vectors. Pass
    /// `None` to skip computing them. The `m × min(m,n)` `u` and the
    /// `min(m,n) × n` `vt` are written column-major.
    Svd {
        a: GpuRef<f32>,
        m: i32,
        n: i32,
        s: GpuRef<f32>,
        u: Option<GpuRef<f32>>,
        vt: Option<GpuRef<f32>>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Symmetric eigendecomposition of an `n × n` matrix `a` stored in
    /// `uplo` triangle. `w` receives the n eigenvalues in ascending
    /// order. When `compute_vectors` is true, `a` is overwritten with
    /// the orthonormal eigenvectors (column-major).
    Syevd {
        a: GpuRef<f32>,
        n: i32,
        uplo: Uplo,
        w: GpuRef<f32>,
        compute_vectors: bool,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct SolverActor {
    inner: SolverInner,
}

struct SendDn(DnHandle);
unsafe impl Send for SendDn {}
unsafe impl Sync for SendDn {}

#[allow(dead_code)]
enum SolverInner {
    Real {
        handle: Mutex<SendDn>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        /// On-demand-grown scratch buffer (in f32 elements). Never
        /// shrunk; rebuilt fresh on context restart.
        workspace: Mutex<Option<CudaSlice<f32>>>,
        /// 1-element `i32` info buffer reused across calls.
        info: Mutex<CudaSlice<i32>>,
    },
    Mock,
}

impl SolverActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            let handle = match DnHandle::new(stream.clone()) {
                Ok(h) => h,
                Err(e) => panic!("ContextPoisoned: DnHandle::new failed: {e}"),
            };
            let info = stream
                .alloc_zeros::<i32>(1)
                .unwrap_or_else(|e| panic!("ContextPoisoned: alloc info: {e}"));
            SolverActor {
                inner: SolverInner::Real {
                    handle: Mutex::new(SendDn(handle)),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                    workspace: Mutex::new(None),
                    info: Mutex::new(info),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| SolverActor {
            inner: SolverInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for SolverActor {
    type Msg = SolverMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SolverMsg) {
        match &self.inner {
            SolverInner::Mock => mock_reply(msg),
            SolverInner::Real {
                handle,
                stream,
                completion,
                workspace,
                info,
                ..
            } => match msg {
                SolverMsg::QrFactorize {
                    a,
                    m,
                    n,
                    tau,
                    reply,
                } => {
                    handle_qr_factorize(
                        handle, stream, completion, workspace, info, a, m, n, tau, reply,
                    );
                }
                SolverMsg::LuFactorize {
                    a,
                    m,
                    n,
                    ipiv,
                    reply,
                } => {
                    handle_lu_factorize(
                        handle, stream, completion, workspace, info, a, m, n, ipiv, reply,
                    );
                }
                SolverMsg::LuSolve {
                    lu,
                    ipiv,
                    b,
                    n,
                    nrhs,
                    trans,
                    reply,
                } => {
                    handle_lu_solve(
                        handle, stream, completion, info, lu, ipiv, b, n, nrhs, trans, reply,
                    );
                }
                SolverMsg::Cholesky { a, n, uplo, reply } => {
                    handle_cholesky(
                        handle, stream, completion, workspace, info, a, n, uplo, reply,
                    );
                }
                SolverMsg::Svd {
                    a,
                    m,
                    n,
                    s,
                    u,
                    vt,
                    reply,
                } => {
                    handle_svd(
                        handle, stream, completion, workspace, info, a, m, n, s, u, vt, reply,
                    );
                }
                SolverMsg::Syevd {
                    a,
                    n,
                    uplo,
                    w,
                    compute_vectors,
                    reply,
                } => {
                    handle_syevd(
                        handle,
                        stream,
                        completion,
                        workspace,
                        info,
                        a,
                        n,
                        uplo,
                        w,
                        compute_vectors,
                        reply,
                    );
                }
            },
        }
    }
}

fn mock_reply(msg: SolverMsg) {
    let err = || GpuError::Unrecoverable("SolverActor in mock mode".into());
    match msg {
        SolverMsg::QrFactorize { reply, .. }
        | SolverMsg::LuFactorize { reply, .. }
        | SolverMsg::LuSolve { reply, .. }
        | SolverMsg::Cholesky { reply, .. }
        | SolverMsg::Svd { reply, .. }
        | SolverMsg::Syevd { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
    }
}

/// Grow `workspace` to at least `needed` f32 elements. Returns the
/// new size in elements, or an error.
fn ensure_workspace(
    workspace: &Mutex<Option<CudaSlice<f32>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    needed_elems: usize,
) -> Result<(), GpuError> {
    let mut g = workspace.lock();
    let cur = g.as_ref().map(|s| s.len()).unwrap_or(0);
    if cur >= needed_elems {
        return Ok(());
    }
    *g =
        Some(stream.alloc_zeros::<f32>(needed_elems).map_err(|e| {
            GpuError::OutOfMemory(format!("solver workspace ({needed_elems}f): {e}"))
        })?);
    Ok(())
}

/// Read back the 1-element info buffer synchronously and translate
/// non-zero values into `LibraryError`.
fn check_info(
    info: &Mutex<CudaSlice<i32>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    op: &'static str,
) -> Result<(), GpuError> {
    let g = info.lock();
    let mut host = vec![0i32; 1];
    stream
        .memcpy_dtoh(&*g, &mut host[..])
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: read info: {e}"),
        })?;
    stream.synchronize().map_err(|e| GpuError::LibraryError {
        lib: LIB,
        msg: format!("{op}: sync after info: {e}"),
    })?;
    if host[0] != 0 {
        return Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: info={}", host[0]),
        });
    }
    Ok(())
}

fn handle_qr_factorize(
    handle: &Mutex<SendDn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<f32>>>,
    info: &Mutex<CudaSlice<i32>>,
    a: GpuRef<f32>,
    m: i32,
    n: i32,
    tau: GpuRef<f32>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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

    // Query workspace size.
    let mut lwork = 0i32;
    let h = handle.lock();
    let (a_ptr, _g1) = a_owned.device_ptr_mut(stream);
    let status = unsafe {
        cusolver_sys::cusolverDnSgeqrf_bufferSize(
            h.0.cu(),
            m,
            n,
            a_ptr as *mut f32,
            m,
            &mut lwork as *mut _,
        )
    };
    drop(_g1);
    if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("Sgeqrf_bufferSize: {status:?}"),
        }));
        return;
    }
    drop(h);

    if let Err(e) = ensure_workspace(workspace, stream, lwork as usize) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    tau.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let info_ref = info;
    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let mut info_lock = info_ref.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (tau_ptr, _g2) = tau_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g3) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            cusolver_sys::cusolverDnSgeqrf(
                h.0.cu(),
                m,
                n,
                a_ptr as *mut f32,
                m,
                tau_ptr as *mut f32,
                ws_ptr as *mut f32,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Sgeqrf: {status:?}"),
            });
        }
        check_info(info_ref, &stream_for_check, "Sgeqrf")?;
        Ok((a_owned, tau_owned))
    });
}

fn handle_lu_factorize(
    handle: &Mutex<SendDn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<f32>>>,
    info: &Mutex<CudaSlice<i32>>,
    a: GpuRef<f32>,
    m: i32,
    n: i32,
    ipiv: GpuRef<i32>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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

    // Query workspace.
    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _g) = a_owned.device_ptr_mut(stream);
        let status = unsafe {
            cusolver_sys::cusolverDnSgetrf_bufferSize(
                h.0.cu(),
                m,
                n,
                a_ptr as *mut f32,
                m,
                &mut lwork as *mut _,
            )
        };
        drop(_g);
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Sgetrf_bufferSize: {status:?}"),
            }));
            return;
        }
    }
    if let Err(e) = ensure_workspace(workspace, stream, lwork as usize) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    ipiv.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let info_ref = info;
    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let mut info_lock = info_ref.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (ipiv_ptr, _g2) = ipiv_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g3) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            cusolver_sys::cusolverDnSgetrf(
                h.0.cu(),
                m,
                n,
                a_ptr as *mut f32,
                m,
                ws_ptr as *mut f32,
                ipiv_ptr as *mut i32,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Sgetrf: {status:?}"),
            });
        }
        check_info(info_ref, &stream_for_check, "Sgetrf")?;
        Ok((a_owned, ipiv_owned))
    });
}

fn handle_lu_solve(
    handle: &Mutex<SendDn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    info: &Mutex<CudaSlice<i32>>,
    lu: GpuRef<f32>,
    ipiv: GpuRef<i32>,
    b: GpuRef<f32>,
    n: i32,
    nrhs: i32,
    trans: bool,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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
        cusolver_sys::cublasOperation_t::CUBLAS_OP_T
    } else {
        cusolver_sys::cublasOperation_t::CUBLAS_OP_N
    };
    b.record_write(stream);

    let handle_clone = handle;
    let info_ref = info;
    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut info_lock = info_ref.lock();
        let (lu_ptr, _g1) = lu_slice.device_ptr(&stream_for_check);
        let (ipiv_ptr, _g2) = ipiv_slice.device_ptr(&stream_for_check);
        let (b_ptr, _g3) = b_owned.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            cusolver_sys::cusolverDnSgetrs(
                h.0.cu(),
                trans_op,
                n,
                nrhs,
                lu_ptr as *const f32,
                n,
                ipiv_ptr as *const i32,
                b_ptr as *mut f32,
                n,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Sgetrs: {status:?}"),
            });
        }
        check_info(info_ref, &stream_for_check, "Sgetrs")?;
        Ok((lu_slice, ipiv_slice, b_owned))
    });
}

fn handle_cholesky(
    handle: &Mutex<SendDn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<f32>>>,
    info: &Mutex<CudaSlice<i32>>,
    a: GpuRef<f32>,
    n: i32,
    uplo: Uplo,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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
            cusolver_sys::cusolverDnSpotrf_bufferSize(
                h.0.cu(),
                fill,
                n,
                a_ptr as *mut f32,
                n,
                &mut lwork as *mut _,
            )
        };
        drop(_g);
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Spotrf_bufferSize: {status:?}"),
            }));
            return;
        }
    }
    if let Err(e) = ensure_workspace(workspace, stream, lwork as usize) {
        let _ = reply.send(Err(e));
        return;
    }
    a.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let info_ref = info;
    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let mut info_lock = info_ref.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g2) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g3) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            cusolver_sys::cusolverDnSpotrf(
                h.0.cu(),
                fill,
                n,
                a_ptr as *mut f32,
                n,
                ws_ptr as *mut f32,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Spotrf: {status:?}"),
            });
        }
        check_info(info_ref, &stream_for_check, "Spotrf")?;
        Ok((a_owned,))
    });
}

#[allow(clippy::too_many_arguments)]
fn handle_svd(
    handle: &Mutex<SendDn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<f32>>>,
    info: &Mutex<CudaSlice<i32>>,
    a: GpuRef<f32>,
    m: i32,
    n: i32,
    s: GpuRef<f32>,
    u: Option<GpuRef<f32>>,
    vt: Option<GpuRef<f32>>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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
    // Optional u / vt buffers. When absent we pass null pointers and
    // jobu/jobvt='N'.
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
        let status = unsafe {
            cusolver_sys::cusolverDnSgesvd_bufferSize(h.0.cu(), m, n, &mut lwork as *mut _)
        };
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Sgesvd_bufferSize: {status:?}"),
            }));
            return;
        }
    }
    if let Err(e) = ensure_workspace(workspace, stream, lwork as usize) {
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

    let handle_clone = handle;
    let workspace_ref = workspace;
    let info_ref = info;
    let stream_for_check = stream.clone();
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

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let mut info_lock = info_ref.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (s_ptr, _g2) = s_owned.device_ptr_mut(&stream_for_check);
        // Take raw u/vt pointers if buffers are present; null
        // otherwise. Lock guards are extracted into Option to keep the
        // device pointers valid through the launch call.
        let (u_ptr, _gu_opt): (*mut f32, _) = match u_owned.as_mut() {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&stream_for_check);
                (p as *mut f32, Some(g))
            }
            None => (std::ptr::null_mut(), None),
        };
        let (vt_ptr, _gvt_opt): (*mut f32, _) = match vt_owned.as_mut() {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&stream_for_check);
                (p as *mut f32, Some(g))
            }
            None => (std::ptr::null_mut(), None),
        };
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g5) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g6) = info_lock.device_ptr_mut(&stream_for_check);
        let ldu = m;
        let ldvt = n;
        let status = unsafe {
            cusolver_sys::cusolverDnSgesvd(
                h.0.cu(),
                jobu,
                jobvt,
                m,
                n,
                a_ptr as *mut f32,
                m,
                s_ptr as *mut f32,
                u_ptr,
                ldu,
                vt_ptr,
                ldvt,
                ws_ptr as *mut f32,
                lwork,
                std::ptr::null_mut(),
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g5, _g6, _gu_opt, _gvt_opt));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Sgesvd: {status:?}"),
            });
        }
        check_info(info_ref, &stream_for_check, "Sgesvd")?;
        Ok((a_owned, s_owned, u_owned, vt_owned))
    });
}

#[allow(clippy::too_many_arguments)]
fn handle_syevd(
    handle: &Mutex<SendDn>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<f32>>>,
    info: &Mutex<CudaSlice<i32>>,
    a: GpuRef<f32>,
    n: i32,
    uplo: Uplo,
    w: GpuRef<f32>,
    compute_vectors: bool,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
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
        cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR
    } else {
        cusolver_sys::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR
    };

    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _ga) = a_owned.device_ptr_mut(stream);
        let (w_ptr, _gw) = w_owned.device_ptr_mut(stream);
        let status = unsafe {
            cusolver_sys::cusolverDnSsyevd_bufferSize(
                h.0.cu(),
                jobz,
                fill,
                n,
                a_ptr as *const f32,
                n,
                w_ptr as *const f32,
                &mut lwork as *mut _,
            )
        };
        drop((_ga, _gw));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            let _ = reply.send(Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Ssyevd_bufferSize: {status:?}"),
            }));
            return;
        }
    }
    if let Err(e) = ensure_workspace(workspace, stream, lwork as usize) {
        let _ = reply.send(Err(e));
        return;
    }

    a.record_write(stream);
    w.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let info_ref = info;
    let stream_for_check = stream.clone();

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let mut info_lock = info_ref.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (w_ptr, _g2) = w_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g3) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _g4) = info_lock.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            cusolver_sys::cusolverDnSsyevd(
                h.0.cu(),
                jobz,
                fill,
                n,
                a_ptr as *mut f32,
                n,
                w_ptr as *mut f32,
                ws_ptr as *mut f32,
                lwork,
                info_ptr as *mut i32,
            )
        };
        drop((_g1, _g2, _g3, _g4));
        if status != cusolver_sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("Ssyevd: {status:?}"),
            });
        }
        check_info(info_ref, &stream_for_check, "Ssyevd")?;
        Ok((a_owned, w_owned))
    });
}
