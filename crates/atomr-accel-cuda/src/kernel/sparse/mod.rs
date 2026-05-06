//! `SparseActor` — wraps cuSPARSE for the full op surface.
//!
//! Phase 4 expands the F-9 CSR-only SpMv/SpMm shape into a full
//! generic-API surface (CSR/COO/CSC/Blocked-ELL/BSR × f32/f64/f16/bf16
//! × i32/i64) plus SpGEMM, SpSV, SDDMM, and dense↔sparse conversion.
//!
//! ## Mailbox shape
//!
//! ```ignore
//! pub enum SparseMsg {
//!     Op(Box<dyn SparseDispatch>),    // canonical
//!     #[deprecated] SpMv { ... },     // legacy CSR-only f32
//!     #[deprecated] SpMm { ... },
//! }
//! ```
//!
//! New callers ship the `Op(Box<…>)` variant produced by an
//! `SpMvRequest::new(…)` / `SpMmRequest::new(…)` / etc. The deprecated
//! typed variants route through the original F-9 implementation
//! directly (CSR-only, f32-only) and remain wire-compatible with the
//! existing `tests/spmv_e2e.rs` end-to-end test.
//!
//! The CudaContext / supervision wiring is unchanged from F-9 — the
//! actor still owns a `cusparseHandle_t` for the lifetime of the
//! current `ContextActor` generation, panics with `"ContextPoisoned"`
//! on init failure, and drops to `Mock` mode when no GPU is present.

pub mod convert;
pub mod descriptor;
pub mod dispatch_impls;
pub mod format;
pub mod sddmm;
pub mod spgemm;
pub mod spmm;
pub mod spmv;
pub mod spsv;

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::cusparse::sys as cusparse_sys;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{SendSparseHandle, SparseDispatch, SparseDispatchCtx};
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "cusparse";

/// Legacy CSR sparse matrix in device memory — kept for back-compat with
/// callers built against F-9. Prefer [`SparseMatrix`] for new code.
#[derive(Clone)]
pub struct CsrMatrix {
    pub row_offsets: GpuRef<i32>,
    pub col_indices: GpuRef<i32>,
    pub values: GpuRef<f32>,
    pub rows: i64,
    pub cols: i64,
    pub nnz: i64,
}

/// Public messages for [`SparseActor`].
///
/// New code uses the canonical `Op(Box<dyn SparseDispatch>)` payload.
/// The two deprecated typed variants are aliases retained for
/// back-compat with F-9 callers and the existing `spmv_e2e` integration
/// test.
pub enum SparseMsg {
    /// Canonical Phase-4 dispatch — generic over dtype/format/index
    /// type via the boxed [`SparseDispatch`].
    Op(Box<dyn SparseDispatch>),

    #[deprecated(
        note = "use SparseMsg::Op(Box::new(SpMvRequest::new(...))) for the dtype-generic path"
    )]
    SpMv {
        csr: CsrMatrix,
        x: GpuRef<f32>,
        y: GpuRef<f32>,
        alpha: f32,
        beta: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },

    #[deprecated(
        note = "use SparseMsg::Op(Box::new(SpMmRequest::new(...))) for the dtype-generic path"
    )]
    SpMm {
        csr: CsrMatrix,
        b: GpuRef<f32>,
        c: GpuRef<f32>,
        b_cols: i64,
        ldb: i64,
        ldc: i64,
        alpha: f32,
        beta: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct SparseActor {
    inner: SparseInner,
}

#[allow(dead_code)]
enum SparseInner {
    Real {
        handle: Mutex<SendSparseHandle>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        /// On-demand-grown external buffer (in u8). Never shrunk.
        workspace: Mutex<Option<CudaSlice<u8>>>,
    },
    Mock,
}

impl Drop for SparseInner {
    fn drop(&mut self) {
        if let SparseInner::Real { handle, .. } = self {
            let h = handle.lock();
            unsafe {
                let _ = cusparse_sys::cusparseDestroy(h.0);
            }
        }
    }
}

impl SparseActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            let mut h: cusparse_sys::cusparseHandle_t = std::ptr::null_mut();
            let s = unsafe { cusparse_sys::cusparseCreate(&mut h as *mut _) };
            if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
                panic!("ContextPoisoned: cusparseCreate failed: {s:?}");
            }
            let s = unsafe { cusparse_sys::cusparseSetStream(h, stream.cu_stream() as *mut _) };
            if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
                unsafe {
                    let _ = cusparse_sys::cusparseDestroy(h);
                }
                panic!("ContextPoisoned: cusparseSetStream failed: {s:?}");
            }
            SparseActor {
                inner: SparseInner::Real {
                    handle: Mutex::new(SendSparseHandle(h)),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                    workspace: Mutex::new(None),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| SparseActor {
            inner: SparseInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for SparseActor {
    type Msg = SparseMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: SparseMsg) {
        match &self.inner {
            SparseInner::Mock => mock_reply(msg),
            SparseInner::Real {
                handle,
                stream,
                completion,
                workspace,
                ..
            } =>
            {
                #[allow(deprecated)]
                match msg {
                    SparseMsg::Op(op) => {
                        let ctx = SparseDispatchCtx {
                            handle,
                            stream,
                            completion,
                            workspace,
                        };
                        op.dispatch(&ctx);
                    }
                    SparseMsg::SpMv {
                        csr,
                        x,
                        y,
                        alpha,
                        beta,
                        reply,
                    } => {
                        handle_spmv(
                            handle, stream, completion, workspace, csr, x, y, alpha, beta, reply,
                        );
                    }
                    SparseMsg::SpMm {
                        csr,
                        b,
                        c,
                        b_cols,
                        ldb,
                        ldc,
                        alpha,
                        beta,
                        reply,
                    } => {
                        handle_spmm(
                            handle, stream, completion, workspace, csr, b, c, b_cols, ldb, ldc,
                            alpha, beta, reply,
                        );
                    }
                }
            }
        }
    }
}

fn mock_reply(msg: SparseMsg) {
    let err = || GpuError::Unrecoverable("SparseActor in mock mode".into());
    #[allow(deprecated)]
    match msg {
        SparseMsg::Op(op) => {
            // We can't dispatch without a handle, so surface the error
            // via the boxed op's own dispatch path. The ctx is unused
            // for the mock case — but we still need to give the op a
            // place to put its reply. Drop the box; the reply oneshot
            // inside is dropped, which the caller observes as a
            // `RecvError`.
            //
            // For symmetry with the typed variants, we surface a typed
            // error via the dispatch trait's own context — but no ctx
            // exists here. Drop is the documented mock-mode behaviour.
            drop(op);
        }
        SparseMsg::SpMv { reply, .. } | SparseMsg::SpMm { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
    }
}

fn ensure_workspace(
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    needed_bytes: usize,
) -> Result<(), GpuError> {
    let mut g = workspace.lock();
    let cur = g.as_ref().map(|s| s.len()).unwrap_or(0);
    if cur >= needed_bytes {
        return Ok(());
    }
    *g = Some(stream.alloc_zeros::<u8>(needed_bytes.max(1)).map_err(|e| {
        GpuError::OutOfMemory(format!("cusparse workspace ({needed_bytes}B): {e}"))
    })?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_spmv(
    handle: &Mutex<SendSparseHandle>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    csr: CsrMatrix,
    x: GpuRef<f32>,
    y: GpuRef<f32>,
    alpha: f32,
    beta: f32,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let row_off = match csr.row_offsets.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let col_idx = match csr.col_indices.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let vals = match csr.values.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
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
    let mut y_owned = match Arc::try_unwrap(y_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SpMv y has multiple live references".into(),
            )));
            return;
        }
    };

    let h = handle.lock();
    let (row_off_ptr, _g0) = row_off.device_ptr(stream);
    let (col_idx_ptr, _g1) = col_idx.device_ptr(stream);
    let (vals_ptr, _g2) = vals.device_ptr(stream);
    let (x_ptr, _g3) = x_slice.device_ptr(stream);
    let (y_ptr, _g4) = y_owned.device_ptr_mut(stream);

    let mut mat_desc: cusparse_sys::cusparseSpMatDescr_t = std::ptr::null_mut();
    let s = unsafe {
        cusparse_sys::cusparseCreateCsr(
            &mut mat_desc as *mut _,
            csr.rows,
            csr.cols,
            csr.nnz,
            row_off_ptr as *mut _,
            col_idx_ptr as *mut _,
            vals_ptr as *mut _,
            cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparse_sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cusparse_sys::cudaDataType::CUDA_R_32F,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateCsr: {s:?}"),
        }));
        return;
    }
    let mut x_desc: cusparse_sys::cusparseDnVecDescr_t = std::ptr::null_mut();
    let s = unsafe {
        cusparse_sys::cusparseCreateDnVec(
            &mut x_desc as *mut _,
            csr.cols,
            x_ptr as *mut _,
            cusparse_sys::cudaDataType::CUDA_R_32F,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        unsafe {
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateDnVec(x): {s:?}"),
        }));
        return;
    }
    let mut y_desc: cusparse_sys::cusparseDnVecDescr_t = std::ptr::null_mut();
    let s = unsafe {
        cusparse_sys::cusparseCreateDnVec(
            &mut y_desc as *mut _,
            csr.rows,
            y_ptr as *mut _,
            cusparse_sys::cudaDataType::CUDA_R_32F,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        unsafe {
            let _ = cusparse_sys::cusparseDestroyDnVec(x_desc);
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateDnVec(y): {s:?}"),
        }));
        return;
    }

    let alpha_h = alpha;
    let beta_h = beta;
    let mut buf_size: usize = 0;
    let s = unsafe {
        cusparse_sys::cusparseSpMV_bufferSize(
            h.0,
            cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_h as *const f32 as *const _,
            mat_desc,
            x_desc,
            &beta_h as *const f32 as *const _,
            y_desc,
            cusparse_sys::cudaDataType::CUDA_R_32F,
            cusparse_sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
            &mut buf_size as *mut _,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        unsafe {
            let _ = cusparse_sys::cusparseDestroyDnVec(y_desc);
            let _ = cusparse_sys::cusparseDestroyDnVec(x_desc);
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("SpMV_bufferSize: {s:?}"),
        }));
        return;
    }
    drop((_g0, _g1, _g2, _g3, _g4));
    drop(h);

    if let Err(e) = ensure_workspace(workspace, stream, buf_size) {
        unsafe {
            let _ = cusparse_sys::cusparseDestroyDnVec(y_desc);
            let _ = cusparse_sys::cusparseDestroyDnVec(x_desc);
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(e));
        return;
    }

    y.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let stream_for_check = stream.clone();
    struct SendDesc<T>(T);
    unsafe impl<T> Send for SendDesc<T> {}
    let mat = SendDesc(mat_desc);
    let xd = SendDesc(x_desc);
    let yd = SendDesc(y_desc);

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let (y_ptr, _g) = y_owned.device_ptr_mut(&stream_for_check);
        let _ = y_ptr;
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _gws) = ws_slice.device_ptr_mut(&stream_for_check);
        let s = unsafe {
            cusparse_sys::cusparseSpMV(
                h.0,
                cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_h as *const f32 as *const _,
                mat.0,
                xd.0,
                &beta_h as *const f32 as *const _,
                yd.0,
                cusparse_sys::cudaDataType::CUDA_R_32F,
                cusparse_sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                ws_ptr as *mut _,
            )
        };
        drop((_g, _gws));
        if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
            unsafe {
                let _ = cusparse_sys::cusparseDestroyDnVec(yd.0);
                let _ = cusparse_sys::cusparseDestroyDnVec(xd.0);
                let _ = cusparse_sys::cusparseDestroySpMat(mat.0);
            }
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("SpMV: {s:?}"),
            });
        }
        struct DescGuard {
            mat: cusparse_sys::cusparseSpMatDescr_t,
            x: cusparse_sys::cusparseDnVecDescr_t,
            y: cusparse_sys::cusparseDnVecDescr_t,
        }
        impl Drop for DescGuard {
            fn drop(&mut self) {
                unsafe {
                    let _ = cusparse_sys::cusparseDestroyDnVec(self.y);
                    let _ = cusparse_sys::cusparseDestroyDnVec(self.x);
                    let _ = cusparse_sys::cusparseDestroySpMat(self.mat);
                }
            }
        }
        unsafe impl Send for DescGuard {}
        let guard = DescGuard {
            mat: mat.0,
            x: xd.0,
            y: yd.0,
        };
        Ok((y_owned, row_off, col_idx, vals, x_slice, guard))
    });
}

#[allow(clippy::too_many_arguments)]
fn handle_spmm(
    handle: &Mutex<SendSparseHandle>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    csr: CsrMatrix,
    b: GpuRef<f32>,
    c: GpuRef<f32>,
    b_cols: i64,
    ldb: i64,
    ldc: i64,
    alpha: f32,
    beta: f32,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let row_off = match csr.row_offsets.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let col_idx = match csr.col_indices.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let vals = match csr.values.access() {
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
    let c_slice = match c.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "SpMm c has multiple live references".into(),
            )));
            return;
        }
    };

    let h = handle.lock();
    let (row_off_ptr, _g0) = row_off.device_ptr(stream);
    let (col_idx_ptr, _g1) = col_idx.device_ptr(stream);
    let (vals_ptr, _g2) = vals.device_ptr(stream);
    let (b_ptr, _g3) = b_slice.device_ptr(stream);
    let (c_ptr, _g4) = c_owned.device_ptr_mut(stream);

    let mut mat_desc: cusparse_sys::cusparseSpMatDescr_t = std::ptr::null_mut();
    let s = unsafe {
        cusparse_sys::cusparseCreateCsr(
            &mut mat_desc as *mut _,
            csr.rows,
            csr.cols,
            csr.nnz,
            row_off_ptr as *mut _,
            col_idx_ptr as *mut _,
            vals_ptr as *mut _,
            cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparse_sys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusparse_sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cusparse_sys::cudaDataType::CUDA_R_32F,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateCsr: {s:?}"),
        }));
        return;
    }
    let mut b_desc: cusparse_sys::cusparseDnMatDescr_t = std::ptr::null_mut();
    let s = unsafe {
        cusparse_sys::cusparseCreateDnMat(
            &mut b_desc as *mut _,
            csr.cols,
            b_cols,
            ldb,
            b_ptr as *mut _,
            cusparse_sys::cudaDataType::CUDA_R_32F,
            cusparse_sys::cusparseOrder_t::CUSPARSE_ORDER_COL,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        unsafe {
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateDnMat(b): {s:?}"),
        }));
        return;
    }
    let mut c_desc: cusparse_sys::cusparseDnMatDescr_t = std::ptr::null_mut();
    let s = unsafe {
        cusparse_sys::cusparseCreateDnMat(
            &mut c_desc as *mut _,
            csr.rows,
            b_cols,
            ldc,
            c_ptr as *mut _,
            cusparse_sys::cudaDataType::CUDA_R_32F,
            cusparse_sys::cusparseOrder_t::CUSPARSE_ORDER_COL,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        unsafe {
            let _ = cusparse_sys::cusparseDestroyDnMat(b_desc);
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("CreateDnMat(c): {s:?}"),
        }));
        return;
    }

    let alpha_h = alpha;
    let beta_h = beta;
    let mut buf_size: usize = 0;
    let s = unsafe {
        cusparse_sys::cusparseSpMM_bufferSize(
            h.0,
            cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_h as *const f32 as *const _,
            mat_desc,
            b_desc,
            &beta_h as *const f32 as *const _,
            c_desc,
            cusparse_sys::cudaDataType::CUDA_R_32F,
            cusparse_sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
            &mut buf_size as *mut _,
        )
    };
    if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        unsafe {
            let _ = cusparse_sys::cusparseDestroyDnMat(c_desc);
            let _ = cusparse_sys::cusparseDestroyDnMat(b_desc);
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("SpMM_bufferSize: {s:?}"),
        }));
        return;
    }
    drop((_g0, _g1, _g2, _g3, _g4));
    drop(h);

    if let Err(e) = ensure_workspace(workspace, stream, buf_size) {
        unsafe {
            let _ = cusparse_sys::cusparseDestroyDnMat(c_desc);
            let _ = cusparse_sys::cusparseDestroyDnMat(b_desc);
            let _ = cusparse_sys::cusparseDestroySpMat(mat_desc);
        }
        let _ = reply.send(Err(e));
        return;
    }

    c.record_write(stream);

    let handle_clone = handle;
    let workspace_ref = workspace;
    let stream_for_check = stream.clone();
    struct SendDesc<T>(T);
    unsafe impl<T> Send for SendDesc<T> {}
    let mat = SendDesc(mat_desc);
    let bd = SendDesc(b_desc);
    let cd = SendDesc(c_desc);

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle_clone.lock();
        let mut ws = workspace_ref.lock();
        let (_c_ptr, _g) = c_owned.device_ptr_mut(&stream_for_check);
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _gws) = ws_slice.device_ptr_mut(&stream_for_check);
        let s = unsafe {
            cusparse_sys::cusparseSpMM(
                h.0,
                cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                cusparse_sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_h as *const f32 as *const _,
                mat.0,
                bd.0,
                &beta_h as *const f32 as *const _,
                cd.0,
                cusparse_sys::cudaDataType::CUDA_R_32F,
                cusparse_sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                ws_ptr as *mut _,
            )
        };
        drop((_g, _gws));
        if s != cusparse_sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
            unsafe {
                let _ = cusparse_sys::cusparseDestroyDnMat(cd.0);
                let _ = cusparse_sys::cusparseDestroyDnMat(bd.0);
                let _ = cusparse_sys::cusparseDestroySpMat(mat.0);
            }
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("SpMM: {s:?}"),
            });
        }
        struct DescGuard {
            mat: cusparse_sys::cusparseSpMatDescr_t,
            b: cusparse_sys::cusparseDnMatDescr_t,
            c: cusparse_sys::cusparseDnMatDescr_t,
        }
        impl Drop for DescGuard {
            fn drop(&mut self) {
                unsafe {
                    let _ = cusparse_sys::cusparseDestroyDnMat(self.c);
                    let _ = cusparse_sys::cusparseDestroyDnMat(self.b);
                    let _ = cusparse_sys::cusparseDestroySpMat(self.mat);
                }
            }
        }
        unsafe impl Send for DescGuard {}
        let guard = DescGuard {
            mat: mat.0,
            b: bd.0,
            c: cd.0,
        };
        Ok((c_owned, row_off, col_idx, vals, b_slice, guard))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// The deprecated typed `SpMv` variant is what the F-9 e2e test
    /// emits — make sure it still constructs cleanly even though new
    /// callers route through `SparseMsg::Op(...)`.
    #[test]
    #[allow(deprecated)]
    fn deprecated_spmv_alias_still_constructs() {
        // We can't mint a real GpuRef without a CudaSlice, so we
        // exercise just the enum match shape.
        let state = Arc::new(DeviceState::new(0));
        // Touch `state` so the unused-binding lint stays quiet on the
        // host-only test path.
        assert_eq!(state.generation(), 0);

        // Compile-only: the enum variant is constructible by name.
        fn _assemble<F>(_f: F) {}
        _assemble(
            |csr: CsrMatrix,
             x: GpuRef<f32>,
             y: GpuRef<f32>,
             reply: oneshot::Sender<Result<(), GpuError>>| {
                SparseMsg::SpMv {
                    csr,
                    x,
                    y,
                    alpha: 1.0,
                    beta: 0.0,
                    reply,
                }
            },
        );
        _assemble(
            |csr: CsrMatrix,
             b: GpuRef<f32>,
             c: GpuRef<f32>,
             reply: oneshot::Sender<Result<(), GpuError>>| {
                SparseMsg::SpMm {
                    csr,
                    b,
                    c,
                    b_cols: 1,
                    ldb: 1,
                    ldc: 1,
                    alpha: 1.0,
                    beta: 0.0,
                    reply,
                }
            },
        );
    }
}
