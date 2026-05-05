//! `cusolverSp` sparse direct solvers.
//!
//! cuSOLVER ships device-side `csrlsv*` routines for solving
//! `A x = b` with `A` in CSR layout via Cholesky (`csrlsvchol`) or
//! QR (`csrlsvqr`). The `lu` variant is host-only in cuSOLVER; we
//! expose it here as a stub that returns a typed
//! `LibraryError("csrlsvluHost: not bridged from host yet")` rather
//! than dropping the message silently — applications that want LU
//! today can route through the dense `LuRequest`.
//!
//! All three requests share the same shape: a CSR-encoded `m × m`
//! matrix and a length-`m` RHS, plus a length-`m` output `x` and a
//! single `i32` "singularity" indicator.

use std::sync::Arc;

use cudarc::cusolver::sys as cs;
use cudarc::cusolver::SpHandle;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use tokio::sync::oneshot;

use crate::dtype::SolverSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::kernel::solver::SendSp;
use crate::sys::cusolver::{status_to_result, SparseSolverScalar, LIB};

use super::{SolverCells, SolverDispatch};

/// CSR matrix passed by reference into a sparse solve.
#[derive(Clone)]
pub struct SparseCsrInput<T: SolverSupported> {
    pub row_ptr: GpuRef<i32>,
    pub col_ind: GpuRef<i32>,
    pub values: GpuRef<T>,
    pub m: i32,
    pub nnz: i32,
}

/// Cholesky-based solve. Requires SPD `A`.
pub struct SparseCholeskyRequest<T: SolverSupported> {
    pub a: SparseCsrInput<T>,
    pub b: GpuRef<T>,
    pub x: GpuRef<T>,
    pub tol: f64,
    /// `0` = no reorder, `1`/`2`/`3` = AMD/RCM/METIS (cuSOLVER docs).
    pub reorder: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// QR-based solve.
pub struct SparseQrRequest<T: SolverSupported> {
    pub a: SparseCsrInput<T>,
    pub b: GpuRef<T>,
    pub x: GpuRef<T>,
    pub tol: f64,
    pub reorder: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// LU-based solve (cuSOLVER only ships `csrlsvluHost`; this request
/// is reserved so applications can express intent without us
/// having to add another message variant later).
pub struct SparseLuRequest<T: SolverSupported> {
    pub a: SparseCsrInput<T>,
    pub b: GpuRef<T>,
    pub x: GpuRef<T>,
    pub tol: f64,
    pub reorder: i32,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// Newtype around `cusparseMatDescr_t` that owns the descriptor for
/// the duration of one solve. Default-initialised with general
/// matrix type and zero-based indexing — cuSOLVER's Sp solvers
/// require both.
struct SpMatDescr(cs::cusparseMatDescr_t);
unsafe impl Send for SpMatDescr {}
impl Drop for SpMatDescr {
    fn drop(&mut self) {
        unsafe {
            // Destroy via the cusparse symbol; cusolver's sys re-
            // exports the same FFI declaration.
            let _ = cudarc::cusparse::sys::cusparseDestroyMatDescr(
                self.0 as cudarc::cusparse::sys::cusparseMatDescr_t,
            );
        }
    }
}

fn make_descr() -> Result<SpMatDescr, GpuError> {
    let mut handle: cudarc::cusparse::sys::cusparseMatDescr_t = std::ptr::null_mut();
    let status = unsafe { cudarc::cusparse::sys::cusparseCreateMatDescr(&mut handle) };
    if status != cudarc::cusparse::sys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        return Err(GpuError::lib(
            LIB,
            format!("cusparseCreateMatDescr: {status:?}"),
        ));
    }
    let _ = unsafe {
        cudarc::cusparse::sys::cusparseSetMatType(
            handle,
            cudarc::cusparse::sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL,
        )
    };
    let _ = unsafe {
        cudarc::cusparse::sys::cusparseSetMatIndexBase(
            handle,
            cudarc::cusparse::sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        )
    };
    Ok(SpMatDescr(handle as cs::cusparseMatDescr_t))
}

fn ensure_sp_handle(
    cell: &parking_lot::Mutex<Option<SendSp>>,
    stream: &Arc<cudarc::driver::CudaStream>,
) -> Result<(), GpuError> {
    let mut g = cell.lock();
    if g.is_some() {
        return Ok(());
    }
    let h = SpHandle::new(stream.clone())
        .map_err(|e| GpuError::lib(LIB, format!("SpHandle::new: {e}")))?;
    *g = Some(SendSp(h));
    Ok(())
}

enum SparseAlgo {
    Cholesky,
    Qr,
}

impl<T> SolverDispatch for SparseCholeskyRequest<T>
where
    T: SolverSupported + SparseSolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let SparseCholeskyRequest {
            a,
            b,
            x,
            tol,
            reorder,
            reply,
        } = *self;
        run_csrlsv::<T>(cells, SparseAlgo::Cholesky, a, b, x, tol, reorder, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

impl<T> SolverDispatch for SparseQrRequest<T>
where
    T: SolverSupported + SparseSolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let SparseQrRequest {
            a,
            b,
            x,
            tol,
            reorder,
            reply,
        } = *self;
        run_csrlsv::<T>(cells, SparseAlgo::Qr, a, b, x, tol, reorder, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

impl<T> SolverDispatch for SparseLuRequest<T>
where
    T: SolverSupported + SparseSolverScalar,
{
    fn dispatch(self: Box<Self>, _cells: SolverCells<'_>) {
        // Device-side `csrlsvlu` does not exist in cuSOLVER 11.x; the
        // host variant requires a host-side staging copy that's out
        // of scope for Phase 1. Surface a typed error so callers
        // know the surface exists but is not yet wired.
        let _ = self.reply.send(Err(GpuError::lib(
            LIB,
            "csrlsvlu: cuSOLVER only ships a host-side LU; pending host bridge",
        )));
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

#[allow(clippy::too_many_arguments)]
fn run_csrlsv<T: SparseSolverScalar>(
    cells: SolverCells<'_>,
    algo: SparseAlgo,
    a: SparseCsrInput<T>,
    b: GpuRef<T>,
    x: GpuRef<T>,
    tol: f64,
    reorder: i32,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        stream,
        completion,
        sp_handle,
        ..
    } = cells;

    if let Err(e) = ensure_sp_handle(sp_handle, stream) {
        let _ = reply.send(Err(e));
        return;
    }

    // Validate refs.
    let row_ptr_slice = match a.row_ptr.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let col_ind_slice = match a.col_ind.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let vals_slice = match a.values.access() {
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
                "Sparse x has multiple live references".into(),
            )));
            return;
        }
    };

    let descr = match make_descr() {
        Ok(d) => d,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };

    x.record_write(stream);

    let stream_for_check = stream.clone();
    let m = a.m;
    let nnz = a.nnz;

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        // Singularity flag — host-side i32, populated synchronously.
        let mut singularity: i32 = 0;
        let g = sp_handle.lock();
        let sp = g.as_ref().expect("sp handle ensured");
        let (row_ptr_p, _g1) = row_ptr_slice.device_ptr(&stream_for_check);
        let (col_ind_p, _g2) = col_ind_slice.device_ptr(&stream_for_check);
        let (vals_p, _g3) = vals_slice.device_ptr(&stream_for_check);
        let (b_ptr, _g4) = b_slice.device_ptr(&stream_for_check);
        let (x_ptr, _g5) = x_owned.device_ptr_mut(&stream_for_check);
        let status = unsafe {
            match algo {
                SparseAlgo::Cholesky => T::csrlsvchol(
                    sp.0.cu(),
                    m,
                    nnz,
                    descr.0,
                    vals_p as *const T,
                    row_ptr_p as *const i32,
                    col_ind_p as *const i32,
                    b_ptr as *const T,
                    tol,
                    reorder,
                    x_ptr as *mut T,
                    &mut singularity as *mut _,
                ),
                SparseAlgo::Qr => T::csrlsvqr(
                    sp.0.cu(),
                    m,
                    nnz,
                    descr.0,
                    vals_p as *const T,
                    row_ptr_p as *const i32,
                    col_ind_p as *const i32,
                    b_ptr as *const T,
                    tol,
                    reorder,
                    x_ptr as *mut T,
                    &mut singularity as *mut _,
                ),
            }
        };
        drop((_g1, _g2, _g3, _g4, _g5));
        let op = match algo {
            SparseAlgo::Cholesky => "csrlsvchol",
            SparseAlgo::Qr => "csrlsvqr",
        };
        status_to_result(status, op)?;
        if singularity >= 0 {
            return Err(GpuError::lib(
                LIB,
                format!("{op}: singularity at row {singularity}"),
            ));
        }
        Ok((
            row_ptr_slice,
            col_ind_slice,
            vals_slice,
            b_slice,
            x_owned,
            descr,
        ))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_cholesky_request_round_trip() {
        fn assert_dispatch<R: SolverDispatch>() {}
        assert_dispatch::<SparseCholeskyRequest<f32>>();
        assert_dispatch::<SparseCholeskyRequest<f64>>();
        assert_dispatch::<SparseQrRequest<f32>>();
        assert_dispatch::<SparseQrRequest<f64>>();
        assert_dispatch::<SparseLuRequest<f32>>();
        assert_dispatch::<SparseLuRequest<f64>>();
    }
}
