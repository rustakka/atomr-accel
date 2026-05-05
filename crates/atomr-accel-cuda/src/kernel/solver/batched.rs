//! Batched cuSOLVER ops.
//!
//! cuSOLVER ships native batched variants for Cholesky
//! (`Spotrf/DpotrfBatched`) and Jacobi SVD
//! (`Sgesvdj/DgesvdjBatched`). The batched LU live on the cuBLAS side
//! (`cublasSgetrf/DgetrfBatched`); we adopt them into `SolverActor` for
//! API symmetry — each request takes a contiguous strided block of
//! `batch_size × n × n` matrices and stages an array-of-pointers on
//! demand.
//!
//! All three requests share the same input shape:
//! - `a: GpuRef<T>` — `batch_size × m × n × stride` packed
//!   column-major, with stride equal to `m * n` (no extra padding).
//! - `batch_size: i32` — number of independent problems.
//!
//! Per-batch info codes are read back into a host vec; any non-zero
//! entry surfaces as a `LibraryError` identifying the failing batch
//! index.

use std::ffi::c_int;
use std::sync::Arc;

use cudarc::cusolver::sys as cs;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::dtype::SolverSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::sys::cusolver::{status_to_result, SolverScalar, LIB};

use super::workspace::{check_info_array, ensure_workspace_bytes, lwork_bytes};
use super::{SolverCells, SolverDispatch, Uplo};

// =====================================================================
// LU batched (`cublasSgetrfBatched` / `cublasDgetrfBatched`)
// =====================================================================

pub struct GetrfBatchedRequest<T: SolverSupported> {
    /// Contiguous `batch_size × n × n` column-major buffer.
    pub a: GpuRef<T>,
    /// Square problem size.
    pub n: i32,
    /// Number of independent problems.
    pub batch_size: i32,
    /// Pivot indices: `batch_size * n` `i32` entries.
    pub ipiv: GpuRef<i32>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// Per-dtype dispatch into `cublas[SD]getrfBatched`. The cuBLAS
/// batched LU lives outside cuSOLVER; we keep the FFI thunks
/// inline (rather than expanding `SolverScalar`) so the cuBLAS
/// symbol surface stays scoped to this module.
trait BatchedLu: SolverScalar {
    unsafe fn getrf_batched(
        handle: cudarc::cublas::sys::cublasHandle_t,
        n: c_int,
        a_array: *const *mut Self,
        lda: c_int,
        pivots: *mut c_int,
        info_array: *mut c_int,
        batch_size: c_int,
    ) -> cudarc::cublas::sys::cublasStatus_t;
}

impl BatchedLu for f32 {
    unsafe fn getrf_batched(
        handle: cudarc::cublas::sys::cublasHandle_t,
        n: c_int,
        a_array: *const *mut Self,
        lda: c_int,
        pivots: *mut c_int,
        info_array: *mut c_int,
        batch_size: c_int,
    ) -> cudarc::cublas::sys::cublasStatus_t {
        cudarc::cublas::sys::cublasSgetrfBatched(
            handle, n, a_array, lda, pivots, info_array, batch_size,
        )
    }
}

impl BatchedLu for f64 {
    unsafe fn getrf_batched(
        handle: cudarc::cublas::sys::cublasHandle_t,
        n: c_int,
        a_array: *const *mut Self,
        lda: c_int,
        pivots: *mut c_int,
        info_array: *mut c_int,
        batch_size: c_int,
    ) -> cudarc::cublas::sys::cublasStatus_t {
        cudarc::cublas::sys::cublasDgetrfBatched(
            handle, n, a_array, lda, pivots, info_array, batch_size,
        )
    }
}

impl<T> SolverDispatch for GetrfBatchedRequest<T>
where
    T: SolverSupported + SolverScalar + BatchedLu,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let GetrfBatchedRequest {
            a,
            n,
            batch_size,
            ipiv,
            reply,
        } = *self;
        run_getrf_batched::<T>(cells, a, n, batch_size, ipiv, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

/// Build a contiguous `Vec<*mut T>` of per-batch starting pointers and
/// upload it as a `CudaSlice<u64>` (raw pointer values). We use the
/// stream's `memcpy_htod` since the device pointer table is unique to
/// this launch and short-lived.
fn upload_pointer_table<T>(
    stream: &Arc<cudarc::driver::CudaStream>,
    base: *mut T,
    batch: i32,
    n: i32,
) -> Result<cudarc::driver::CudaSlice<u64>, GpuError> {
    let count = batch.max(0) as usize;
    let stride_bytes = (n.max(0) as usize) * (n.max(0) as usize) * std::mem::size_of::<T>();
    let mut ptrs = Vec::with_capacity(count);
    for i in 0..count {
        let p = (base as usize).saturating_add(i * stride_bytes);
        ptrs.push(p as u64);
    }
    let mut buf = stream
        .alloc_zeros::<u64>(count.max(1))
        .map_err(|e| GpuError::OutOfMemory(format!("ptr table ({count}): {e}")))?;
    stream
        .memcpy_htod(&ptrs, &mut buf)
        .map_err(|e| GpuError::lib(LIB, format!("upload ptr table: {e}")))?;
    Ok(buf)
}

fn run_getrf_batched<T: SolverScalar + BatchedLu>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    n: i32,
    batch_size: i32,
    ipiv: GpuRef<i32>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        stream, completion, ..
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
                "GetrfBatched a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut ipiv_owned = match Arc::try_unwrap(ipiv_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GetrfBatched ipiv has multiple live references".into(),
            )));
            return;
        }
    };

    // Lazy cuBLAS handle: cuSOLVER doesn't ship batched LU, so we
    // bind cuBLAS into the cuSOLVER actor's stream just for this
    // op. Created/destroyed locally so it doesn't outlive the
    // launch. cudarc 0.19's `CudaBlas::new` pins the handle to the
    // supplied stream — exactly what we need.
    let blas = match cudarc::cublas::CudaBlas::new(stream.clone()) {
        Ok(b) => b,
        Err(e) => {
            let _ = reply.send(Err(GpuError::lib(LIB, format!("CudaBlas::new: {e}"))));
            return;
        }
    };
    let blas_handle = *blas.handle();

    // Build pointer table.
    let (a_base_ptr, _g_base) = a_owned.device_ptr_mut(stream);
    let ptr_table = match upload_pointer_table::<T>(stream, a_base_ptr as *mut T, batch_size, n) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    drop(_g_base);

    // info array: one int per batch entry, allocated fresh.
    let info_array = match stream.alloc_zeros::<i32>(batch_size.max(1) as usize) {
        Ok(b) => b,
        Err(e) => {
            let _ = reply.send(Err(GpuError::OutOfMemory(format!(
                "GetrfBatched info: {e}"
            ))));
            return;
        }
    };

    a.record_write(stream);
    ipiv.record_write(stream);

    let stream_for_check = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let (ptrs_dev, _gp) = ptr_table.device_ptr(&stream_for_check);
        let (ipiv_ptr, _gpiv) = ipiv_owned.device_ptr_mut(&stream_for_check);
        let (info_ptr, _ginfo) = info_array.device_ptr(&stream_for_check);
        let status = unsafe {
            T::getrf_batched(
                blas_handle,
                n,
                ptrs_dev as *const *mut T,
                n,
                ipiv_ptr as *mut c_int,
                info_ptr as *mut c_int,
                batch_size,
            )
        };
        drop((_gp, _gpiv, _ginfo));
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(GpuError::lib(LIB, format!("getrfBatched: {status:?}")));
        }
        check_info_array(
            &info_array,
            &stream_for_check,
            "getrfBatched",
            batch_size.max(0) as usize,
        )?;
        // Keep blas + table alive until completion.
        Ok((a_owned, ipiv_owned, ptr_table, info_array, blas))
    });
}

// =====================================================================
// Cholesky batched (`cusolverDn[SD]potrfBatched`)
// =====================================================================

pub struct PotrfBatchedRequest<T: SolverSupported> {
    pub a: GpuRef<T>,
    pub n: i32,
    pub batch_size: i32,
    pub uplo: Uplo,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

impl<T> SolverDispatch for PotrfBatchedRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let PotrfBatchedRequest {
            a,
            n,
            batch_size,
            uplo,
            reply,
        } = *self;
        run_potrf_batched::<T>(cells, a, n, batch_size, uplo, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_potrf_batched<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    n: i32,
    batch_size: i32,
    uplo: Uplo,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
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
                "PotrfBatched a has multiple live references".into(),
            )));
            return;
        }
    };

    let (a_base_ptr, _g_base) = a_owned.device_ptr_mut(stream);
    let ptr_table = match upload_pointer_table::<T>(stream, a_base_ptr as *mut T, batch_size, n) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    drop(_g_base);

    let info_array = match stream.alloc_zeros::<i32>(batch_size.max(1) as usize) {
        Ok(b) => b,
        Err(e) => {
            let _ = reply.send(Err(GpuError::OutOfMemory(format!(
                "PotrfBatched info: {e}"
            ))));
            return;
        }
    };

    a.record_write(stream);
    let fill = uplo.as_cusolver_fill();
    let stream_for_check = stream.clone();

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let (ptrs_dev, _gp) = ptr_table.device_ptr(&stream_for_check);
        let (info_ptr, _ginfo) = info_array.device_ptr(&stream_for_check);
        let status = unsafe {
            T::potrf_batched(
                h.0.cu(),
                fill,
                n,
                ptrs_dev as *mut *mut T,
                n,
                info_ptr as *mut i32,
                batch_size,
            )
        };
        drop((_gp, _ginfo));
        status_to_result(status, "potrfBatched")?;
        check_info_array(
            &info_array,
            &stream_for_check,
            "potrfBatched",
            batch_size.max(0) as usize,
        )?;
        Ok((a_owned, ptr_table, info_array))
    });
}

// =====================================================================
// Batched Jacobi SVD (`cusolverDn[SD]gesvdjBatched`)
// =====================================================================

pub struct GesvdjBatchedRequest<T: SolverSupported> {
    /// Contiguous `batch_size × m × n` column-major buffer.
    pub a: GpuRef<T>,
    pub m: i32,
    pub n: i32,
    pub batch_size: i32,
    /// Singular values: `batch_size * min(m, n)` entries.
    pub s: GpuRef<T>,
    /// Left singular vectors (`batch_size × m × m`). When `None`,
    /// `jobz = NOVECTOR`.
    pub u: Option<GpuRef<T>>,
    /// Right singular vectors (`batch_size × n × n`). When `None`,
    /// `jobz = NOVECTOR`.
    pub v: Option<GpuRef<T>>,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
}

/// `gesvdjInfo_t` is non-Send by default (raw pointer); we wrap so
/// the keep-alive can ride the post-launch task.
struct GesvdjParams(cs::gesvdjInfo_t);
unsafe impl Send for GesvdjParams {}
impl Drop for GesvdjParams {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cusolverDnDestroyGesvdjInfo(self.0);
        }
    }
}

impl<T> SolverDispatch for GesvdjBatchedRequest<T>
where
    T: SolverSupported + SolverScalar,
{
    fn dispatch(self: Box<Self>, cells: SolverCells<'_>) {
        let GesvdjBatchedRequest {
            a,
            m,
            n,
            batch_size,
            s,
            u,
            v,
            reply,
        } = *self;
        run_gesvdj_batched::<T>(cells, a, m, n, batch_size, s, u, v, reply);
    }

    fn dispatch_mock(self: Box<Self>) {
        let _ = self.reply.send(Err(GpuError::Unrecoverable(
            "SolverActor in mock mode".into(),
        )));
    }
}

fn run_gesvdj_batched<T: SolverScalar>(
    cells: SolverCells<'_>,
    a: GpuRef<T>,
    m: i32,
    n: i32,
    batch_size: i32,
    s: GpuRef<T>,
    u: Option<GpuRef<T>>,
    v: Option<GpuRef<T>>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    let SolverCells {
        handle,
        stream,
        completion,
        workspace,
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
                "GesvdjBatched a has multiple live references".into(),
            )));
            return;
        }
    };
    let mut s_owned = match Arc::try_unwrap(s_slice) {
        Ok(sl) => sl,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "GesvdjBatched s has multiple live references".into(),
            )));
            return;
        }
    };

    let mut u_owned = match u.as_ref().map(|g| g.access().map(|sl| sl.clone())) {
        Some(Ok(sl)) => match Arc::try_unwrap(sl) {
            Ok(o) => Some(o),
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "GesvdjBatched u has multiple live references".into(),
                )));
                return;
            }
        },
        Some(Err(e)) => {
            let _ = reply.send(Err(e));
            return;
        }
        None => None,
    };
    let mut v_owned = match v.as_ref().map(|g| g.access().map(|sl| sl.clone())) {
        Some(Ok(sl)) => match Arc::try_unwrap(sl) {
            Ok(o) => Some(o),
            Err(_) => {
                let _ = reply.send(Err(GpuError::Unrecoverable(
                    "GesvdjBatched v has multiple live references".into(),
                )));
                return;
            }
        },
        Some(Err(e)) => {
            let _ = reply.send(Err(e));
            return;
        }
        None => None,
    };

    // Create gesvdj params. Defaults are fine for Phase 1.
    let mut info_handle: cs::gesvdjInfo_t = std::ptr::null_mut();
    let st = unsafe { cs::cusolverDnCreateGesvdjInfo(&mut info_handle as *mut _) };
    if let Err(e) = status_to_result(st, "CreateGesvdjInfo") {
        let _ = reply.send(Err(e));
        return;
    }
    let params = GesvdjParams(info_handle);

    // info array: batch_size + 1 (matches cuSOLVER reference; we
    // size it to batch_size for the basic check).
    let info_array = match stream.alloc_zeros::<i32>(batch_size.max(1) as usize) {
        Ok(b) => b,
        Err(e) => {
            let _ = reply.send(Err(GpuError::OutOfMemory(format!(
                "GesvdjBatched info: {e}"
            ))));
            return;
        }
    };

    let jobz = if u_owned.is_some() && v_owned.is_some() {
        cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR
    } else {
        cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR
    };

    // Workspace query.
    let ldu = m;
    let ldv = n;
    let mut lwork = 0i32;
    {
        let h = handle.lock();
        let (a_ptr, _ga) = a_owned.device_ptr(stream);
        let (s_ptr, _gs) = s_owned.device_ptr(stream);
        let u_ptr: *const T = match u_owned.as_ref() {
            Some(o) => {
                let (p, _g) = o.device_ptr(stream);
                p as *const T
            }
            None => std::ptr::null(),
        };
        let v_ptr: *const T = match v_owned.as_ref() {
            Some(o) => {
                let (p, _g) = o.device_ptr(stream);
                p as *const T
            }
            None => std::ptr::null(),
        };
        let status = unsafe {
            T::gesvdj_batched_buffer_size(
                h.0.cu(),
                jobz,
                m,
                n,
                a_ptr as *const T,
                m,
                s_ptr as *const T,
                u_ptr,
                ldu,
                v_ptr,
                ldv,
                &mut lwork as *mut _,
                params.0,
                batch_size,
            )
        };
        drop((_ga, _gs));
        if let Err(e) = status_to_result(status, "gesvdjBatched_bufferSize") {
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
    if let Some(g) = &v {
        g.record_write(stream);
    }

    let stream_for_check = stream.clone();
    let workspace_ref: &Mutex<Option<cudarc::driver::CudaSlice<u8>>> = workspace;

    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let h = handle.lock();
        let mut ws = workspace_ref.lock();
        let (a_ptr, _g1) = a_owned.device_ptr_mut(&stream_for_check);
        let (s_ptr, _g2) = s_owned.device_ptr_mut(&stream_for_check);
        let (u_ptr, _gu_opt): (*mut T, _) = match u_owned.as_mut() {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&stream_for_check);
                (p as *mut T, Some(g))
            }
            None => (std::ptr::null_mut(), None),
        };
        let (v_ptr, _gv_opt): (*mut T, _) = match v_owned.as_mut() {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&stream_for_check);
                (p as *mut T, Some(g))
            }
            None => (std::ptr::null_mut(), None),
        };
        let ws_slice = ws.as_mut().expect("workspace ensured");
        let (ws_ptr, _g5) = ws_slice.device_ptr_mut(&stream_for_check);
        let (info_ptr, _ginfo) = info_array.device_ptr(&stream_for_check);
        let status = unsafe {
            T::gesvdj_batched(
                h.0.cu(),
                jobz,
                m,
                n,
                a_ptr as *mut T,
                m,
                s_ptr as *mut T,
                u_ptr,
                ldu,
                v_ptr,
                ldv,
                ws_ptr as *mut T,
                lwork,
                info_ptr as *mut i32,
                params.0,
                batch_size,
            )
        };
        drop((_g1, _g2, _g5, _ginfo, _gu_opt, _gv_opt));
        status_to_result(status, "gesvdjBatched")?;
        check_info_array(
            &info_array,
            &stream_for_check,
            "gesvdjBatched",
            batch_size.max(0) as usize,
        )?;
        Ok((a_owned, s_owned, u_owned, v_owned, info_array, params))
    });
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_request_round_trip() {
        fn assert_dispatch<R: SolverDispatch>() {}
        assert_dispatch::<GetrfBatchedRequest<f32>>();
        assert_dispatch::<GetrfBatchedRequest<f64>>();
        assert_dispatch::<PotrfBatchedRequest<f32>>();
        assert_dispatch::<PotrfBatchedRequest<f64>>();
        assert_dispatch::<GesvdjBatchedRequest<f32>>();
        assert_dispatch::<GesvdjBatchedRequest<f64>>();
    }
}
