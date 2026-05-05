//! Shared scratch / info helpers for `kernel::solver::*`.
//!
//! Phase 1 widens the dense workspace from `CudaSlice<f32>` to
//! `CudaSlice<u8>` so the same buffer can serve f32 / f64 (and
//! eventually c32 / c64) launches. Each op converts its
//! `lwork: i32` (in `T` elements) to bytes via `lwork *
//! size_of::<T>()` before calling [`ensure_workspace_bytes`].

use std::mem::size_of;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use parking_lot::Mutex;

use crate::error::GpuError;
use crate::sys::cusolver::LIB;

/// Grow `workspace` to at least `needed_bytes` bytes. Returns `()`
/// on success; the caller is expected to lock and `as_mut()` the
/// buffer when actually launching.
pub fn ensure_workspace_bytes(
    workspace: &Mutex<Option<CudaSlice<u8>>>,
    stream: &Arc<cudarc::driver::CudaStream>,
    needed_bytes: usize,
) -> Result<(), GpuError> {
    let mut g = workspace.lock();
    let cur = g.as_ref().map(|s| s.len()).unwrap_or(0);
    if cur >= needed_bytes {
        return Ok(());
    }
    *g =
        Some(stream.alloc_zeros::<u8>(needed_bytes).map_err(|e| {
            GpuError::OutOfMemory(format!("solver workspace ({needed_bytes}B): {e}"))
        })?);
    Ok(())
}

/// Compute `lwork * sizeof::<T>()` with explicit overflow handling.
pub fn lwork_bytes<T>(lwork: i32) -> usize {
    let lwork = lwork.max(0) as usize;
    lwork.saturating_mul(size_of::<T>())
}

/// Read back the 1-element info buffer synchronously and translate
/// non-zero values into `LibraryError`.
pub fn check_info(
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

/// Read back an `n`-element info array (used by batched factorisations
/// where each problem instance writes its own info code).
pub fn check_info_array(
    info: &CudaSlice<i32>,
    stream: &Arc<cudarc::driver::CudaStream>,
    op: &'static str,
    n: usize,
) -> Result<(), GpuError> {
    let mut host = vec![0i32; n];
    stream
        .memcpy_dtoh(info, &mut host[..])
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: read info array: {e}"),
        })?;
    stream.synchronize().map_err(|e| GpuError::LibraryError {
        lib: LIB,
        msg: format!("{op}: sync after info array: {e}"),
    })?;
    if let Some((idx, code)) = host.iter().enumerate().find(|(_, c)| **c != 0) {
        return Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: batch[{idx}] info={code}"),
        });
    }
    Ok(())
}
