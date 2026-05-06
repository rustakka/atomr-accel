//! Sys-level safe wrappers for the cuBLAS entry points cudarc 0.19
//! doesn't expose through its safe layer.
//!
//! Wrapped today (Phase 1 cuBLAS slice):
//! - `cublasGemmEx`, `cublasGemmStridedBatchedEx`
//! - `cublasAxpyEx`, `cublasScalEx`, `cublasNrm2Ex`, `cublasDotEx`
//! - `cublasIamaxEx`, `cublasIaminEx`, `cublasAsumEx`
//! - `cublasCopyEx`, `cublasSwapEx`, `cublasRotEx`
//! - `cublasGemv_v2`/`cublasDgemv_v2`, `cublasSger_v2`/`cublasDger_v2`
//! - `cublasSgeam`/`cublasDgeam`
//! - `cublasSsyrk_v2`/`cublasDsyrk_v2`
//! - `cublasStrsm_v2`/`cublasDtrsm_v2`
//!
//! All callers must hold the cuBLAS handle's stream current on the
//! same OS thread. The atomr-accel-cuda actor pipeline guarantees
//! that via `GpuDispatcher`.

#![allow(non_snake_case)]

use core::ffi::{c_int, c_longlong};

use cudarc::cublas::sys::{
    self, cublasComputeType_t, cublasDiagType_t, cublasFillMode_t, cublasGemmAlgo_t,
    cublasHandle_t, cublasOperation_t, cublasSideMode_t, cudaDataType,
};
use cudarc::driver::sys::CUdeviceptr;

use crate::error::GpuError;

const LIB: &str = "cublas";

#[inline]
fn check(status: sys::cublasStatus_t, op: &'static str) -> Result<(), GpuError> {
    match status {
        sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
        e => Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: {e:?}"),
        }),
    }
}

// ───────────────────────────── L3 ─────────────────────────────

/// `cublasGemmEx` — typed-erased gemm with a separate compute type.
///
/// # Safety
/// `a`/`b`/`c` must point to device buffers with the dtypes encoded
/// in `a_type`/`b_type`/`c_type` and the sizes implied by
/// `(m,n,k,lda,ldb,ldc)`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const core::ffi::c_void,
    a: CUdeviceptr,
    a_type: cudaDataType,
    lda: c_int,
    b: CUdeviceptr,
    b_type: cudaDataType,
    ldb: c_int,
    beta: *const core::ffi::c_void,
    c: CUdeviceptr,
    c_type: cudaDataType,
    ldc: c_int,
    compute_type: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> Result<(), GpuError> {
    let status = sys::cublasGemmEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a as *const _,
        a_type,
        lda,
        b as *const _,
        b_type,
        ldb,
        beta,
        c as *mut _,
        c_type,
        ldc,
        compute_type,
        algo,
    );
    check(status, "cublasGemmEx")
}

/// `cublasGemmStridedBatchedEx` — typed-erased strided-batched gemm.
///
/// # Safety
/// Like [`gemm_ex`], plus `stride_*` describes the byte stride between
/// consecutive batch entries inside a single allocation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_strided_batched_ex(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const core::ffi::c_void,
    a: CUdeviceptr,
    a_type: cudaDataType,
    lda: c_int,
    stride_a: c_longlong,
    b: CUdeviceptr,
    b_type: cudaDataType,
    ldb: c_int,
    stride_b: c_longlong,
    beta: *const core::ffi::c_void,
    c: CUdeviceptr,
    c_type: cudaDataType,
    ldc: c_int,
    stride_c: c_longlong,
    batch_count: c_int,
    compute_type: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> Result<(), GpuError> {
    let status = sys::cublasGemmStridedBatchedEx(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a as *const _,
        a_type,
        lda,
        stride_a,
        b as *const _,
        b_type,
        ldb,
        stride_b,
        beta,
        c as *mut _,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
    );
    check(status, "cublasGemmStridedBatchedEx")
}

/// `cublasSgeam` / `cublasDgeam` — matrix add/scale: `C = α·op(A) + β·op(B)`.
///
/// # Safety
/// Pointers must be valid for `(m,n)` matrices with leading dims
/// `lda`/`ldb`/`ldc` in column-major layout.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: CUdeviceptr,
    lda: c_int,
    beta: *const f32,
    b: CUdeviceptr,
    ldb: c_int,
    c: CUdeviceptr,
    ldc: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasSgeam(
        handle,
        transa,
        transb,
        m,
        n,
        alpha,
        a as *const _,
        lda,
        beta,
        b as *const _,
        ldb,
        c as *mut _,
        ldc,
    );
    check(status, "cublasSgeam")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: CUdeviceptr,
    lda: c_int,
    beta: *const f64,
    b: CUdeviceptr,
    ldb: c_int,
    c: CUdeviceptr,
    ldc: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasDgeam(
        handle,
        transa,
        transb,
        m,
        n,
        alpha,
        a as *const _,
        lda,
        beta,
        b as *const _,
        ldb,
        c as *mut _,
        ldc,
    );
    check(status, "cublasDgeam")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn ssyrk(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: CUdeviceptr,
    lda: c_int,
    beta: *const f32,
    c: CUdeviceptr,
    ldc: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasSsyrk_v2(
        handle,
        uplo,
        trans,
        n,
        k,
        alpha,
        a as *const _,
        lda,
        beta,
        c as *mut _,
        ldc,
    );
    check(status, "cublasSsyrk_v2")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dsyrk(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: CUdeviceptr,
    lda: c_int,
    beta: *const f64,
    c: CUdeviceptr,
    ldc: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasDsyrk_v2(
        handle,
        uplo,
        trans,
        n,
        k,
        alpha,
        a as *const _,
        lda,
        beta,
        c as *mut _,
        ldc,
    );
    check(status, "cublasDsyrk_v2")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn strsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: CUdeviceptr,
    lda: c_int,
    b: CUdeviceptr,
    ldb: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasStrsm_v2(
        handle,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        a as *const _,
        lda,
        b as *mut _,
        ldb,
    );
    check(status, "cublasStrsm_v2")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dtrsm(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: CUdeviceptr,
    lda: c_int,
    b: CUdeviceptr,
    ldb: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasDtrsm_v2(
        handle,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        alpha,
        a as *const _,
        lda,
        b as *mut _,
        ldb,
    );
    check(status, "cublasDtrsm_v2")
}

// ───────────────────────────── L2 ─────────────────────────────

#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: CUdeviceptr,
    lda: c_int,
    x: CUdeviceptr,
    incx: c_int,
    beta: *const f32,
    y: CUdeviceptr,
    incy: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasSgemv_v2(
        handle,
        trans,
        m,
        n,
        alpha,
        a as *const _,
        lda,
        x as *const _,
        incx,
        beta,
        y as *mut _,
        incy,
    );
    check(status, "cublasSgemv_v2")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemv(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: CUdeviceptr,
    lda: c_int,
    x: CUdeviceptr,
    incx: c_int,
    beta: *const f64,
    y: CUdeviceptr,
    incy: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasDgemv_v2(
        handle,
        trans,
        m,
        n,
        alpha,
        a as *const _,
        lda,
        x as *const _,
        incx,
        beta,
        y as *mut _,
        incy,
    );
    check(status, "cublasDgemv_v2")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn sger(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    x: CUdeviceptr,
    incx: c_int,
    y: CUdeviceptr,
    incy: c_int,
    a: CUdeviceptr,
    lda: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasSger_v2(
        handle,
        m,
        n,
        alpha,
        x as *const _,
        incx,
        y as *const _,
        incy,
        a as *mut _,
        lda,
    );
    check(status, "cublasSger_v2")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dger(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    x: CUdeviceptr,
    incx: c_int,
    y: CUdeviceptr,
    incy: c_int,
    a: CUdeviceptr,
    lda: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasDger_v2(
        handle,
        m,
        n,
        alpha,
        x as *const _,
        incx,
        y as *const _,
        incy,
        a as *mut _,
        lda,
    );
    check(status, "cublasDger_v2")
}

// ───────────────────────────── L1 ─────────────────────────────

#[allow(clippy::too_many_arguments)]
pub unsafe fn axpy_ex(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const core::ffi::c_void,
    alpha_type: cudaDataType,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    y: CUdeviceptr,
    y_type: cudaDataType,
    incy: c_int,
    execution_type: cudaDataType,
) -> Result<(), GpuError> {
    let status = sys::cublasAxpyEx(
        handle,
        n,
        alpha,
        alpha_type,
        x as *const _,
        x_type,
        incx,
        y as *mut _,
        y_type,
        incy,
        execution_type,
    );
    check(status, "cublasAxpyEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn scal_ex(
    handle: cublasHandle_t,
    n: c_int,
    alpha: *const core::ffi::c_void,
    alpha_type: cudaDataType,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    execution_type: cudaDataType,
) -> Result<(), GpuError> {
    let status = sys::cublasScalEx(
        handle,
        n,
        alpha,
        alpha_type,
        x as *mut _,
        x_type,
        incx,
        execution_type,
    );
    check(status, "cublasScalEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn nrm2_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    result: *mut core::ffi::c_void,
    result_type: cudaDataType,
    execution_type: cudaDataType,
) -> Result<(), GpuError> {
    let status = sys::cublasNrm2Ex(
        handle,
        n,
        x as *const _,
        x_type,
        incx,
        result,
        result_type,
        execution_type,
    );
    check(status, "cublasNrm2Ex")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn dot_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    y: CUdeviceptr,
    y_type: cudaDataType,
    incy: c_int,
    result: *mut core::ffi::c_void,
    result_type: cudaDataType,
    execution_type: cudaDataType,
) -> Result<(), GpuError> {
    let status = sys::cublasDotEx(
        handle,
        n,
        x as *const _,
        x_type,
        incx,
        y as *const _,
        y_type,
        incy,
        result,
        result_type,
        execution_type,
    );
    check(status, "cublasDotEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn iamax_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    result: *mut c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasIamaxEx(handle, n, x as *const _, x_type, incx, result);
    check(status, "cublasIamaxEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn iamin_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    result: *mut c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasIaminEx(handle, n, x as *const _, x_type, incx, result);
    check(status, "cublasIaminEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn asum_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    result: *mut core::ffi::c_void,
    result_type: cudaDataType,
    execution_type: cudaDataType,
) -> Result<(), GpuError> {
    let status = sys::cublasAsumEx(
        handle,
        n,
        x as *const _,
        x_type,
        incx,
        result,
        result_type,
        execution_type,
    );
    check(status, "cublasAsumEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn copy_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    y: CUdeviceptr,
    y_type: cudaDataType,
    incy: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasCopyEx(
        handle,
        n,
        x as *const _,
        x_type,
        incx,
        y as *mut _,
        y_type,
        incy,
    );
    check(status, "cublasCopyEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn swap_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    y: CUdeviceptr,
    y_type: cudaDataType,
    incy: c_int,
) -> Result<(), GpuError> {
    let status = sys::cublasSwapEx(
        handle,
        n,
        x as *mut _,
        x_type,
        incx,
        y as *mut _,
        y_type,
        incy,
    );
    check(status, "cublasSwapEx")
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn rot_ex(
    handle: cublasHandle_t,
    n: c_int,
    x: CUdeviceptr,
    x_type: cudaDataType,
    incx: c_int,
    y: CUdeviceptr,
    y_type: cudaDataType,
    incy: c_int,
    cs: *const core::ffi::c_void,
    s: *const core::ffi::c_void,
    cs_type: cudaDataType,
    execution_type: cudaDataType,
) -> Result<(), GpuError> {
    let status = sys::cublasRotEx(
        handle,
        n,
        x as *mut _,
        x_type,
        incx,
        y as *mut _,
        y_type,
        incy,
        cs,
        s,
        cs_type,
        execution_type,
    );
    check(status, "cublasRotEx")
}
