//! Crate-private FFI thunks around `cudarc::cusolver::sys`.
//!
//! cudarc 0.19's safe layer covers handle construction (`DnHandle`,
//! `SpHandle`) but leaves every dense / sparse / batched factorisation
//! behind raw `extern "C"` declarations under `cusolver::sys::lib`. The
//! [`SolverScalar`] trait lifts the per-prefix C entry points
//! (`Sgeqrf`, `Dgeqrf`, …) onto a uniform Rust surface so the actor's
//! per-op handlers can be written generically over `T: SolverSupported`.
//!
//! All unsafe FFI is contained inside this module — handlers in
//! `kernel::solver::*` only see the typed wrappers and a single
//! [`status_to_result`] adapter.

use cudarc::cusolver::sys as cs;

use crate::dtype::SolverSupported;
use crate::error::GpuError;

pub const LIB: &str = "cusolver";

/// Translate a `cusolverStatus_t` into our typed error.
pub fn status_to_result(status: cs::cusolverStatus_t, op: &'static str) -> Result<(), GpuError> {
    if status == cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: {status:?}"),
        })
    }
}

/// Per-dtype dispatcher to the cuSOLVER `S/D/C/Z` entry points.
///
/// Methods take raw pointers because the actor side has already
/// extracted them via `device_ptr_mut`; threading lifetimes through
/// `&mut CudaSlice<T>` would force every wrapper to be generic over
/// keep-alive guards. The actor is responsible for keeping the slices
/// alive for the duration of the call.
///
/// # Safety
///
/// Each `unsafe` method is safe to call when:
/// - the `handle` is a live `DnHandle::cu()` value bound to the same
///   stream the slices were allocated/written through,
/// - all device pointers reference the buffer sizes implied by the
///   `(m, n, lda)` triple per the cuSOLVER reference,
/// - `lwork` matches what the corresponding `*_buffer_size` returned,
/// - `info` points to at least one writable `i32`.
pub trait SolverScalar: SolverSupported {
    /// QR `geqrf`: workspace query.
    unsafe fn geqrf_buffer_size(
        handle: cs::cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        lwork: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn geqrf(
        handle: cs::cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        tau: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// LU `getrf`: workspace query.
    unsafe fn getrf_buffer_size(
        handle: cs::cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        lwork: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn getrf(
        handle: cs::cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        work: *mut Self,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn getrs(
        handle: cs::cusolverDnHandle_t,
        trans: cs::cublasOperation_t,
        n: i32,
        nrhs: i32,
        a: *const Self,
        lda: i32,
        ipiv: *const i32,
        b: *mut Self,
        ldb: i32,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// Cholesky `potrf`.
    unsafe fn potrf_buffer_size(
        handle: cs::cusolverDnHandle_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a: *mut Self,
        lda: i32,
        lwork: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn potrf(
        handle: cs::cusolverDnHandle_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a: *mut Self,
        lda: i32,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// SVD `gesvd`. (`gesvd` workspace query takes only m/n.)
    unsafe fn gesvd_buffer_size(
        handle: cs::cusolverDnHandle_t,
        m: i32,
        n: i32,
        lwork: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn gesvd(
        handle: cs::cusolverDnHandle_t,
        jobu: i8,
        jobvt: i8,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        vt: *mut Self,
        ldvt: i32,
        work: *mut Self,
        lwork: i32,
        rwork: *mut Self,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// Symmetric eigendecomposition `syevd`.
    unsafe fn syevd_buffer_size(
        handle: cs::cusolverDnHandle_t,
        jobz: cs::cusolverEigMode_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a: *const Self,
        lda: i32,
        w: *const Self,
        lwork: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn syevd(
        handle: cs::cusolverDnHandle_t,
        jobz: cs::cusolverEigMode_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a: *mut Self,
        lda: i32,
        w: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// Generalized symmetric eigendecomposition `sygvd` (real)
    /// / `hegvd` (complex). Phase 1 routes both messages through
    /// the same trait method since the f32/f64 surface is purely
    /// real.
    unsafe fn sygvd_buffer_size(
        handle: cs::cusolverDnHandle_t,
        itype: cs::cusolverEigType_t,
        jobz: cs::cusolverEigMode_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a: *const Self,
        lda: i32,
        b: *const Self,
        ldb: i32,
        w: *const Self,
        lwork: *mut i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn sygvd(
        handle: cs::cusolverDnHandle_t,
        itype: cs::cusolverEigType_t,
        jobz: cs::cusolverEigMode_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a: *mut Self,
        lda: i32,
        b: *mut Self,
        ldb: i32,
        w: *mut Self,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// Batched Cholesky `potrfBatched`. cuSOLVER takes an
    /// array-of-pointers to `n × n` matrices; the actor stages
    /// them into a contiguous device buffer of pointers.
    unsafe fn potrf_batched(
        handle: cs::cusolverDnHandle_t,
        uplo: cs::cublasFillMode_t,
        n: i32,
        a_array: *mut *mut Self,
        lda: i32,
        info_array: *mut i32,
        batch_size: i32,
    ) -> cs::cusolverStatus_t;

    /// Batched Jacobi SVD `gesvdjBatched` workspace query + launch.
    unsafe fn gesvdj_batched_buffer_size(
        handle: cs::cusolverDnHandle_t,
        jobz: cs::cusolverEigMode_t,
        m: i32,
        n: i32,
        a: *const Self,
        lda: i32,
        s: *const Self,
        u: *const Self,
        ldu: i32,
        v: *const Self,
        ldv: i32,
        lwork: *mut i32,
        params: cs::gesvdjInfo_t,
        batch_size: i32,
    ) -> cs::cusolverStatus_t;
    unsafe fn gesvdj_batched(
        handle: cs::cusolverDnHandle_t,
        jobz: cs::cusolverEigMode_t,
        m: i32,
        n: i32,
        a: *mut Self,
        lda: i32,
        s: *mut Self,
        u: *mut Self,
        ldu: i32,
        v: *mut Self,
        ldv: i32,
        work: *mut Self,
        lwork: i32,
        info: *mut i32,
        params: cs::gesvdjInfo_t,
        batch_size: i32,
    ) -> cs::cusolverStatus_t;
}

macro_rules! impl_solver_scalar {
    (
        $T:ty,
        geqrf:           $geqrf:ident,        $geqrf_bs:ident;
        getrf:           $getrf:ident,        $getrf_bs:ident,        $getrs:ident;
        potrf:           $potrf:ident,        $potrf_bs:ident;
        gesvd:           $gesvd:ident,        $gesvd_bs:ident;
        syevd:           $syevd:ident,        $syevd_bs:ident;
        sygvd:           $sygvd:ident,        $sygvd_bs:ident;
        potrf_batched:   $potrf_b:ident;
        gesvdj_batched:  $gesvdj_b:ident,     $gesvdj_b_bs:ident;
    ) => {
        impl SolverScalar for $T {
            unsafe fn geqrf_buffer_size(
                handle: cs::cusolverDnHandle_t,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                lwork: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$geqrf_bs(handle, m, n, a, lda, lwork)
            }
            unsafe fn geqrf(
                handle: cs::cusolverDnHandle_t,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                tau: *mut Self,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$geqrf(handle, m, n, a, lda, tau, work, lwork, info)
            }

            unsafe fn getrf_buffer_size(
                handle: cs::cusolverDnHandle_t,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                lwork: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$getrf_bs(handle, m, n, a, lda, lwork)
            }
            unsafe fn getrf(
                handle: cs::cusolverDnHandle_t,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                work: *mut Self,
                ipiv: *mut i32,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$getrf(handle, m, n, a, lda, work, ipiv, info)
            }
            unsafe fn getrs(
                handle: cs::cusolverDnHandle_t,
                trans: cs::cublasOperation_t,
                n: i32,
                nrhs: i32,
                a: *const Self,
                lda: i32,
                ipiv: *const i32,
                b: *mut Self,
                ldb: i32,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$getrs(handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)
            }

            unsafe fn potrf_buffer_size(
                handle: cs::cusolverDnHandle_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a: *mut Self,
                lda: i32,
                lwork: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$potrf_bs(handle, uplo, n, a, lda, lwork)
            }
            unsafe fn potrf(
                handle: cs::cusolverDnHandle_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a: *mut Self,
                lda: i32,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$potrf(handle, uplo, n, a, lda, work, lwork, info)
            }

            unsafe fn gesvd_buffer_size(
                handle: cs::cusolverDnHandle_t,
                m: i32,
                n: i32,
                lwork: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$gesvd_bs(handle, m, n, lwork)
            }
            unsafe fn gesvd(
                handle: cs::cusolverDnHandle_t,
                jobu: i8,
                jobvt: i8,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                s: *mut Self,
                u: *mut Self,
                ldu: i32,
                vt: *mut Self,
                ldvt: i32,
                work: *mut Self,
                lwork: i32,
                rwork: *mut Self,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$gesvd(
                    handle, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork,
                    info,
                )
            }

            unsafe fn syevd_buffer_size(
                handle: cs::cusolverDnHandle_t,
                jobz: cs::cusolverEigMode_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a: *const Self,
                lda: i32,
                w: *const Self,
                lwork: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$syevd_bs(handle, jobz, uplo, n, a, lda, w, lwork)
            }
            unsafe fn syevd(
                handle: cs::cusolverDnHandle_t,
                jobz: cs::cusolverEigMode_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a: *mut Self,
                lda: i32,
                w: *mut Self,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$syevd(handle, jobz, uplo, n, a, lda, w, work, lwork, info)
            }

            unsafe fn sygvd_buffer_size(
                handle: cs::cusolverDnHandle_t,
                itype: cs::cusolverEigType_t,
                jobz: cs::cusolverEigMode_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a: *const Self,
                lda: i32,
                b: *const Self,
                ldb: i32,
                w: *const Self,
                lwork: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$sygvd_bs(handle, itype, jobz, uplo, n, a, lda, b, ldb, w, lwork)
            }
            unsafe fn sygvd(
                handle: cs::cusolverDnHandle_t,
                itype: cs::cusolverEigType_t,
                jobz: cs::cusolverEigMode_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a: *mut Self,
                lda: i32,
                b: *mut Self,
                ldb: i32,
                w: *mut Self,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$sygvd(
                    handle, itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info,
                )
            }

            unsafe fn potrf_batched(
                handle: cs::cusolverDnHandle_t,
                uplo: cs::cublasFillMode_t,
                n: i32,
                a_array: *mut *mut Self,
                lda: i32,
                info_array: *mut i32,
                batch_size: i32,
            ) -> cs::cusolverStatus_t {
                cs::$potrf_b(handle, uplo, n, a_array, lda, info_array, batch_size)
            }

            unsafe fn gesvdj_batched_buffer_size(
                handle: cs::cusolverDnHandle_t,
                jobz: cs::cusolverEigMode_t,
                m: i32,
                n: i32,
                a: *const Self,
                lda: i32,
                s: *const Self,
                u: *const Self,
                ldu: i32,
                v: *const Self,
                ldv: i32,
                lwork: *mut i32,
                params: cs::gesvdjInfo_t,
                batch_size: i32,
            ) -> cs::cusolverStatus_t {
                cs::$gesvdj_b_bs(
                    handle, jobz, m, n, a, lda, s, u, ldu, v, ldv, lwork, params, batch_size,
                )
            }
            unsafe fn gesvdj_batched(
                handle: cs::cusolverDnHandle_t,
                jobz: cs::cusolverEigMode_t,
                m: i32,
                n: i32,
                a: *mut Self,
                lda: i32,
                s: *mut Self,
                u: *mut Self,
                ldu: i32,
                v: *mut Self,
                ldv: i32,
                work: *mut Self,
                lwork: i32,
                info: *mut i32,
                params: cs::gesvdjInfo_t,
                batch_size: i32,
            ) -> cs::cusolverStatus_t {
                cs::$gesvdj_b(
                    handle, jobz, m, n, a, lda, s, u, ldu, v, ldv, work, lwork, info, params,
                    batch_size,
                )
            }
        }
    };
}

impl_solver_scalar!(
    f32,
    geqrf:          cusolverDnSgeqrf,            cusolverDnSgeqrf_bufferSize;
    getrf:          cusolverDnSgetrf,            cusolverDnSgetrf_bufferSize, cusolverDnSgetrs;
    potrf:          cusolverDnSpotrf,            cusolverDnSpotrf_bufferSize;
    gesvd:          cusolverDnSgesvd,            cusolverDnSgesvd_bufferSize;
    syevd:          cusolverDnSsyevd,            cusolverDnSsyevd_bufferSize;
    sygvd:          cusolverDnSsygvd,            cusolverDnSsygvd_bufferSize;
    potrf_batched:  cusolverDnSpotrfBatched;
    gesvdj_batched: cusolverDnSgesvdjBatched,    cusolverDnSgesvdjBatched_bufferSize;
);

impl_solver_scalar!(
    f64,
    geqrf:          cusolverDnDgeqrf,            cusolverDnDgeqrf_bufferSize;
    getrf:          cusolverDnDgetrf,            cusolverDnDgetrf_bufferSize, cusolverDnDgetrs;
    potrf:          cusolverDnDpotrf,            cusolverDnDpotrf_bufferSize;
    gesvd:          cusolverDnDgesvd,            cusolverDnDgesvd_bufferSize;
    syevd:          cusolverDnDsyevd,            cusolverDnDsyevd_bufferSize;
    sygvd:          cusolverDnDsygvd,            cusolverDnDsygvd_bufferSize;
    potrf_batched:  cusolverDnDpotrfBatched;
    gesvdj_batched: cusolverDnDgesvdjBatched,    cusolverDnDgesvdjBatched_bufferSize;
);

/// Sparse cuSOLVER (`cusolverSp`) entry points.
///
/// Only the device-side `csrlsv*` triplet (Cholesky / QR) is exposed
/// here — `csrlsvluHost` is host-only and out of scope. Each method
/// solves `A x = b` in one shot for an `m × m` CSR matrix.
#[cfg(feature = "cusolver-sp")]
pub trait SparseSolverScalar: SolverSupported {
    /// `csrlsvchol`: SPD CSR system via Cholesky.
    unsafe fn csrlsvchol(
        handle: cs::cusolverSpHandle_t,
        m: i32,
        nnz: i32,
        descr_a: cs::cusparseMatDescr_t,
        csr_val: *const Self,
        csr_row_ptr: *const i32,
        csr_col_ind: *const i32,
        b: *const Self,
        tol: f64,
        reorder: i32,
        x: *mut Self,
        singularity: *mut i32,
    ) -> cs::cusolverStatus_t;

    /// `csrlsvqr`: general CSR system via QR.
    unsafe fn csrlsvqr(
        handle: cs::cusolverSpHandle_t,
        m: i32,
        nnz: i32,
        descr_a: cs::cusparseMatDescr_t,
        csr_val: *const Self,
        csr_row_ptr: *const i32,
        csr_col_ind: *const i32,
        b: *const Self,
        tol: f64,
        reorder: i32,
        x: *mut Self,
        singularity: *mut i32,
    ) -> cs::cusolverStatus_t;
}

#[cfg(feature = "cusolver-sp")]
macro_rules! impl_sparse_solver_scalar {
    ($T:ty, $tol:ty, chol: $chol:ident, qr: $qr:ident) => {
        impl SparseSolverScalar for $T {
            unsafe fn csrlsvchol(
                handle: cs::cusolverSpHandle_t,
                m: i32,
                nnz: i32,
                descr_a: cs::cusparseMatDescr_t,
                csr_val: *const Self,
                csr_row_ptr: *const i32,
                csr_col_ind: *const i32,
                b: *const Self,
                tol: f64,
                reorder: i32,
                x: *mut Self,
                singularity: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$chol(
                    handle,
                    m,
                    nnz,
                    descr_a,
                    csr_val,
                    csr_row_ptr,
                    csr_col_ind,
                    b,
                    tol as $tol,
                    reorder,
                    x,
                    singularity,
                )
            }

            unsafe fn csrlsvqr(
                handle: cs::cusolverSpHandle_t,
                m: i32,
                nnz: i32,
                descr_a: cs::cusparseMatDescr_t,
                csr_val: *const Self,
                csr_row_ptr: *const i32,
                csr_col_ind: *const i32,
                b: *const Self,
                tol: f64,
                reorder: i32,
                x: *mut Self,
                singularity: *mut i32,
            ) -> cs::cusolverStatus_t {
                cs::$qr(
                    handle,
                    m,
                    nnz,
                    descr_a,
                    csr_val,
                    csr_row_ptr,
                    csr_col_ind,
                    b,
                    tol as $tol,
                    reorder,
                    x,
                    singularity,
                )
            }
        }
    };
}

#[cfg(feature = "cusolver-sp")]
impl_sparse_solver_scalar!(f32, f32, chol: cusolverSpScsrlsvchol, qr: cusolverSpScsrlsvqr);
#[cfg(feature = "cusolver-sp")]
impl_sparse_solver_scalar!(f64, f64, chol: cusolverSpDcsrlsvchol, qr: cusolverSpDcsrlsvqr);
