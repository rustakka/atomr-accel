//! Thin safe-ish wrappers over `cudarc::cusparse::sys` entry points
//! Phase 4 needs but the safe layer doesn't provide.
//!
//! Every wrapper:
//! 1. Translates a `cusparseStatus_t` into a [`crate::error::GpuError::LibraryError`]
//!    tagged `"cusparse"`.
//! 2. Funnels `cusparseDestroy*` calls into a `Drop` so descriptors do
//!    not leak when an error short-circuits the pipeline.
//!
//! Higher-level orchestration (descriptor caches, workspace pooling)
//! lives in `crate::kernel::sparse`.

use cudarc::cusparse::sys as cs;

use crate::error::GpuError;

const LIB: &str = "cusparse";

/// Convert a cuSPARSE status into a `Result<(), GpuError>`.
#[inline]
pub fn ok(status: cs::cusparseStatus_t, what: &'static str) -> Result<(), GpuError> {
    if status == cs::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{what}: {status:?}"),
        })
    }
}

/// RAII guard for a `cusparseSpMatDescr_t`.
pub struct SpMatGuard(pub cs::cusparseSpMatDescr_t);
unsafe impl Send for SpMatGuard {}
impl Drop for SpMatGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cusparseDestroySpMat(self.0);
        }
    }
}

/// RAII guard for a `cusparseDnVecDescr_t`.
pub struct DnVecGuard(pub cs::cusparseDnVecDescr_t);
unsafe impl Send for DnVecGuard {}
impl Drop for DnVecGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cusparseDestroyDnVec(self.0);
        }
    }
}

/// RAII guard for a `cusparseDnMatDescr_t`.
pub struct DnMatGuard(pub cs::cusparseDnMatDescr_t);
unsafe impl Send for DnMatGuard {}
impl Drop for DnMatGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cusparseDestroyDnMat(self.0);
        }
    }
}

/// RAII guard for a `cusparseSpGEMMDescr_t`.
pub struct SpGemmDescGuard(pub cs::cusparseSpGEMMDescr_t);
unsafe impl Send for SpGemmDescGuard {}
impl Drop for SpGemmDescGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cusparseSpGEMM_destroyDescr(self.0);
        }
    }
}

/// RAII guard for a `cusparseSpSVDescr_t`.
pub struct SpSvDescGuard(pub cs::cusparseSpSVDescr_t);
unsafe impl Send for SpSvDescGuard {}
impl Drop for SpSvDescGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = cs::cusparseSpSV_destroyDescr(self.0);
        }
    }
}

/// Algorithm tag to use for `cusparseSpMV`. Public so `kernel/sparse/spmv.rs`
/// can plumb it through the request struct in a future phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpMvAlg {
    Default,
    Csr,
    Coo,
}

impl SpMvAlg {
    pub fn raw(self) -> cs::cusparseSpMVAlg_t {
        match self {
            SpMvAlg::Default => cs::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
            SpMvAlg::Csr => cs::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1,
            SpMvAlg::Coo => cs::cusparseSpMVAlg_t::CUSPARSE_SPMV_COO_ALG1,
        }
    }
}

/// Algorithm tag for `cusparseSpMM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpMmAlg {
    Default,
    Csr,
    BlockedEll,
}

impl SpMmAlg {
    pub fn raw(self) -> cs::cusparseSpMMAlg_t {
        match self {
            SpMmAlg::Default => cs::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
            SpMmAlg::Csr => cs::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG2,
            SpMmAlg::BlockedEll => cs::cusparseSpMMAlg_t::CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
        }
    }
}

/// Algorithm tag for `cusparseSpGEMM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpGemmAlg {
    Default,
}

impl SpGemmAlg {
    pub fn raw(self) -> cs::cusparseSpGEMMAlg_t {
        match self {
            SpGemmAlg::Default => cs::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT,
        }
    }
}

/// Algorithm tag for `cusparseSDDMM`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SddmmAlg {
    Default,
}

impl SddmmAlg {
    pub fn raw(self) -> cs::cusparseSDDMMAlg_t {
        match self {
            SddmmAlg::Default => cs::cusparseSDDMMAlg_t::CUSPARSE_SDDMM_ALG_DEFAULT,
        }
    }
}

/// Algorithm tag for `cusparseSpSV`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpSvAlg {
    Default,
}

impl SpSvAlg {
    pub fn raw(self) -> cs::cusparseSpSVAlg_t {
        match self {
            SpSvAlg::Default => cs::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        }
    }
}
