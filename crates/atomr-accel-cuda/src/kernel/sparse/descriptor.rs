//! Generic-API descriptor lifecycle for cuSPARSE.
//!
//! `SpMatDescr`, `DnVecDescr`, `DnMatDescr` are the three primary
//! descriptor types. Each is created once per op (cuSPARSE doesn't
//! cache them) and destroyed via the matching `cusparseDestroy*`. RAII
//! guards live in `crate::sys::cusparse` — this module only adds
//! type-aware constructors that pull `cudaDataType_t` from `T: CudaDtype`
//! and `cusparseIndexType_t` from `I: SparseIndex`.

use cudarc::cusparse::sys as cs;

use crate::dtype::{CudaDtype, SparseIndex, SparseSupported};
use crate::error::GpuError;
use crate::sys::cusparse as sys_cs;

/// Build a CSR `cusparseSpMatDescr_t` from raw device pointers.
///
/// SAFETY: pointers must be valid device pointers for at least the
/// lifetime of the returned guard. The caller owns the underlying
/// allocations (typically as `GpuRef`s tracked by the actor).
#[allow(clippy::too_many_arguments)]
pub unsafe fn create_csr<T: SparseSupported, I: SparseIndex>(
    rows: i64,
    cols: i64,
    nnz: i64,
    row_offsets: *mut std::ffi::c_void,
    col_indices: *mut std::ffi::c_void,
    values: *mut std::ffi::c_void,
) -> Result<sys_cs::SpMatGuard, GpuError> {
    let mut descr: cs::cusparseSpMatDescr_t = std::ptr::null_mut();
    let s = cs::cusparseCreateCsr(
        &mut descr as *mut _,
        rows,
        cols,
        nnz,
        row_offsets,
        col_indices,
        values,
        I::cusparse_index_type(),
        I::cusparse_index_type(),
        cs::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        unsafe { std::mem::transmute::<u32, cs::cudaDataType_t>(T::cuda_data_type() as u32) },
    );
    sys_cs::ok(s, "cusparseCreateCsr")?;
    Ok(sys_cs::SpMatGuard(descr))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn create_coo<T: SparseSupported, I: SparseIndex>(
    rows: i64,
    cols: i64,
    nnz: i64,
    row_indices: *mut std::ffi::c_void,
    col_indices: *mut std::ffi::c_void,
    values: *mut std::ffi::c_void,
) -> Result<sys_cs::SpMatGuard, GpuError> {
    let mut descr: cs::cusparseSpMatDescr_t = std::ptr::null_mut();
    let s = cs::cusparseCreateCoo(
        &mut descr as *mut _,
        rows,
        cols,
        nnz,
        row_indices,
        col_indices,
        values,
        I::cusparse_index_type(),
        cs::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        unsafe { std::mem::transmute::<u32, cs::cudaDataType_t>(T::cuda_data_type() as u32) },
    );
    sys_cs::ok(s, "cusparseCreateCoo")?;
    Ok(sys_cs::SpMatGuard(descr))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn create_csc<T: SparseSupported, I: SparseIndex>(
    rows: i64,
    cols: i64,
    nnz: i64,
    col_offsets: *mut std::ffi::c_void,
    row_indices: *mut std::ffi::c_void,
    values: *mut std::ffi::c_void,
) -> Result<sys_cs::SpMatGuard, GpuError> {
    let mut descr: cs::cusparseSpMatDescr_t = std::ptr::null_mut();
    let s = cs::cusparseCreateCsc(
        &mut descr as *mut _,
        rows,
        cols,
        nnz,
        col_offsets,
        row_indices,
        values,
        I::cusparse_index_type(),
        I::cusparse_index_type(),
        cs::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        unsafe { std::mem::transmute::<u32, cs::cudaDataType_t>(T::cuda_data_type() as u32) },
    );
    sys_cs::ok(s, "cusparseCreateCsc")?;
    Ok(sys_cs::SpMatGuard(descr))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn create_blocked_ell<T: SparseSupported, I: SparseIndex>(
    rows: i64,
    cols: i64,
    ell_block_size: i64,
    ell_cols: i64,
    col_indices: *mut std::ffi::c_void,
    values: *mut std::ffi::c_void,
) -> Result<sys_cs::SpMatGuard, GpuError> {
    let mut descr: cs::cusparseSpMatDescr_t = std::ptr::null_mut();
    let s = cs::cusparseCreateBlockedEll(
        &mut descr as *mut _,
        rows,
        cols,
        ell_block_size,
        ell_cols,
        col_indices,
        values,
        I::cusparse_index_type(),
        cs::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        unsafe { std::mem::transmute::<u32, cs::cudaDataType_t>(T::cuda_data_type() as u32) },
    );
    sys_cs::ok(s, "cusparseCreateBlockedEll")?;
    Ok(sys_cs::SpMatGuard(descr))
}

/// Build a dense vector descriptor.
pub unsafe fn create_dn_vec<T: CudaDtype>(
    size: i64,
    values: *mut std::ffi::c_void,
) -> Result<sys_cs::DnVecGuard, GpuError> {
    let mut descr: cs::cusparseDnVecDescr_t = std::ptr::null_mut();
    let s = cs::cusparseCreateDnVec(&mut descr as *mut _, size, values, unsafe { std::mem::transmute::<u32, cs::cudaDataType_t>(T::cuda_data_type() as u32) });
    sys_cs::ok(s, "cusparseCreateDnVec")?;
    Ok(sys_cs::DnVecGuard(descr))
}

/// Build a dense matrix descriptor.
pub unsafe fn create_dn_mat<T: CudaDtype>(
    rows: i64,
    cols: i64,
    ld: i64,
    values: *mut std::ffi::c_void,
    order: cs::cusparseOrder_t,
) -> Result<sys_cs::DnMatGuard, GpuError> {
    let mut descr: cs::cusparseDnMatDescr_t = std::ptr::null_mut();
    let s = cs::cusparseCreateDnMat(
        &mut descr as *mut _,
        rows,
        cols,
        ld,
        values,
        unsafe { std::mem::transmute::<u32, cs::cudaDataType_t>(T::cuda_data_type() as u32) },
        order,
    );
    sys_cs::ok(s, "cusparseCreateDnMat")?;
    Ok(sys_cs::DnMatGuard(descr))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the descriptor-builder signatures compile across all four
    /// supported value dtypes paired with both index dtypes — this is
    /// the only way without a GPU to assert that
    /// `T: SparseSupported, I: SparseIndex` resolves cleanly.
    #[test]
    fn descriptor_signatures_compile() {
        fn _ct<T: SparseSupported, I: SparseIndex>() {}
        _ct::<f32, i32>();
        _ct::<f64, i32>();
        _ct::<f32, i64>();
        _ct::<f64, i64>();
        #[cfg(feature = "f16")]
        {
            _ct::<half::f16, i32>();
            _ct::<half::bf16, i32>();
            _ct::<half::f16, i64>();
            _ct::<half::bf16, i64>();
        }
    }
}
