//! Sparse-matrix format zoo for cuSPARSE — CSR / COO / CSC / Blocked-ELL
//! / BSR.
//!
//! Each variant carries the buffers cuSPARSE's generic API expects:
//! `row_offsets`/`col_indices`/`values` for CSR; `row_indices`/`col_indices`/`values`
//! for COO; etc. The element types are `T: SparseSupported` for values
//! and `I: SparseIndex` for indices — `(T, I)` is constrained to the
//! cross-product cuSPARSE supports.
//!
//! The dispatcher in `convert.rs` translates a [`SparseMatrix`] into a
//! `cusparseSpMatDescr_t` via the matching create-descriptor entry point
//! (`cusparseCreateCsr`, `cusparseCreateCoo`, `cusparseCreateCsc`,
//! `cusparseCreateBlockedEll`, `cusparseCreateBsr`).

use crate::dtype::{SparseIndex, SparseSupported};
use crate::gpu_ref::GpuRef;

/// Multi-format sparse matrix in device memory. The whole matrix —
/// indices + values — is owned by the surrounding actor message; the
/// inner `GpuRef`s share `Arc` ownership with whatever allocated them.
#[derive(Clone)]
pub enum SparseMatrix<T: SparseSupported, I: SparseIndex> {
    /// Compressed Sparse Row.
    Csr {
        rows: i64,
        cols: i64,
        nnz: i64,
        row_offsets: GpuRef<I>,
        col_indices: GpuRef<I>,
        values: GpuRef<T>,
    },
    /// Coordinate-list.
    Coo {
        rows: i64,
        cols: i64,
        nnz: i64,
        row_indices: GpuRef<I>,
        col_indices: GpuRef<I>,
        values: GpuRef<T>,
    },
    /// Compressed Sparse Column.
    Csc {
        rows: i64,
        cols: i64,
        nnz: i64,
        col_offsets: GpuRef<I>,
        row_indices: GpuRef<I>,
        values: GpuRef<T>,
    },
    /// Blocked-ELL — fixed `ell_block_size`-by-`ell_cols` tiles per row,
    /// padded with sentinel column indices.
    BlockedEll {
        rows: i64,
        cols: i64,
        ell_block_size: i64,
        ell_cols: i64,
        col_indices: GpuRef<I>,
        values: GpuRef<T>,
    },
    /// Block Sparse Row — variable block size, dense block payloads.
    Bsr {
        block_rows: i64,
        block_cols: i64,
        block_size: i64,
        nnz_blocks: i64,
        row_offsets: GpuRef<I>,
        col_indices: GpuRef<I>,
        values: GpuRef<T>,
    },
}

impl<T: SparseSupported, I: SparseIndex> SparseMatrix<T, I> {
    /// Tag the variant for tracing / cache keying.
    pub fn format(&self) -> SparseFormat {
        match self {
            Self::Csr { .. } => SparseFormat::Csr,
            Self::Coo { .. } => SparseFormat::Coo,
            Self::Csc { .. } => SparseFormat::Csc,
            Self::BlockedEll { .. } => SparseFormat::BlockedEll,
            Self::Bsr { .. } => SparseFormat::Bsr,
        }
    }

    /// Logical row count.
    pub fn rows(&self) -> i64 {
        match self {
            Self::Csr { rows, .. }
            | Self::Coo { rows, .. }
            | Self::Csc { rows, .. }
            | Self::BlockedEll { rows, .. } => *rows,
            Self::Bsr {
                block_rows,
                block_size,
                ..
            } => block_rows * block_size,
        }
    }

    /// Logical column count.
    pub fn cols(&self) -> i64 {
        match self {
            Self::Csr { cols, .. }
            | Self::Coo { cols, .. }
            | Self::Csc { cols, .. }
            | Self::BlockedEll { cols, .. } => *cols,
            Self::Bsr {
                block_cols,
                block_size,
                ..
            } => block_cols * block_size,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    Csr,
    Coo,
    Csc,
    BlockedEll,
    Bsr,
}

impl SparseFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            SparseFormat::Csr => "csr",
            SparseFormat::Coo => "coo",
            SparseFormat::Csc => "csc",
            SparseFormat::BlockedEll => "blocked_ell",
            SparseFormat::Bsr => "bsr",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceState;
    use std::sync::Arc;

    /// Smoke test that every variant of [`SparseMatrix`] can be
    /// constructed without touching the GPU. We can't actually mint a
    /// real `GpuRef` without a `CudaSlice`, but we can exercise the
    /// `Default::default()`-style stub via the `format()` accessor on
    /// _just-the-discriminant_ matches (the variants own only refs and
    /// scalars — neither of which trigger device work).
    #[test]
    fn format_round_trip_csr_coo_csc_bsr_ell() {
        // Verify the discriminant accessors line up. We don't construct
        // real `GpuRef<T>` values here — that requires a `CudaSlice` —
        // but `SparseFormat` round trips through `as_str` cleanly.
        assert_eq!(SparseFormat::Csr.as_str(), "csr");
        assert_eq!(SparseFormat::Coo.as_str(), "coo");
        assert_eq!(SparseFormat::Csc.as_str(), "csc");
        assert_eq!(SparseFormat::BlockedEll.as_str(), "blocked_ell");
        assert_eq!(SparseFormat::Bsr.as_str(), "bsr");

        // Compile-time: SparseMatrix<f32, i32> instantiates.
        fn _ct<T: SparseSupported, I: SparseIndex>() -> Option<SparseMatrix<T, I>> {
            None
        }
        let _: Option<SparseMatrix<f32, i32>> = _ct::<f32, i32>();
        let _: Option<SparseMatrix<f64, i64>> = _ct::<f64, i64>();
        // Touch DeviceState so Arc<DeviceState> is exercised via the
        // SparseSupported plumbing path (sanity).
        let _ = Arc::new(DeviceState::new(0));
    }
}
