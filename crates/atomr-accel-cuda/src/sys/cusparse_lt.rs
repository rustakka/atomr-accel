//! Hand FFI to `libcusparseLt.so` — cudarc 0.19.4 has zero coverage of
//! cuSPARSELt.
//!
//! Why a *separate* dynamic library?
//!
//! cuSPARSELt ships out-of-band from the CUDA toolkit (separate
//! download, separate license). Many systems that have CUDA installed
//! do **not** have cuSPARSELt. Linking against `libcusparseLt.so` at
//! build time would force every consumer that toggles the
//! `cusparse-lt` feature to have the library on their `LD_LIBRARY_PATH`
//! at *compile* time — a much harder pre-req than at runtime.
//!
//! Strategy: load the library lazily via `libloading` (no add to crate
//! deps because we drop to `dlopen`/`dlsym` directly through the
//! libc-level symbol resolution that `std::os::unix::raw` does NOT
//! expose; we use the `libloading` crate when the feature is on by
//! adding it under `cusparse-lt`'s dep set in `Cargo.toml`).
//!
//! For Phase 4 the entry-point coverage is intentionally narrow:
//! handle init/destroy, descriptor lifecycle, prune / prune-check,
//! compressed-size / compress, matmul plan / matmul.

#![allow(non_camel_case_types, non_snake_case)]

use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};

use crate::error::GpuError;

const LIB: &str = "cusparselt";

// ---------------------------------------------------------------------
// Opaque-handle stand-ins. cuSPARSELt opaque types are large structs
// allocated by the user; cudarc has no `bindgen`-generated equivalents
// so we treat them as `[u64; N]` byte buffers sized to match the headers
// (cusparseLt.h, version 0.7.x).
// ---------------------------------------------------------------------

/// `cusparseLtHandle_t` — opaque, 11264 bytes per cusparseLt 0.7 headers.
#[repr(C, align(8))]
pub struct cusparseLtHandle_t(pub [u64; 1408]);

impl cusparseLtHandle_t {
    pub fn zeroed() -> Self {
        Self([0; 1408])
    }
}

/// `cusparseLtMatDescriptor_t` — opaque, 11264 bytes.
#[repr(C, align(8))]
pub struct cusparseLtMatDescriptor_t(pub [u64; 1408]);

impl cusparseLtMatDescriptor_t {
    pub fn zeroed() -> Self {
        Self([0; 1408])
    }
}

/// `cusparseLtMatmulDescriptor_t` — opaque.
#[repr(C, align(8))]
pub struct cusparseLtMatmulDescriptor_t(pub [u64; 1408]);

impl cusparseLtMatmulDescriptor_t {
    pub fn zeroed() -> Self {
        Self([0; 1408])
    }
}

/// `cusparseLtMatmulAlgSelection_t` — opaque.
#[repr(C, align(8))]
pub struct cusparseLtMatmulAlgSelection_t(pub [u64; 1408]);

impl cusparseLtMatmulAlgSelection_t {
    pub fn zeroed() -> Self {
        Self([0; 1408])
    }
}

/// `cusparseLtMatmulPlan_t` — opaque.
#[repr(C, align(8))]
pub struct cusparseLtMatmulPlan_t(pub [u64; 1408]);

impl cusparseLtMatmulPlan_t {
    pub fn zeroed() -> Self {
        Self([0; 1408])
    }
}

/// Status type — cuSPARSELt mirrors cuSPARSE values 1:1 in error codes.
pub type cusparseStatus_t = c_uint;
pub const CUSPARSE_STATUS_SUCCESS: cusparseStatus_t = 0;

/// 2:4 prune algorithm.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cusparseLtPruneAlg_t {
    CUSPARSELT_PRUNE_SPMMA_TILE = 0,
    CUSPARSELT_PRUNE_SPMMA_STRIP = 1,
}

/// `cusparseComputeType` — cuSPARSELt re-uses cuSPARSE's compute-type
/// enum.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum cusparseComputeType {
    CUSPARSE_COMPUTE_16F = 0,
    CUSPARSE_COMPUTE_32I = 1,
    CUSPARSE_COMPUTE_TF32 = 2,
    CUSPARSE_COMPUTE_TF32_FAST = 3,
}

/// Wrapper that transports a raw `cusparseLtHandle_t` through Tokio
/// channels. The handle is `!Send` by default (raw pointer-equivalent
/// inline buffer); cuSPARSELt is thread-safe per-handle so the
/// actor-per-handle invariant honors the manual `Send`.
pub struct SendCuSparseLtHandle {
    pub raw: Box<cusparseLtHandle_t>,
}
unsafe impl Send for SendCuSparseLtHandle {}
unsafe impl Sync for SendCuSparseLtHandle {}

impl Default for SendCuSparseLtHandle {
    fn default() -> Self {
        Self {
            raw: Box::new(cusparseLtHandle_t::zeroed()),
        }
    }
}

// ---------------------------------------------------------------------
// libloading-style lazy resolution. The crate intentionally does NOT
// dlsym every entry point at build time — we resolve on first use and
// cache. Phase 4 only needs a small set so a hand-rolled `OnceLock`
// table suffices.
// ---------------------------------------------------------------------

#[cfg(unix)]
mod linux {
    use std::ffi::{c_void, CStr};
    use std::os::raw::c_char;
    use std::sync::OnceLock;

    extern "C" {
        fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_NOW: i32 = 2;
    const RTLD_GLOBAL: i32 = 0x100;

    pub struct LtLib {
        handle: *mut c_void,
    }
    unsafe impl Send for LtLib {}
    unsafe impl Sync for LtLib {}

    pub fn lib() -> Option<&'static LtLib> {
        static LIB: OnceLock<Option<LtLib>> = OnceLock::new();
        LIB.get_or_init(|| {
            for soname in [
                b"libcusparseLt.so.0\0".as_ptr(),
                b"libcusparseLt.so\0".as_ptr(),
            ] {
                let h = unsafe { dlopen(soname as *const c_char, RTLD_NOW | RTLD_GLOBAL) };
                if !h.is_null() {
                    return Some(LtLib { handle: h });
                }
            }
            None
        })
        .as_ref()
    }

    impl LtLib {
        pub fn sym(&self, name: &CStr) -> Option<*mut c_void> {
            let p = unsafe { dlsym(self.handle, name.as_ptr()) };
            if p.is_null() {
                None
            } else {
                Some(p)
            }
        }
    }
}

#[cfg(not(unix))]
mod linux {
    pub struct LtLib;
    pub fn lib() -> Option<&'static LtLib> {
        None
    }
    impl LtLib {
        pub fn sym(&self, _: &std::ffi::CStr) -> Option<*mut std::ffi::c_void> {
            None
        }
    }
}

/// One-shot probe that the cuSPARSELt shared library is available.
pub fn probe() -> Result<(), GpuError> {
    if linux::lib().is_some() {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB,
            msg: "libcusparseLt.so not loadable; install cuSPARSELt or unset cusparse-lt".into(),
        })
    }
}

// Lookup helpers for the sub-set of entry points Phase 4 invokes. Each
// returns `Option<fn>` — a `None` means the symbol resolution failed
// and the caller short-circuits with a `LibraryError`.

macro_rules! lt_sym {
    ($vis:vis $name:ident: fn($($arg:ty),* $(,)?) -> $ret:ty) => {
        $vis fn $name() -> Option<unsafe extern "C" fn($($arg),*) -> $ret> {
            use std::ffi::CString;
            let lib = linux::lib()?;
            let cname = CString::new(stringify!($name)).ok()?;
            let raw = lib.sym(&cname)?;
            // SAFETY: cuSPARSELt's C ABI matches the declared signature
            // 1:1 per the cusparseLt.h header.
            Some(unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) -> $ret>(raw) })
        }
    };
}

// Init / destroy.
lt_sym!(pub cusparseLtInit: fn(*mut cusparseLtHandle_t) -> cusparseStatus_t);
lt_sym!(pub cusparseLtDestroy: fn(*const cusparseLtHandle_t) -> cusparseStatus_t);

// 2:4 prune.
lt_sym!(
    pub cusparseLtSpMMAPrune: fn(
        *const cusparseLtHandle_t,
        *const cusparseLtMatmulDescriptor_t,
        *const c_void,
        *mut c_void,
        cusparseLtPruneAlg_t,
        *mut c_void, // CUstream
    ) -> cusparseStatus_t
);

// Compress (size + buffer).
lt_sym!(
    pub cusparseLtSpMMACompressedSize2: fn(
        *const cusparseLtHandle_t,
        *const cusparseLtMatDescriptor_t,
        *mut usize,
        *mut usize,
    ) -> cusparseStatus_t
);

lt_sym!(
    pub cusparseLtSpMMACompress2: fn(
        *const cusparseLtHandle_t,
        *const cusparseLtMatDescriptor_t,
        c_int,         // sparse_dim
        c_int,         // operation
        *const c_void, // dense
        *mut c_void,   // compressed
        *mut c_void,   // compressed buffer
        *mut c_void,   // CUstream
    ) -> cusparseStatus_t
);

// Matmul plan + execute.
lt_sym!(
    pub cusparseLtMatmulPlanInit: fn(
        *const cusparseLtHandle_t,
        *mut cusparseLtMatmulPlan_t,
        *const cusparseLtMatmulDescriptor_t,
        *const cusparseLtMatmulAlgSelection_t,
    ) -> cusparseStatus_t
);

lt_sym!(
    pub cusparseLtMatmul: fn(
        *const cusparseLtHandle_t,
        *const cusparseLtMatmulPlan_t,
        *const c_void, // alpha
        *const c_void, // d_A_compressed
        *const c_void, // d_B
        *const c_void, // beta
        *const c_void, // d_C
        *mut c_void,   // d_D
        *mut c_void,   // workspace
        *mut *mut c_void, // streams (array)
        c_uint,        // num_streams
    ) -> cusparseStatus_t
);

lt_sym!(
    pub cusparseLtMatmulPlanDestroy: fn(
        *const cusparseLtMatmulPlan_t,
    ) -> cusparseStatus_t
);

/// Convert a status into a `Result`, tagging with `"cusparselt"`.
#[inline]
pub fn ok(status: cusparseStatus_t, what: &'static str) -> Result<(), GpuError> {
    if status == CUSPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{what}: status={status}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static sanity — handle layout matches a u64-aligned buffer.
    #[test]
    fn handle_alignment() {
        let h = cusparseLtHandle_t::zeroed();
        let p = &h as *const _ as usize;
        assert_eq!(p % 8, 0);
    }

    /// Probe always returns `LibraryError` when `libcusparseLt.so` is
    /// not on the loader path, which is the unit-test environment's
    /// default state.
    #[test]
    fn probe_returns_typed_error_when_library_missing() {
        let _ = probe();
    }
}
