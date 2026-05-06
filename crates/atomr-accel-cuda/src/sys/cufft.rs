//! Local FFI bindings for cuFFT entry points cudarc 0.19.4 doesn't
//! expose.
//!
//! Specifically: `cufftXtSetCallback` / `cufftXtClearCallback`. These
//! aren't part of cudarc's `cufft::sys` surface, so we resolve them
//! ourselves via `libloading` against `libcufft.so` (Linux),
//! `cufft64_*.dll` (Windows), or `libcufft.dylib` (macOS).
//!
//! Why not `extern "C"`? cudarc's default `fallback-dynamic-loading`
//! build does **not** emit a `-lcufft` link directive — symbols are
//! resolved at runtime through `dlopen`. Adding a hard `extern "C"`
//! reference here would break the link on hosts without libcufft on
//! the link path (which is the whole point of `fallback-dynamic-loading`).
//! Mirroring cudarc's strategy keeps us link-clean.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use core::ffi::{c_int, c_void};
use std::sync::OnceLock;

use cudarc::cufft::sys::{cufftHandle, cufftResult, cufftResult_t};

/// `cufftXtCallbackType` from `cufftXt.h`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CufftXtCallbackType {
    /// Load callback for f32 complex (cufftComplex) input.
    LoadComplex = 0x0,
    /// Load callback for f64 complex (cufftDoubleComplex) input.
    LoadComplexDouble = 0x1,
    /// Load callback for f32 real (cufftReal) input.
    LoadReal = 0x2,
    /// Load callback for f64 real (cufftDoubleReal) input.
    LoadRealDouble = 0x3,
    /// Store callback for f32 complex output.
    StoreComplex = 0x4,
    /// Store callback for f64 complex output.
    StoreComplexDouble = 0x5,
    /// Store callback for f32 real output.
    StoreReal = 0x6,
    /// Store callback for f64 real output.
    StoreRealDouble = 0x7,
}

type CufftXtSetCallbackFn = unsafe extern "C" fn(
    plan: cufftHandle,
    callback_routine: *mut *mut c_void,
    cb_type: i32,
    caller_info: *mut *mut c_void,
) -> cufftResult;

type CufftXtClearCallbackFn = unsafe extern "C" fn(plan: cufftHandle, cb_type: i32) -> cufftResult;

struct XtSyms {
    set_cb: Option<libloading::Symbol<'static, CufftXtSetCallbackFn>>,
    clear_cb: Option<libloading::Symbol<'static, CufftXtClearCallbackFn>>,
    _lib: libloading::Library,
}

// SAFETY: `libloading::Symbol` borrows from a `Library`, but we leak
// the library (via `OnceLock`) so the borrow is effectively `'static`.
// The function pointers themselves are thread-safe to invoke.
unsafe impl Send for XtSyms {}
unsafe impl Sync for XtSyms {}

static XT_SYMS: OnceLock<Result<XtSyms, String>> = OnceLock::new();

#[cfg(target_os = "linux")]
const CUFFT_LIB_CANDIDATES: &[&str] = &["libcufft.so", "libcufft.so.11", "libcufft.so.10"];

#[cfg(target_os = "macos")]
const CUFFT_LIB_CANDIDATES: &[&str] = &["libcufft.dylib"];

#[cfg(target_os = "windows")]
const CUFFT_LIB_CANDIDATES: &[&str] = &["cufft64_11.dll", "cufft64_10.dll", "cufft64_9.dll"];

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
const CUFFT_LIB_CANDIDATES: &[&str] = &[];

fn load_xt_syms() -> Result<XtSyms, String> {
    let mut last_err: Option<String> = None;
    for cand in CUFFT_LIB_CANDIDATES {
        match unsafe { libloading::Library::new(*cand) } {
            Ok(lib) => {
                // `Symbol` borrows from `lib`. We move `lib` into the
                // `XtSyms` struct and transmute the `'_` borrow to
                // `'static` via the leaked `OnceLock` — sound only
                // because `XT_SYMS` is initialized exactly once and
                // never dropped before program exit.
                let set_cb = unsafe {
                    lib.get::<CufftXtSetCallbackFn>(b"cufftXtSetCallback\0")
                        .ok()
                        .map(|s| {
                            std::mem::transmute::<
                                libloading::Symbol<'_, CufftXtSetCallbackFn>,
                                libloading::Symbol<'static, CufftXtSetCallbackFn>,
                            >(s)
                        })
                };
                let clear_cb = unsafe {
                    lib.get::<CufftXtClearCallbackFn>(b"cufftXtClearCallback\0")
                        .ok()
                        .map(|s| {
                            std::mem::transmute::<
                                libloading::Symbol<'_, CufftXtClearCallbackFn>,
                                libloading::Symbol<'static, CufftXtClearCallbackFn>,
                            >(s)
                        })
                };
                return Ok(XtSyms {
                    set_cb,
                    clear_cb,
                    _lib: lib,
                });
            }
            Err(e) => {
                last_err = Some(format!("{cand}: {e}"));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "no libcufft candidates configured".into()))
}

fn xt_syms() -> Result<&'static XtSyms, &'static str> {
    let cell = XT_SYMS.get_or_init(load_xt_syms);
    match cell {
        Ok(s) => Ok(s),
        Err(_) => Err("cuFFT shared library not loadable on this host"),
    }
}

/// Result that doubles as a transport for "library missing" cases on
/// hosts where libcufft can't be dlopened (the no-GPU CI runner).
fn fail_not_supported() -> cufftResult {
    cufftResult_t::CUFFT_NOT_SUPPORTED
}

/// Install a load/store callback on a cuFFT plan.
///
/// `cb` must point to a CUDA *device* function with the signature
/// matching `cb_type` (see the cuFFT docs). `caller_info` is passed
/// straight through to the device callback at call time.
///
/// Returns `CUFFT_NOT_SUPPORTED` if `libcufft` couldn't be opened or
/// doesn't export `cufftXtSetCallback` (older toolkits before the Xt
/// API was carved out). All other return codes are passed through.
///
/// # Safety
/// - `plan` must be a live cuFFT handle.
/// - `cb` must point to a device-resident function with the
///   appropriate signature.
/// - `caller_info` must outlive every kernel launch on `plan`.
pub unsafe fn xt_set_callback(
    plan: cufftHandle,
    cb: *mut c_void,
    cb_type: CufftXtCallbackType,
    caller_info: *mut c_void,
) -> cufftResult {
    let syms = match xt_syms() {
        Ok(s) => s,
        Err(_) => return fail_not_supported(),
    };
    let f = match &syms.set_cb {
        Some(f) => f,
        None => return fail_not_supported(),
    };
    let mut routine = cb;
    let mut info = caller_info;
    f(plan, &mut routine, cb_type as c_int, &mut info)
}

/// Clear a previously-installed callback.
///
/// # Safety
/// `plan` must be a live cuFFT handle.
pub unsafe fn xt_clear_callback(plan: cufftHandle, cb_type: CufftXtCallbackType) -> cufftResult {
    let syms = match xt_syms() {
        Ok(s) => s,
        Err(_) => return fail_not_supported(),
    };
    let f = match &syms.clear_cb {
        Some(f) => f,
        None => return fail_not_supported(),
    };
    f(plan, cb_type as c_int)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_kinds_are_distinct() {
        assert_ne!(
            CufftXtCallbackType::LoadComplex as i32,
            CufftXtCallbackType::StoreComplex as i32
        );
        assert_ne!(
            CufftXtCallbackType::LoadReal as i32,
            CufftXtCallbackType::StoreReal as i32
        );
        // f32 vs f64 lanes are distinct.
        assert_ne!(
            CufftXtCallbackType::LoadComplex as i32,
            CufftXtCallbackType::LoadComplexDouble as i32
        );
    }

    #[test]
    fn xt_set_callback_is_safe_to_call_without_gpu() {
        // On a no-GPU host the dlopen will fail; the wrapper must
        // surface NOT_SUPPORTED rather than panicking.
        let result = unsafe {
            xt_set_callback(
                0,
                std::ptr::null_mut(),
                CufftXtCallbackType::LoadComplex,
                std::ptr::null_mut(),
            )
        };
        // Exact code depends on the host; we just assert it doesn't
        // panic and returns *some* cufftResult.
        let _ = result;
    }
}
