//! Local sys-level wrappers around `cudarc::cutensor::sys` for the
//! cuTENSOR entry points the safe `cudarc::cutensor::result` module
//! does not expose (Reduce/ElementwiseBinary/ElementwiseTrinary
//! create+execute, Permutation create+execute, predefined compute
//! descriptors).
//!
//! Every function here is `unsafe` and takes the raw cuTENSOR enum
//! types from `cudarc::cutensor::sys`. The entire crate's actor layer
//! drives these through `kernel/tensor/`.
//!
//! All wrappers convert `cutensorStatus_t` into a thin
//! [`CutensorError`] that mirrors what `cudarc::cutensor::result`
//! emits; callers that already use `cudarc::cutensor::result` can
//! interleave both freely.

use core::ffi::c_void;
use core::mem::MaybeUninit;

use cudarc::cutensor::sys as ct_sys;

/// Error wrapper around a `cutensorStatus_t`. Mirrors
/// `cudarc::cutensor::result::CutensorError` so error messages are
/// consistent across the safe/sys boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutensorError(pub ct_sys::cutensorStatus_t);

impl std::fmt::Display for CutensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cutensor status: {:?}", self.0)
    }
}

impl std::error::Error for CutensorError {}

#[inline]
fn check(status: ct_sys::cutensorStatus_t) -> Result<(), CutensorError> {
    match status {
        ct_sys::cutensorStatus_t::CUTENSOR_STATUS_SUCCESS => Ok(()),
        e => Err(CutensorError(e)),
    }
}

// ---------------------------------------------------------------------
// Predefined compute descriptors.
//
// libcutensor exports a handful of `cutensorComputeDescriptor_t` global
// constants (`CUTENSOR_R_MIN_32F`, `CUTENSOR_R_MIN_64F`, etc.). cudarc
// 0.19 doesn't surface them. Linking them as `extern "C" { static ... }`
// requires libcutensor.so at link time, which fails on no-GPU hosts —
// instead, we resolve them at first use through `libloading`. The
// returned pointer is cached in a `OnceLock` per symbol.
// ---------------------------------------------------------------------

use std::sync::OnceLock;

struct ComputeDescriptors {
    r_min_32f: ct_sys::cutensorComputeDescriptor_t,
    r_min_64f: ct_sys::cutensorComputeDescriptor_t,
    r_min_16f: ct_sys::cutensorComputeDescriptor_t,
    r_min_16bf: ct_sys::cutensorComputeDescriptor_t,
    r_min_tf32: ct_sys::cutensorComputeDescriptor_t,
    r_32f: ct_sys::cutensorComputeDescriptor_t,
    r_64f: ct_sys::cutensorComputeDescriptor_t,
    c_32f: ct_sys::cutensorComputeDescriptor_t,
}

unsafe impl Send for ComputeDescriptors {}
unsafe impl Sync for ComputeDescriptors {}

static DESCRIPTORS: OnceLock<ComputeDescriptors> = OnceLock::new();

fn load_descriptors() -> ComputeDescriptors {
    // libcutensor candidates. Mirrors cudarc's lookup but kept local so
    // we don't reach into cudarc's private `culib()`.
    let candidates = [
        "libcutensor.so.2",
        "libcutensor.so.1",
        "libcutensor.so",
        "cutensor.dll",
    ];
    for cand in candidates.iter() {
        let lib = unsafe { libloading::Library::new(*cand) };
        let Ok(lib) = lib else { continue };
        let read = |name: &[u8]| -> Option<ct_sys::cutensorComputeDescriptor_t> {
            unsafe {
                // The exported symbol is the variable itself; libloading
                // returns a `Symbol<T>` whose `Deref` yields `T`. We
                // ask for `T = *const cutensorComputeDescriptor_t` so
                // `*s` is the variable's address, and one further deref
                // (`**s`) reads the descriptor pointer value out of it.
                let s: libloading::Symbol<*const ct_sys::cutensorComputeDescriptor_t> =
                    lib.get(name).ok()?;
                Some(**s)
            }
        };
        let r_min_32f = read(b"CUTENSOR_R_MIN_32F\0");
        let r_min_64f = read(b"CUTENSOR_R_MIN_64F\0");
        let r_min_16f = read(b"CUTENSOR_R_MIN_16F\0");
        let r_min_16bf = read(b"CUTENSOR_R_MIN_16BF\0");
        let r_min_tf32 = read(b"CUTENSOR_R_MIN_TF32\0");
        let r_32f = read(b"CUTENSOR_R_32F\0");
        let r_64f = read(b"CUTENSOR_R_64F\0");
        let c_32f = read(b"CUTENSOR_C_32F\0");
        // Require at least the f32/f64 min descriptors — every supported
        // dtype routes through one of those by default.
        if let (Some(a), Some(b)) = (r_min_32f, r_min_64f) {
            // Forget the library so its destructor doesn't unload while
            // we're still holding pointers into it.
            std::mem::forget(lib);
            return ComputeDescriptors {
                r_min_32f: a,
                r_min_64f: b,
                r_min_16f: r_min_16f.unwrap_or(a),
                r_min_16bf: r_min_16bf.unwrap_or(a),
                r_min_tf32: r_min_tf32.unwrap_or(a),
                r_32f: r_32f.unwrap_or(a),
                r_64f: r_64f.unwrap_or(b),
                c_32f: c_32f.unwrap_or(a),
            };
        }
    }
    panic!(
        "ContextPoisoned: failed to dlopen libcutensor.so / locate \
         CUTENSOR_R_MIN_32F (compute descriptor symbol). cuTENSOR \
         must be installed on the host for cutensor-feature builds."
    );
}

#[inline]
fn descriptors() -> &'static ComputeDescriptors {
    DESCRIPTORS.get_or_init(load_descriptors)
}

pub fn r_min_32f() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_min_32f
}
pub fn r_min_64f() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_min_64f
}
pub fn r_min_16f() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_min_16f
}
pub fn r_min_16bf() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_min_16bf
}
pub fn r_min_tf32() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_min_tf32
}
pub fn r_32f() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_32f
}
pub fn r_64f() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().r_64f
}
pub fn c_32f() -> ct_sys::cutensorComputeDescriptor_t {
    descriptors().c_32f
}

// ---------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------

/// Wraps `cutensorReduce` (post-plan execution).
///
/// # Safety
/// All pointers must be valid. `workspace` must hold at least
/// `workspace_size` bytes. `stream` must outlive the call.
pub unsafe fn reduce(
    handle: ct_sys::cutensorHandle_t,
    plan: ct_sys::cutensorPlan_t,
    alpha: *const c_void,
    a: *const c_void,
    beta: *const c_void,
    c: *const c_void,
    d: *mut c_void,
    workspace: *mut c_void,
    workspace_size: u64,
    stream: ct_sys::cudaStream_t,
) -> Result<(), CutensorError> {
    check(ct_sys::cutensorReduce(
        handle,
        plan,
        alpha,
        a,
        beta,
        c,
        d,
        workspace,
        workspace_size,
        stream,
    ))
}

// ---------------------------------------------------------------------
// Elementwise binary
// ---------------------------------------------------------------------

/// Create a binary elementwise operation descriptor. Wraps
/// `cutensorCreateElementwiseBinary`.
///
/// # Safety
/// All handles/descriptors must be valid; mode arrays must align with
/// each tensor descriptor's rank.
pub unsafe fn create_elementwise_binary(
    handle: ct_sys::cutensorHandle_t,
    desc_a: ct_sys::cutensorTensorDescriptor_t,
    mode_a: *const i32,
    op_a: ct_sys::cutensorOperator_t,
    desc_c: ct_sys::cutensorTensorDescriptor_t,
    mode_c: *const i32,
    op_c: ct_sys::cutensorOperator_t,
    desc_d: ct_sys::cutensorTensorDescriptor_t,
    mode_d: *const i32,
    op_ac: ct_sys::cutensorOperator_t,
    desc_compute: ct_sys::cutensorComputeDescriptor_t,
) -> Result<ct_sys::cutensorOperationDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    check(ct_sys::cutensorCreateElementwiseBinary(
        handle,
        desc.as_mut_ptr(),
        desc_a,
        mode_a,
        op_a,
        desc_c,
        mode_c,
        op_c,
        desc_d,
        mode_d,
        op_ac,
        desc_compute,
    ))?;
    Ok(desc.assume_init())
}

/// Execute a previously-planned binary elementwise op.
///
/// # Safety
/// `handle`, `plan`, all data pointers, and `stream` must be valid.
pub unsafe fn elementwise_binary_execute(
    handle: ct_sys::cutensorHandle_t,
    plan: ct_sys::cutensorPlan_t,
    alpha: *const c_void,
    a: *const c_void,
    gamma: *const c_void,
    c: *const c_void,
    d: *mut c_void,
    stream: ct_sys::cudaStream_t,
) -> Result<(), CutensorError> {
    check(ct_sys::cutensorElementwiseBinaryExecute(
        handle, plan, alpha, a, gamma, c, d, stream,
    ))
}

// ---------------------------------------------------------------------
// Elementwise trinary
// ---------------------------------------------------------------------

/// Create a trinary elementwise operation descriptor. Wraps
/// `cutensorCreateElementwiseTrinary`.
///
/// # Safety
/// As [`create_elementwise_binary`].
pub unsafe fn create_elementwise_trinary(
    handle: ct_sys::cutensorHandle_t,
    desc_a: ct_sys::cutensorTensorDescriptor_t,
    mode_a: *const i32,
    op_a: ct_sys::cutensorOperator_t,
    desc_b: ct_sys::cutensorTensorDescriptor_t,
    mode_b: *const i32,
    op_b: ct_sys::cutensorOperator_t,
    desc_c: ct_sys::cutensorTensorDescriptor_t,
    mode_c: *const i32,
    op_c: ct_sys::cutensorOperator_t,
    desc_d: ct_sys::cutensorTensorDescriptor_t,
    mode_d: *const i32,
    op_ab: ct_sys::cutensorOperator_t,
    op_abc: ct_sys::cutensorOperator_t,
    desc_compute: ct_sys::cutensorComputeDescriptor_t,
) -> Result<ct_sys::cutensorOperationDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    check(ct_sys::cutensorCreateElementwiseTrinary(
        handle,
        desc.as_mut_ptr(),
        desc_a,
        mode_a,
        op_a,
        desc_b,
        mode_b,
        op_b,
        desc_c,
        mode_c,
        op_c,
        desc_d,
        mode_d,
        op_ab,
        op_abc,
        desc_compute,
    ))?;
    Ok(desc.assume_init())
}

/// Execute a previously-planned trinary elementwise op.
///
/// # Safety
/// As [`elementwise_binary_execute`].
pub unsafe fn elementwise_trinary_execute(
    handle: ct_sys::cutensorHandle_t,
    plan: ct_sys::cutensorPlan_t,
    alpha: *const c_void,
    a: *const c_void,
    beta: *const c_void,
    b: *const c_void,
    gamma: *const c_void,
    c: *const c_void,
    d: *mut c_void,
    stream: ct_sys::cudaStream_t,
) -> Result<(), CutensorError> {
    check(ct_sys::cutensorElementwiseTrinaryExecute(
        handle, plan, alpha, a, beta, b, gamma, c, d, stream,
    ))
}

// ---------------------------------------------------------------------
// Permutation
// ---------------------------------------------------------------------

/// Create a permutation operation descriptor. Wraps
/// `cutensorCreatePermutation`.
///
/// # Safety
/// As [`create_elementwise_binary`].
pub unsafe fn create_permutation(
    handle: ct_sys::cutensorHandle_t,
    desc_a: ct_sys::cutensorTensorDescriptor_t,
    mode_a: *const i32,
    op_a: ct_sys::cutensorOperator_t,
    desc_b: ct_sys::cutensorTensorDescriptor_t,
    mode_b: *const i32,
    desc_compute: ct_sys::cutensorComputeDescriptor_t,
) -> Result<ct_sys::cutensorOperationDescriptor_t, CutensorError> {
    let mut desc = MaybeUninit::uninit();
    check(ct_sys::cutensorCreatePermutation(
        handle,
        desc.as_mut_ptr(),
        desc_a,
        mode_a,
        op_a,
        desc_b,
        mode_b,
        desc_compute,
    ))?;
    Ok(desc.assume_init())
}

/// Execute a previously-planned permutation.
///
/// # Safety
/// As [`elementwise_binary_execute`].
pub unsafe fn permute(
    handle: ct_sys::cutensorHandle_t,
    plan: ct_sys::cutensorPlan_t,
    alpha: *const c_void,
    a: *const c_void,
    b: *mut c_void,
    stream: ct_sys::cudaStream_t,
) -> Result<(), CutensorError> {
    check(ct_sys::cutensorPermute(handle, plan, alpha, a, b, stream))
}

// ---------------------------------------------------------------------
// Plan-preference attribute setters used by autotune.
// ---------------------------------------------------------------------

/// Set the pinned algorithm on a plan-preference object. Used by the
/// contraction autotune to probe a specific `cutensorAlgo_t` value.
///
/// # Safety
/// `handle` and `pref` must be valid.
pub unsafe fn plan_preference_set_algo(
    handle: ct_sys::cutensorHandle_t,
    pref: ct_sys::cutensorPlanPreference_t,
    algo: ct_sys::cutensorAlgo_t,
) -> Result<(), CutensorError> {
    let value = algo as i32;
    check(ct_sys::cutensorPlanPreferenceSetAttribute(
        handle,
        pref,
        ct_sys::cutensorPlanPreferenceAttribute_t::CUTENSOR_PLAN_PREFERENCE_ALGO,
        &value as *const i32 as *const c_void,
        std::mem::size_of::<i32>(),
    ))
}
