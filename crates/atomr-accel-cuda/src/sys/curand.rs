//! Thin Rust-level wrappers over [`cudarc::curand::sys`] for the
//! cuRAND host-API entry points that aren't surfaced by the safe
//! [`cudarc::curand::CudaRng`] handle:
//!
//! * generator creation by **explicit** `curandRngType_t`
//!   (`Philox4_32_10`, `XORWOW`, `MTGP32`, `MRG32K3A`, plus all four
//!   Sobol variants);
//! * the **host-API** generator pair (`curandCreateGeneratorHost` —
//!   fills *host* buffers, copies internally);
//! * `curandSetQuasiRandomGeneratorDimensions`,
//!   `curandSetGeneratorOrdering`, `curandSetGeneratorOffset`;
//! * the Poisson and bit-generator families
//!   (`curandGeneratePoisson`, `curandGenerate`,
//!   `curandGenerateLongLong`).
//!
//! These functions are **unsafe** — they take the raw
//! `curandGenerator_t` and require the caller to keep the generator
//! alive, valid, and bound to a stream that the destination pointer
//! lives on. The actor in `kernel/rng/*` is the only intended caller.

use std::mem::MaybeUninit;

use cudarc::curand::result::CurandError;
use cudarc::curand::sys;

/// Public mirror of [`sys::curandRngType_t`] so callers don't have to
/// take a `cudarc::curand::sys::*` symbol on their public API. The
/// numeric values match cuRAND.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum RngGeneratorKind {
    /// `CURAND_RNG_PSEUDO_DEFAULT` — XORWOW today; defined by cuRAND.
    #[default]
    PseudoDefault,
    /// `CURAND_RNG_PSEUDO_PHILOX4_32_10`. Recommended high-quality
    /// pseudo-RNG. Counter-based, friendly to SIMD/SIMT.
    Philox4_32_10,
    /// `CURAND_RNG_PSEUDO_XORWOW`. Default in cuRAND <= 11.
    XorWow,
    /// `CURAND_RNG_PSEUDO_MRG32K3A`. L'Ecuyer's MRG.
    Mrg32K3A,
    /// `CURAND_RNG_PSEUDO_MTGP32`. Mersenne Twister 32-bit.
    Mtgp32,
    /// `CURAND_RNG_QUASI_SOBOL32`. Quasi-random 32-bit Sobol.
    Sobol32,
    /// `CURAND_RNG_QUASI_SCRAMBLED_SOBOL32`.
    ScrambledSobol32,
    /// `CURAND_RNG_QUASI_SOBOL64`.
    Sobol64,
    /// `CURAND_RNG_QUASI_SCRAMBLED_SOBOL64`.
    ScrambledSobol64,
}

impl RngGeneratorKind {
    /// Whether this kind is a quasi-random (Sobol) generator. Quasi
    /// generators must be configured with
    /// [`set_quasi_random_dimensions`] before any fill, and they do
    /// not accept a pseudo-random seed.
    pub fn is_quasi(self) -> bool {
        matches!(
            self,
            Self::Sobol32 | Self::ScrambledSobol32 | Self::Sobol64 | Self::ScrambledSobol64
        )
    }

    /// 64-bit Sobol vs. 32-bit Sobol. Kept separate from [`Self::is_quasi`]
    /// so callers can pick the matching `curandSetQuasiRandomGeneratorDimensions`
    /// argument width without re-matching.
    pub fn is_quasi_64(self) -> bool {
        matches!(self, Self::Sobol64 | Self::ScrambledSobol64)
    }

    pub fn to_sys(self) -> sys::curandRngType_t {
        match self {
            Self::PseudoDefault => sys::curandRngType_t::CURAND_RNG_PSEUDO_DEFAULT,
            Self::Philox4_32_10 => sys::curandRngType_t::CURAND_RNG_PSEUDO_PHILOX4_32_10,
            Self::XorWow => sys::curandRngType_t::CURAND_RNG_PSEUDO_XORWOW,
            Self::Mrg32K3A => sys::curandRngType_t::CURAND_RNG_PSEUDO_MRG32K3A,
            Self::Mtgp32 => sys::curandRngType_t::CURAND_RNG_PSEUDO_MTGP32,
            Self::Sobol32 => sys::curandRngType_t::CURAND_RNG_QUASI_SOBOL32,
            Self::ScrambledSobol32 => sys::curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
            Self::Sobol64 => sys::curandRngType_t::CURAND_RNG_QUASI_SOBOL64,
            Self::ScrambledSobol64 => sys::curandRngType_t::CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
        }
    }
}

/// Create a device-side generator of the given `kind`
/// (`curandCreateGenerator`).
///
/// # Safety
/// The returned handle must be released with
/// [`destroy_generator`] before drop, and a stream must be bound via
/// [`set_stream`] before any fill is enqueued.
pub unsafe fn create_generator(
    kind: RngGeneratorKind,
) -> Result<sys::curandGenerator_t, CurandError> {
    let mut g = MaybeUninit::uninit();
    sys::curandCreateGenerator(g.as_mut_ptr(), kind.to_sys()).result()?;
    Ok(g.assume_init())
}

/// Create a host-API generator of the given `kind`
/// (`curandCreateGeneratorHost`). The handle's `curandGenerate*`
/// targets must be **host-resident** memory.
///
/// # Safety
/// Same lifecycle rules as [`create_generator`]; in addition the
/// destination pointers passed to subsequent generate calls must be
/// host pointers, *not* device pointers.
#[cfg(feature = "curand-host")]
pub unsafe fn create_generator_host(
    kind: RngGeneratorKind,
) -> Result<sys::curandGenerator_t, CurandError> {
    let mut g = MaybeUninit::uninit();
    sys::curandCreateGeneratorHost(g.as_mut_ptr(), kind.to_sys()).result()?;
    Ok(g.assume_init())
}

/// Bind `gen` to `stream` (`curandSetStream`).
///
/// # Safety
/// `gen` must be live; `stream` must be the raw cuStream pointer from
/// a `cudarc::driver::CudaStream` whose context owns `gen`.
pub unsafe fn set_stream(
    gen: sys::curandGenerator_t,
    stream: sys::cudaStream_t,
) -> Result<(), CurandError> {
    sys::curandSetStream(gen, stream).result()
}

/// `curandSetPseudoRandomGeneratorSeed`.
///
/// # Safety
/// `gen` must be live and pseudo-random.
pub unsafe fn set_seed(gen: sys::curandGenerator_t, seed: u64) -> Result<(), CurandError> {
    sys::curandSetPseudoRandomGeneratorSeed(gen, seed).result()
}

/// `curandSetGeneratorOffset`.
///
/// # Safety
/// `gen` must be live.
pub unsafe fn set_offset(gen: sys::curandGenerator_t, offset: u64) -> Result<(), CurandError> {
    sys::curandSetGeneratorOffset(gen, offset).result()
}

/// `curandSetQuasiRandomGeneratorDimensions`.
///
/// # Safety
/// `gen` must be a live quasi-random generator. `dimensions` must be
/// >= 1. Length-allocated buffer per dimension is implementation-defined
/// (usually 20000 for Sobol32, fewer for Sobol64).
#[cfg(feature = "curand-quasirandom")]
pub unsafe fn set_quasi_random_dimensions(
    gen: sys::curandGenerator_t,
    dimensions: u32,
) -> Result<(), CurandError> {
    sys::curandSetQuasiRandomGeneratorDimensions(gen, dimensions).result()
}

/// `curandDestroyGenerator`.
///
/// # Safety
/// `gen` must not have been destroyed already. After this call the
/// pointer is dangling.
pub unsafe fn destroy_generator(gen: sys::curandGenerator_t) -> Result<(), CurandError> {
    sys::curandDestroyGenerator(gen).result()
}

/// `curandGeneratePoisson` — fill `out` with `n` u32 values drawn
/// from a Poisson distribution parameterised by `lambda` (f64).
///
/// # Safety
/// `gen` must be live and bound to the same stream/context as the
/// device pointer `out`. `out` must point to at least `n` u32 slots.
pub unsafe fn generate_poisson_u32(
    gen: sys::curandGenerator_t,
    out: *mut u32,
    n: usize,
    lambda: f64,
) -> Result<(), CurandError> {
    sys::curandGeneratePoisson(gen, out, n, lambda).result()
}

/// `curandGenerate` — raw u32 bit fill (used as the building block
/// for any custom transform).
///
/// # Safety
/// Same invariants as [`generate_poisson_u32`].
pub unsafe fn generate_u32(
    gen: sys::curandGenerator_t,
    out: *mut u32,
    n: usize,
) -> Result<(), CurandError> {
    sys::curandGenerate(gen, out, n).result()
}

/// `curandGenerateLongLong` — raw u64 bit fill.
///
/// # Safety
/// `gen` must be a 64-bit quasi-random or pseudo-random generator
/// supporting long-long output.
pub unsafe fn generate_u64(
    gen: sys::curandGenerator_t,
    out: *mut u64,
    n: usize,
) -> Result<(), CurandError> {
    sys::curandGenerateLongLong(gen, out as *mut std::os::raw::c_ulonglong, n).result()
}
