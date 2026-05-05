//! Sobol / Scrambled-Sobol quasi-random support.
//!
//! Quasi-random generators in cuRAND share the same `curandGenerate*`
//! surface as the pseudo-random families but require:
//!
//! 1. Construction via `curandCreateGenerator` with one of
//!    `CURAND_RNG_QUASI_*` rng-types (mapped from
//!    [`super::RngGeneratorKind::Sobol32`] et al);
//! 2. A call to `curandSetQuasiRandomGeneratorDimensions` *before* the
//!    first generate call;
//! 3. **No pseudo-seed** — `curandSetPseudoRandomGeneratorSeed` would
//!    return `CURAND_STATUS_TYPE_ERROR`.
//!
//! [`super::RngActor`] handles (1) and (3) automatically when the
//! caller picks a Sobol kind via `RngMsg::SetGenerator`. (2) is
//! exposed here as a small helper so callers can change the
//! dimension count without dropping to raw `cudarc::curand::sys`.

#![cfg(feature = "curand-quasirandom")]

use crate::error::GpuError;
use crate::sys::curand as csys;
use crate::sys::curand::RngGeneratorKind;

use super::LIB;

/// Configure the dimension of an active quasi-random generator. Must
/// be called between `RngMsg::SetGenerator { kind: Sobol* }` and the
/// first `RngMsg::Fill` against that generator.
///
/// # Safety
/// The supplied `gen` must be the *current* generator owned by an
/// `RngActor` *and* of a quasi family — passing a pseudo handle here
/// will return `CURAND_STATUS_TYPE_ERROR`. In practice callers go
/// through [`RngActor::set_quasi_dimensions`] (see
/// [`super::RngActor`]).
pub unsafe fn set_dimensions(
    gen: cudarc::curand::sys::curandGenerator_t,
    dimensions: u32,
) -> Result<(), GpuError> {
    csys::set_quasi_random_dimensions(gen, dimensions).map_err(|e| GpuError::LibraryError {
        lib: LIB,
        msg: format!("set_quasi_random_dimensions({dimensions}): {e}"),
    })
}

impl super::RngActor {
    /// Configure the active quasi-random generator's dimension count.
    /// Returns an error if the current generator isn't Sobol-flavoured.
    pub fn set_quasi_dimensions(&self, dimensions: u32) -> Result<(), GpuError> {
        let (gen_lock, kind_lock) = match &self.inner {
            super::RngInner::Real { gen, kind, .. } => (gen, kind),
            super::RngInner::Mock => {
                return Err(GpuError::Unrecoverable("RngActor in mock mode".into()))
            }
        };
        let active = *kind_lock.lock();
        if !active.is_quasi() {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!(
                    "set_quasi_dimensions called on non-quasi generator ({active:?}); \
                     SetGenerator {{ kind: Sobol* }} first"
                ),
            });
        }
        let g = gen_lock.lock();
        unsafe { set_dimensions(g.0, dimensions) }
    }
}

/// Helper: which dimension widths the four Sobol variants accept.
/// cuRAND 32-bit Sobol supports up to 20000 dimensions; 64-bit Sobol
/// supports up to ~21201. Used by docs and validation; not enforced
/// here because the cuRAND error path is cheaper than a Rust-side
/// table.
pub fn max_dimensions(kind: RngGeneratorKind) -> Option<u32> {
    match kind {
        RngGeneratorKind::Sobol32 | RngGeneratorKind::ScrambledSobol32 => Some(20_000),
        RngGeneratorKind::Sobol64 | RngGeneratorKind::ScrambledSobol64 => Some(20_000),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobol32_sobol64_kind_correct() {
        // Mapping check: Sobol32 → 201, ScrambledSobol32 → 202,
        // Sobol64 → 203, ScrambledSobol64 → 204. Numeric IDs come
        // straight from the cuRAND header.
        assert_eq!(RngGeneratorKind::Sobol32.to_sys() as u32, 201);
        assert_eq!(RngGeneratorKind::ScrambledSobol32.to_sys() as u32, 202);
        assert_eq!(RngGeneratorKind::Sobol64.to_sys() as u32, 203);
        assert_eq!(RngGeneratorKind::ScrambledSobol64.to_sys() as u32, 204);

        for k in [
            RngGeneratorKind::Sobol32,
            RngGeneratorKind::ScrambledSobol32,
            RngGeneratorKind::Sobol64,
            RngGeneratorKind::ScrambledSobol64,
        ] {
            assert!(k.is_quasi());
            assert_eq!(max_dimensions(k), Some(20_000));
        }
        for k in [
            RngGeneratorKind::PseudoDefault,
            RngGeneratorKind::Philox4_32_10,
            RngGeneratorKind::XorWow,
            RngGeneratorKind::Mrg32K3A,
            RngGeneratorKind::Mtgp32,
        ] {
            assert!(!k.is_quasi());
            assert_eq!(max_dimensions(k), None);
        }
    }
}
