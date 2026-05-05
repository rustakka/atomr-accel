//! Host-API generator path.
//!
//! cuRAND exposes two parallel generator constructors:
//!
//! * `curandCreateGenerator` (device) — fills *device* buffers; what
//!   [`super::RngActor`] uses by default.
//! * `curandCreateGeneratorHost` (host) — same RNG types, but
//!   `curandGenerate*` writes to *host*-resident memory and the call
//!   blocks until completion.
//!
//! The host path is useful for:
//!
//! * generating small reproducible streams for tests / examples
//!   without round-tripping through a `GpuRef` allocation;
//! * pre-staging deterministic seeds the device-API actor can later
//!   copy in.
//!
//! Gated under the `curand-host` feature so the unsafe FFI doesn't
//! land in the default build surface.

#![cfg(feature = "curand-host")]

use cudarc::curand::result::CurandError;
use cudarc::curand::sys;

use crate::sys::curand as csys;
use crate::sys::curand::RngGeneratorKind;

/// Owned host-API cuRAND generator. Drops the underlying handle on
/// destruction.
pub struct HostRng {
    gen: sys::curandGenerator_t,
    kind: RngGeneratorKind,
}

// SAFETY: a host-API generator has no implicit stream affinity; its
// only mutating operation is the `curandGenerate*` family which we
// thread through `&mut self`. The raw pointer is opaque to anything
// outside cuRAND so cross-thread move is sound provided the user
// upholds the cuRAND host-API contract (single concurrent caller).
unsafe impl Send for HostRng {}

impl HostRng {
    /// Build a host-API generator and (for pseudo families) seed it.
    pub fn new(kind: RngGeneratorKind, seed: u64) -> Result<Self, CurandError> {
        let gen = unsafe { csys::create_generator_host(kind)? };
        if !kind.is_quasi() {
            unsafe { csys::set_seed(gen, seed)? };
        }
        Ok(Self { gen, kind })
    }

    pub fn kind(&self) -> RngGeneratorKind {
        self.kind
    }

    /// Re-seed (no-op for quasi).
    pub fn set_seed(&mut self, seed: u64) -> Result<(), CurandError> {
        if self.kind.is_quasi() {
            return Ok(());
        }
        unsafe { csys::set_seed(self.gen, seed) }
    }

    /// Fill `out` with `out.len()` u32 values from the raw bit
    /// generator (`curandGenerate`).
    pub fn fill_u32(&mut self, out: &mut [u32]) -> Result<(), CurandError> {
        let n = out.len();
        unsafe { csys::generate_u32(self.gen, out.as_mut_ptr(), n) }
    }

    /// Fill `out` with `out.len()` u64 values
    /// (`curandGenerateLongLong`).
    pub fn fill_u64(&mut self, out: &mut [u64]) -> Result<(), CurandError> {
        let n = out.len();
        unsafe { csys::generate_u64(self.gen, out.as_mut_ptr(), n) }
    }

    /// Fill `out` with `out.len()` f32 uniform samples
    /// (`curandGenerateUniform`).
    pub fn fill_uniform_f32(&mut self, out: &mut [f32]) -> Result<(), CurandError> {
        let n = out.len();
        unsafe { sys::curandGenerateUniform(self.gen, out.as_mut_ptr(), n).result() }
    }

    /// Fill `out` with `out.len()` f64 uniform samples
    /// (`curandGenerateUniformDouble`).
    pub fn fill_uniform_f64(&mut self, out: &mut [f64]) -> Result<(), CurandError> {
        let n = out.len();
        unsafe { sys::curandGenerateUniformDouble(self.gen, out.as_mut_ptr(), n).result() }
    }

    /// Fill `out` with `out.len()` Normal(mean, std) f32 samples.
    pub fn fill_normal_f32(
        &mut self,
        out: &mut [f32],
        mean: f32,
        std: f32,
    ) -> Result<(), CurandError> {
        let n = out.len();
        unsafe { sys::curandGenerateNormal(self.gen, out.as_mut_ptr(), n, mean, std).result() }
    }

    /// Fill `out` with `out.len()` Normal(mean, std) f64 samples.
    pub fn fill_normal_f64(
        &mut self,
        out: &mut [f64],
        mean: f64,
        std: f64,
    ) -> Result<(), CurandError> {
        let n = out.len();
        unsafe {
            sys::curandGenerateNormalDouble(self.gen, out.as_mut_ptr(), n, mean, std).result()
        }
    }

    /// Fill `out` with `out.len()` Poisson(lambda) u32 samples.
    pub fn fill_poisson_u32(&mut self, out: &mut [u32], lambda: f64) -> Result<(), CurandError> {
        let n = out.len();
        unsafe { csys::generate_poisson_u32(self.gen, out.as_mut_ptr(), n, lambda) }
    }
}

impl Drop for HostRng {
    fn drop(&mut self) {
        if !self.gen.is_null() {
            let _ = unsafe { csys::destroy_generator(self.gen) };
            self.gen = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static-only test — locks in the `HostRng::new` constructor
    /// signature. We can't actually call `new` on a host without
    /// cuRAND symbols loaded, but referring to the function pointer
    /// is enough to verify the public API and force monomorphization.
    #[test]
    fn host_api_generator_constructs() {
        let f: fn(RngGeneratorKind, u64) -> Result<HostRng, CurandError> = HostRng::new;
        let _ = f;
    }
}
