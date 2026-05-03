//! NVRTC kernel sources for the realtime actors.
//!
//! Each constant is the verbatim CUDA-C source for one of the four
//! realtime actors that historically shipped a CPU reference. The
//! sources live under `crates/rakka-accel-cuda-realtime/kernels/` and are
//! `include_str!`-bundled here so that downstream code (e.g. an
//! actor's `with_nvrtc(...)` constructor) can hand them to a
//! [`rakka_accel_cuda::kernel::NvrtcMsg::Compile`] without filesystem
//! access at runtime.
//!
//! These are exposed unconditionally — the `nvrtc` cargo feature only
//! gates the actors' constructor variants that depend on
//! `rakka_accel_cuda::kernel::NvrtcActor`. The sources themselves are pure
//! data and harmless to ship in every build.

/// COO-format sparse matrix-vector multiply. Mirrors the CPU
/// reference in [`crate::sparse::GpuSparseStructureActor::handle`].
/// Kernel name: `coo_spmv`.
pub const COO_SPMV_SRC: &str = include_str!("../kernels/coo_spmv.cu");

/// Velocity-Verlet particle integration with optional drag and
/// bounding-box reflection. Mirrors
/// [`crate::particle::ParticleSystemActor::step`]. Kernel name:
/// `particle_step`.
pub const PARTICLE_STEP_SRC: &str = include_str!("../kernels/particle_step.cu");

/// Verlet cloth integration + structural-spring constraint pass.
/// Exposes two kernels: `cloth_verlet` and `cloth_constrain_pass`.
/// Mirrors [`crate::cloth::ClothSimulationActor::step`].
pub const CLOTH_SPRINGS_SRC: &str = include_str!("../kernels/cloth_springs.cu");

/// Open-addressing hashmap probe. Exposes two kernels:
/// `hashmap_lookup` and `hashmap_insert`. Mirrors the CPU table in
/// [`crate::hashmap::GpuHashMapActor`].
pub const HASHMAP_PROBE_SRC: &str = include_str!("../kernels/hashmap_probe.cu");

#[cfg(test)]
mod tests {
    use super::*;

    /// Lightweight check that each `include_str!` reached its file
    /// and the source contains the expected `extern "C"` entry-point
    /// markers. Catches accidental rename / move regressions without
    /// requiring NVRTC to be available.
    #[test]
    fn kernel_sources_have_entry_points() {
        assert!(COO_SPMV_SRC.contains("coo_spmv"));
        assert!(PARTICLE_STEP_SRC.contains("particle_step"));
        assert!(CLOTH_SPRINGS_SRC.contains("cloth_verlet"));
        assert!(CLOTH_SPRINGS_SRC.contains("cloth_constrain_pass"));
        assert!(HASHMAP_PROBE_SRC.contains("hashmap_lookup"));
        assert!(HASHMAP_PROBE_SRC.contains("hashmap_insert"));
    }
}
