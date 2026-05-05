//! Mixed-precision compute descriptors used by every cuTENSOR op.
//!
//! cuTENSOR's `cutensorComputeDescriptor_t` is opaque; the library
//! exports a fixed set of named globals (`CUTENSOR_R_MIN_32F`, …).
//! We expose them as a Rust enum and resolve to the underlying
//! pointer at the call site.

use cudarc::cutensor::sys as ct_sys;

use crate::sys::cutensor as ct_local;

/// Selects the cuTENSOR mixed-precision compute path. `MinF32` is
/// the natural pairing for f16/bf16 inputs that accumulate in f32.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ComputeDesc {
    /// `CUTENSOR_R_32F` — full fp32, no min-precision.
    F32,
    /// `CUTENSOR_R_64F` — full fp64.
    F64,
    /// `CUTENSOR_R_MIN_32F` — fp32 accumulation with min-precision
    /// kernels. Default for f32 / f16 / bf16 inputs.
    MinF32,
    /// `CUTENSOR_R_MIN_64F` — fp64 accumulation, min-precision.
    MinF64,
    /// `CUTENSOR_R_MIN_16F` — pure half-precision compute.
    MinF16,
    /// `CUTENSOR_R_MIN_16BF` — pure bf16 compute.
    MinBf16,
    /// `CUTENSOR_R_MIN_TF32` — TF32 accumulation (Ampere+).
    Tf32,
    /// `CUTENSOR_C_32F` — complex fp32 inputs/compute.
    C32F,
}

impl ComputeDesc {
    pub fn tag(self) -> &'static str {
        match self {
            ComputeDesc::F32 => "F32",
            ComputeDesc::F64 => "F64",
            ComputeDesc::MinF32 => "MinF32",
            ComputeDesc::MinF64 => "MinF64",
            ComputeDesc::MinF16 => "MinF16",
            ComputeDesc::MinBf16 => "MinBf16",
            ComputeDesc::Tf32 => "Tf32",
            ComputeDesc::C32F => "C32F",
        }
    }
}

/// Stable u32 fingerprint of a `ComputeDesc` for plan-cache keys.
/// Different from `tag()` so two compute descs are guaranteed to
/// hash distinctly without relying on string hashing.
pub fn compute_desc_tag(c: ComputeDesc) -> u32 {
    match c {
        ComputeDesc::F32 => 0x01,
        ComputeDesc::F64 => 0x02,
        ComputeDesc::MinF32 => 0x04,
        ComputeDesc::MinF64 => 0x08,
        ComputeDesc::MinF16 => 0x10,
        ComputeDesc::MinBf16 => 0x20,
        ComputeDesc::Tf32 => 0x40,
        ComputeDesc::C32F => 0x80,
    }
}

/// Resolve a [`ComputeDesc`] to the corresponding extern global
/// pointer cuTENSOR expects.
pub fn resolve_compute_desc(c: ComputeDesc) -> ct_sys::cutensorComputeDescriptor_t {
    match c {
        ComputeDesc::F32 => ct_local::r_32f(),
        ComputeDesc::F64 => ct_local::r_64f(),
        ComputeDesc::MinF32 => ct_local::r_min_32f(),
        ComputeDesc::MinF64 => ct_local::r_min_64f(),
        ComputeDesc::MinF16 => ct_local::r_min_16f(),
        ComputeDesc::MinBf16 => ct_local::r_min_16bf(),
        ComputeDesc::Tf32 => ct_local::r_min_tf32(),
        ComputeDesc::C32F => ct_local::c_32f(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_desc_tags_are_unique() {
        let descs = [
            ComputeDesc::F32,
            ComputeDesc::F64,
            ComputeDesc::MinF32,
            ComputeDesc::MinF64,
            ComputeDesc::MinF16,
            ComputeDesc::MinBf16,
            ComputeDesc::Tf32,
            ComputeDesc::C32F,
        ];
        let tags: Vec<u32> = descs.iter().copied().map(compute_desc_tag).collect();
        let mut sorted = tags.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), tags.len(), "tags must all be distinct");
    }

    #[test]
    fn compute_desc_tag_strs() {
        assert_eq!(ComputeDesc::F32.tag(), "F32");
        assert_eq!(ComputeDesc::MinF32.tag(), "MinF32");
        assert_eq!(ComputeDesc::Tf32.tag(), "Tf32");
    }
}
