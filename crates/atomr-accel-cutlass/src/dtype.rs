//! Local dtype enum used by CUTLASS template messages.
//!
//! `atomr-accel-cuda` is expected to expose a richer `CudaDtype` (with
//! 16 capability markers and fp8 / fp4 wrappers); on this branch the
//! cutlass crate ships a minimal mirror that covers the surface needed
//! for template instantiation. Once the upstream `CudaDtype` lands, the
//! re-export here can be replaced with a `pub use
//! atomr_accel_cuda::dtype::CudaDtype as CutlassDtype` alias without
//! changing the public API of this crate.

use core::fmt;

/// CUTLASS-side dtype tag. Each variant maps 1-to-1 onto a concrete
/// CUTLASS C++ scalar type via [`CutlassDtype::as_cutlass_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutlassDtype {
    /// `float` / `cutlass::float32_t`.
    F32,
    /// `double` / `cutlass::float64_t`.
    F64,
    /// `cutlass::half_t`.
    F16,
    /// `cutlass::bfloat16_t`.
    Bf16,
    /// `cutlass::float_e4m3_t` (Hopper / Blackwell fp8).
    F8E4m3,
    /// `cutlass::float_e5m2_t` (Hopper / Blackwell fp8).
    F8E5m2,
    /// `cutlass::float_e2m1_t` (Blackwell fp4).
    F4E2m1,
    /// `int8_t` (CUTLASS quantized GEMM lane).
    I8,
    /// `int32_t` (accumulator).
    I32,
    /// `uint8_t`.
    U8,
}

impl CutlassDtype {
    /// CUTLASS C++ type spelling for use inside a generated template
    /// instantiation. Used by the `.cu` source emitter.
    pub fn as_cutlass_type(self) -> &'static str {
        match self {
            CutlassDtype::F32 => "float",
            CutlassDtype::F64 => "double",
            CutlassDtype::F16 => "cutlass::half_t",
            CutlassDtype::Bf16 => "cutlass::bfloat16_t",
            CutlassDtype::F8E4m3 => "cutlass::float_e4m3_t",
            CutlassDtype::F8E5m2 => "cutlass::float_e5m2_t",
            CutlassDtype::F4E2m1 => "cutlass::float_e2m1_t",
            CutlassDtype::I8 => "int8_t",
            CutlassDtype::I32 => "int32_t",
            CutlassDtype::U8 => "uint8_t",
        }
    }

    /// Stable short name used in plan-cache keys and log output.
    pub fn short_name(self) -> &'static str {
        match self {
            CutlassDtype::F32 => "f32",
            CutlassDtype::F64 => "f64",
            CutlassDtype::F16 => "f16",
            CutlassDtype::Bf16 => "bf16",
            CutlassDtype::F8E4m3 => "f8e4m3",
            CutlassDtype::F8E5m2 => "f8e5m2",
            CutlassDtype::F4E2m1 => "f4e2m1",
            CutlassDtype::I8 => "i8",
            CutlassDtype::I32 => "i32",
            CutlassDtype::U8 => "u8",
        }
    }

    /// Element size in bits. fp4 is the only sub-byte dtype.
    pub fn size_bits(self) -> u32 {
        match self {
            CutlassDtype::F64 => 64,
            CutlassDtype::F32 | CutlassDtype::I32 => 32,
            CutlassDtype::F16 | CutlassDtype::Bf16 => 16,
            CutlassDtype::F8E4m3 | CutlassDtype::F8E5m2 | CutlassDtype::I8 | CutlassDtype::U8 => 8,
            CutlassDtype::F4E2m1 => 4,
        }
    }
}

impl fmt::Display for CutlassDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

/// Compute architectures the GEMM template emitter knows how to target.
///
/// Mirrors the per-arch toolchain keys used by `NvrtcActor`. Adding a
/// variant here is a non-breaking change — downstream code matches via
/// the helper predicates below.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmArch {
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm90a,
    Sm100,
    Sm120,
}

impl SmArch {
    pub fn nvrtc_flag(self) -> &'static str {
        match self {
            SmArch::Sm80 => "--gpu-architecture=compute_80",
            SmArch::Sm86 => "--gpu-architecture=compute_86",
            SmArch::Sm89 => "--gpu-architecture=compute_89",
            SmArch::Sm90 => "--gpu-architecture=compute_90",
            SmArch::Sm90a => "--gpu-architecture=compute_90a",
            SmArch::Sm100 => "--gpu-architecture=compute_100",
            SmArch::Sm120 => "--gpu-architecture=compute_120",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            SmArch::Sm80 => "sm_80",
            SmArch::Sm86 => "sm_86",
            SmArch::Sm89 => "sm_89",
            SmArch::Sm90 => "sm_90",
            SmArch::Sm90a => "sm_90a",
            SmArch::Sm100 => "sm_100",
            SmArch::Sm120 => "sm_120",
        }
    }

    pub fn supports_fp8(self) -> bool {
        matches!(
            self,
            SmArch::Sm89 | SmArch::Sm90 | SmArch::Sm90a | SmArch::Sm100 | SmArch::Sm120
        )
    }

    pub fn supports_fp4(self) -> bool {
        matches!(self, SmArch::Sm100 | SmArch::Sm120)
    }

    pub fn supports_persistent_kernels(self) -> bool {
        matches!(
            self,
            SmArch::Sm90 | SmArch::Sm90a | SmArch::Sm100 | SmArch::Sm120
        )
    }
}

impl fmt::Display for SmArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

/// Marker trait: types that a CUTLASS GEMM template can instantiate
/// against. The trait body is empty; the dtype tag returned by
/// [`GemmSupported::DTYPE`] drives the template emitter.
///
/// Implemented for `f32`, `f64`, `i8`, `i32`, `u8`, plus the fp8 / fp4 /
/// f16 / bf16 wrapper structs in this module.
pub trait GemmSupported: Copy + Send + Sync + 'static {
    const DTYPE: CutlassDtype;
}

impl GemmSupported for f32 {
    const DTYPE: CutlassDtype = CutlassDtype::F32;
}
impl GemmSupported for f64 {
    const DTYPE: CutlassDtype = CutlassDtype::F64;
}
impl GemmSupported for i8 {
    const DTYPE: CutlassDtype = CutlassDtype::I8;
}
impl GemmSupported for i32 {
    const DTYPE: CutlassDtype = CutlassDtype::I32;
}
impl GemmSupported for u8 {
    const DTYPE: CutlassDtype = CutlassDtype::U8;
}

/// Local f16 marker. Wraps a `u16` to avoid pulling `half` into the
/// crate's dependency tree. Matches the wrapper layout used by the
/// upstream `atomr-accel-cuda::dtype` module so users that pass through
/// our actor surface don't observe any divergence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F16(pub u16);
impl GemmSupported for F16 {
    const DTYPE: CutlassDtype = CutlassDtype::F16;
}

/// Local bf16 marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Bf16(pub u16);
impl GemmSupported for Bf16 {
    const DTYPE: CutlassDtype = CutlassDtype::Bf16;
}

/// Local fp8 e4m3 marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F8E4m3(pub u8);
impl GemmSupported for F8E4m3 {
    const DTYPE: CutlassDtype = CutlassDtype::F8E4m3;
}

/// Local fp8 e5m2 marker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F8E5m2(pub u8);
impl GemmSupported for F8E5m2 {
    const DTYPE: CutlassDtype = CutlassDtype::F8E5m2;
}

/// Local fp4 e2m1 marker. Stored as `u8` because Rust has no native
/// `u4`; the lower nibble is the value, the upper nibble is unused.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct F4E2m1(pub u8);
impl GemmSupported for F4E2m1 {
    const DTYPE: CutlassDtype = CutlassDtype::F4E2m1;
}

/// Returns true if `dtype` can be instantiated on `arch` by a CUTLASS
/// GEMM template. fp8 is sm_89+; fp4 is sm_100+; everything else is
/// supported on every modern Tensor-Core arch.
pub fn is_supported_for(dtype: CutlassDtype, arch: SmArch) -> bool {
    match dtype {
        CutlassDtype::F8E4m3 | CutlassDtype::F8E5m2 => arch.supports_fp8(),
        CutlassDtype::F4E2m1 => arch.supports_fp4(),
        _ => true,
    }
}

/// Convenience predicate for fp8-only callers.
pub fn is_fp8_supported(arch: SmArch) -> bool {
    arch.supports_fp8()
}

/// Convenience predicate for fp4-only callers.
pub fn is_fp4_supported(arch: SmArch) -> bool {
    arch.supports_fp4()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_capability_predicates() {
        assert!(!SmArch::Sm80.supports_fp8());
        assert!(SmArch::Sm89.supports_fp8());
        assert!(SmArch::Sm90a.supports_fp8());
        assert!(SmArch::Sm100.supports_fp4());
        assert!(!SmArch::Sm89.supports_fp4());
        assert!(SmArch::Sm90a.supports_persistent_kernels());
        assert!(!SmArch::Sm80.supports_persistent_kernels());
    }

    #[test]
    fn dtype_short_names_unique() {
        let all = [
            CutlassDtype::F32,
            CutlassDtype::F64,
            CutlassDtype::F16,
            CutlassDtype::Bf16,
            CutlassDtype::F8E4m3,
            CutlassDtype::F8E5m2,
            CutlassDtype::F4E2m1,
            CutlassDtype::I8,
            CutlassDtype::I32,
            CutlassDtype::U8,
        ];
        let mut seen: Vec<&'static str> = Vec::new();
        for dt in all {
            assert!(!seen.contains(&dt.short_name()));
            seen.push(dt.short_name());
        }
    }

    #[test]
    fn is_supported_for_matrix() {
        assert!(is_supported_for(CutlassDtype::F32, SmArch::Sm80));
        assert!(!is_supported_for(CutlassDtype::F8E4m3, SmArch::Sm80));
        assert!(is_supported_for(CutlassDtype::F8E4m3, SmArch::Sm90a));
        assert!(!is_supported_for(CutlassDtype::F4E2m1, SmArch::Sm89));
        assert!(is_supported_for(CutlassDtype::F4E2m1, SmArch::Sm100));
    }
}
