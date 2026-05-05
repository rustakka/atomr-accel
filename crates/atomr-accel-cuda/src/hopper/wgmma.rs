//! WGMMA (warp-group matrix multiply accumulate) intrinsic shim.
//!
//! Hopper's `wgmma.mma_async.sync` instruction is issued from a
//! 128-thread warpgroup; the host side has nothing to call, but NVRTC
//! kernels embed the intrinsics through PTX inline assembly. This
//! module ships a small set of macro shims (in `atomr_hopper.cuh`) that
//! pin the asm constraints to the supported `(M, N, K, dtype-A,
//! dtype-B, dtype-D)` shapes and give Rust callers symbolic names for
//! the descriptors they have to build host-side.
//!
//! Only the most common matmul variants are wrapped. Adding a new
//! variant means adding a new `WGMMA_MMA_ASYNC_*` macro in
//! `atomr_hopper.cuh` and a constant in [`WgmmaShape`].

/// Subset of WGMMA matmul shapes commonly exercised by attention /
/// matmul kernels. The numeric tuple is `(M, N, K)` (row × col × inner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgmmaShape {
    /// `m64n64k16` — most-common fp16 variant.
    M64N64K16,
    /// `m64n128k16` — wider tile, same fp16.
    M64N128K16,
    /// `m64n256k16` — full warpgroup output tile.
    M64N256K16,
    /// `m64n64k32` — fp8 (e4m3/e5m2) variant.
    M64N64K32,
    /// `m64n128k32` — fp8 wider.
    M64N128K32,
    /// `m64n256k32` — fp8 full.
    M64N256K32,
}

impl WgmmaShape {
    /// `(M, N, K)` decomposition.
    pub fn dims(self) -> (u32, u32, u32) {
        match self {
            WgmmaShape::M64N64K16 => (64, 64, 16),
            WgmmaShape::M64N128K16 => (64, 128, 16),
            WgmmaShape::M64N256K16 => (64, 256, 16),
            WgmmaShape::M64N64K32 => (64, 64, 32),
            WgmmaShape::M64N128K32 => (64, 128, 32),
            WgmmaShape::M64N256K32 => (64, 256, 32),
        }
    }

    /// Macro name (matches `atomr_hopper.cuh`).
    pub fn macro_name(self) -> &'static str {
        match self {
            WgmmaShape::M64N64K16 => "ATOMR_WGMMA_F16_M64N64K16",
            WgmmaShape::M64N128K16 => "ATOMR_WGMMA_F16_M64N128K16",
            WgmmaShape::M64N256K16 => "ATOMR_WGMMA_F16_M64N256K16",
            WgmmaShape::M64N64K32 => "ATOMR_WGMMA_F8_M64N64K32",
            WgmmaShape::M64N128K32 => "ATOMR_WGMMA_F8_M64N128K32",
            WgmmaShape::M64N256K32 => "ATOMR_WGMMA_F8_M64N256K32",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_round_trip() {
        assert_eq!(WgmmaShape::M64N64K16.dims(), (64, 64, 16));
        assert_eq!(WgmmaShape::M64N256K32.dims(), (64, 256, 32));
    }

    #[test]
    fn macro_names_match_header() {
        assert_eq!(
            WgmmaShape::M64N64K16.macro_name(),
            "ATOMR_WGMMA_F16_M64N64K16"
        );
    }
}
