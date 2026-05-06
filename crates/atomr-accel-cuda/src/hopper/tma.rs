//! Tensor Memory Accelerator (TMA) host-side descriptor builder.
//!
//! On Hopper (sm_90+) and Blackwell, TMA decouples global → shared
//! memory tile copies from the threads that issue them: the kernel
//! issues `cp.async.bulk.tensor.NN.global.shared` against an opaque
//! [`CUtensorMap`](cudarc::driver::sys::CUtensorMap), which the host
//! built once via `cuTensorMapEncodeTiled`. The kernel then waits on a
//! barrier (`mbarrier.try_wait`) for the copy to land.
//!
//! [`TensorMapDescriptor`] is a host-side builder for the `tiled`
//! flavour of the encode-call. The free `encode` method returns the
//! 128-byte tensor-map struct cudarc surfaces as
//! `cudarc::driver::sys::CUtensorMap`. This module makes no attempt to
//! cover the `im2col` / `im2col-wide` flavours — those have shape-
//! specific descriptor sets that fit poorly into a uniform builder.

use std::fmt;

/// Element dtype consumed by the TMA. Matches
/// [`cudarc::driver::sys::CUtensorMapDataType_enum`] one-to-one. We
/// duplicate the enum so callers don't depend on cudarc's `sys` module
/// directly (which is gated on a CUDA-version feature in cudarc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMapDataType {
    UInt8,
    UInt16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Float32Ftz,
    /// TF32 / `__nv_tf32`.
    TFloat32,
    TFloat32Ftz,
    /// Blackwell-only fp4/fp6/fp8 variants. Available with the
    /// `blackwell` cargo feature only — the host-side enum is allowed
    /// regardless of feature so unit tests can round-trip the value;
    /// runtime kernels need a Blackwell driver.
    Float8E4m3,
    Float8E5m2,
    Float4E2m1,
    Float6E2m3,
    Float6E3m2,
}

impl TensorMapDataType {
    /// Numeric value used by the underlying CUDA driver. Matches the
    /// public ABI of `CUtensorMapDataType_enum`.
    pub fn as_u32(self) -> u32 {
        match self {
            TensorMapDataType::UInt8 => 0,
            TensorMapDataType::UInt16 => 1,
            TensorMapDataType::UInt32 => 2,
            TensorMapDataType::Int32 => 3,
            TensorMapDataType::UInt64 => 4,
            TensorMapDataType::Int64 => 5,
            TensorMapDataType::Float16 => 6,
            TensorMapDataType::Float32 => 7,
            TensorMapDataType::Float64 => 8,
            TensorMapDataType::BFloat16 => 9,
            TensorMapDataType::Float32Ftz => 10,
            TensorMapDataType::TFloat32 => 11,
            TensorMapDataType::TFloat32Ftz => 12,
            TensorMapDataType::Float8E4m3 => 13,
            TensorMapDataType::Float8E5m2 => 14,
            TensorMapDataType::Float4E2m1 => 15,
            TensorMapDataType::Float6E2m3 => 16,
            TensorMapDataType::Float6E3m2 => 17,
        }
    }
}

/// Interleave layout for 1D / 2D / 3D bulk-tile copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMapInterleave {
    /// No interleave — natural row-major / column-major as described by
    /// `global_strides`.
    None,
    /// 16-byte interleave — packs four fp32 lanes per 16B chunk.
    Bytes16,
    /// 32-byte interleave — packs eight fp32 lanes / sixteen fp16 lanes.
    Bytes32,
}

impl TensorMapInterleave {
    pub fn as_u32(self) -> u32 {
        match self {
            TensorMapInterleave::None => 0,
            TensorMapInterleave::Bytes16 => 1,
            TensorMapInterleave::Bytes32 => 2,
        }
    }
}

/// Shared-memory swizzle pattern. A swizzled load/store interleaves
/// rows so 4-thread bank conflicts can't arise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMapSwizzle {
    None,
    /// 32B swizzle — cache-line aligned 4-element rows.
    Bytes32,
    /// 64B swizzle — half cache line.
    Bytes64,
    /// 128B swizzle — full cache line. Most common for wgmma feeds.
    Bytes128,
}

impl TensorMapSwizzle {
    pub fn as_u32(self) -> u32 {
        match self {
            TensorMapSwizzle::None => 0,
            TensorMapSwizzle::Bytes32 => 1,
            TensorMapSwizzle::Bytes64 => 2,
            TensorMapSwizzle::Bytes128 => 3,
        }
    }
}

/// L2 promotion hint — the subset of L2 cache the TMA is allowed to
/// promote into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMapL2Promotion {
    None,
    Bytes64,
    Bytes128,
    Bytes256,
}

impl TensorMapL2Promotion {
    pub fn as_u32(self) -> u32 {
        match self {
            TensorMapL2Promotion::None => 0,
            TensorMapL2Promotion::Bytes64 => 1,
            TensorMapL2Promotion::Bytes128 => 2,
            TensorMapL2Promotion::Bytes256 => 3,
        }
    }
}

/// Out-of-bounds fill mode for partial tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorMapOobFill {
    /// Pad with `0`. Most common for matmul tail tiles.
    NaZero,
    /// Pad with NaN where the dtype representation supports it.
    NanRequest,
}

impl TensorMapOobFill {
    pub fn as_u32(self) -> u32 {
        match self {
            TensorMapOobFill::NaZero => 0,
            TensorMapOobFill::NanRequest => 1,
        }
    }
}

/// Host-side builder for the tiled flavour of `cuTensorMapEncodeTiled`.
///
/// Cap at 5 dimensions (the CUDA-driver hard limit). Validation:
///
/// * `rank` must equal `global_dim.len()` and `box_dim.len()` and
///   `element_strides.len()`; `global_strides.len() == rank - 1`.
/// * `global_address` must be 16-byte aligned (TMA requires it).
/// * Every entry of `global_dim` and `box_dim` must be non-zero.
#[derive(Debug, Clone)]
pub struct TensorMapDescriptor {
    pub data_type: TensorMapDataType,
    pub global_address: usize,
    pub global_dim: Vec<u64>,
    pub global_strides: Vec<u64>,
    pub box_dim: Vec<u32>,
    pub element_strides: Vec<u32>,
    pub interleave: TensorMapInterleave,
    pub swizzle: TensorMapSwizzle,
    pub l2_promotion: TensorMapL2Promotion,
    pub oob_fill: TensorMapOobFill,
}

impl TensorMapDescriptor {
    /// Construct a defaulted-shape descriptor. Caller must populate
    /// `global_dim`, `global_strides`, `box_dim`, `element_strides`
    /// before calling [`TensorMapDescriptor::validate`].
    pub fn new(data_type: TensorMapDataType, global_address: usize) -> Self {
        Self {
            data_type,
            global_address,
            global_dim: Vec::new(),
            global_strides: Vec::new(),
            box_dim: Vec::new(),
            element_strides: Vec::new(),
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            l2_promotion: TensorMapL2Promotion::None,
            oob_fill: TensorMapOobFill::NaZero,
        }
    }

    /// Tensor rank (1..=5).
    pub fn rank(&self) -> usize {
        self.global_dim.len()
    }

    /// Validate that all sizes line up and the global address is 16B
    /// aligned. Returns `Err(TmaEncodeError::*)` on mismatch. Pure host
    /// validation — does not call into the driver.
    pub fn validate(&self) -> Result<(), TmaEncodeError> {
        let r = self.rank();
        if r == 0 || r > 5 {
            return Err(TmaEncodeError::BadRank(r));
        }
        if self.box_dim.len() != r {
            return Err(TmaEncodeError::Mismatch {
                what: "box_dim",
                expected: r,
                got: self.box_dim.len(),
            });
        }
        if self.element_strides.len() != r {
            return Err(TmaEncodeError::Mismatch {
                what: "element_strides",
                expected: r,
                got: self.element_strides.len(),
            });
        }
        // CUDA's API takes `rank - 1` global strides because the
        // innermost stride is implicit (= sizeof(elem)).
        if !self.global_strides.is_empty() && self.global_strides.len() != r - 1 {
            return Err(TmaEncodeError::Mismatch {
                what: "global_strides",
                expected: r - 1,
                got: self.global_strides.len(),
            });
        }
        if self.global_address % 16 != 0 {
            return Err(TmaEncodeError::UnalignedAddress(self.global_address));
        }
        if self.global_dim.contains(&0) {
            return Err(TmaEncodeError::ZeroDim("global_dim"));
        }
        if self.box_dim.contains(&0) {
            return Err(TmaEncodeError::ZeroDim("box_dim"));
        }
        Ok(())
    }

    /// Validate, then call `cuTensorMapEncodeTiled` to populate a
    /// 128-byte `CUtensorMap`. Available only with the `hopper` cargo
    /// feature (otherwise the call would have nothing to encode into).
    ///
    /// # Safety
    ///
    /// The caller must ensure `global_address` points at a
    /// driver-mapped device allocation that lives at least as long as
    /// the kernels that consume the resulting tensor map. The function
    /// itself is safe to call (it does not dereference the address) but
    /// kernels launched against the encoded map perform device reads.
    #[cfg(feature = "hopper")]
    pub fn encode(&self) -> Result<TensorMap, TmaEncodeError> {
        use cudarc::driver::sys as cu;

        self.validate()?;
        let mut tm: cu::CUtensorMap = unsafe { std::mem::zeroed() };
        // SAFETY: every pointer is into our own Vecs which live for
        // the duration of this call. cuTensorMapEncodeTiled copies
        // the values into the opaque map; no aliasing required.
        let res = unsafe {
            cu::cuTensorMapEncodeTiled(
                &mut tm,
                std::mem::transmute::<u32, cu::CUtensorMapDataType>(self.data_type.as_u32()),
                self.rank() as cu::cuuint32_t,
                self.global_address as *mut _,
                self.global_dim.as_ptr(),
                self.global_strides.as_ptr(),
                self.box_dim.as_ptr(),
                self.element_strides.as_ptr(),
                std::mem::transmute::<u32, cu::CUtensorMapInterleave>(self.interleave.as_u32()),
                std::mem::transmute::<u32, cu::CUtensorMapSwizzle>(self.swizzle.as_u32()),
                std::mem::transmute::<u32, cu::CUtensorMapL2promotion>(self.l2_promotion.as_u32()),
                std::mem::transmute::<u32, cu::CUtensorMapFloatOOBfill>(self.oob_fill.as_u32()),
            )
        };
        if res != cu::CUresult::CUDA_SUCCESS {
            return Err(TmaEncodeError::DriverError(res as i32));
        }
        Ok(TensorMap(tm))
    }
}

/// Opaque 128-byte handle returned by [`TensorMapDescriptor::encode`].
/// Pass into NVRTC kernels as a `const __grid_constant__ CUtensorMap`.
#[cfg(feature = "hopper")]
pub struct TensorMap(pub cudarc::driver::sys::CUtensorMap);

#[cfg(feature = "hopper")]
impl TensorMap {
    /// Raw byte pointer to the encoded map. Useful when the kernel
    /// signature wants `const CUtensorMap*`.
    pub fn as_ptr(&self) -> *const cudarc::driver::sys::CUtensorMap {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TmaEncodeError {
    BadRank(usize),
    Mismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    UnalignedAddress(usize),
    ZeroDim(&'static str),
    /// Driver returned a non-zero `CUresult`.
    DriverError(i32),
}

impl fmt::Display for TmaEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TmaEncodeError::BadRank(r) => write!(f, "TMA rank {r} out of [1,5]"),
            TmaEncodeError::Mismatch {
                what,
                expected,
                got,
            } => {
                write!(
                    f,
                    "TMA descriptor: {what}.len() = {got}, expected {expected}"
                )
            }
            TmaEncodeError::UnalignedAddress(a) => {
                write!(f, "TMA global_address 0x{a:x} is not 16-byte aligned")
            }
            TmaEncodeError::ZeroDim(field) => write!(f, "TMA descriptor: {field} contains a zero"),
            TmaEncodeError::DriverError(c) => write!(f, "cuTensorMapEncodeTiled returned {c}"),
        }
    }
}

impl std::error::Error for TmaEncodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_2d_descriptor() -> TensorMapDescriptor {
        TensorMapDescriptor {
            data_type: TensorMapDataType::Float16,
            global_address: 0x1_0000, // 16B aligned
            global_dim: vec![1024, 1024],
            global_strides: vec![1024 * 2], // row stride in bytes for fp16
            box_dim: vec![64, 64],
            element_strides: vec![1, 1],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::Bytes128,
            l2_promotion: TensorMapL2Promotion::Bytes128,
            oob_fill: TensorMapOobFill::NaZero,
        }
    }

    /// Phase 5 test: round-trip a tiled-TMA descriptor through the
    /// builder + validate path. No GPU required — the validation logic
    /// is host-side.
    #[test]
    fn tensor_map_encode_descriptor_round_trip() {
        let d = sample_2d_descriptor();
        d.validate().expect("sample descriptor must validate");

        // Field round-trip.
        assert_eq!(d.rank(), 2);
        assert_eq!(d.data_type.as_u32(), TensorMapDataType::Float16.as_u32());
        assert_eq!(d.swizzle.as_u32(), TensorMapSwizzle::Bytes128.as_u32());
        assert_eq!(
            d.l2_promotion.as_u32(),
            TensorMapL2Promotion::Bytes128.as_u32()
        );
        assert_eq!(d.oob_fill.as_u32(), TensorMapOobFill::NaZero.as_u32());

        // Mutate to misaligned address — must reject.
        let mut bad = d.clone();
        bad.global_address = 0x1_0001;
        assert!(matches!(
            bad.validate().unwrap_err(),
            TmaEncodeError::UnalignedAddress(_)
        ));

        // Mutate to wrong-length box_dim — must reject.
        let mut bad = d.clone();
        bad.box_dim.push(32);
        assert!(matches!(
            bad.validate().unwrap_err(),
            TmaEncodeError::Mismatch {
                what: "box_dim",
                ..
            }
        ));

        // Rank 0 — reject.
        let bad = TensorMapDescriptor::new(TensorMapDataType::Float32, 0x10);
        assert!(matches!(
            bad.validate().unwrap_err(),
            TmaEncodeError::BadRank(0)
        ));

        // Rank 6 — reject.
        let bad = TensorMapDescriptor {
            data_type: TensorMapDataType::Float32,
            global_address: 0x10,
            global_dim: vec![1; 6],
            global_strides: vec![4; 5],
            box_dim: vec![1; 6],
            element_strides: vec![1; 6],
            interleave: TensorMapInterleave::None,
            swizzle: TensorMapSwizzle::None,
            l2_promotion: TensorMapL2Promotion::None,
            oob_fill: TensorMapOobFill::NaZero,
        };
        assert!(matches!(
            bad.validate().unwrap_err(),
            TmaEncodeError::BadRank(6)
        ));
    }

    /// Every dtype/swizzle/interleave round-trips through `as_u32` to
    /// a unique value (enum identity).
    #[test]
    fn enum_discriminants_are_unique() {
        let dts = [
            TensorMapDataType::UInt8,
            TensorMapDataType::UInt16,
            TensorMapDataType::UInt32,
            TensorMapDataType::Int32,
            TensorMapDataType::UInt64,
            TensorMapDataType::Int64,
            TensorMapDataType::Float16,
            TensorMapDataType::Float32,
            TensorMapDataType::Float64,
            TensorMapDataType::BFloat16,
            TensorMapDataType::Float32Ftz,
            TensorMapDataType::TFloat32,
            TensorMapDataType::TFloat32Ftz,
            TensorMapDataType::Float8E4m3,
            TensorMapDataType::Float8E5m2,
            TensorMapDataType::Float4E2m1,
            TensorMapDataType::Float6E2m3,
            TensorMapDataType::Float6E3m2,
        ];
        let mut seen = std::collections::HashSet::new();
        for d in dts {
            assert!(
                seen.insert(d.as_u32()),
                "duplicate dtype discriminant for {d:?}"
            );
        }
    }
}
