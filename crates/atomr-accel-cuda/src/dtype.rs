//! Minimal CUDA-side dtype trait (Phase 0.4 scaffolding).
//!
//! This is a tightly-scoped subset of the eventual cross-cutting
//! `AccelDtype` / `CudaDtype` traits described in the roadmap. The
//! Phase 0.4 collapse needs:
//!
//! 1. A runtime `DType` tag so a boxed `*Dispatch` trait object can
//!    advertise the concrete dtype it is carrying (for tracing,
//!    metrics, and the `dtype()` accessor required by the dispatch
//!    trait surface).
//! 2. A blanket bound (`CudaDtype`) that subsumes the cudarc trait
//!    bounds previously written long-hand (`DeviceRepr +
//!    ValidAsZeroBits + Send + Sync + 'static`) so the alloc/copy
//!    helpers in `context_actor` and the new generic dispatch types
//!    in `device_actor` stay terse.
//!
//! Future phases (0.1, full CudaDtype implementation) will move this
//! into `crates/atomr-accel/src/dtype.rs` and grow the trait surface
//! with cuBLAS / cuDNN / FFT / NCCL capability markers. This file is
//! kept deliberately small so that migration is mechanical.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

/// Runtime dtype tag. Mirrors the eventual `atomr_accel::DType` from
/// Phase 0.1 — the variants here are the ones currently reachable via
/// `DeviceMsg::Allocate*` / `CopyToHost*` / `CopyFromHost*`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    Bf16,
    I8,
    I32,
    I64,
    U8,
    U32,
    U64,
}

impl DType {
    /// Stable string name. Useful in tracing / log fields.
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::Bf16 => "bf16",
            DType::I8 => "i8",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U32 => "u32",
            DType::U64 => "u64",
        }
    }

    /// Element size in bytes.
    pub const fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F16 | DType::Bf16 => 2,
            DType::I8 | DType::U8 => 1,
        }
    }
}

/// Phase 0.4 minimal CUDA dtype trait.
///
/// Implementations are provided for every primitive dtype reachable
/// from the existing `Allocate*` variants. Future phases will lift the
/// trait into the backend-agnostic core crate and add cuBLAS /
/// cuDNN / FFT capability markers — this file moves wholesale into
/// the new home at that point.
pub trait CudaDtype: DeviceRepr + ValidAsZeroBits + Send + Sync + 'static {
    const KIND: DType;
}

impl CudaDtype for f32 {
    const KIND: DType = DType::F32;
}
impl CudaDtype for f64 {
    const KIND: DType = DType::F64;
}
impl CudaDtype for i8 {
    const KIND: DType = DType::I8;
}
impl CudaDtype for i32 {
    const KIND: DType = DType::I32;
}
impl CudaDtype for i64 {
    const KIND: DType = DType::I64;
}
impl CudaDtype for u8 {
    const KIND: DType = DType::U8;
}
impl CudaDtype for u32 {
    const KIND: DType = DType::U32;
}
impl CudaDtype for u64 {
    const KIND: DType = DType::U64;
}
#[cfg(feature = "f16")]
impl CudaDtype for half::f16 {
    const KIND: DType = DType::F16;
}
#[cfg(feature = "f16")]
impl CudaDtype for half::bf16 {
    const KIND: DType = DType::Bf16;
}
