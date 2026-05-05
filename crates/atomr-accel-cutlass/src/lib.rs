//! # atomr-accel-cutlass
//!
//! CUTLASS kernel-template instantiation for `atomr-accel-cuda`.
//!
//! Phase 6 of the CUDA-coverage roadmap. This crate exposes a thin
//! actor-friendly facade over CUTLASS GEMM, grouped-GEMM, and
//! implicit-GEMM convolution templates. Two compilation strategies are
//! supported:
//!
//! ## Strategy A (default — NVRTC at runtime)
//!
//! User code constructs a [`gemm::GemmRequest<T>`] (or grouped /
//! conv equivalent) and forwards it to a [`actor::CutlassActor`].
//! The actor builds a small `.cu` translation unit that
//! `#include`s the vendored CUTLASS headers (under
//! `crates/atomr-accel-cutlass/cutlass/include/`) and instantiates
//! the requested template, then hands the source to
//! `atomr_accel_cuda::kernel::NvrtcActor` for compilation.
//! Compilation cost is amortized via the per-arch disk cache that
//! `NvrtcActor` already maintains and via the in-process
//! [`plan_cache::PlanCache`] keyed by
//! `(template_id, shape, dtype, arch)`.
//!
//! ## Strategy B (`cutlass-prebuilt` feature — nvcc at build time)
//!
//! `build.rs` walks a generator that emits a static library of
//! pre-instantiated kernels for a fixed `(op × dtype × arch)` matrix.
//! When the feature is OFF the `build.rs` is a no-op probe so the
//! crate still builds on hosts without `nvcc`. The contract for
//! Strategy B is documented in [`build.rs`] and in the crate
//! README — full implementation is a follow-up; we wire the toggle
//! and the empty hooks here.
//!
//! ## Coverage
//!
//! | op                | dtype             | arch                      |
//! |-------------------|-------------------|---------------------------|
//! | GEMM              | fp32, fp16, bf16  | sm_80, sm_86, sm_89       |
//! | GEMM              | fp8 e4m3 / e5m2   | sm_89, sm_90a, sm_100     |
//! | GEMM              | fp4 e2m1          | sm_100, sm_120            |
//! | grouped GEMM      | fp16, bf16, fp8   | sm_90a, sm_100            |
//! | conv2d fwd / dgrad / wgrad (implicit-GEMM) | fp16, bf16, fp32, fp8 | sm_80, sm_86, sm_89, sm_90a |
//!
//! Compute targets: Ampere (sm_80, fp16/bf16/fp32), Hopper (sm_90a,
//! fp8 e4m3/e5m2 + fp16/bf16, persistent kernels), Blackwell
//! (sm_100, fp8/fp4 + EVT).
//!
//! ## Crate layout
//!
//! ```text
//! crates/atomr-accel-cutlass/
//! ├── Cargo.toml
//! ├── cutlass/                  # vendored CUTLASS headers (BSD-3-Clause)
//! ├── include/                  # local template adapters
//! ├── build.rs                  # gated on cutlass-prebuilt feature
//! ├── examples/
//! │   ├── cutlass_gemm_fp8.rs
//! │   └── cutlass_grouped_gemm.rs
//! └── src/
//!     ├── lib.rs                # CutlassActor, CutlassMsg, props
//!     ├── actor.rs              # actor surface
//!     ├── gemm.rs               # GemmRequest<T>, CutlassGemmDispatch
//!     ├── grouped_gemm.rs       # GroupedGemmRequest<T>
//!     ├── conv.rs               # ConvFwd / Dgrad / Wgrad requests
//!     ├── evt.rs                # EpilogueVisitorTree builder
//!     ├── plan_cache.rs         # template plan cache
//!     └── kernels/              # generated .cu sources at runtime
//! ```

#![deny(rust_2018_idioms)]

pub mod actor;
pub mod conv;
pub mod dtype;
pub mod gemm;
pub mod plan_cache;

#[cfg(feature = "evt")]
pub mod evt;

#[cfg(feature = "grouped")]
pub mod grouped_gemm;

mod kernels;

pub use actor::{CutlassActor, CutlassInner, CutlassMsg};
pub use conv::{
    ConvDgradRequest, ConvFwdRequest, ConvLayout, ConvShape, ConvWgradRequest, CutlassConvDispatch,
};
pub use dtype::{
    is_fp4_supported, is_fp8_supported, is_supported_for, CutlassDtype, GemmSupported, SmArch,
};
pub use gemm::{CutlassGemmDispatch, GemmEpilogue, GemmLayout, GemmRequest, GemmShape, RefitMsg};
pub use plan_cache::{PlanCache, PlanKey};

#[cfg(feature = "evt")]
pub use evt::{EpilogueOp, EpilogueVisitorTree, EvtBuilder};

#[cfg(feature = "grouped")]
pub use grouped_gemm::{
    CutlassGroupedGemmDispatch, GroupedGemmRequest, GroupedGemmShape, GroupedLayout,
};

/// Entry point used by the `cutlass` cargo feature on
/// `atomr-accel-cuda`. Returns the human-readable crate version
/// string. Exposed for completeness so that downstream re-exports can
/// version-pin against the cutlass crate without an extra cargo
/// metadata call.
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Convenience: build a [`CutlassActor`] props value. Mirrors the
/// `props` constructors used by the rest of `atomr-accel-cuda` so
/// downstream code can write
/// `system.actor_of(atomr_accel_cutlass::props(...), "cutlass")`.
pub fn props(plan_cache_capacity: usize) -> actor::CutlassProps {
    actor::CutlassProps::new(plan_cache_capacity)
}
