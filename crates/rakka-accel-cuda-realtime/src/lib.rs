//! Interactive-rate GPU actor blueprints on rakka-accel-cuda.
//!
//! ```ignore
//! use rakka_accel_cuda_realtime::prelude::*;
//! ```
//!
//! Ships CPU reference implementations plus the bundled NVRTC
//! source for each actor's GPU fast path (see [`kernels`]).
//! Per-actor `with_nvrtc(...)` constructors are gated on the
//! `nvrtc` feature.

pub mod cloth;
pub mod fluid;
pub mod hashmap;
pub mod image_filter;
pub mod kernels;
pub mod multi_pass;
pub mod particle;
pub mod reduction;
pub mod sparse;
pub mod spatial_index;
pub mod video_effects;

pub mod prelude {
    //! Canonical re-exports. `use rakka_accel_cuda_realtime::prelude::*;`.
    pub use crate::cloth::{ClothConfig, ClothMsg, ClothSimulationActor};
    pub use crate::fluid::{FluidConfig, FluidMsg, FluidSimulationActor};
    pub use crate::hashmap::{
        GpuHashMapActor, GpuHashMapConfig, GpuHashMapMsg, GpuHashMapStats,
    };
    pub use crate::image_filter::{ImageFilterConfig, ImageFilterMsg, ImageFilterPipeline};
    pub use crate::multi_pass::{MultiPassAnalysisActor, MultiPassConfig, MultiPassMsg};
    pub use crate::particle::{
        Particle, ParticleMsg, ParticleSystemActor, ParticleSystemConfig, Vec3,
    };
    pub use crate::reduction::{ReductionAnalysisActor, ReductionKind, ReductionMsg};
    pub use crate::sparse::{
        CooEntry, GpuSparseStructureActor, SparseConfig, SparseMsg, SparseStats,
    };
    pub use crate::spatial_index::{
        CellKey, Point3, SpatialIndexActor, SpatialIndexConfig, SpatialMsg, SpatialStats,
    };
    pub use crate::video_effects::{VideoEffectsConfig, VideoEffectsGraph, VideoEffectsMsg};
}
