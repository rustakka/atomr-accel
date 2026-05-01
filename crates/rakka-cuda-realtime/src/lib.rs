//! Interactive-rate GPU actor blueprints (§7.5).
//!
//! F3 ships:
//! - [`image_filter::ImageFilterPipeline`] — H2D → Conv → Activation
//!   → D2H pipeline, optimized for per-frame replay via `GraphActor`.
//! - [`hashmap::GpuHashMapActor`] — open-addressing hashmap on the
//!   GPU (skeleton; the actual NVRTC kernel lives in
//!   `nvrtc_sources/gpu_hashmap.cu` once F3 NVRTC is wired in).
//!
//! F5 adds: ParticleSystemActor, SpatialIndexActor, VideoEffects,
//! cloth/fluid sims.

pub mod cloth;
pub mod fluid;
pub mod hashmap;
pub mod image_filter;
pub mod multi_pass;
pub mod particle;
pub mod reduction;
pub mod sparse;
pub mod spatial_index;
pub mod video_effects;
