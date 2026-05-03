//! Multi-stream pipeline pattern.
//!
//! Lets users wire `Source -> KernelStage -> ... -> Sink` such that
//! stage K+1 begins as soon as stage K's GPU work is complete, with
//! cross-stage handoff via [`cudarc::driver::CudaEvent`] — no host
//! roundtrip between stages.
//!
//! # Stage shape
//!
//! Implement [`PipelineStage`] for any kernel-actor adapter:
//!
//! ```ignore
//! struct BlasSgemmStage { /* ... */ }
//! impl PipelineStage for BlasSgemmStage {
//!     type In = (GpuRef<f32>, GpuRef<f32>);
//!     type Out = GpuRef<f32>;
//!     fn enqueue(
//!         &mut self, stream, wait_for, (a, b)
//!     ) -> Result<(CudaEvent, GpuRef<f32>), GpuError> {
//!         if let Some(ev) = wait_for { stream.wait(ev)?; }
//!         /* enqueue cuBLAS gemm via record-mode contract */
//!         let ev = stream.record_event(None)?;
//!         Ok((ev, c))
//!     }
//! }
//! ```
//!
//! F2 ships the trait + a thin executor; the full
//! `PipelineBuilder<I, O>` type-state DSL with Source / Sink wrappers
//! lands in F3 once we have more concrete patterns demanding it.

mod executor;
mod sink;
mod stage;

pub use executor::{run_pipeline, BoxedStage, PipelineExecutor, PipelineExecutorN, StageBox};
pub use sink::{spawn_pipeline, PipelineSink, PipelineSource};
pub use stage::PipelineStage;
