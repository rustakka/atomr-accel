//! # rakka-accel-cuda
//!
//! GPU acceleration via the actor model. Wraps NVIDIA CUDA libraries as
//! actors on top of [`rakka`](../rakka). See `README.md` and the
//! architecture document under `docs/` for the full design.
//!
//! ## Foundation Phase F1 (current)
//!
//! - Two-tier supervision: [`device::DeviceActor`] (stable address) ↔
//!   [`device::ContextActor`] (owns `Arc<CudaContext>`, restartable).
//! - [`gpu_ref::GpuRef`] with generation-token validity checks.
//! - [`dispatcher::GpuDispatcher`] pinning actor execution to a single
//!   OS thread.
//! - [`completion::HostFnCompletion`] for sub-microsecond stream
//!   completion via `cuLaunchHostFunc`.
//! - [`stream::PerActorAllocator`] as the default §5.7 strategy.
//! - [`kernel::BlasActor`] performing cuBLAS SGEMM as the canonical
//!   demo.
//!
//! Phases F2–F5 (cuDNN, cuFFT, NCCL, TensorRT, the `PythonGpuBridge`)
//! and the four blueprint sub-crates are deferred.

pub mod completion;
pub mod device;
pub mod dispatcher;
pub mod error;
pub mod gpu_ref;
pub mod graph;
pub mod host;
pub mod kernel;
pub mod memory;
#[cfg(feature = "nccl")]
pub mod multi_device;
pub mod p2p;
pub mod pipeline;
pub mod placement;
pub mod prelude;
pub mod replay;
pub mod stream;
#[cfg(feature = "streams")]
pub mod streams_pipeline;
#[cfg(feature = "telemetry")]
pub mod observability;
