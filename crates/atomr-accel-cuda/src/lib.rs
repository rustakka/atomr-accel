//! # atomr-accel-cuda
//!
//! GPU acceleration via the actor model. Wraps NVIDIA CUDA libraries as
//! actors on top of [`atomr`](../atomr). See `README.md` and the
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

// Subjective clippy lints that fight the actor-message design:
// * `type_complexity` — actor messages and kernel envelopes return
//   tuples of typed `Arc<CudaSlice<T>>` keep-alives; refactoring to
//   `type` aliases would worsen the public API.
// * `too_many_arguments` — kernel-launcher fns mirror the underlying
//   CUDA library entry points (cuDNN conv, cuSPARSE SpMV) which take
//   8–10 args; collapsing to a config struct just moves the fields.
// * `arc_with_non_send_sync` — CUDA driver handles (CudaGraph,
//   cudnnHandle) are `!Send` by design and only ever shared inside
//   the producing actor.
// * `large_enum_variant` — kernel-message enums have one large
//   conv-descriptor variant; boxing it would fragment the hot path.
#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::arc_with_non_send_sync,
    clippy::large_enum_variant
)]

pub mod completion;
pub mod device;
pub mod dispatcher;
pub mod dtype;
pub mod error;
pub mod gpu_ref;
pub mod graph;
pub mod host;
pub mod kernel;
pub mod memory;
pub mod sys;
#[cfg(feature = "nccl")]
pub mod multi_device;
#[cfg(feature = "telemetry")]
pub mod observability;
pub mod p2p;
pub mod pipeline;
pub mod placement;
pub mod prelude;
pub mod replay;
pub mod stream;
#[cfg(feature = "streams")]
pub mod streams_pipeline;
