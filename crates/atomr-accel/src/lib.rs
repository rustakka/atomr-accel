//! # atomr-accel
//!
//! Actor-shaped face for compute-acceleration backends, on top of the
//! [atomr](../../atomr) actor runtime. NVIDIA CUDA is the first
//! shipping implementation ([`atomr-accel-cuda`](../atomr_accel_cuda));
//! the same trait surface accommodates AMD ROCm, Apple Metal, Intel
//! oneAPI, and Vulkan compute when those land.
//!
//! ```toml
//! [dependencies]
//! atomr-accel = { version = "0.0", features = ["cuda"] }
//! ```
//!
//! ```ignore
//! use atomr_accel::prelude::*;
//! use atomr_accel::cuda;          // re-export of `atomr-accel-cuda`
//! ```
//!
//! ## What this crate is
//!
//! A **thin core** that names the abstractions every backend has to
//! satisfy:
//!
//! - [`AccelBackend`] — marker trait identifying a backend, with
//!   associated `Device`, `Stream`, `Event`, `Error` types.
//! - [`AccelRef`] — generation-validated typed device pointer
//!   parametric over the backend.
//! - [`AccelError`] — typed error enum, `#[non_exhaustive]` so
//!   backends can add `LibraryError` variants without breaking core.
//! - [`CompletionStrategy`] — async wakeup contract for kernel
//!   completion (host-fn callback, sync, polled).
//! - [`KernelOp`] — marker trait for typed op envelopes (Sgemm,
//!   RngFillUniform, etc.).
//!
//! The core deliberately ships **no concrete actors**. Each backend
//! crate provides its own `DeviceActor`, `KernelActor` family, and
//! library wrappers. The umbrella re-exports the active backend at
//! `atomr_accel::cuda` (and, eventually, `atomr_accel::rocm`,
//! `atomr_accel::metal`, etc.) so users have one stable import path.
//!
//! ## What this crate is not
//!
//! - A least-common-denominator API. Backends expose more than the
//!   trait surface — `atomr_accel::cuda::kernel::CudnnActor` has a
//!   richer message set than `KernelOp` knows about, and that's
//!   fine. The trait surface is for portable code; backend-specific
//!   work uses the concrete crate directly.
//! - A device-abstraction layer like wgpu or SYCL. We don't try to
//!   compile one shader to many targets. We supervise the right
//!   library on the right hardware.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod backend;
pub mod completion;
pub mod error;
pub mod gpu_ref;
pub mod kernel;

pub use backend::{AccelBackend, AccelDevice, AccelStream};
pub use completion::CompletionStrategy;
pub use error::AccelError;
pub use gpu_ref::AccelRef;
pub use kernel::KernelOp;

/// Re-export the CUDA backend at a stable path. Active when the
/// `cuda` feature is on.
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub use atomr_accel_cuda as cuda;

pub mod prelude {
    //! Canonical re-exports. `use atomr_accel::prelude::*;`.
    pub use crate::backend::{AccelBackend, AccelDevice, AccelStream};
    pub use crate::completion::CompletionStrategy;
    pub use crate::error::{AccelError, AccelResult};
    pub use crate::gpu_ref::AccelRef;
    pub use crate::kernel::KernelOp;
}
