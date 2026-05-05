//! # atomr-accel-telemetry
//!
//! GPU observability for `atomr-accel`. Three independent feature
//! flags slice the surface so consumers pay only for what they use:
//!
//! * `nvtx`  — [`nvtx::NvtxKernelTrace`], an implementation of the
//!   [`KernelTrace`] hook that pushes / pops NVTX ranges around every
//!   kernel enqueue. Wired through `atomr-accel-cuda`'s kernel
//!   envelope (the `nvtx-trace` feature on that crate).
//! * `nvml`  — [`nvml::NvmlActor`], polling `libnvidia-ml.so` for
//!   power, temperature, clocks, ECC, throttle reasons, PCIe Tx/Rx,
//!   memory utilisation, processes, and MIG configuration. Each
//!   metric is pushed into atomr-telemetry as a probe via
//!   [`nvml::probes::register_all`].
//! * `cupti` — [`cupti::CuptiSession`], a Tokio actor that drives
//!   CUPTI's activity API + range profiler. Activity records flow
//!   through an mpsc channel; `Drain` collects them.
//!
//! The crate is opt-in; with no features enabled the build is a
//! no-op. With every feature enabled it pulls cudarc's `nvtx`/`cupti`
//! bindings and `libloading`.
//!
//! ## Forward compatibility with `atomr-accel-cuda::kernel::envelope`
//!
//! Phase 0.7 of the CUDA-coverage roadmap lands `KernelTrace` +
//! `KernelInfo` inside
//! `atomr_accel_cuda::kernel::envelope`. Until that change merges,
//! the trait + struct live here so this crate compiles standalone.
//! When Phase 0.7 is wired in, this module continues to compile —
//! the local definitions stay as the canonical home for trace impls
//! and re-export from `atomr-accel-cuda` is additive.

#![deny(rust_2018_idioms)]
#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

pub mod trace;

#[cfg(feature = "nvtx")]
pub mod nvtx;

#[cfg(feature = "nvml")]
pub mod nvml;

#[cfg(feature = "cupti")]
pub mod cupti;

pub use trace::{KernelInfo, KernelTrace, NoopKernelTrace};

/// Convenience alias for shared trace handles. Most callers store a
/// `SharedTrace` on their actor state and clone it into the
/// completion task.
pub type SharedTrace = Arc<dyn KernelTrace>;
