//! Record-side `GraphOp` adapters for each kernel actor.
//!
//! These types wrap a typed kernel-actor request and call the
//! kernel's existing record-mode (capture-safe) entry point against
//! the captured stream. They're designed to be inserted into a
//! [`crate::graph::GraphMsg::Record`] script alongside the
//! Phase 0.5 `GraphOp` variants.
//!
//! Submodule layout:
//! - [`cudnn`] — `ConvForwardOp` / `ActivationOp` / `SoftmaxOp`
//!   (gated `cudnn`)
//! - [`cusparse`] — `SpMvOp` / `SpMmOp` (gated `cusparse`)
//! - [`cutensor`] — `ContractOp` (gated `cutensor`)
//! - [`nccl`] — `AllReduceOp` / `BroadcastOp` (gated `nccl`)
//! - [`nvrtc`] — `LaunchOp` (gated `nvrtc`)

#[cfg(feature = "cudnn")]
pub mod cudnn;
#[cfg(feature = "cusparse")]
pub mod cusparse;
#[cfg(feature = "cutensor")]
pub mod cutensor;
#[cfg(feature = "nccl")]
pub mod nccl;
#[cfg(feature = "nvrtc")]
pub mod nvrtc;
