//! # atomr-accel-tensorrt
//!
//! TensorRT engine builder + runtime as supervised atomr actors.
//! Wraps NVIDIA's libnvinfer (and optionally libnvonnxparser) at
//! runtime — the library itself is **not** vendored because it is
//! proprietary; users opt in via the `tensorrt-link` feature and
//! either install TensorRT system-wide or set `LIBNVINFER_PATH`.
//!
//! ## Features
//!
//! - `tensorrt-link` — actually link libnvinfer at build time.
//!   Off-by-default so the crate compiles on hosts without
//!   TensorRT (used by CI + unit tests).
//! - `tensorrt-onnx` — pull in `nvonnxparser` for ONNX import.
//! - `tensorrt-plugin` — `IPluginV3` Rust trampolines.
//! - `tensorrt-int8` — INT8 calibration helpers (entropy / minmax).
//! - `tensorrt-fp8` — FP8 PTQ helpers (Hopper-class GPUs).
//!
//! ## Public surface
//!
//! - [`actor::TrtActor`] / [`actor::TrtMsg`] — sibling actor to
//!   `atomr_accel_cuda::DeviceActor`. Shares `Arc<CudaStream>` with
//!   the device actor so inference rides the same execution
//!   timeline.
//! - [`builder::IBuilderConfig`] — pure-Rust mirror of the TensorRT
//!   builder config, with knobs for precision, DLA, structured
//!   sparsity, tactic sources, timing cache, and engine refit.
//! - [`engine::TrtEngine`] — owned, immutable engine handle that's
//!   `Send + Sync` via newtype.
//! - [`runtime::TrtRuntime`] / [`runtime::ExecutionContext`] — load
//!   serialised plans + drive `enqueueV3` on a shared CUDA stream.
//! - [`onnx::OnnxParser`] — gated on `tensorrt-onnx`.
//! - [`calibration`] — gated on `tensorrt-int8` / `tensorrt-fp8`.
//! - [`plugin`] — gated on `tensorrt-plugin`.

#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::arc_with_non_send_sync
)]

pub mod actor;
pub mod builder;
pub mod engine;
pub mod error;
pub mod runtime;
pub mod sys;

#[cfg(feature = "tensorrt-onnx")]
pub mod onnx;

#[cfg(feature = "tensorrt-int8")]
pub mod calibration;

#[cfg(feature = "tensorrt-plugin")]
pub mod plugin;

pub use actor::{
    BuildFromOnnxReply, BuildReply, CreateContextReply, DeserializeReply, EnqueueReply,
    ExecuteReply, NetworkSource, RefitReply, RefitWeights, TrtActor, TrtMsg,
};
pub use builder::{
    BuilderFlags, DeviceType, IBuilderConfig, Precision, RefitPolicy, TacticSources,
};
pub use engine::{EnginePlan, TrtEngine, TrtRefitter};
pub use error::TrtError;
pub use runtime::{EnqueueRequest, ExecutionBindings, ExecutionContext, TensorShape, TrtRuntime};
