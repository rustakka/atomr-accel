//! `TensorRt` — Python handle wrapping `atomr_accel_tensorrt::TrtActor`.
//!
//! Phase 4 ships this handle as a structural anchor only. The
//! TensorRT actor's mailbox (`TrtMsg`) is dominated by request types
//! that don't currently round-trip through PyO3 cleanly:
//!
//!   - `Build` takes a boxed `IBuilderConfig` plus a `NetworkSource`
//!     enum whose `Onnx(Vec<u8>)` variant is gated behind the
//!     `tensorrt-onnx` feature on the wrapped crate.
//!   - `Deserialize` returns an `Arc<TrtEngine>` whose lifetime is
//!     entangled with the actor's cached runtime.
//!   - `EnqueueOnStream` requires an `Arc<CudaStream>` borrowed from
//!     `atomr-accel-cuda::DeviceActor` — exposing it would mean
//!     marshalling raw CUDA stream handles across the language
//!     boundary, which Phase 4 doesn't tackle.
//!
//! What *does* work today: probing whether the host has libnvinfer
//! linked in (via `tensorrt-link` on the wrapped crate). We expose a
//! single `runtime_ready` method that asks the actor to lazily
//! construct its runtime. Without `tensorrt-link` it returns `False`;
//! with the feature, the answer reflects whether libnvinfer.so was
//! found at link time. This gives Python callers a way to feature-
//! detect TensorRT without crashing and is the recommended entry point
//! before any Phase 4.5 typed methods land.
//!
//! Build / Deserialize / CreateContext / EnqueueOnStream / Refit
//! follow in the Phase 4.5 TensorRT tracking issue.
//
// TODO Phase 4.5: typed `build_from_onnx`, `deserialize_plan`,
// `create_context`, `enqueue_on_stream`, `refit` once the upstream
// crate exposes engine handles in a shape that survives the PyO3
// boundary.

#![cfg(feature = "tensorrt")]

use std::sync::Arc;

use pyo3::prelude::*;

use atomr_accel_tensorrt::TrtActor;

#[pyclass(name = "TensorRt", module = "atomr_accel._native")]
pub struct PyTensorRt {
    actor: Arc<TrtActor>,
}

impl PyTensorRt {
    pub fn new(actor: Arc<TrtActor>) -> Self {
        Self { actor }
    }
}

#[pymethods]
impl PyTensorRt {
    /// Lazily construct the TensorRT runtime and return whether the
    /// link succeeded. On wheels built without the upstream
    /// `tensorrt-link` feature this is always `False` (the runtime
    /// constructor returns `TrtError::NotLinked`); with the feature
    /// on, the answer reflects whether `libnvinfer.so` was located at
    /// runtime.
    fn runtime_ready(&self, py: Python<'_>) -> bool {
        let actor = self.actor.clone();
        py.allow_threads(|| actor.ensure_runtime().is_ok())
    }

    fn __repr__(&self) -> &'static str {
        "TensorRt(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensorRt>()?;
    Ok(())
}
