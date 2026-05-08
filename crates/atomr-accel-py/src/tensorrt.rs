//! `TensorRt` / `TrtEngine` — Python handles around
//! `atomr_accel_tensorrt`.
//!
//! Phase 4.5 ships the load + introspection slice of the TensorRT
//! surface. The wrapped crate's `TrtMsg` mailbox is design-only (no
//! actor run loop ships in `atomr-accel-tensorrt`), so the typed
//! `Build` / `EnqueueOnStream` paths can't ride the actor; instead we
//! drive the synchronous `TrtRuntime` API directly. That gives us:
//!
//! - `TensorRt.load_engine(path)` — read a serialised plan file, hand
//!   it to `TrtRuntime::deserialize`, and hand back an opaque
//!   `TrtEngine` Python class wrapping the resulting
//!   `Arc<TrtEngine>`.
//! - `TensorRt.runtime_ready()` — feature probe (Phase 4 anchor;
//!   preserved).
//! - `TrtEngine.is_loaded()` / `num_io_tensors` / `__repr__` — opaque
//!   handle introspection.
//!
//! Gaps (documented; tracked in the Phase 4.5 TensorRT issue):
//!
//! - **No `build_engine_from_onnx`.** The upstream `TrtActor::Build`
//!   path is design-only — there is no actor run loop in
//!   `atomr-accel-tensorrt`, and the `IBuilder` / `nvonnxparser` FFI
//!   shims are gated behind the upstream `tensorrt-link` +
//!   `tensorrt-onnx` cargo features that don't pass through to this
//!   crate. Wiring this requires either (a) an upstream `TrtRuntime`
//!   safe wrapper for the build/parse path, or (b) a feature
//!   pass-through on `atomr-accel-py` itself. Both are out of scope
//!   for the Phase 4.5 deliverable.
//! - **No `engine.execute(...)`.** Inference needs three things the
//!   PyO3 boundary doesn't surface today: an `Arc<CudaStream>` from
//!   `DeviceActor` (no Python accessor), raw `CUdeviceptr` values
//!   from `GpuBuffer*` handles (cudarc keeps `cu_device_ptr`
//!   crate-private), and `IExecutionContext::enqueueV3` access
//!   (gated on `tensorrt-link`). Closing this gap is the next
//!   tracked Phase 4.5 task.
//! - **No `binding_info()` dtype/shape.** The upstream `sys`
//!   declarations only expose `num_io_tensors` + `io_tensor_name`;
//!   tensor dtype, shape, and direction queries aren't on the C-ABI
//!   shim yet. `num_io_tensors` is exposed; per-tensor names follow
//!   when the shim grows.
//!
//! What ships today is the smallest cleanly-typed slice that survives
//! the PyO3 boundary on hosts *without* libnvinfer (graceful
//! `NotLinked` errors) and on hosts *with* libnvinfer (real
//! deserialise via the upstream safe wrapper).

#![cfg(feature = "tensorrt")]

use std::sync::Arc;

use pyo3::prelude::*;

use atomr_accel_tensorrt::{TrtActor, TrtEngine, TrtRuntime};

use crate::errors;

/// Opaque Python handle wrapping `Arc<TrtEngine>`.
///
/// Engines are immutable post-build (the upstream `TrtEngine` is
/// `Send + Sync` via its newtype; multiple `IExecutionContext`s can
/// share it). On the Python side this class is intentionally narrow:
/// it carries the `Arc` so callers can pass it back into future
/// `engine.execute(...)` / `engine.refit(...)` calls when those land,
/// and it exposes the cached `num_io_tensors` count for sanity
/// checks.
#[pyclass(name = "TrtEngine", module = "atomr_accel._native")]
pub struct PyTrtEngine {
    inner: Arc<TrtEngine>,
}

impl PyTrtEngine {
    pub fn new(engine: Arc<TrtEngine>) -> Self {
        Self { inner: engine }
    }

    /// Borrow the underlying shared engine. Exposed for future
    /// methods on `PyTensorRt` (e.g. `execute`) that take an engine
    /// argument.
    pub fn shared(&self) -> Arc<TrtEngine> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTrtEngine {
    /// `True` if the underlying engine pointer is non-null. With
    /// `tensorrt-link` off the upstream `TrtRuntime::deserialize`
    /// returns `NotLinked` before we ever construct a `TrtEngine`,
    /// so any handle that *does* exist on the Python side is loaded
    /// by construction. The probe is kept so callers can write
    /// defensive code that mirrors the C++ surface.
    fn is_loaded(&self) -> bool {
        !self.inner.raw().is_null()
    }

    /// Number of input + output tensor bindings on this engine.
    /// Cached at deserialise time from
    /// `atomr_trt_engine_num_io_tensors`.
    #[getter]
    fn num_io_tensors(&self) -> usize {
        self.inner.num_io_tensors()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrtEngine(loaded={}, io_tensors={})",
            !self.inner.raw().is_null(),
            self.inner.num_io_tensors(),
        )
    }
}

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

    /// Load a previously-serialised TensorRT plan file from disk and
    /// return an opaque [`PyTrtEngine`] handle.
    ///
    /// Errors:
    /// - `GpuRuntimeError("libnvinfer not available: ...")` when the
    ///   wheel's `atomr-accel-tensorrt` was built without the
    ///   `tensorrt-link` feature (graceful — no panic, no
    ///   segfault).
    /// - `GpuRuntimeError("...")` for IO errors reading the plan
    ///   file.
    /// - `GpuRuntimeError("TensorRT runtime failed: ...")` when
    ///   libnvinfer rejects the plan blob (e.g. version skew, wrong
    ///   GPU arch, corrupt bytes).
    ///
    /// The runtime is constructed *per call* on the synchronous
    /// `TrtRuntime` path rather than reused from the actor's cached
    /// runtime, because the actor's mailbox doesn't drive a run loop
    /// in the current upstream skeleton. The deserialise itself runs
    /// under `py.allow_threads(...)` so the GIL is released for the
    /// duration of the FFI call.
    #[staticmethod]
    fn load_engine(py: Python<'_>, path: &str) -> PyResult<PyTrtEngine> {
        // Read the plan file on the Python thread; cheap, IO-bound,
        // and lets us surface `FileNotFoundError`-shaped errors
        // before we ever talk to libnvinfer.
        let plan_bytes = std::fs::read(path).map_err(|e| {
            errors::map_str(format!("failed to read TensorRT plan file {path:?}: {e}"))
        })?;

        py.allow_threads(move || {
            let runtime = TrtRuntime::new().map_err(errors::map_str)?;
            let engine = runtime.deserialize(&plan_bytes).map_err(errors::map_str)?;
            Ok(PyTrtEngine::new(engine.into_shared()))
        })
    }

    fn __repr__(&self) -> &'static str {
        "TensorRt(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensorRt>()?;
    m.add_class::<PyTrtEngine>()?;
    Ok(())
}
