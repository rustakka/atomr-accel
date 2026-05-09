//! `TensorRt` / `TrtEngine` — Python handles around
//! `atomr_accel_tensorrt`.
//!
//! Phase 4.5++ extends the original load + introspection slice with:
//!
//! - `Tensorrt.build_engine_from_onnx(onnx_path, output_path, fp16=…,
//!   int8=…, workspace_bytes=…)` — drive `IBuilder +
//!   nvonnxparser` end-to-end and write the resulting plan to disk.
//! - `engine.execute(inputs, outputs, input_shapes=…)` — bind every
//!   I/O tensor's `CUdeviceptr`, set dynamic shapes, and call
//!   `enqueueV3` on the device's primary `CudaStream`.
//! - `engine.binding_info()` — list per-tensor names + `is_input`
//!   flags via the upstream `io_tensor_name` accessor.
//!
//! Each method gracefully degrades when the wheel was built without
//! `tensorrt-link` / `tensorrt-onnx`: the upstream actor returns
//! `TrtError::NotLinked`, and we surface that as `GpuRuntimeError`
//! whose message contains "libnvinfer not available".
//!
//! ## Boundary notes
//!
//! - Raw `CUdeviceptr` values flow through `GpuRef::raw_device_ptr`
//!   (added in Phase 4.5++ alongside this commit).
//! - The shared `Arc<CudaStream>` flows through the new
//!   `DeviceMsg::SnapshotStream` mailbox; `Device::snapshot_stream`
//!   is a private accessor reused here.
//! - `engine.execute` doesn't currently mint a fresh
//!   `IExecutionContext` per call from a long-lived `TrtActor` —
//!   the synchronous `TrtActor::execute` helper builds + tears down
//!   the context inline. Callers that need pooled contexts can
//!   wrap this surface; the actor message variant exists for that
//!   future case.

#![cfg(feature = "tensorrt")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_tensorrt::{IBuilderConfig, Precision, TrtActor, TrtEngine, TrtRuntime};

use crate::buffer::{
    PyGpuBufferC128, PyGpuBufferC64, PyGpuBufferF32, PyGpuBufferF64, PyGpuBufferI32,
    PyGpuBufferU32, PyGpuBufferU8,
};
use crate::device::PyDevice;
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

    /// Borrow the underlying shared engine.
    pub fn shared(&self) -> Arc<TrtEngine> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTrtEngine {
    /// `True` if the underlying engine pointer is non-null.
    fn is_loaded(&self) -> bool {
        !self.inner.raw().is_null()
    }

    /// Number of input + output tensor bindings on this engine.
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

    /// Phase 4.5++ — list per-tensor names. Each entry is a dict with
    /// `name` (str) and `index` (int). `dtype` / `shape` / `is_input`
    /// land when the upstream FFI shim grows the matching accessors;
    /// today only the name is queryable.
    ///
    /// On wheels without `tensorrt-link` this returns an empty list
    /// (the underlying engine pointer is null in that branch — the
    /// load_engine path returned a clean `NotLinked` error before
    /// any `PyTrtEngine` was ever constructed).
    fn binding_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let n = self.inner.num_io_tensors();
        let list = PyList::empty_bound(py);
        for i in 0..n {
            let d = PyDict::new_bound(py);
            d.set_item("index", i)?;
            match self.inner.io_tensor_name(i) {
                Some(name) => d.set_item("name", name)?,
                None => d.set_item("name", py.None())?,
            }
            list.append(d)?;
        }
        Ok(list)
    }

    /// Phase 4.5++ — run inference on this engine.
    ///
    /// `inputs` and `outputs` map tensor name → device buffer
    /// (`GpuBufferF32` only for now; the FFI shim is dtype-blind once
    /// it has the raw `CUdeviceptr`, but typing the Python surface
    /// against a single buffer class keeps the wrapper short — typed
    /// dispatchers can land later).
    ///
    /// `input_shapes` is an optional `dict[str, list[int]]` for engines
    /// with dynamic input shapes; entries are forwarded to
    /// `IExecutionContext::setInputShape` before tensor address binding.
    /// Engines with fully-fixed shapes can omit this argument.
    ///
    /// `device` must be the same `Device` that allocated the buffers
    /// — we use its primary `Arc<CudaStream>` for `enqueueV3`.
    ///
    /// Returns `None`. Real GPU completion is observed by the
    /// device's existing completion strategy on the shared stream.
    #[pyo3(signature = (device, inputs, outputs, input_shapes=None, timeout_secs=60.0))]
    fn execute(
        &self,
        py: Python<'_>,
        device: Py<PyDevice>,
        inputs: &Bound<'_, PyDict>,
        outputs: &Bound<'_, PyDict>,
        input_shapes: Option<&Bound<'_, PyDict>>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let device_borrow = device.borrow(py);

        // Step 1: snapshot the device's primary stream.
        let stream = device_borrow
            .snapshot_stream(py, timeout_secs)?
            .ok_or_else(|| errors::map_str("device stream not ready (mock mode or pre-init)"))?;

        // Step 2: collect bindings from inputs + outputs. Each entry
        // is `(tensor_name, CUdeviceptr_u64)`. We accept the dtype-
        // tagged buffer wrappers and pull the raw pointer out of the
        // underlying GpuRef.
        let mut bindings: Vec<(String, u64)> = Vec::with_capacity(inputs.len() + outputs.len());
        collect_bindings(inputs, &mut bindings)?;
        collect_bindings(outputs, &mut bindings)?;

        // Step 3: collect input_shapes (optional).
        let mut shape_pairs: Vec<(String, Vec<i32>)> = Vec::new();
        if let Some(map) = input_shapes {
            for (k, v) in map.iter() {
                let name: String = k.extract()?;
                let dims: Vec<i32> = v.extract()?;
                shape_pairs.push((name, dims));
            }
        }

        // Step 4: drive the synchronous `TrtActor::execute` helper.
        // The underlying call sits inside `py.allow_threads` so the
        // GIL is released for the duration of the FFI sequence.
        let actor = Arc::new(TrtActor::new());
        let engine = self.inner.clone();
        py.allow_threads(move || {
            actor
                .execute(&engine, &bindings, &shape_pairs, &stream)
                .map_err(errors::map_str)
        })
    }
}

/// Helper: pull `(name, raw_device_ptr)` out of a Python dict mapping
/// `str → GpuBuffer*` and append to `out`. Accepts every dtype-tagged
/// buffer class we ship.
fn collect_bindings(map: &Bound<'_, PyDict>, out: &mut Vec<(String, u64)>) -> PyResult<()> {
    for (k, v) in map.iter() {
        let name: String = k.extract()?;
        // Try each typed buffer class until one matches.
        macro_rules! try_cast {
            ($ty:ty, $rust:ty) => {{
                if let Ok(buf) = v.extract::<Py<$ty>>() {
                    let g: GpuRef<$rust> = Python::with_gil(|py| buf.borrow(py).clone_ref())
                        .ok_or_else(|| errors::map_str(format!("buffer {name:?} consumed")))?;
                    let ptr = g.raw_device_ptr().map_err(errors::map_gpu)?;
                    out.push((name, ptr));
                    continue;
                }
            }};
        }
        try_cast!(PyGpuBufferF32, f32);
        try_cast!(PyGpuBufferF64, f64);
        try_cast!(PyGpuBufferI32, i32);
        try_cast!(PyGpuBufferU32, u32);
        try_cast!(PyGpuBufferU8, u8);
        try_cast!(PyGpuBufferC64, atomr_accel_cuda::dtype::C32);
        try_cast!(PyGpuBufferC128, atomr_accel_cuda::dtype::C64);
        return Err(errors::map_str(format!(
            "binding {name:?}: unsupported buffer type (expected GpuBufferF32/F64/I32/U32/U8/C64/C128)"
        )));
    }
    Ok(())
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
    /// `tensorrt-link` feature this is always `False`.
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
    ///   `tensorrt-link` feature.
    /// - `GpuRuntimeError("...")` for IO errors reading the plan
    ///   file.
    /// - `GpuRuntimeError("TensorRT runtime failed: ...")` when
    ///   libnvinfer rejects the plan blob (e.g. version skew, wrong
    ///   GPU arch, corrupt bytes).
    #[staticmethod]
    fn load_engine(py: Python<'_>, path: &str) -> PyResult<PyTrtEngine> {
        let plan_bytes = std::fs::read(path).map_err(|e| {
            errors::map_str(format!("failed to read TensorRT plan file {path:?}: {e}"))
        })?;

        py.allow_threads(move || {
            let runtime = TrtRuntime::new().map_err(errors::map_str)?;
            let engine = runtime.deserialize(&plan_bytes).map_err(errors::map_str)?;
            Ok(PyTrtEngine::new(engine.into_shared()))
        })
    }

    /// Phase 4.5++ — parse an ONNX model + build a TensorRT engine
    /// plan, writing the resulting plan blob to `output_path`.
    ///
    /// Errors:
    /// - `GpuRuntimeError("libnvinfer not available: ...")` when the
    ///   wheel was built without `tensorrt-link` *or* `tensorrt-onnx`.
    /// - `GpuRuntimeError("...")` for IO / parser / build failures.
    ///
    /// Builder knobs (`fp16`, `int8`, `workspace_bytes`) map directly
    /// onto `IBuilderConfig` flags. Future calls can layer richer
    /// config (DLA, refit, calibrator) by accepting an explicit
    /// config dict — this surface keeps the common path one-liner.
    #[staticmethod]
    #[pyo3(signature = (
        onnx_path,
        output_path,
        fp16=false,
        int8=false,
        workspace_bytes=1usize << 30,
        timeout_secs=600.0,
    ))]
    fn build_engine_from_onnx(
        py: Python<'_>,
        onnx_path: &str,
        output_path: &str,
        fp16: bool,
        int8: bool,
        workspace_bytes: usize,
        timeout_secs: f64,
    ) -> PyResult<String> {
        let _ = timeout_secs; // synchronous path; arg reserved for the actor variant
        let onnx_bytes = std::fs::read(onnx_path)
            .map_err(|e| errors::map_str(format!("failed to read ONNX file {onnx_path:?}: {e}")))?;

        let precision = match (fp16, int8) {
            (false, false) => Precision::Fp32,
            (true, false) => Precision::Fp16,
            (false, true) => Precision::Int8,
            (true, true) => Precision::Best,
        };
        let config = IBuilderConfig::new()
            .with_precision(precision)
            .with_workspace_bytes(workspace_bytes);

        let plan = py.allow_threads(move || {
            let actor = TrtActor::new();
            actor
                .build_from_onnx(&onnx_bytes, &config)
                .map_err(errors::map_str)
        })?;

        std::fs::write(output_path, plan.as_slice()).map_err(|e| {
            errors::map_str(format!("failed to write plan to {output_path:?}: {e}"))
        })?;

        Ok(output_path.to_string())
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
