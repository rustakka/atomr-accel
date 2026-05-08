//! `Device` — Python wrapper around `ActorRef<DeviceMsg>`.
//!
//! Every method here is **blocking from the Python side** (the actor
//! reply is awaited on the shared tokio runtime via
//! `Runtime::block_on`). This keeps Python code straight-line and
//! GIL-friendly; long calls release the GIL through
//! `py.allow_threads`. Async wrappers can be layered in later by
//! returning the underlying tokio future via
//! `pyo3_async_runtimes::tokio::future_into_py`.
#![allow(deprecated)]

use std::sync::Arc;
use std::time::Duration;

use numpy::{Element, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::device::{DeviceLoad, DeviceMsg, HostBuf, KernelChildren, SgemmRequest};
use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::{BlasMsg, GemmRequest};
use atomr_core::actor::ActorRef;

use crate::buffer::{
    PyGpuBufferF32, PyGpuBufferF64, PyGpuBufferI32, PyGpuBufferU32, PyGpuBufferU8,
};
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Device", module = "atomr_accel._native")]
pub struct PyDevice {
    pub(crate) actor_ref: ActorRef<DeviceMsg>,
    pub(crate) device_id: u32,
}

impl PyDevice {
    pub fn new(actor_ref: ActorRef<DeviceMsg>, device_id: u32) -> Self {
        Self {
            actor_ref,
            device_id,
        }
    }

    /// Snapshot the kernel children (cuBLAS / cuDNN / cuFFT / cuRAND
    /// actor refs). Returns `None` if the context isn't ready yet.
    /// Used internally by `Device.cudnn()`, `.fft()`, etc. handle
    /// constructors.
    pub(crate) fn snapshot_children(
        &self,
        py: Python<'_>,
        timeout_secs: f64,
    ) -> PyResult<Option<KernelChildren>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(DeviceMsg::SnapshotChildren { reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(children)) => Ok(children),
                    Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                    Err(_) => Err(errors::map_str("snapshot_children timed out")),
                }
            })
        })
    }
}

#[pymethods]
impl PyDevice {
    #[getter]
    fn device_id(&self) -> u32 {
        self.device_id
    }

    // ─── Allocation: one method per supported dtype ──────────────

    /// Allocate `len` `f32` elements on-device. Returns a `GpuBufferF32`.
    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_f32(
        &self,
        py: Python<'_>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGpuBufferF32>> {
        let g = ask_alloc::<f32>(py, &self.actor_ref, len, timeout_secs)?;
        Py::new(py, PyGpuBufferF32::new(g))
    }

    /// Allocate `len` `f64` elements on-device. Returns a `GpuBufferF64`.
    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_f64(
        &self,
        py: Python<'_>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGpuBufferF64>> {
        let g = ask_alloc::<f64>(py, &self.actor_ref, len, timeout_secs)?;
        Py::new(py, PyGpuBufferF64::new(g))
    }

    /// Allocate `len` `i32` elements on-device. Returns a `GpuBufferI32`.
    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_i32(
        &self,
        py: Python<'_>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGpuBufferI32>> {
        let g = ask_alloc::<i32>(py, &self.actor_ref, len, timeout_secs)?;
        Py::new(py, PyGpuBufferI32::new(g))
    }

    /// Allocate `len` `u32` elements on-device. Returns a `GpuBufferU32`.
    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_u32(
        &self,
        py: Python<'_>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGpuBufferU32>> {
        let g = ask_alloc::<u32>(py, &self.actor_ref, len, timeout_secs)?;
        Py::new(py, PyGpuBufferU32::new(g))
    }

    /// Allocate `len` `u8` elements on-device. Returns a `GpuBufferU8`.
    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_u8(
        &self,
        py: Python<'_>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGpuBufferU8>> {
        let g = ask_alloc::<u8>(py, &self.actor_ref, len, timeout_secs)?;
        Py::new(py, PyGpuBufferU8::new(g))
    }

    // ─── Host ↔ device copies (typed) ────────────────────────────

    /// Upload a numpy `float32` array into a device buffer (f32).
    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy(
        &self,
        py: Python<'_>,
        dst: Py<PyGpuBufferF32>,
        src: PyReadonlyArray1<'_, f32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        copy_from_numpy_typed::<f32, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    /// Upload a numpy `float64` array into a device buffer (f64).
    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_f64(
        &self,
        py: Python<'_>,
        dst: Py<PyGpuBufferF64>,
        src: PyReadonlyArray1<'_, f64>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        copy_from_numpy_typed::<f64, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    /// Upload a numpy `int32` array into a device buffer (i32).
    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_i32(
        &self,
        py: Python<'_>,
        dst: Py<PyGpuBufferI32>,
        src: PyReadonlyArray1<'_, i32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        copy_from_numpy_typed::<i32, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    /// Upload a numpy `uint32` array into a device buffer (u32).
    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_u32(
        &self,
        py: Python<'_>,
        dst: Py<PyGpuBufferU32>,
        src: PyReadonlyArray1<'_, u32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        copy_from_numpy_typed::<u32, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    /// Upload a numpy `uint8` array into a device buffer (u8).
    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_u8(
        &self,
        py: Python<'_>,
        dst: Py<PyGpuBufferU8>,
        src: PyReadonlyArray1<'_, u8>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        copy_from_numpy_typed::<u8, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    /// Download an f32 device buffer into a fresh numpy array.
    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        copy_to_numpy_with::<f32, PyGpuBufferF32>(py, &self.actor_ref, &src, timeout_secs)
    }

    /// Download an f64 device buffer into a fresh numpy array.
    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_f64<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferF64>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        copy_to_numpy_with::<f64, PyGpuBufferF64>(py, &self.actor_ref, &src, timeout_secs)
    }

    /// Download an i32 device buffer into a fresh numpy array.
    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_i32<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferI32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        copy_to_numpy_with::<i32, PyGpuBufferI32>(py, &self.actor_ref, &src, timeout_secs)
    }

    /// Download a u32 device buffer into a fresh numpy array.
    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_u32<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferU32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        copy_to_numpy_with::<u32, PyGpuBufferU32>(py, &self.actor_ref, &src, timeout_secs)
    }

    /// Download a u8 device buffer into a fresh numpy array.
    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_u8<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferU8>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        copy_to_numpy_with::<u8, PyGpuBufferU8>(py, &self.actor_ref, &src, timeout_secs)
    }

    // ─── BLAS: SGEMM (legacy alias) + typed gemm ─────────────────

    /// SGEMM (f32): `c = alpha * a · b + beta * c`. `m × k`, `k × n`,
    /// `m × n` (column-major). Kept as an alias for back-compat;
    /// `gemm_f32` is equivalent.
    #[pyo3(signature = (a, b, c, m, n, k, alpha=1.0, beta=0.0, timeout_secs=60.0))]
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
                    a,
                    b,
                    c,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    reply: tx,
                })));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                    Err(_) => Err(errors::map_str("sgemm timed out")),
                }
            })
        })
    }

    // ─── Per-device probes / lifecycle ───────────────────────────

    /// Pull the latest device load snapshot (free VRAM, queue depth,
    /// active streams).
    #[pyo3(signature = (timeout_secs=2.0))]
    fn stats(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<DeviceLoadDict> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        let load: DeviceLoad = py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(DeviceMsg::Stats { reply: tx });
                tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx)
                    .await
                    .map_err(|_| errors::map_str("stats timed out"))?
                    .map_err(|_| errors::map_str("device dropped reply"))
            })
        })?;
        Ok(DeviceLoadDict {
            free_bytes: load.free_bytes as u64,
            total_bytes: load.total_bytes as u64,
            active_streams: load.active_streams,
            queue_depth: load.queue_depth,
            compute_cap_major: load.compute_cap.0,
            compute_cap_minor: load.compute_cap.1,
        })
    }

    /// Probe whether the kernel children (cuBLAS / cuDNN / cuFFT /
    /// cuRAND actor refs) are ready. Returns a dict mapping the
    /// library name to `True` if its actor was spawned. In mock mode
    /// this is empty; on real hardware it reflects the active
    /// `EnabledLibraries` set.
    #[pyo3(signature = (timeout_secs=2.0))]
    fn libraries_ready<'py>(
        &self,
        py: Python<'py>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let kc = self.snapshot_children(py, timeout_secs)?;
        let dict = pyo3::types::PyDict::new_bound(py);
        match kc {
            Some(children) => {
                dict.set_item("blas", true)?;
                #[cfg(feature = "cudnn")]
                {
                    dict.set_item("cudnn", children.cudnn.is_some())?;
                }
                #[cfg(not(feature = "cudnn"))]
                {
                    let _ = &children;
                    dict.set_item("cudnn", false)?;
                }
                #[cfg(feature = "cufft")]
                {
                    dict.set_item("cufft", children.fft.is_some())?;
                }
                #[cfg(not(feature = "cufft"))]
                {
                    dict.set_item("cufft", false)?;
                }
                #[cfg(feature = "curand")]
                {
                    dict.set_item("curand", children.rng.is_some())?;
                }
                #[cfg(not(feature = "curand"))]
                {
                    dict.set_item("curand", false)?;
                }
                #[cfg(feature = "cusolver")]
                {
                    dict.set_item("cusolver", children.solver.is_some())?;
                }
                #[cfg(not(feature = "cusolver"))]
                {
                    dict.set_item("cusolver", false)?;
                }
                dict.set_item("extras", children.extras_len())?;
            }
            None => {
                dict.set_item("blas", false)?;
                dict.set_item("cudnn", false)?;
                dict.set_item("cufft", false)?;
                dict.set_item("curand", false)?;
                dict.set_item("cusolver", false)?;
                dict.set_item("extras", 0usize)?;
                dict.set_item("ready", false)?;
            }
        }
        Ok(dict)
    }

    // ─── Kernel-actor handles ────────────────────────────────────

    /// Borrow the `Blas` handle. Returns the wrapper class on success
    /// (the actor exists once the context is ready). Raises
    /// `GpuRuntimeError` when the device's children haven't initialized
    /// yet — call `libraries_ready()` to probe first.
    #[pyo3(signature = (timeout_secs=2.0))]
    fn blas(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<Py<crate::blas::PyBlas>> {
        let kc = self
            .snapshot_children(py, timeout_secs)?
            .ok_or_else(|| errors::map_str("device children not ready"))?;
        Py::new(py, crate::blas::PyBlas::new(kc.blas.clone()))
    }

    /// Borrow the `Cudnn` handle. Requires the `cudnn` cargo feature
    /// at build time *and* `EnabledLibraries::CUDNN` on this device.
    #[cfg(feature = "cudnn")]
    #[pyo3(signature = (timeout_secs=2.0))]
    fn cudnn(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<Py<crate::cudnn::PyCudnn>> {
        let kc = self
            .snapshot_children(py, timeout_secs)?
            .ok_or_else(|| errors::map_str("device children not ready"))?;
        let h = kc
            .cudnn
            .clone()
            .ok_or_else(|| errors::map_str("cuDNN actor not enabled on this device"))?;
        Py::new(py, crate::cudnn::PyCudnn::new(h))
    }

    /// Borrow the `Fft` handle. Requires the `cufft` cargo feature at
    /// build time *and* `EnabledLibraries::CUFFT` on this device.
    #[cfg(feature = "cufft")]
    #[pyo3(signature = (timeout_secs=2.0))]
    fn fft(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<Py<crate::fft::PyFft>> {
        let kc = self
            .snapshot_children(py, timeout_secs)?
            .ok_or_else(|| errors::map_str("device children not ready"))?;
        let h = kc
            .fft
            .clone()
            .ok_or_else(|| errors::map_str("cuFFT actor not enabled on this device"))?;
        Py::new(py, crate::fft::PyFft::new(h, self.actor_ref.clone()))
    }

    /// Borrow the `RngGenerator` handle. Requires the `curand` cargo
    /// feature at build time *and* `EnabledLibraries::CURAND` on this
    /// device.
    #[cfg(feature = "curand")]
    #[pyo3(signature = (timeout_secs=2.0))]
    fn rng(
        &self,
        py: Python<'_>,
        timeout_secs: f64,
    ) -> PyResult<Py<crate::rng::PyRngGenerator>> {
        let kc = self
            .snapshot_children(py, timeout_secs)?
            .ok_or_else(|| errors::map_str("device children not ready"))?;
        let h = kc
            .rng
            .clone()
            .ok_or_else(|| errors::map_str("cuRAND actor not enabled on this device"))?;
        Py::new(py, crate::rng::PyRngGenerator::new(h))
    }

    /// Borrow the `Solver` handle. Requires the `cusolver` cargo
    /// feature at build time *and* `EnabledLibraries::CUSOLVER` on
    /// this device.
    #[cfg(feature = "cusolver")]
    #[pyo3(signature = (timeout_secs=2.0))]
    fn solver(
        &self,
        py: Python<'_>,
        timeout_secs: f64,
    ) -> PyResult<Py<crate::solver::PySolver>> {
        let kc = self
            .snapshot_children(py, timeout_secs)?
            .ok_or_else(|| errors::map_str("device children not ready"))?;
        let h = kc
            .solver
            .clone()
            .ok_or_else(|| errors::map_str("cuSOLVER actor not enabled on this device"))?;
        Py::new(py, crate::solver::PySolver::new(h))
    }

    fn __repr__(&self) -> String {
        format!("Device(id={})", self.device_id)
    }

    // ─── Async (asyncio) variants ────────────────────────────────
    //
    // Each `_async` method mirrors its blocking counterpart but
    // returns a Python awaitable via
    // `pyo3_async_runtimes::tokio::future_into_py`. The synchronous
    // setup (cloning ActorRef / extracting `GpuRef<T>` from the
    // Python buffer wrappers) happens before entering the async
    // block — the actor pipeline runs without holding the GIL.

    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_f32_async<'py>(
        &self,
        py: Python<'py>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let g = ask_alloc_async::<f32>(actor, len, timeout_secs).await?;
            Python::with_gil(|py| Py::new(py, PyGpuBufferF32::new(g)))
        })
    }

    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_f64_async<'py>(
        &self,
        py: Python<'py>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let g = ask_alloc_async::<f64>(actor, len, timeout_secs).await?;
            Python::with_gil(|py| Py::new(py, PyGpuBufferF64::new(g)))
        })
    }

    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_i32_async<'py>(
        &self,
        py: Python<'py>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let g = ask_alloc_async::<i32>(actor, len, timeout_secs).await?;
            Python::with_gil(|py| Py::new(py, PyGpuBufferI32::new(g)))
        })
    }

    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_u32_async<'py>(
        &self,
        py: Python<'py>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let g = ask_alloc_async::<u32>(actor, len, timeout_secs).await?;
            Python::with_gil(|py| Py::new(py, PyGpuBufferU32::new(g)))
        })
    }

    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_u8_async<'py>(
        &self,
        py: Python<'py>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let g = ask_alloc_async::<u8>(actor, len, timeout_secs).await?;
            Python::with_gil(|py| Py::new(py, PyGpuBufferU8::new(g)))
        })
    }

    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_async<'py>(
        &self,
        py: Python<'py>,
        dst: Py<PyGpuBufferF32>,
        src: PyReadonlyArray1<'_, f32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_from_numpy_async_typed::<f32, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_f64_async<'py>(
        &self,
        py: Python<'py>,
        dst: Py<PyGpuBufferF64>,
        src: PyReadonlyArray1<'_, f64>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_from_numpy_async_typed::<f64, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_i32_async<'py>(
        &self,
        py: Python<'py>,
        dst: Py<PyGpuBufferI32>,
        src: PyReadonlyArray1<'_, i32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_from_numpy_async_typed::<i32, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_u32_async<'py>(
        &self,
        py: Python<'py>,
        dst: Py<PyGpuBufferU32>,
        src: PyReadonlyArray1<'_, u32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_from_numpy_async_typed::<u32, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy_u8_async<'py>(
        &self,
        py: Python<'py>,
        dst: Py<PyGpuBufferU8>,
        src: PyReadonlyArray1<'_, u8>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_from_numpy_async_typed::<u8, _>(py, &self.actor_ref, &dst, src, timeout_secs)
    }

    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_async<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferF32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_to_numpy_async_with::<f32, PyGpuBufferF32>(py, &self.actor_ref, &src, timeout_secs)
    }

    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_f64_async<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferF64>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_to_numpy_async_with::<f64, PyGpuBufferF64>(py, &self.actor_ref, &src, timeout_secs)
    }

    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_i32_async<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferI32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_to_numpy_async_with::<i32, PyGpuBufferI32>(py, &self.actor_ref, &src, timeout_secs)
    }

    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_u32_async<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferU32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_to_numpy_async_with::<u32, PyGpuBufferU32>(py, &self.actor_ref, &src, timeout_secs)
    }

    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy_u8_async<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBufferU8>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        copy_to_numpy_async_with::<u8, PyGpuBufferU8>(py, &self.actor_ref, &src, timeout_secs)
    }

    #[pyo3(signature = (a, b, c, m, n, k, alpha=1.0, beta=0.0, timeout_secs=60.0))]
    #[allow(clippy::too_many_arguments)]
    fn sgemm_async<'py>(
        &self,
        py: Python<'py>,
        a: Py<PyGpuBufferF32>,
        b: Py<PyGpuBufferF32>,
        c: Py<PyGpuBufferF32>,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let a = a
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("a consumed"))?;
        let b = b
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("b consumed"))?;
        let c = c
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("c consumed"))?;
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
                a,
                b,
                c,
                m,
                n,
                k,
                alpha,
                beta,
                reply: tx,
            })));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("sgemm timed out")),
            }
        })
    }

    #[pyo3(signature = (timeout_secs=2.0))]
    fn stats_async<'py>(
        &self,
        py: Python<'py>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::Stats { reply: tx });
            let load: DeviceLoad = tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx)
                .await
                .map_err(|_| errors::map_str("stats timed out"))?
                .map_err(|_| errors::map_str("device dropped reply"))?;
            Ok(DeviceLoadDict {
                free_bytes: load.free_bytes as u64,
                total_bytes: load.total_bytes as u64,
                active_streams: load.active_streams,
                queue_depth: load.queue_depth,
                compute_cap_major: load.compute_cap.0,
                compute_cap_minor: load.compute_cap.1,
            })
        })
    }
}

/// Plain-data wrapper for `DeviceLoad`. Exposed as a Python class so
/// callers can access fields by name.
#[pyclass(name = "DeviceLoad", module = "atomr_accel._native")]
#[derive(Clone)]
pub struct DeviceLoadDict {
    #[pyo3(get)]
    pub free_bytes: u64,
    #[pyo3(get)]
    pub total_bytes: u64,
    #[pyo3(get)]
    pub active_streams: u32,
    #[pyo3(get)]
    pub queue_depth: u32,
    #[pyo3(get)]
    pub compute_cap_major: i32,
    #[pyo3(get)]
    pub compute_cap_minor: i32,
}

#[pymethods]
impl DeviceLoadDict {
    fn __repr__(&self) -> String {
        format!(
            "DeviceLoad(free={}, total={}, streams={}, queue={}, sm={}.{})",
            self.free_bytes,
            self.total_bytes,
            self.active_streams,
            self.queue_depth,
            self.compute_cap_major,
            self.compute_cap_minor
        )
    }
}

// ─── Internal helpers ────────────────────────────────────────────

fn ask_alloc<T>(
    py: Python<'_>,
    actor: &ActorRef<DeviceMsg>,
    len: usize,
    timeout_secs: f64,
) -> PyResult<GpuRef<T>>
where
    T: CudaDtype,
{
    let actor = actor.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::alloc::<T>(len, tx));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(g))) => Ok(g),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("allocate timed out")),
            }
        })
    })
}

/// Async counterpart of [`ask_alloc`]. The caller is responsible for
/// having already cloned the `ActorRef` and extracted any other
/// non-Send args from the GIL — this function only does the actor
/// round-trip.
async fn ask_alloc_async<T>(
    actor: ActorRef<DeviceMsg>,
    len: usize,
    timeout_secs: f64,
) -> PyResult<GpuRef<T>>
where
    T: CudaDtype,
{
    let (tx, rx) = oneshot::channel();
    actor.tell(DeviceMsg::alloc::<T>(len, tx));
    match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
        Ok(Ok(Ok(g))) => Ok(g),
        Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
        Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
        Err(_) => Err(errors::map_str("allocate timed out")),
    }
}

/// Trait for the per-dtype Python buffer wrappers. Every concrete
/// `PyGpuBuffer*` implements it via the `clone_ref` inherent method
/// already declared on the wrapper. We bind it here so generic helpers
/// (copy_from_numpy / copy_to_numpy / gemm) can talk about "any buffer
/// of dtype T" uniformly.
pub trait BorrowGpuRef<T> {
    fn borrow_ref(&self, py: Python<'_>) -> Option<GpuRef<T>>;
}

macro_rules! impl_borrow {
    ($Ty:ty, $rust:ty) => {
        impl BorrowGpuRef<$rust> for Py<$Ty> {
            fn borrow_ref(&self, py: Python<'_>) -> Option<GpuRef<$rust>> {
                self.borrow(py).clone_ref()
            }
        }
    };
}

impl_borrow!(PyGpuBufferF32, f32);
impl_borrow!(PyGpuBufferF64, f64);
impl_borrow!(PyGpuBufferI32, i32);
impl_borrow!(PyGpuBufferU32, u32);
impl_borrow!(PyGpuBufferU8, u8);

fn copy_from_numpy_typed<T, B>(
    py: Python<'_>,
    actor: &ActorRef<DeviceMsg>,
    dst: &Py<B>,
    src: PyReadonlyArray1<'_, T>,
    timeout_secs: f64,
) -> PyResult<()>
where
    T: CudaDtype + Element + Copy + 'static,
    Py<B>: BorrowGpuRef<T>,
    B: pyo3::PyClass,
{
    let host = src.as_slice().map_err(errors::map_str)?.to_vec();
    let g = dst
        .borrow_ref(py)
        .ok_or_else(|| errors::map_str("destination buffer has been consumed"))?;
    if host.len() != g.len() {
        return Err(errors::map_str(format!(
            "copy_from_numpy: src len {} != dst len {}",
            host.len(),
            g.len()
        )));
    }
    let actor = actor.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::copy_from_host::<T>(HostBuf::Owned(host), g, tx));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(_))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("copy_from_numpy timed out")),
            }
        })
    })
}

/// Inner copy-to-numpy worker. Generic over the buffer type; the
/// per-dtype methods on `PyDevice` invoke this directly.
fn copy_to_numpy_with<'py, T, B>(
    py: Python<'py>,
    actor: &ActorRef<DeviceMsg>,
    src: &Py<B>,
    timeout_secs: f64,
) -> PyResult<Bound<'py, PyArray1<T>>>
where
    T: CudaDtype + Element + Copy + Default + 'static,
    Py<B>: BorrowGpuRef<T>,
    B: pyo3::PyClass,
{
    let g = src
        .borrow_ref(py)
        .ok_or_else(|| errors::map_str("source buffer has been consumed"))?;
    let len = g.len();
    let actor = actor.clone();
    let rt = runtime();
    let host = py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::copy_to_host::<T>(
                g,
                HostBuf::Owned(vec![T::default(); len]),
                tx,
            ));
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(HostBuf::Owned(v)))) => Ok(v),
                Ok(Ok(Ok(_))) => Err(errors::map_str("unexpected pinned reply")),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("copy_to_numpy timed out")),
            }
        })
    })?;
    Ok(PyArray1::from_vec_bound(py, host))
}

/// Async copy_from_numpy. Reads the host slice + clones the actor
/// ref/destination GpuRef synchronously (under the GIL), then returns
/// a Python awaitable that completes when the actor replies.
fn copy_from_numpy_async_typed<'py, T, B>(
    py: Python<'py>,
    actor: &ActorRef<DeviceMsg>,
    dst: &Py<B>,
    src: PyReadonlyArray1<'_, T>,
    timeout_secs: f64,
) -> PyResult<Bound<'py, PyAny>>
where
    T: CudaDtype + Element + Copy + Send + 'static,
    Py<B>: BorrowGpuRef<T>,
    B: pyo3::PyClass,
{
    let host = src.as_slice().map_err(errors::map_str)?.to_vec();
    let g = dst
        .borrow_ref(py)
        .ok_or_else(|| errors::map_str("destination buffer has been consumed"))?;
    if host.len() != g.len() {
        return Err(errors::map_str(format!(
            "copy_from_numpy: src len {} != dst len {}",
            host.len(),
            g.len()
        )));
    }
    let actor = actor.clone();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (tx, rx) = oneshot::channel();
        actor.tell(DeviceMsg::copy_from_host::<T>(HostBuf::Owned(host), g, tx));
        match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
            Ok(Ok(Ok(_))) => Ok(()),
            Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
            Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
            Err(_) => Err(errors::map_str("copy_from_numpy timed out")),
        }
    })
}

/// Async copy_to_numpy. The future runs without the GIL; once the
/// actor reply lands we briefly re-acquire the GIL to wrap the result
/// `Vec<T>` into a `Py<PyArray1<T>>` for return.
fn copy_to_numpy_async_with<'py, T, B>(
    py: Python<'py>,
    actor: &ActorRef<DeviceMsg>,
    src: &Py<B>,
    timeout_secs: f64,
) -> PyResult<Bound<'py, PyAny>>
where
    T: CudaDtype + Element + Copy + Default + Send + 'static,
    Py<B>: BorrowGpuRef<T>,
    B: pyo3::PyClass,
{
    let g = src
        .borrow_ref(py)
        .ok_or_else(|| errors::map_str("source buffer has been consumed"))?;
    let len = g.len();
    let actor = actor.clone();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let (tx, rx) = oneshot::channel();
        actor.tell(DeviceMsg::copy_to_host::<T>(
            g,
            HostBuf::Owned(vec![T::default(); len]),
            tx,
        ));
        let host = match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
            Ok(Ok(Ok(HostBuf::Owned(v)))) => v,
            Ok(Ok(Ok(_))) => return Err(errors::map_str("unexpected pinned reply")),
            Ok(Ok(Err(e))) => return Err(errors::map_gpu(e)),
            Ok(Err(_)) => return Err(errors::map_str("device dropped reply")),
            Err(_) => return Err(errors::map_str("copy_to_numpy timed out")),
        };
        Python::with_gil(|py| Ok(PyArray1::from_vec_bound(py, host).unbind()))
    })
}

#[allow(dead_code)]
fn _typecheck() {
    let _ = GpuError::Timeout;
    let _ = std::any::TypeId::of::<GemmRequest<f32>>();
    let _ = std::any::TypeId::of::<BlasMsg>();
    let _ = std::any::TypeId::of::<Arc<()>>();
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<DeviceLoadDict>()?;
    Ok(())
}
