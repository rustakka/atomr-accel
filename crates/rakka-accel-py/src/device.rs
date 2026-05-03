//! `Device` — Python wrapper around `ActorRef<DeviceMsg>`.
//!
//! Every method here is **blocking from the Python side** (the actor
//! reply is awaited on the shared tokio runtime via
//! `Runtime::block_on`). This keeps Python code straight-line and
//! GIL-friendly; long calls release the GIL through
//! `py.allow_threads`. Async wrappers can be layered in later by
//! returning the underlying tokio future via
//! `pyo3_async_runtimes::tokio::future_into_py`.

use std::time::Duration;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use tokio::sync::oneshot;

use rakka_accel_cuda::device::{DeviceLoad, DeviceMsg, HostBuf, SgemmRequest};
use rakka_accel_cuda::error::GpuError;
use rakka_accel_cuda::gpu_ref::GpuRef;
use rakka_core::actor::ActorRef;

use crate::buffer::PyGpuBuffer;
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Device", module = "rakka_accel._native")]
pub struct PyDevice {
    actor_ref: ActorRef<DeviceMsg>,
    device_id: u32,
}

impl PyDevice {
    pub fn new(actor_ref: ActorRef<DeviceMsg>, device_id: u32) -> Self {
        Self {
            actor_ref,
            device_id,
        }
    }
}

#[pymethods]
impl PyDevice {
    #[getter]
    fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Allocate `len` `f32` elements on-device. Returns a
    /// [`GpuBuffer`].
    #[pyo3(signature = (len, timeout_secs=10.0))]
    fn allocate_f32(
        &self,
        py: Python<'_>,
        len: usize,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGpuBuffer>> {
        let g = ask_alloc_f32(py, &self.actor_ref, len, timeout_secs)?;
        Py::new(py, PyGpuBuffer::new(g))
    }

    /// Upload a numpy `float32` array into a device buffer. The
    /// array must be contiguous and the same length as `dst`.
    #[pyo3(signature = (dst, src, timeout_secs=10.0))]
    fn copy_from_numpy(
        &self,
        py: Python<'_>,
        dst: Py<PyGpuBuffer>,
        src: PyReadonlyArray1<'_, f32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let host = src.as_slice().map_err(errors::map_str)?.to_vec();
        let g = dst
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("GpuBuffer has been consumed"))?;
        if host.len() != g.len() {
            return Err(errors::map_str(format!(
                "copy_from_numpy: src len {} != dst len {}",
                host.len(),
                g.len()
            )));
        }
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(DeviceMsg::CopyFromHostF32 {
                    src: HostBuf::Owned(host),
                    dst: g,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(_))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                    Err(_) => Err(errors::map_str("copy_from_numpy timed out")),
                }
            })
        })
    }

    /// Download a device buffer into a fresh numpy `float32` array.
    #[pyo3(signature = (src, timeout_secs=10.0))]
    fn copy_to_numpy<'py>(
        &self,
        py: Python<'py>,
        src: Py<PyGpuBuffer>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let g = src
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("GpuBuffer has been consumed"))?;
        let len = g.len();
        let actor = self.actor_ref.clone();
        let rt = runtime();
        let host = py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(DeviceMsg::CopyToHostF32 {
                    src: g,
                    dst: HostBuf::Owned(vec![0.0f32; len]),
                    reply: tx,
                });
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

    /// SGEMM: `c = alpha * a · b + beta * c`. All buffers are
    /// `m × k`, `k × n`, and `m × n` respectively (column-major).
    #[pyo3(signature = (a, b, c, m, n, k, alpha=1.0, beta=0.0, timeout_secs=60.0))]
    #[allow(clippy::too_many_arguments)]
    fn sgemm(
        &self,
        py: Python<'_>,
        a: Py<PyGpuBuffer>,
        b: Py<PyGpuBuffer>,
        c: Py<PyGpuBuffer>,
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
            active_streams: load.active_streams as u32,
            queue_depth: load.queue_depth,
            compute_cap_major: load.compute_cap.0,
            compute_cap_minor: load.compute_cap.1,
        })
    }

    fn __repr__(&self) -> String {
        format!("Device(id={})", self.device_id)
    }
}

/// Plain-data wrapper for `Stats`. Exposed as a Python class so
/// callers can access fields by name.
#[pyclass(name = "DeviceLoad", module = "rakka_accel._native")]
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

fn ask_alloc_f32(
    py: Python<'_>,
    actor: &ActorRef<DeviceMsg>,
    len: usize,
    timeout_secs: f64,
) -> PyResult<GpuRef<f32>> {
    let actor = actor.clone();
    let rt = runtime();
    py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(DeviceMsg::AllocateF32 { len, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(g))) => Ok(g),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("device dropped reply")),
                Err(_) => Err(errors::map_str("allocate_f32 timed out")),
            }
        })
    })
}

#[allow(dead_code)]
fn _typecheck() {
    let _ = GpuError::Timeout;
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<DeviceLoadDict>()?;
    Ok(())
}
