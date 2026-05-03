//! `GpuBuffer` — opaque handle wrapping `GpuRef<f32>`.
//!
//! Python callers receive these from `Device.allocate_f32(n)` or any
//! actor reply that mints a fresh device buffer. The class is
//! deliberately thin: it exposes length, device id, and the
//! generation token, plus an `is_stale()` probe. Reading / writing
//! the contents goes through `Device.copy_to_numpy(buf)` /
//! `Device.copy_from_numpy(buf, arr)`.

use parking_lot::Mutex;
use pyo3::prelude::*;

use rakka_accel_cuda::gpu_ref::GpuRef;

#[pyclass(name = "GpuBuffer", module = "rakka_accel._native")]
pub struct PyGpuBuffer {
    inner: Mutex<Option<GpuRef<f32>>>,
}

impl PyGpuBuffer {
    pub fn new(g: GpuRef<f32>) -> Self {
        Self {
            inner: Mutex::new(Some(g)),
        }
    }

    /// Borrow the underlying GpuRef. Returns None if the buffer was
    /// already moved out (e.g. by a previous typed op).
    pub fn clone_ref(&self) -> Option<GpuRef<f32>> {
        self.inner.lock().clone()
    }
}

#[pymethods]
impl PyGpuBuffer {
    #[getter]
    fn len(&self) -> usize {
        self.inner.lock().as_ref().map(|g| g.len()).unwrap_or(0)
    }

    #[getter]
    fn device_id(&self) -> Option<u32> {
        self.inner.lock().as_ref().and_then(|g| g.device_id())
    }

    /// Probe whether the buffer is still valid against the current
    /// `DeviceState::generation`. Returns `True` if a follow-up
    /// op would surface `GpuRefStale`.
    fn is_stale(&self) -> bool {
        match self.inner.lock().as_ref() {
            Some(g) => g.access().is_err(),
            None => true,
        }
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        let g = self.inner.lock();
        match g.as_ref() {
            Some(r) => format!("GpuBuffer(len={}, device={:?})", r.len(), r.device_id()),
            None => "GpuBuffer(consumed)".to_string(),
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuBuffer>()?;
    Ok(())
}
