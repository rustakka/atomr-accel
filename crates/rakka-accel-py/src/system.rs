//! `System` — wraps a `rakka_core::actor::ActorSystem` for Python
//! callers. Lifecycle is sync (`open` / `close`) for ergonomics in
//! Python scripts; the underlying actor system is async.

use pyo3::prelude::*;

use rakka_config::Config;
use rakka_core::actor::ActorSystem as RustSystem;

use crate::device::PyDevice;
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "System", module = "rakka_accel._native")]
pub struct PySystem {
    pub(crate) inner: RustSystem,
}

#[pymethods]
impl PySystem {
    /// Open a new actor system. Blocking; the underlying tokio
    /// runtime is created the first time this is called.
    #[staticmethod]
    #[pyo3(signature = (name="rakka-accel".to_string()))]
    pub fn open(py: Python<'_>, name: String) -> PyResult<Py<Self>> {
        let rt = runtime();
        let inner = py
            .allow_threads(|| rt.block_on(RustSystem::create(name, Config::empty())))
            .map_err(errors::map_str)?;
        Py::new(py, PySystem { inner })
    }

    /// Spawn a `DeviceActor` against the given CUDA device id. With
    /// `mock=True` the actor runs without touching the CUDA driver
    /// (every kernel reply is a synthetic `Unrecoverable("...mock
    /// mode")`); useful for tests on hosts without a GPU.
    #[pyo3(signature = (device_id=0, name=None, mock=false))]
    pub fn spawn_device(
        &self,
        py: Python<'_>,
        device_id: u32,
        name: Option<String>,
        mock: bool,
    ) -> PyResult<Py<PyDevice>> {
        use rakka_accel_cuda::device::{DeviceActor, DeviceConfig};
        let cfg = if mock {
            DeviceConfig::mock(device_id)
        } else {
            DeviceConfig::new(device_id)
        };
        let actor_name = name.unwrap_or_else(|| format!("device-{device_id}"));
        let actor_ref = {
            let _guard = runtime().enter();
            self.inner
                .actor_of(DeviceActor::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyDevice::new(actor_ref, device_id))
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Terminate the actor system. After this call the System is
    /// unusable — drop the Python reference.
    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let rt = runtime();
        py.allow_threads(|| rt.block_on(inner.terminate()));
        Ok(())
    }

    fn __enter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!("System(name='{}')", self.inner.name())
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySystem>()?;
    Ok(())
}
