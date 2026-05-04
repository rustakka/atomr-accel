//! `NvrtcKernel` — Python wrapper for an NVRTC-compiled kernel
//! handle. Gated on the `nvrtc` feature.
//!
//! Compilation happens through `Device.compile_kernel(src,
//! kernel_name)` (added in a follow-up that wires
//! `ContextActor::SnapshotChildren` through the Python facade); the
//! resulting `NvrtcKernel` is launchable via `device.launch_kernel`.

use atomr_accel_cuda::kernel::KernelHandle;
use pyo3::prelude::*;

#[pyclass(name = "NvrtcKernel", module = "atomr_accel._native")]
pub struct PyNvrtcKernel {
    #[allow(dead_code)]
    pub(crate) handle: KernelHandle,
}

#[pymethods]
impl PyNvrtcKernel {
    #[getter]
    fn name(&self) -> String {
        self.handle.name.clone()
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.handle.generation()
    }

    fn __repr__(&self) -> String {
        format!(
            "NvrtcKernel(name='{}', generation={})",
            self.handle.name,
            self.handle.generation()
        )
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNvrtcKernel>()?;
    Ok(())
}
