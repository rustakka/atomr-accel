//! `GpuBuffer*` — opaque, dtype-tagged Python handles around `GpuRef<T>`.
//!
//! Phase 1 ships one Python class per supported numpy-friendly dtype:
//! `GpuBufferF32`, `GpuBufferF64`, `GpuBufferI32`, `GpuBufferU32`,
//! `GpuBufferU8`. Phase 1.5++ adds typed complex buffers
//! (`GpuBufferC64` over `C32 = [f32; 2]`, `GpuBufferC128` over
//! `C64 = [f64; 2]`) for the cuFFT typed dispatch path. The original
//! `GpuBuffer` (alias for `GpuBufferF32`) is kept so existing scripts
//! don't break.
//!
//! Each class wraps `Mutex<Option<GpuRef<T>>>` — the `Option` lets a
//! follow-up op move the inner `GpuRef` out (e.g. into a kernel
//! keep-alive); the `Mutex` makes the wrapper `Send` for `#[pyclass]`.
//! `len`, `device_id`, `is_stale()` are zero-cost probes; reads/writes
//! go through `Device.copy_*_numpy_*`.

use parking_lot::Mutex;
use pyo3::prelude::*;

use atomr_accel_cuda::dtype::{C32, C64};
use atomr_accel_cuda::gpu_ref::GpuRef;

macro_rules! py_gpu_buffer {
    ($PyName:ident, $py_class:literal, $rust_ty:ty) => {
        #[pyclass(name = $py_class, module = "atomr_accel._native")]
        pub struct $PyName {
            inner: Mutex<Option<GpuRef<$rust_ty>>>,
        }

        impl $PyName {
            pub fn new(g: GpuRef<$rust_ty>) -> Self {
                Self {
                    inner: Mutex::new(Some(g)),
                }
            }

            /// Borrow the underlying `GpuRef`. `None` if a previous typed
            /// op already moved it out.
            pub fn clone_ref(&self) -> Option<GpuRef<$rust_ty>> {
                self.inner.lock().clone()
            }
        }

        #[pymethods]
        impl $PyName {
            #[getter]
            fn len(&self) -> usize {
                self.inner.lock().as_ref().map(|g| g.len()).unwrap_or(0)
            }

            #[getter]
            fn device_id(&self) -> Option<u32> {
                self.inner.lock().as_ref().and_then(|g| g.device_id())
            }

            #[getter]
            fn dtype(&self) -> &'static str {
                stringify!($rust_ty)
            }

            /// Probe whether the buffer is still valid against the current
            /// `DeviceState::generation`. Returns `True` if a follow-up op
            /// would surface `GpuRefStale`.
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
                    Some(r) => format!(
                        concat!($py_class, "(len={}, device={:?})"),
                        r.len(),
                        r.device_id()
                    ),
                    None => concat!($py_class, "(consumed)").to_string(),
                }
            }
        }
    };
}

py_gpu_buffer!(PyGpuBufferF32, "GpuBufferF32", f32);
py_gpu_buffer!(PyGpuBufferF64, "GpuBufferF64", f64);
py_gpu_buffer!(PyGpuBufferI32, "GpuBufferI32", i32);
py_gpu_buffer!(PyGpuBufferU32, "GpuBufferU32", u32);
py_gpu_buffer!(PyGpuBufferU8, "GpuBufferU8", u8);

// Phase 1.5++ — typed complex buffers for the cuFFT Path A dispatch.
// `C32` / `C64` are `#[repr(transparent)]` over `[f32; 2]` / `[f64; 2]`
// (defined in `atomr-accel-cuda::dtype`); they map to
// `numpy.complex64` / `numpy.complex128` and to cuFFT's `cuComplex` /
// `cuDoubleComplex`.
py_gpu_buffer!(PyGpuBufferC64, "GpuBufferC64", C32);
py_gpu_buffer!(PyGpuBufferC128, "GpuBufferC128", C64);

/// Back-compat alias: `atomr_accel.GpuBuffer == GpuBufferF32`.
pub type PyGpuBuffer = PyGpuBufferF32;

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGpuBufferF32>()?;
    m.add_class::<PyGpuBufferF64>()?;
    m.add_class::<PyGpuBufferI32>()?;
    m.add_class::<PyGpuBufferU32>()?;
    m.add_class::<PyGpuBufferU8>()?;
    m.add_class::<PyGpuBufferC64>()?;
    m.add_class::<PyGpuBufferC128>()?;
    // Add the back-compat name "GpuBuffer" as an alias for GpuBufferF32.
    let f32_cls = m.py().get_type_bound::<PyGpuBufferF32>();
    m.add("GpuBuffer", f32_cls)?;
    Ok(())
}
