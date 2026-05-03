//! Python exception hierarchy for the rakka-accel-cuda bridge.
//!
//! ```text
//! GpuError                  (base, subclass of Python Exception)
//!   ├── ContextPoisoned     (CUDA context poisoned; supervisor will restart)
//!   ├── OutOfMemory         (device-side OOM; supervisor resumes)
//!   ├── Unrecoverable       (hardware fault / past retry budget; device stops)
//!   ├── GpuRefStale         (buffer used after context rebuild)
//!   ├── LibraryError        (cuBLAS/cuDNN/etc. error with `lib` tag)
//!   └── Timeout             (ask exceeded its budget)
//! ```

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use rakka_accel_cuda::error::GpuError;

create_exception!(
    rakka_accel,
    GpuRuntimeError,
    PyException,
    "Base rakka-accel error."
);
create_exception!(
    rakka_accel,
    ContextPoisoned,
    GpuRuntimeError,
    "CUDA context poisoned; supervisor will restart."
);
create_exception!(
    rakka_accel,
    OutOfMemory,
    GpuRuntimeError,
    "Device-side OOM; supervisor resumes the actor."
);
create_exception!(
    rakka_accel,
    Unrecoverable,
    GpuRuntimeError,
    "Hardware fault or past retry budget; device stops."
);
create_exception!(
    rakka_accel,
    GpuRefStale,
    GpuRuntimeError,
    "GpuRef used after context rebuild."
);
create_exception!(
    rakka_accel,
    LibraryError,
    GpuRuntimeError,
    "Library-level CUDA error (cuBLAS/cuDNN/...)."
);
create_exception!(
    rakka_accel,
    AskTimeout,
    GpuRuntimeError,
    "Ask exceeded its timeout budget."
);

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "GpuRuntimeError",
        m.py().get_type_bound::<GpuRuntimeError>(),
    )?;
    m.add(
        "ContextPoisoned",
        m.py().get_type_bound::<ContextPoisoned>(),
    )?;
    m.add("OutOfMemory", m.py().get_type_bound::<OutOfMemory>())?;
    m.add("Unrecoverable", m.py().get_type_bound::<Unrecoverable>())?;
    m.add("GpuRefStale", m.py().get_type_bound::<GpuRefStale>())?;
    m.add("LibraryError", m.py().get_type_bound::<LibraryError>())?;
    m.add("AskTimeout", m.py().get_type_bound::<AskTimeout>())?;
    Ok(())
}

/// Map a `GpuError` to the most specific Python exception subclass.
pub fn map_gpu(e: GpuError) -> PyErr {
    let msg = e.to_string();
    match e {
        GpuError::ContextPoisoned(_) => PyErr::new::<ContextPoisoned, _>(msg),
        GpuError::OutOfMemory(_) => PyErr::new::<OutOfMemory, _>(msg),
        GpuError::Unrecoverable(_) => PyErr::new::<Unrecoverable, _>(msg),
        GpuError::GpuRefStale(_) => PyErr::new::<GpuRefStale, _>(msg),
        GpuError::LibraryError { .. } | GpuError::Driver(_) => PyErr::new::<LibraryError, _>(msg),
        GpuError::Timeout => PyErr::new::<AskTimeout, _>(msg),
        #[allow(deprecated)]
        GpuError::Cublas(_) => PyErr::new::<LibraryError, _>(msg),
    }
}

/// Map any string-displayable error to the generic `GpuRuntimeError`.
pub fn map_str<E: std::fmt::Display>(e: E) -> PyErr {
    PyErr::new::<GpuRuntimeError, _>(e.to_string())
}
