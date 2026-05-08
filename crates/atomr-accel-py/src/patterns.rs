//! `atomr_accel_patterns` Python wrappers.
//!
//! Phase 2 ships the **structural anchors** for the pattern actors —
//! every actor in `atomr-accel-patterns` is generic over a user-supplied
//! `Req`/`Resp`, expert protocol, backend protocol, or callback closure
//! (`BatchFn`, `DraftFn`, `VerifierFn`, `GateFn`, …). Bridging those
//! into Python requires a typed marshal layer that's still being
//! designed; until then the PyClasses below exist as `isinstance`
//! anchors so downstream Python code can `from atomr_accel.patterns
//! import DynamicBatchingServer` and pass instances around without
//! upcasting.
//!
//! TODO Phase 2.5: per-actor `spawn(...)` constructors and
//! representative `submit` / `decode` / `route` methods once the
//! Python-side `Req` / `Resp` typed-bytes contract lands.

use pyo3::prelude::*;

#[pyclass(name = "DynamicBatchingServer", module = "atomr_accel._native")]
pub struct PyDynamicBatchingServer {}

#[pymethods]
impl PyDynamicBatchingServer {
    fn __repr__(&self) -> &'static str {
        "DynamicBatchingServer(handle, structural-anchor)"
    }
}

#[pyclass(name = "InferenceCascade", module = "atomr_accel._native")]
pub struct PyInferenceCascade {}

#[pymethods]
impl PyInferenceCascade {
    fn __repr__(&self) -> &'static str {
        "InferenceCascade(handle, structural-anchor)"
    }
}

#[pyclass(name = "ModelReplicaPool", module = "atomr_accel._native")]
pub struct PyModelReplicaPool {}

#[pymethods]
impl PyModelReplicaPool {
    fn __repr__(&self) -> &'static str {
        "ModelReplicaPool(handle, structural-anchor)"
    }
}

#[pyclass(name = "FairShareScheduler", module = "atomr_accel._native")]
pub struct PyFairShareScheduler {}

#[pymethods]
impl PyFairShareScheduler {
    fn __repr__(&self) -> &'static str {
        "FairShareScheduler(handle, structural-anchor)"
    }
}

#[pyclass(name = "HotSwapServer", module = "atomr_accel._native")]
pub struct PyHotSwapServer {}

#[pymethods]
impl PyHotSwapServer {
    fn __repr__(&self) -> &'static str {
        "HotSwapServer(handle, structural-anchor)"
    }
}

#[pyclass(name = "SpeculativeDecoder", module = "atomr_accel._native")]
pub struct PySpeculativeDecoder {}

#[pymethods]
impl PySpeculativeDecoder {
    fn __repr__(&self) -> &'static str {
        "SpeculativeDecoder(handle, structural-anchor)"
    }
}

#[pyclass(name = "MoeRouter", module = "atomr_accel._native")]
pub struct PyMoeRouter {}

#[pymethods]
impl PyMoeRouter {
    fn __repr__(&self) -> &'static str {
        "MoeRouter(handle, structural-anchor)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDynamicBatchingServer>()?;
    m.add_class::<PyInferenceCascade>()?;
    m.add_class::<PyModelReplicaPool>()?;
    m.add_class::<PyFairShareScheduler>()?;
    m.add_class::<PyHotSwapServer>()?;
    m.add_class::<PySpeculativeDecoder>()?;
    m.add_class::<PyMoeRouter>()?;
    Ok(())
}
