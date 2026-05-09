//! Python bindings for `atomr-accel-telemetry` — NVTX kernel ranges,
//! NVML actor probes, and CUPTI activity sessions.
//!
//! The underlying crate is feature-sliced (`nvtx` / `nvml` / `cupti`);
//! this module mirrors the same shape: each sub-feature on
//! `atomr-accel-py` (`telemetry-nvtx`, `telemetry-nvml`,
//! `telemetry-cupti`) gates the corresponding `PyClass`.
//!
//! Phase 3 ships representative methods that exercise the public
//! surface end-to-end on hosts where the underlying NVIDIA libraries
//! are present, and surface `Unrecoverable("not supported in mock
//! mode")` (or the equivalent typed error) on hosts without the
//! library installed. Method-level depth (full NVML field projection,
//! per-category activity decoders, range profiler metric selection)
//! is tracked in the Phase 3.5 telemetry-coverage issue.

#![cfg(feature = "telemetry")]

use pyo3::prelude::*;

#[cfg(feature = "telemetry-nvtx")]
mod nvtx_impl {
    use std::any::Any;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Mutex;

    use pyo3::prelude::*;
    use pyo3::types::PyType;

    use atomr_accel_telemetry::nvtx::NvtxKernelTrace;
    use atomr_accel_telemetry::trace::{KernelInfo, KernelTrace};

    use crate::errors;

    /// `with telemetry.NvtxKernelTrace("span"): ...` emits an NVTX
    /// domain range around the body. Constructs a fresh
    /// `NvtxKernelTrace` lazily on `__enter__` (NVTX domain creation
    /// goes through cudarc's libloading wrapper, which panics on
    /// hosts without `libnvToolsExt.so`); the panic is caught and
    /// surfaced as `Unrecoverable`.
    ///
    /// The span name is leaked to a `&'static str` because
    /// `KernelInfo::op_name` is `&'static str`. Span names are
    /// expected to be a small bounded set (per-actor / per-op
    /// labels), so the leak is intentional and bounded.
    #[pyclass(name = "NvtxKernelTrace", module = "atomr_accel._native")]
    pub struct PyNvtxKernelTrace {
        domain_name: String,
        span_name: String,
        /// Active trace + cookie pair, populated on `__enter__`.
        active: Mutex<Option<ActiveSpan>>,
    }

    struct ActiveSpan {
        trace: NvtxKernelTrace,
        cookie: Option<Box<dyn Any + Send>>,
        info: KernelInfo,
    }

    #[pymethods]
    impl PyNvtxKernelTrace {
        /// Construct with a span name. `domain` defaults to
        /// `"atomr-accel"`; pass a per-actor name to give each
        /// actor its own swim lane in Nsight.
        #[new]
        #[pyo3(signature = (name="span", domain="atomr-accel"))]
        fn new(name: &str, domain: &str) -> Self {
            Self {
                domain_name: domain.to_owned(),
                span_name: name.to_owned(),
                active: Mutex::new(None),
            }
        }

        fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
            let mut guard = slf
                .active
                .lock()
                .map_err(|_| errors::map_str("nvtx span mutex poisoned"))?;
            if guard.is_some() {
                return Err(errors::map_str("NvtxKernelTrace already entered"));
            }
            // Construct the trace + push a range. Both go through
            // cudarc's libloading wrapper which panics on hosts
            // without NVTX; catch the panic and remap.
            let domain_name = slf.domain_name.clone();
            let span_name: &'static str = Box::leak(slf.span_name.clone().into_boxed_str());
            let info = KernelInfo {
                lib_tag: "py-nvtx",
                op_name: span_name,
                device_index: None,
                correlation_id: None,
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                let trace = NvtxKernelTrace::with_domain_name(&domain_name);
                let cookie = trace.before_enqueue(&info);
                (trace, cookie)
            }));
            match result {
                Ok((trace, cookie)) => {
                    *guard = Some(ActiveSpan {
                        trace,
                        cookie: Some(cookie),
                        info,
                    });
                    drop(guard);
                    Ok(slf)
                }
                Err(_) => Err(PyErr::new::<errors::Unrecoverable, _>(
                    "NVTX not supported in mock mode (libnvToolsExt unavailable)",
                )),
            }
        }

        #[pyo3(signature = (_exc_type=None, _exc_val=None, _exc_tb=None))]
        fn __exit__(
            &self,
            _exc_type: Option<Bound<'_, PyType>>,
            _exc_val: Option<Bound<'_, PyAny>>,
            _exc_tb: Option<Bound<'_, PyAny>>,
        ) -> PyResult<bool> {
            let mut guard = self
                .active
                .lock()
                .map_err(|_| errors::map_str("nvtx span mutex poisoned"))?;
            if let Some(mut active) = guard.take() {
                if let Some(cookie) = active.cookie.take() {
                    // `after_complete` calls into NVTX through
                    // libloading; on hosts where NVTX vanished
                    // mid-span we still want to surface a clean
                    // exception rather than abort.
                    let trace_ref = &active.trace;
                    let info = active.info.clone();
                    let _ = catch_unwind(AssertUnwindSafe(|| {
                        trace_ref.after_complete(&info, cookie, std::time::Duration::ZERO);
                    }));
                }
            }
            Ok(false)
        }

        fn __repr__(&self) -> String {
            format!(
                "NvtxKernelTrace(name={:?}, domain={:?})",
                self.span_name, self.domain_name
            )
        }
    }
}

#[cfg(feature = "telemetry-nvml")]
mod nvml_impl {
    use std::time::Duration;

    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};
    use tokio::sync::oneshot;

    use atomr_accel_telemetry::nvml::{NvmlActor, NvmlConfig, NvmlError, NvmlMsg};

    use crate::errors;
    use crate::runtime::runtime;

    /// Python handle around `NvmlActor`. Construction loads
    /// `libnvidia-ml.so.1` via libloading; on hosts without NVML the
    /// constructor raises `Unrecoverable`. `read()` returns the most
    /// recent snapshot as a Python dict.
    #[pyclass(name = "NvmlActor", module = "atomr_accel._native")]
    pub struct PyNvmlActor {
        actor: NvmlActor,
    }

    #[pymethods]
    impl PyNvmlActor {
        /// Spawn an NVML polling actor with `interval_secs` cadence
        /// (default 1.0s). On hosts where `libnvidia-ml.so.1` is not
        /// installed (mock-mode CI, WSL consumer setups) raises
        /// `Unrecoverable`.
        #[new]
        #[pyo3(signature = (interval_secs=1.0))]
        fn new(interval_secs: f64) -> PyResult<Self> {
            let config = NvmlConfig {
                interval: Duration::from_secs_f64(interval_secs),
                library_path: None,
            };
            // The actor spawns a tokio task; we need the runtime
            // initialised so the task gets scheduled.
            let _rt = runtime();
            let _guard = _rt.enter();
            match NvmlActor::try_new(config) {
                Ok(actor) => Ok(Self { actor }),
                Err(NvmlError::LibraryUnavailable(msg)) => {
                    Err(PyErr::new::<errors::Unrecoverable, _>(format!(
                        "NVML not supported in mock mode: {msg}"
                    )))
                }
                Err(e) => Err(errors::map_str(e)),
            }
        }

        /// Most recent NVML snapshot. Returns a dict with `devices`
        /// (list of per-device dicts) and `generated_at_unix_nanos`.
        /// Each device dict carries the metrics NVML exposed at the
        /// last polling tick.
        #[pyo3(signature = (timeout_secs=2.0))]
        fn read<'py>(&self, py: Python<'py>, timeout_secs: f64) -> PyResult<Bound<'py, PyDict>> {
            let sender = self.actor.sender();
            let rt = runtime();
            let snap = py.allow_threads(|| {
                rt.block_on(async move {
                    let (tx, rx) = oneshot::channel();
                    if sender.send(NvmlMsg::Snapshot { reply: tx }).await.is_err() {
                        return Err(errors::map_str("nvml actor channel closed"));
                    }
                    match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                        Ok(Ok(Ok(snap))) => Ok(snap),
                        Ok(Ok(Err(e))) => Err(errors::map_str(e)),
                        Ok(Err(_)) => Err(errors::map_str("nvml actor dropped reply")),
                        Err(_) => Err(errors::map_str("nvml read timed out")),
                    }
                })
            })?;

            let dict = PyDict::new_bound(py);
            dict.set_item(
                "generated_at_unix_nanos",
                snap.generated_at_unix_nanos as u64,
            )?;
            let devices = PyList::empty_bound(py);
            for d in snap.devices.iter() {
                let dd = PyDict::new_bound(py);
                dd.set_item("device_index", d.device_index)?;
                dd.set_item("uuid", d.uuid.clone())?;
                dd.set_item("name", d.name.clone())?;
                dd.set_item("power_milliwatts", d.power_milliwatts)?;
                dd.set_item("temperature_gpu_c", d.temperature_gpu_c)?;
                dd.set_item("clock_sm_mhz", d.clock_sm_mhz)?;
                dd.set_item("clock_mem_mhz", d.clock_mem_mhz)?;
                dd.set_item("mem_total_bytes", d.mem_total_bytes)?;
                dd.set_item("mem_used_bytes", d.mem_used_bytes)?;
                devices.append(dd)?;
            }
            dict.set_item("devices", devices)?;
            Ok(dict)
        }

        /// Convenience accessor: current power draw of device 0 in
        /// watts, or `None` if the field is unavailable / no devices
        /// were enumerated.
        #[pyo3(signature = (timeout_secs=2.0))]
        fn power_w(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<Option<f64>> {
            let sender = self.actor.sender();
            let rt = runtime();
            let snap = py.allow_threads(|| {
                rt.block_on(async move {
                    let (tx, rx) = oneshot::channel();
                    if sender.send(NvmlMsg::Snapshot { reply: tx }).await.is_err() {
                        return Err(errors::map_str("nvml actor channel closed"));
                    }
                    match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                        Ok(Ok(Ok(snap))) => Ok(snap),
                        Ok(Ok(Err(e))) => Err(errors::map_str(e)),
                        Ok(Err(_)) => Err(errors::map_str("nvml actor dropped reply")),
                        Err(_) => Err(errors::map_str("nvml power_w timed out")),
                    }
                })
            })?;
            Ok(snap
                .devices
                .first()
                .and_then(|d| d.power_milliwatts)
                .map(|mw| f64::from(mw) / 1000.0))
        }

        fn __repr__(&self) -> &'static str {
            "NvmlActor(handle)"
        }
    }
}

#[cfg(feature = "telemetry-cupti")]
mod cupti_impl {
    use std::time::Duration;

    use pyo3::prelude::*;
    use pyo3::types::PyList;
    use tokio::sync::oneshot;

    use atomr_accel_telemetry::cupti::{ActivityCategory, CuptiMsg, CuptiSession};

    use crate::errors;
    use crate::runtime::runtime;

    /// Python handle around `CuptiSession`. Spawns the actor on
    /// construction; `start(categories)` enables the requested
    /// activity kinds, `stop()` flushes, `drain()` returns the
    /// buffered records as a list of dicts.
    ///
    /// On hosts without CUPTI installed the spawn itself succeeds —
    /// the underlying actor stores the requested categories without
    /// actually wiring CUPTI's FFI when libcupti is missing — but
    /// `drain()` will simply return an empty list.
    #[pyclass(name = "CuptiSession", module = "atomr_accel._native")]
    pub struct PyCuptiSession {
        session: CuptiSession,
    }

    fn parse_category(s: &str) -> PyResult<ActivityCategory> {
        match s.to_ascii_lowercase().as_str() {
            "kernel" | "kernel_launch" | "kernellaunch" => Ok(ActivityCategory::KernelLaunch),
            "memcpy" => Ok(ActivityCategory::Memcpy),
            "driver" | "driver_api" | "driverapi" => Ok(ActivityCategory::DriverApi),
            "runtime" | "runtime_api" | "runtimeapi" => Ok(ActivityCategory::RuntimeApi),
            "range" | "range_profiler" | "rangeprofiler" => Ok(ActivityCategory::RangeProfiler),
            other => Err(errors::map_str(format!(
                "unknown CUPTI activity category {other:?}"
            ))),
        }
    }

    #[pymethods]
    impl PyCuptiSession {
        /// Spawn a CUPTI session actor. Construction does *not*
        /// dlopen libcupti — that's done by `CuptiBootstrap` which
        /// must run before `cuInit`. On a mock-mode host this is a
        /// pure-Tokio noop session that still accepts `start` /
        /// `stop` / `drain`.
        #[new]
        fn new() -> Self {
            let _rt = runtime();
            let _guard = _rt.enter();
            Self {
                session: CuptiSession::spawn(),
            }
        }

        /// Enable the requested activity categories. Accepts a list
        /// of strings: `"kernel"`, `"memcpy"`, `"driver"`,
        /// `"runtime"`, `"range"`.
        #[pyo3(signature = (categories=None, timeout_secs=2.0))]
        fn start(
            &self,
            py: Python<'_>,
            categories: Option<Vec<String>>,
            timeout_secs: f64,
        ) -> PyResult<()> {
            let cats = match categories {
                Some(strs) => strs
                    .iter()
                    .map(|s| parse_category(s))
                    .collect::<PyResult<Vec<_>>>()?,
                None => vec![
                    ActivityCategory::KernelLaunch,
                    ActivityCategory::Memcpy,
                    ActivityCategory::DriverApi,
                    ActivityCategory::RuntimeApi,
                ],
            };
            let sender = self.session.sender();
            let rt = runtime();
            py.allow_threads(|| {
                rt.block_on(async move {
                    let (tx, rx) = oneshot::channel();
                    if sender
                        .send(CuptiMsg::Start {
                            categories: cats,
                            reply: tx,
                        })
                        .await
                        .is_err()
                    {
                        return Err(errors::map_str("cupti session channel closed"));
                    }
                    match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                        Ok(Ok(Ok(()))) => Ok(()),
                        Ok(Ok(Err(e))) => Err(errors::map_str(e)),
                        Ok(Err(_)) => Err(errors::map_str("cupti session dropped reply")),
                        Err(_) => Err(errors::map_str("cupti start timed out")),
                    }
                })
            })
        }

        /// Flush + disable every active category.
        #[pyo3(signature = (timeout_secs=2.0))]
        fn stop(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<()> {
            let sender = self.session.sender();
            let rt = runtime();
            py.allow_threads(|| {
                rt.block_on(async move {
                    let (tx, rx) = oneshot::channel();
                    if sender.send(CuptiMsg::Stop { reply: tx }).await.is_err() {
                        return Err(errors::map_str("cupti session channel closed"));
                    }
                    match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                        Ok(Ok(Ok(()))) => Ok(()),
                        Ok(Ok(Err(e))) => Err(errors::map_str(e)),
                        Ok(Err(_)) => Err(errors::map_str("cupti session dropped reply")),
                        Err(_) => Err(errors::map_str("cupti stop timed out")),
                    }
                })
            })
        }

        /// Drain every buffered activity record. Returns a list of
        /// dicts; on mock-mode hosts (no libcupti) this is always
        /// empty.
        #[pyo3(signature = (timeout_secs=2.0))]
        fn drain<'py>(&self, py: Python<'py>, timeout_secs: f64) -> PyResult<Bound<'py, PyList>> {
            let sender = self.session.sender();
            let rt = runtime();
            let records = py.allow_threads(|| {
                rt.block_on(async move {
                    let (tx, rx) = oneshot::channel();
                    if sender.send(CuptiMsg::Drain { reply: tx }).await.is_err() {
                        return Err(errors::map_str("cupti session channel closed"));
                    }
                    match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                        Ok(Ok(Ok(records))) => Ok(records),
                        Ok(Ok(Err(e))) => Err(errors::map_str(e)),
                        Ok(Err(_)) => Err(errors::map_str("cupti session dropped reply")),
                        Err(_) => Err(errors::map_str("cupti drain timed out")),
                    }
                })
            })?;
            let out = PyList::empty_bound(py);
            for r in records.iter() {
                out.append(format!("{r:?}"))?;
            }
            Ok(out)
        }

        fn __repr__(&self) -> &'static str {
            "CuptiSession(handle)"
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "telemetry-nvtx")]
    m.add_class::<nvtx_impl::PyNvtxKernelTrace>()?;
    #[cfg(feature = "telemetry-nvml")]
    m.add_class::<nvml_impl::PyNvmlActor>()?;
    #[cfg(feature = "telemetry-cupti")]
    m.add_class::<cupti_impl::PyCuptiSession>()?;
    let _ = m;
    Ok(())
}
