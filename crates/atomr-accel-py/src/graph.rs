//! `GraphCapture` / `GraphHandle` / `GraphScript` — Python wrappers
//! around the [`atomr_accel_cuda::graph`] actor surface.
//!
//! Phase 1.5 — CUDA graphs structural anchor. The wrapper spawns a
//! [`GraphActor`] in **mock mode** and exposes the full
//! `Record` / `Launch` message path plus the helper accessors
//! ([`GraphHandle::export_dot`]). This lets Python callers exercise
//! the surface (script construction, message round-trip, error
//! mapping) without a CUDA driver.
//!
//! Real-mode capture requires a `CudaStream` + `CompletionStrategy`
//! + `DeviceState`, which today live behind the `Device` actor and
//! are not handed out as kernel children. Wiring `Device.graph()`
//! lives in Phase 5 alongside the Phase-5 device.rs changes —
//! once landed, [`PyGraphCapture::spawn_for_device`] can replace
//! the `mock_props` call and the rest of the surface keeps working
//! verbatim.
//!
//! ```text
//! GraphScript                     # builds Vec<Box<dyn GraphOp>>
//!     add_memcpy(src, dst)
//!     add_sgemm(a, b, c, m, n, k, alpha, beta)
//! GraphCapture.spawn(system)      # spawns GraphActor (mock mode)
//!     .record(script) -> GraphHandle
//!     .launch(handle) -> ()
//! GraphHandle.export_dot(flags)   # Graphviz DOT round-trip
//! ```
//!
//! Mock mode replies are routed by the actor: `Record` and `Launch`
//! both surface as `Unrecoverable("GraphActor in mock mode")`. The
//! Python caller sees a typed `Unrecoverable` exception. This faithful
//! mock behaviour is what cuDNN / FlashAttention / etc. Python
//! wrappers also rely on for their CI test paths.

use std::sync::Mutex;
use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::graph::{
    export_dot, DotFlags, GraphActor, GraphHandle, GraphMsg, GraphOp, MemcpyOp, SgemmOp,
};
use atomr_core::actor::ActorRef;

use crate::buffer::PyGpuBufferF32;
use crate::errors;
use crate::runtime::runtime;
use crate::system::PySystem;

// ─── GraphHandle ──────────────────────────────────────────────────

/// A captured-and-instantiated CUDA graph. Returned by
/// [`PyGraphCapture::record`] (or [`PyGraphHandle::synthetic`] for
/// tests). The handle is opaque from Python; pass it back into
/// [`PyGraphCapture::launch`] to replay, or call
/// [`PyGraphHandle::export_dot`] to dump the topology as Graphviz.
#[pyclass(name = "GraphHandle", module = "atomr_accel._native")]
pub struct PyGraphHandle {
    pub(crate) inner: GraphHandle,
}

impl PyGraphHandle {
    pub(crate) fn new(inner: GraphHandle) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGraphHandle {
    /// Build a synthetic handle with null sys-level CUgraph /
    /// CUgraphExec pointers. Useful for surface tests on hosts
    /// without a CUDA driver — `export_dot` will surface
    /// `LibraryError` (driver loadable) or `Unrecoverable`
    /// (driver missing) without panicking.
    #[staticmethod]
    fn synthetic(py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(py, PyGraphHandle::new(GraphHandle::synthetic_for_tests()))
    }

    /// `DeviceState` generation at capture time. Used by the
    /// `GraphActor` to reject `Launch` requests against rebuilt
    /// contexts.
    #[getter]
    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    /// Export the graph as a Graphviz DOT string via
    /// `cuGraphDebugDotPrint`. `flags` is a bitmask of
    /// `CU_GRAPH_DEBUG_DOT_FLAGS_*`:
    ///
    ///   1   verbose
    ///   4   kernel nodes
    ///   8   memcpy nodes
    ///  16   memset nodes
    ///  32   host nodes
    ///  64   sub-graph nodes
    ///
    /// `flags=0` (default) emits a minimal DOT.
    #[pyo3(signature = (flags=0))]
    fn export_dot(&self, py: Python<'_>, flags: u32) -> PyResult<String> {
        let inner = self.inner.clone();
        py.allow_threads(|| {
            let dot_flags = DotFlags::from_bits_truncate(flags);
            export_dot(&inner, dot_flags).map_err(errors::map_gpu)
        })
    }

    fn __repr__(&self) -> String {
        format!("GraphHandle(generation={})", self.inner.generation())
    }
}

// ─── GraphScript ──────────────────────────────────────────────────

/// Builder accumulating `Box<dyn GraphOp>` ops to send through
/// [`PyGraphCapture::record`]. Each `add_*` method consumes the
/// underlying `GpuRef` from its [`PyGpuBufferF32`] argument(s) the
/// same way [`crate::cudnn::PyCudnn`] / [`crate::blas::PyBlas`] do —
/// so a buffer used in a graph script can't be re-used afterwards.
///
/// Mock-mode capture rejects any non-empty script: the `Record`
/// reply surfaces as `Unrecoverable("GraphActor in mock mode")`.
/// Real-mode wiring lands in Phase 5 alongside the device-side
/// stream accessor.
#[pyclass(name = "GraphScript", module = "atomr_accel._native")]
pub struct PyGraphScript {
    /// Mutex guard keeps the builder `Send + Sync` for pyo3 while
    /// allowing `take_ops()` to drain the vector exactly once
    /// during `record(script)`.
    ops: Mutex<Vec<Box<dyn GraphOp>>>,
}

impl PyGraphScript {
    /// Drain the accumulated ops. After this call the script is
    /// empty; the typical caller is [`PyGraphCapture::record`]
    /// which moves the script into the actor message.
    fn take_ops(&self) -> Vec<Box<dyn GraphOp>> {
        let mut g = self.ops.lock().expect("graph script mutex poisoned");
        std::mem::take(&mut *g)
    }
}

#[pymethods]
impl PyGraphScript {
    #[new]
    fn new() -> Self {
        Self {
            ops: Mutex::new(Vec::new()),
        }
    }

    /// Number of ops currently buffered. Resets to zero after
    /// `record(script)` consumes the script.
    fn __len__(&self) -> usize {
        self.ops.lock().expect("graph script mutex poisoned").len()
    }

    /// Append a device-to-device f32 memcpy op. Both `src` and
    /// `dst` are consumed (subsequent uses raise
    /// `GpuRuntimeError("... consumed")`).
    fn add_memcpy(
        &self,
        py: Python<'_>,
        src: Py<PyGpuBufferF32>,
        dst: Py<PyGpuBufferF32>,
    ) -> PyResult<()> {
        let src = src
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("src consumed"))?;
        let dst = dst
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dst consumed"))?;
        let op = MemcpyOp::new(src, dst);
        self.ops
            .lock()
            .expect("graph script mutex poisoned")
            .push(Box::new(op));
        Ok(())
    }

    /// Append an SGEMM op: `c := alpha · a · b + beta · c`,
    /// column-major, no transpose. Requires the recording
    /// `GraphActor` to have a cuBLAS handle bound to the captured
    /// stream — mock mode does not, so `record(script)` will surface
    /// `Unrecoverable`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (a, b, c, m, n, k, alpha=1.0, beta=0.0))]
    fn add_sgemm(
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
        let op = SgemmOp::new(a, b, c, m, n, k, alpha, beta);
        self.ops
            .lock()
            .expect("graph script mutex poisoned")
            .push(Box::new(op));
        Ok(())
    }

    fn __repr__(&self) -> String {
        let n = self.ops.lock().map(|g| g.len()).unwrap_or(0);
        format!("GraphScript(ops={n})")
    }
}

// ─── GraphCapture ──────────────────────────────────────────────────

/// Python handle around an `ActorRef<GraphMsg>`. Phase 1.5 ships a
/// mock-mode constructor only — the realistic real-mode constructor
/// (`spawn_for_device`) lands once Phase 5 wires
/// `Device.graph(stream)`.
#[pyclass(name = "GraphCapture", module = "atomr_accel._native")]
pub struct PyGraphCapture {
    actor_ref: ActorRef<GraphMsg>,
}

#[pymethods]
impl PyGraphCapture {
    /// Spawn a `GraphActor` in mock mode. `Record` and `Launch`
    /// reply `Unrecoverable("GraphActor in mock mode")` — the
    /// surface exists for CI / surface tests on hosts without a
    /// CUDA driver. Real-mode capture is gated behind Phase 5's
    /// `Device.graph()` accessor (TODO).
    #[staticmethod]
    #[pyo3(signature = (system, name=None))]
    fn spawn(py: Python<'_>, system: &PySystem, name: Option<String>) -> PyResult<Py<Self>> {
        let actor_name = name.unwrap_or_else(|| "graph-capture".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(GraphActor::mock_props(), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyGraphCapture { actor_ref })
    }

    /// Record `script`'s ops into a CUDA graph and return a
    /// `GraphHandle`. The script is *consumed* by this call; on
    /// return its `__len__` is zero.
    ///
    /// In mock mode the actor replies
    /// `Unrecoverable("GraphActor in mock mode")` regardless of
    /// script contents.
    #[pyo3(signature = (script, timeout_secs=30.0))]
    fn record(
        &self,
        py: Python<'_>,
        script: &PyGraphScript,
        timeout_secs: f64,
    ) -> PyResult<Py<PyGraphHandle>> {
        let ops = script.take_ops();
        let actor = self.actor_ref.clone();
        let rt = runtime();
        let handle = py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(GraphMsg::Record {
                    script: ops,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(h))) => Ok(h),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("graph dropped reply")),
                    Err(_) => Err(errors::map_str("record timed out")),
                }
            })
        })?;
        Py::new(py, PyGraphHandle::new(handle))
    }

    /// Replay a previously-recorded `GraphHandle`. Awaits stream
    /// completion before returning. In mock mode replies
    /// `Unrecoverable`.
    #[pyo3(signature = (handle, timeout_secs=30.0))]
    fn launch(&self, py: Python<'_>, handle: Py<PyGraphHandle>, timeout_secs: f64) -> PyResult<()> {
        let h = handle.borrow(py).inner.clone();
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(GraphMsg::Launch {
                    handle: h,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("graph dropped reply")),
                    Err(_) => Err(errors::map_str("launch timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "GraphCapture(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphCapture>()?;
    m.add_class::<PyGraphHandle>()?;
    m.add_class::<PyGraphScript>()?;
    Ok(())
}
