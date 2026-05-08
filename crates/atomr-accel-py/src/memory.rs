//! `Memory` — Python handle wrapping `ActorRef<ManagedMsg>`.
//!
//! Phase 1.5 — CUDA memory ops Python wrapper. Surfaces the
//! Phase 0.4 / Phase 3 driver-API memory primitives:
//!
//! - `cudaMallocManaged` (managed / unified memory) via the typed
//!   `ManagedAllocatorActor` actor — `Memory.allocate_managed_f32`.
//! - `cuMemPrefetchAsync_v2` and `cuMemAdvise_v2` over a managed
//!   allocation — `Memory.prefetch_f32` / `Memory.advise_f32`.
//! - `cuIpcGetMemHandle` / `cuIpcOpenMemHandle_v2` /
//!   `cuIpcCloseMemHandle` (only when the wheel was built with
//!   `--features cuda-ipc`) — module-level `ipc_get_mem_handle` /
//!   `ipc_open_mem_handle`, plus `IpcMemHandle` and `OpenedMem`
//!   PyClasses.
//!
//! `Memory` is a standalone actor (it does NOT belong to a
//! `Device`'s child set); spawn it directly under the `System`:
//!
//! ```python
//! import atomr_accel
//! from atomr_accel import memory as mem
//!
//! with atomr_accel.System.open("memory-demo") as sys:
//!     m = mem.Memory.spawn(sys)
//!     buf = m.allocate_managed_f32(1024)              # -> ManagedBufferF32
//!     m.advise_f32(buf, "set_read_mostly")
//!     m.prefetch_f32(buf, target="cpu")
//! ```
//!
//! On hosts without a CUDA driver, `allocate_managed_f32` returns
//! `OutOfMemory` (the underlying actor catches the cudarc-loader
//! panic and converts it). Prefetch / advise return `Unrecoverable`
//! or `LibraryError` for invalid pointers; both are mock-mode safe.

use std::time::Duration;

use parking_lot::Mutex;
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::memory::{
    advise::MemAdvice, ManagedAllocatorActor, ManagedFlags, ManagedMsg, ManagedRef, ManagedStats,
    PrefetchTarget,
};
use atomr_core::actor::ActorRef;

use crate::errors;
use crate::runtime::runtime;
use crate::system::PySystem;

// ─── helpers ─────────────────────────────────────────────────────

fn flags_from_str(s: &str) -> PyResult<ManagedFlags> {
    match s {
        "global" | "Global" | "attach_global" | "AttachGlobal" => Ok(ManagedFlags::AttachGlobal),
        "host" | "Host" | "attach_host" | "AttachHost" => Ok(ManagedFlags::AttachHost),
        other => Err(errors::map_str(format!(
            "unknown managed flags: {other:?} (expected 'global' or 'host')"
        ))),
    }
}

fn target_from_args(target: &str, device_id: u32) -> PyResult<PrefetchTarget> {
    match target {
        "cpu" | "Cpu" | "CPU" | "host" | "Host" => Ok(PrefetchTarget::Cpu),
        "device" | "Device" | "gpu" | "Gpu" | "GPU" => Ok(PrefetchTarget::Device(device_id)),
        other => Err(errors::map_str(format!(
            "unknown prefetch target: {other:?} (expected 'cpu' or 'device')"
        ))),
    }
}

fn advice_from_args(advice: &str, target: Option<&str>, device_id: u32) -> PyResult<MemAdvice> {
    let needs_target = matches!(
        advice,
        "set_preferred_location"
            | "SetPreferredLocation"
            | "set_accessed_by"
            | "SetAccessedBy"
            | "unset_accessed_by"
            | "UnsetAccessedBy"
    );
    let resolved_target = if needs_target {
        let t = target.ok_or_else(|| {
            errors::map_str(format!(
                "advice {advice:?} requires a `target` ('cpu' or 'device')"
            ))
        })?;
        Some(target_from_args(t, device_id)?)
    } else {
        None
    };
    match advice {
        "set_read_mostly" | "SetReadMostly" => Ok(MemAdvice::SetReadMostly),
        "unset_read_mostly" | "UnsetReadMostly" => Ok(MemAdvice::UnsetReadMostly),
        "set_preferred_location" | "SetPreferredLocation" => Ok(MemAdvice::SetPreferredLocation(
            resolved_target.expect("target required (validated above)"),
        )),
        "unset_preferred_location" | "UnsetPreferredLocation" => {
            Ok(MemAdvice::UnsetPreferredLocation)
        }
        "set_accessed_by" | "SetAccessedBy" => Ok(MemAdvice::SetAccessedBy(
            resolved_target.expect("target required (validated above)"),
        )),
        "unset_accessed_by" | "UnsetAccessedBy" => Ok(MemAdvice::UnsetAccessedBy(
            resolved_target.expect("target required (validated above)"),
        )),
        other => Err(errors::map_str(format!(
            "unknown mem advice: {other:?} (expected one of: set_read_mostly, \
             unset_read_mostly, set_preferred_location, unset_preferred_location, \
             set_accessed_by, unset_accessed_by)"
        ))),
    }
}

// ─── ManagedBufferF32 ─────────────────────────────────────────────

/// Opaque token wrapping a `ManagedRef<f32>` returned from
/// `Memory.allocate_managed_f32`. Cloning is cheap (Arc-clone of the
/// underlying inner). Drop releases the host's strong ref; the
/// allocator's `post_stop` releases the master ref.
#[pyclass(name = "ManagedBufferF32", module = "atomr_accel._native")]
pub struct PyManagedBufferF32 {
    inner: Mutex<Option<ManagedRef<f32>>>,
}

impl PyManagedBufferF32 {
    fn new(m: ManagedRef<f32>) -> Self {
        Self {
            inner: Mutex::new(Some(m)),
        }
    }

    /// Borrow a clone of the underlying `ManagedRef`. `None` if the
    /// buffer was already consumed (we never consume in this module —
    /// kept for symmetry with `PyGpuBufferF32`).
    fn clone_ref(&self) -> Option<ManagedRef<f32>> {
        self.inner.lock().clone()
    }
}

#[pymethods]
impl PyManagedBufferF32 {
    /// Element count.
    #[getter]
    fn len(&self) -> usize {
        self.inner.lock().as_ref().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the underlying allocation is still live (the allocator
    /// actor hasn't run `post_stop`).
    fn is_valid(&self) -> bool {
        self.inner
            .lock()
            .as_ref()
            .map(|m| m.is_valid())
            .unwrap_or(false)
    }

    /// Raw device pointer as an integer. Useful for handing off to
    /// other CUDA tooling (e.g. cupy) that accepts `int` device
    /// pointers. Returns 0 if the buffer is consumed.
    fn device_ptr(&self) -> usize {
        self.inner
            .lock()
            .as_ref()
            .map(|m| m.as_ptr() as usize)
            .unwrap_or(0)
    }

    /// Element-size dtype tag — always `"f32"` for this class.
    #[getter]
    fn dtype(&self) -> &'static str {
        "f32"
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        let g = self.inner.lock();
        match g.as_ref() {
            Some(m) => format!(
                "ManagedBufferF32(len={}, valid={}, ptr=0x{:x})",
                m.len(),
                m.is_valid(),
                m.as_ptr() as usize
            ),
            None => "ManagedBufferF32(consumed)".to_string(),
        }
    }
}

// ─── Memory (handle wrapping ManagedAllocatorActor) ───────────────

#[pyclass(name = "Memory", module = "atomr_accel._native")]
pub struct PyMemory {
    actor_ref: ActorRef<ManagedMsg>,
}

#[pymethods]
impl PyMemory {
    /// Spawn a `ManagedAllocatorActor` under `system`. Returns a
    /// handle wrapping its `ActorRef<ManagedMsg>`.
    #[staticmethod]
    #[pyo3(signature = (system, name=None))]
    fn spawn(py: Python<'_>, system: &PySystem, name: Option<String>) -> PyResult<Py<Self>> {
        let actor_name = name.unwrap_or_else(|| "managed-allocator".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(ManagedAllocatorActor::props(), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyMemory { actor_ref })
    }

    /// Allocate `len` f32 elements of managed (unified) memory.
    /// `flags` is one of `"global"` (default — `cudaMemAttachGlobal`)
    /// or `"host"` (`cudaMemAttachHost`).
    ///
    /// Raises `OutOfMemory` if `cudaMallocManaged` fails (including
    /// on no-driver hosts where cudarc's loader panics — the actor
    /// catches and converts it).
    #[pyo3(signature = (len, flags="global", timeout_secs=10.0))]
    fn allocate_managed_f32(
        &self,
        py: Python<'_>,
        len: usize,
        flags: &str,
        timeout_secs: f64,
    ) -> PyResult<Py<PyManagedBufferF32>> {
        let parsed_flags = flags_from_str(flags)?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        let mref = py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ManagedMsg::AllocateManagedF32 {
                    len,
                    flags: parsed_flags,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(m))) => Ok(m),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("managed allocator dropped reply")),
                    Err(_) => Err(errors::map_str("allocate_managed_f32 timed out")),
                }
            })
        })?;
        Py::new(py, PyManagedBufferF32::new(mref))
    }

    /// Issue `cuMemPrefetchAsync_v2` over the managed allocation
    /// `mem`. `target` is one of `"cpu"` or `"device"`. When
    /// `target == "device"`, `device_id` selects the CUDA device
    /// (defaults to 0).
    ///
    /// On no-driver hosts this returns `Unrecoverable`; on real
    /// hardware with an invalid pointer, `LibraryError`.
    #[pyo3(signature = (mem, target="cpu", device_id=0, timeout_secs=10.0))]
    fn prefetch_f32(
        &self,
        py: Python<'_>,
        mem: Py<PyManagedBufferF32>,
        target: &str,
        device_id: u32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let target = target_from_args(target, device_id)?;
        let mref = mem
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("managed buffer consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ManagedMsg::PrefetchF32 {
                    mem: mref,
                    target,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("managed allocator dropped reply")),
                    Err(_) => Err(errors::map_str("prefetch_f32 timed out")),
                }
            })
        })
    }

    /// Issue `cuMemAdvise_v2` over the managed allocation `mem`.
    /// `advice` is one of:
    ///   - `"set_read_mostly"` / `"unset_read_mostly"`
    ///   - `"set_preferred_location"` / `"unset_preferred_location"`
    ///   - `"set_accessed_by"` / `"unset_accessed_by"`
    ///
    /// Variants that take a target location (`set_preferred_location`,
    /// `set_accessed_by`, `unset_accessed_by`) require `target` to be
    /// supplied (`"cpu"` or `"device"`); `device_id` selects the CUDA
    /// device when `target == "device"` (defaults to 0).
    #[pyo3(signature = (mem, advice, target=None, device_id=0, timeout_secs=10.0))]
    fn advise_f32(
        &self,
        py: Python<'_>,
        mem: Py<PyManagedBufferF32>,
        advice: &str,
        target: Option<&str>,
        device_id: u32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let advice = advice_from_args(advice, target, device_id)?;
        let mref = mem
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("managed buffer consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ManagedMsg::AdviseF32 {
                    mem: mref,
                    advice,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("managed allocator dropped reply")),
                    Err(_) => Err(errors::map_str("advise_f32 timed out")),
                }
            })
        })
    }

    /// Snapshot `ManagedStats` — returns `(allocations, bytes_allocated)`.
    #[pyo3(signature = (timeout_secs=2.0))]
    fn stats(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<(usize, usize)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ManagedMsg::Stats { reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(ManagedStats {
                        allocations,
                        bytes_allocated,
                    })) => Ok((allocations, bytes_allocated)),
                    Ok(Err(_)) => Err(errors::map_str("managed allocator dropped reply")),
                    Err(_) => Err(errors::map_str("stats timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "Memory(handle)"
    }
}

// ─── IPC handles (cuda-ipc-feature-gated) ────────────────────────

#[cfg(feature = "cuda-ipc")]
mod ipc {
    use super::*;
    use atomr_accel_cuda::memory::ipc::{
        get_mem_handle as rs_get_mem_handle, open_mem_handle as rs_open_mem_handle, IpcMemHandle,
        OpenedMem,
    };

    /// Opaque cross-process IPC handle (64 bytes). Round-trip through
    /// `bytes()` / `from_bytes()` to ship via your application's IPC
    /// channel.
    #[pyclass(name = "IpcMemHandle", module = "atomr_accel._native")]
    pub struct PyIpcMemHandle {
        pub(crate) inner: IpcMemHandle,
    }

    #[pymethods]
    impl PyIpcMemHandle {
        /// Construct a handle from its 64-byte serialized form.
        #[staticmethod]
        fn from_bytes(bytes: [u8; 64]) -> Self {
            Self {
                inner: IpcMemHandle::from_bytes(bytes),
            }
        }

        /// 64 raw bytes for cross-process transmission.
        fn bytes(&self) -> [u8; 64] {
            self.inner.as_bytes()
        }

        fn __repr__(&self) -> &'static str {
            "IpcMemHandle(64-bytes)"
        }
    }

    /// Imported memory handle. `Drop` calls `cuIpcCloseMemHandle`.
    /// Use `dev_ptr()` and `bytes()` to forward into other CUDA
    /// tooling.
    #[pyclass(name = "IpcOpenedMem", module = "atomr_accel._native")]
    pub struct PyOpenedMem {
        // Wrapped in Option so we can take/drop on close-from-Python.
        inner: Mutex<Option<OpenedMem>>,
    }

    #[pymethods]
    impl PyOpenedMem {
        /// Raw device pointer as an integer.
        fn dev_ptr(&self) -> usize {
            self.inner
                .lock()
                .as_ref()
                .map(|m| m.dev_ptr() as usize)
                .unwrap_or(0)
        }

        /// Allocation size in bytes.
        #[getter]
        fn bytes(&self) -> usize {
            self.inner
                .lock()
                .as_ref()
                .map(|m| m.bytes())
                .unwrap_or(0)
        }

        /// Explicitly close the handle. Idempotent — calling twice is
        /// fine. The underlying `cuIpcCloseMemHandle` is also called
        /// from `Drop`, so this is purely for callers that want to
        /// force the release ahead of GC.
        fn close(&self) {
            let _ = self.inner.lock().take();
        }

        fn __repr__(&self) -> String {
            let g = self.inner.lock();
            match g.as_ref() {
                Some(m) => format!("IpcOpenedMem(ptr=0x{:x}, bytes={})", m.dev_ptr() as usize, m.bytes()),
                None => "IpcOpenedMem(closed)".to_string(),
            }
        }
    }

    /// Export an IPC handle for the device pointer `dev_ptr`. The
    /// pointer must come from a CUDA allocation in the current
    /// process (e.g. via `cudaMalloc` or `cudaMallocManaged`).
    /// Returns an `IpcMemHandle` that can be serialized to 64 bytes
    /// and shipped to a peer process.
    #[pyfunction(name = "ipc_get_mem_handle")]
    fn py_ipc_get_mem_handle(dev_ptr: usize) -> PyResult<PyIpcMemHandle> {
        let raw = dev_ptr as cudarc::driver::sys::CUdeviceptr;
        rs_get_mem_handle(raw)
            .map(|inner| PyIpcMemHandle { inner })
            .map_err(errors::map_gpu)
    }

    /// Open a previously-exported IPC handle in the current process.
    /// `bytes` is the original allocation size — propagated through
    /// to the returned `IpcOpenedMem`.
    #[pyfunction(name = "ipc_open_mem_handle")]
    fn py_ipc_open_mem_handle(handle: &PyIpcMemHandle, bytes: usize) -> PyResult<PyOpenedMem> {
        rs_open_mem_handle(handle.inner, bytes)
            .map(|m| PyOpenedMem {
                inner: Mutex::new(Some(m)),
            })
            .map_err(errors::map_gpu)
    }

    pub(super) fn register_ipc(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyIpcMemHandle>()?;
        m.add_class::<PyOpenedMem>()?;
        m.add_function(pyo3::wrap_pyfunction!(py_ipc_get_mem_handle, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_ipc_open_mem_handle, m)?)?;
        Ok(())
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyManagedBufferF32>()?;
    m.add_class::<PyMemory>()?;
    #[cfg(feature = "cuda-ipc")]
    ipc::register_ipc(_py, m)?;
    Ok(())
}
