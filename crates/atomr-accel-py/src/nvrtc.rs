//! `NvrtcKernel` — Python wrapper for an NVRTC-compiled kernel
//! handle. Gated on the `nvrtc` feature.
//!
//! Compile a kernel via `Device.compile_kernel(name, src, ...)`; launch
//! it via `NvrtcKernel.launch(grid, block, args, ...)`.
//!
//! ## `KernelArgPy` — typed kernel-arg payloads
//!
//! Python passes a list of [`PyKernelArg`] values to `launch`; each
//! variant maps onto a [`atomr_accel_cuda::kernel::KernelArg`] form:
//!
//! * Buffer variants (one per supported dtype) → `KernelArg::DevSlice`.
//! * Scalar variants (`i32`, `i64`, `f32`, `f64`, `u32`, `u64`) →
//!   `KernelArg::Scalar`.
//!
//! `KernelArg::DevSlice` for `i64`-typed buffers is intentionally *not*
//! exposed because no `PyGpuBufferI64` exists yet (Phase 1 only ships
//! f32/f64/i32/u32/u8 buffer wrappers). When a GpuBufferI64 lands the
//! variant is one new arm.
//!
//! TODO Phase 1.5++ — wire up `KernelArg::Usize` once the device side
//! grows a portable representation. The Rust enum has the variant; we
//! just don't expose it from Python because `usize` round-trips
//! ambiguously through pyo3.

use std::time::Duration;

use atomr_accel_cuda::kernel::{KernelArg, KernelHandle, NvrtcMsg, NvrtcOpts};
use cudarc::driver::LaunchConfig;
use pyo3::prelude::*;
use pyo3::types::PyList;
use tokio::sync::oneshot;

use crate::buffer::{
    PyGpuBufferF32, PyGpuBufferF64, PyGpuBufferI32, PyGpuBufferU32, PyGpuBufferU8,
};
use crate::errors;
use crate::runtime::runtime;

/// Python-visible enum-equivalent: a single typed kernel argument.
///
/// Constructed via the `KernelArg.{buffer_*,scalar_*}` static methods,
/// each of which stashes the value in the inner Rust enum. The inner
/// payload is `Mutex<Option<…>>`-style: each variant takes the value
/// out of the wrapper at marshalling time, so each `PyKernelArg`
/// instance is consumed by exactly one `launch()`.
#[pyclass(name = "KernelArg", module = "atomr_accel._native")]
pub struct PyKernelArg {
    inner: parking_lot::Mutex<Option<KernelArgPyInner>>,
}

/// Inner Rust enum. `KernelArg::DevSlice` carries
/// `Box<dyn DevSliceArg>` (no `Clone`), so we keep the typed handles
/// here and box at marshalling time. Scalars are trivially `Clone`.
enum KernelArgPyInner {
    BufferF32(Py<PyGpuBufferF32>),
    BufferF64(Py<PyGpuBufferF64>),
    BufferI32(Py<PyGpuBufferI32>),
    BufferU32(Py<PyGpuBufferU32>),
    BufferU8(Py<PyGpuBufferU8>),
    ScalarI32(i32),
    ScalarI64(i64),
    ScalarF32(f32),
    ScalarF64(f64),
    ScalarU32(u32),
    ScalarU64(u64),
}

impl PyKernelArg {
    fn wrap(inner: KernelArgPyInner) -> Self {
        Self {
            inner: parking_lot::Mutex::new(Some(inner)),
        }
    }
}

#[pymethods]
impl PyKernelArg {
    /// Wrap an `f32` device buffer.
    #[staticmethod]
    fn buffer_f32(b: Py<PyGpuBufferF32>) -> Self {
        Self::wrap(KernelArgPyInner::BufferF32(b))
    }

    /// Wrap an `f64` device buffer.
    #[staticmethod]
    fn buffer_f64(b: Py<PyGpuBufferF64>) -> Self {
        Self::wrap(KernelArgPyInner::BufferF64(b))
    }

    /// Wrap an `i32` device buffer.
    #[staticmethod]
    fn buffer_i32(b: Py<PyGpuBufferI32>) -> Self {
        Self::wrap(KernelArgPyInner::BufferI32(b))
    }

    /// Wrap a `u32` device buffer.
    #[staticmethod]
    fn buffer_u32(b: Py<PyGpuBufferU32>) -> Self {
        Self::wrap(KernelArgPyInner::BufferU32(b))
    }

    /// Wrap a `u8` device buffer.
    #[staticmethod]
    fn buffer_u8(b: Py<PyGpuBufferU8>) -> Self {
        Self::wrap(KernelArgPyInner::BufferU8(b))
    }

    /// Wrap an `i32` scalar.
    #[staticmethod]
    fn scalar_i32(v: i32) -> Self {
        Self::wrap(KernelArgPyInner::ScalarI32(v))
    }

    /// Wrap an `i64` scalar.
    #[staticmethod]
    fn scalar_i64(v: i64) -> Self {
        Self::wrap(KernelArgPyInner::ScalarI64(v))
    }

    /// Wrap an `f32` scalar.
    #[staticmethod]
    fn scalar_f32(v: f32) -> Self {
        Self::wrap(KernelArgPyInner::ScalarF32(v))
    }

    /// Wrap an `f64` scalar.
    #[staticmethod]
    fn scalar_f64(v: f64) -> Self {
        Self::wrap(KernelArgPyInner::ScalarF64(v))
    }

    /// Wrap a `u32` scalar.
    #[staticmethod]
    fn scalar_u32(v: u32) -> Self {
        Self::wrap(KernelArgPyInner::ScalarU32(v))
    }

    /// Wrap a `u64` scalar.
    #[staticmethod]
    fn scalar_u64(v: u64) -> Self {
        Self::wrap(KernelArgPyInner::ScalarU64(v))
    }

    fn __repr__(&self) -> String {
        let g = self.inner.lock();
        match g.as_ref() {
            None => "KernelArg(consumed)".into(),
            Some(KernelArgPyInner::BufferF32(_)) => "KernelArg(buffer_f32)".into(),
            Some(KernelArgPyInner::BufferF64(_)) => "KernelArg(buffer_f64)".into(),
            Some(KernelArgPyInner::BufferI32(_)) => "KernelArg(buffer_i32)".into(),
            Some(KernelArgPyInner::BufferU32(_)) => "KernelArg(buffer_u32)".into(),
            Some(KernelArgPyInner::BufferU8(_)) => "KernelArg(buffer_u8)".into(),
            Some(KernelArgPyInner::ScalarI32(v)) => format!("KernelArg(scalar_i32={v})"),
            Some(KernelArgPyInner::ScalarI64(v)) => format!("KernelArg(scalar_i64={v})"),
            Some(KernelArgPyInner::ScalarF32(v)) => format!("KernelArg(scalar_f32={v})"),
            Some(KernelArgPyInner::ScalarF64(v)) => format!("KernelArg(scalar_f64={v})"),
            Some(KernelArgPyInner::ScalarU32(v)) => format!("KernelArg(scalar_u32={v})"),
            Some(KernelArgPyInner::ScalarU64(v)) => format!("KernelArg(scalar_u64={v})"),
        }
    }
}

impl PyKernelArg {
    /// Marshal this Python-side arg into a `KernelArg` for the Rust
    /// actor. Buffer variants pull the underlying `GpuRef<T>` out of
    /// the typed Python wrapper; scalar variants box the value. Each
    /// `PyKernelArg` is consumed by a single launch — calling this a
    /// second time returns an error.
    fn take_kernel_arg(&self, py: Python<'_>) -> PyResult<KernelArg> {
        let inner = self
            .inner
            .lock()
            .take()
            .ok_or_else(|| errors::map_str("kernel arg already consumed by a prior launch"))?;
        Ok(match inner {
            KernelArgPyInner::BufferF32(b) => {
                let g = b
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("kernel arg: f32 buffer consumed"))?;
                KernelArg::DevSlice(Box::new(g))
            }
            KernelArgPyInner::BufferF64(b) => {
                let g = b
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("kernel arg: f64 buffer consumed"))?;
                KernelArg::DevSlice(Box::new(g))
            }
            KernelArgPyInner::BufferI32(b) => {
                let g = b
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("kernel arg: i32 buffer consumed"))?;
                KernelArg::DevSlice(Box::new(g))
            }
            KernelArgPyInner::BufferU32(b) => {
                let g = b
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("kernel arg: u32 buffer consumed"))?;
                KernelArg::DevSlice(Box::new(g))
            }
            KernelArgPyInner::BufferU8(b) => {
                let g = b
                    .borrow(py)
                    .clone_ref()
                    .ok_or_else(|| errors::map_str("kernel arg: u8 buffer consumed"))?;
                KernelArg::DevSlice(Box::new(g))
            }
            KernelArgPyInner::ScalarI32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArgPyInner::ScalarI64(v) => KernelArg::Scalar(Box::new(v)),
            KernelArgPyInner::ScalarF32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArgPyInner::ScalarF64(v) => KernelArg::Scalar(Box::new(v)),
            KernelArgPyInner::ScalarU32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArgPyInner::ScalarU64(v) => KernelArg::Scalar(Box::new(v)),
        })
    }
}

#[pyclass(name = "NvrtcKernel", module = "atomr_accel._native")]
pub struct PyNvrtcKernel {
    pub(crate) handle: KernelHandle,
    /// Actor ref retained so `launch()` can re-enqueue work without
    /// having to round-trip through `Device.snapshot_children` again.
    pub(crate) actor_ref: atomr_core::actor::ActorRef<NvrtcMsg>,
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

    /// Launch the compiled kernel.
    ///
    /// `grid` and `block` are 3-tuples; `shared` is the dynamic shared
    /// memory size in bytes. `args` is a list of `KernelArg` values
    /// (constructed via `KernelArg.buffer_*` / `KernelArg.scalar_*`).
    /// Returns once the launch has been enqueued *and* the configured
    /// completion strategy has signalled the launch finished. In mock
    /// mode this returns `Unrecoverable("NvrtcActor in mock mode")`.
    #[pyo3(signature = (grid, block, args, shared=0, timeout_secs=60.0))]
    fn launch(
        &self,
        py: Python<'_>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &Bound<'_, PyList>,
        shared: u32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        // Marshal the Python-side `KernelArg` list into the Rust enum.
        let mut rust_args: Vec<KernelArg> = Vec::with_capacity(args.len());
        for item in args.iter() {
            let arg: PyRef<PyKernelArg> = item.extract()?;
            rust_args.push(arg.take_kernel_arg(py)?);
        }

        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared,
        };
        let handle = self.handle.clone();
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(NvrtcMsg::Launch {
                    kernel: handle,
                    args: rust_args,
                    cfg,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("nvrtc dropped reply")),
                    Err(_) => Err(errors::map_str("nvrtc launch timed out")),
                }
            })
        })
    }
}

/// Send a `NvrtcMsg::Compile` to the supplied actor and wrap the
/// resulting handle in a `PyNvrtcKernel`. Used by
/// `PyDevice::compile_kernel`.
pub(crate) fn compile_via_actor(
    py: Python<'_>,
    actor: atomr_core::actor::ActorRef<NvrtcMsg>,
    name: String,
    src: String,
    timeout_secs: f64,
) -> PyResult<Py<PyNvrtcKernel>> {
    // NvrtcOpts are deliberately default for now; the Phase-5 builder
    // surface (LTO / `--std=c++17` / SmArch / name expressions) can be
    // exposed in a follow-up by accepting a Python `dict`.
    let opts = NvrtcOpts::default();
    let actor_for_msg = actor.clone();
    let rt = runtime();
    let handle = py.allow_threads(|| {
        rt.block_on(async move {
            let (tx, rx) = oneshot::channel();
            actor_for_msg.tell(NvrtcMsg::Compile {
                src,
                kernel_name: name,
                opts,
                reply: tx,
            });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(h))) => Ok(h),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("nvrtc dropped reply")),
                Err(_) => Err(errors::map_str("nvrtc compile timed out")),
            }
        })
    })?;
    Py::new(
        py,
        PyNvrtcKernel {
            handle,
            actor_ref: actor,
        },
    )
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNvrtcKernel>()?;
    m.add_class::<PyKernelArg>()?;
    Ok(())
}
