//! `atomr_accel_cuda_realtime` Python wrappers.
//!
//! Phase 2 ships handle classes for the realtime / simulation actors.
//! All six are non-generic with simple message types, so each gets a
//! `spawn(...)` constructor plus one representative method exercising
//! the dispatch path. Mock-mode replies surface naturally because the
//! underlying actors run their CPU reference logic without touching
//! CUDA.
//!
//! TODO Phase 2.5: widen the surface to cover `Snapshot` /
//! `QueryNeighbors` / `Lookup` / `ProcessGpu` / `UpdateConfig` etc.

use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda_realtime::cloth::{ClothConfig, ClothMsg, ClothSimulationActor};
use atomr_accel_cuda_realtime::fluid::{FluidConfig, FluidMsg, FluidSimulationActor};
use atomr_accel_cuda_realtime::hashmap::{
    GpuHashMapActor, GpuHashMapConfig, GpuHashMapMsg,
};
use atomr_accel_cuda_realtime::image_filter::{
    ImageFilterConfig, ImageFilterMsg, ImageFilterPipeline,
};
use atomr_accel_cuda_realtime::particle::{
    ParticleSystemActor, ParticleSystemConfig, ParticleMsg, Vec3,
};
use atomr_accel_cuda_realtime::spatial_index::{
    SpatialIndexActor, SpatialIndexConfig, SpatialMsg,
};
use atomr_core::actor::ActorRef;

use crate::errors;
use crate::runtime::runtime;
use crate::system::PySystem;

// ─── ClothSimulationActor ────────────────────────────────────────

#[pyclass(name = "ClothSimulationActor", module = "atomr_accel._native")]
pub struct PyClothSimulationActor {
    actor_ref: ActorRef<ClothMsg>,
}

#[pymethods]
impl PyClothSimulationActor {
    /// Spawn a verlet cloth grid. Defaults: 16×16, spacing 0.1,
    /// gravity (0, -9.8, 0), 4 inner iterations, no pinned points.
    #[staticmethod]
    #[pyo3(signature = (
        system, width=16, height=16, spacing=0.1, stiffness=0.5,
        iterations=4, name=None,
    ))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        width: usize,
        height: usize,
        spacing: f32,
        stiffness: f32,
        iterations: u32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let cfg = ClothConfig {
            width,
            height,
            spacing,
            gravity: Vec3 {
                x: 0.0,
                y: -9.8,
                z: 0.0,
            },
            pinned: vec![false; width * height],
            stiffness,
            iterations,
        };
        let actor_name = name.unwrap_or_else(|| "cloth".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(ClothSimulationActor::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyClothSimulationActor { actor_ref })
    }

    /// Advance the cloth simulation by `dt` seconds.
    #[pyo3(signature = (dt, timeout_secs=5.0))]
    fn step(&self, py: Python<'_>, dt: f32, timeout_secs: f64) -> PyResult<()> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ClothMsg::Step { dt, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cloth dropped reply")),
                    Err(_) => Err(errors::map_str("step timed out")),
                }
            })
        })
    }

    /// Async counterpart of `step`.
    #[pyo3(signature = (dt, timeout_secs=5.0))]
    fn step_async<'py>(
        &self,
        py: Python<'py>,
        dt: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(ClothMsg::Step { dt, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("cloth dropped reply")),
                Err(_) => Err(errors::map_str("step timed out")),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "ClothSimulationActor(handle)"
    }
}

// ─── FluidSimulationActor ────────────────────────────────────────

#[pyclass(name = "FluidSimulationActor", module = "atomr_accel._native")]
pub struct PyFluidSimulationActor {
    actor_ref: ActorRef<FluidMsg>,
}

#[pymethods]
impl PyFluidSimulationActor {
    /// Spawn a 2D Eulerian fluid grid.
    #[staticmethod]
    #[pyo3(signature = (system, width=32, height=32, viscosity=0.0, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        width: usize,
        height: usize,
        viscosity: f32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let cfg = FluidConfig {
            width,
            height,
            viscosity,
        };
        let actor_name = name.unwrap_or_else(|| "fluid".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(FluidSimulationActor::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyFluidSimulationActor { actor_ref })
    }

    /// Advance the fluid simulation by `dt` seconds.
    #[pyo3(signature = (dt, timeout_secs=5.0))]
    fn step(&self, py: Python<'_>, dt: f32, timeout_secs: f64) -> PyResult<()> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(FluidMsg::Step { dt, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("fluid dropped reply")),
                    Err(_) => Err(errors::map_str("step timed out")),
                }
            })
        })
    }

    /// Async counterpart of `step`.
    #[pyo3(signature = (dt, timeout_secs=5.0))]
    fn step_async<'py>(
        &self,
        py: Python<'py>,
        dt: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(FluidMsg::Step { dt, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(()))) => Ok(()),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("fluid dropped reply")),
                Err(_) => Err(errors::map_str("step timed out")),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "FluidSimulationActor(handle)"
    }
}

// ─── ParticleSystemActor ────────────────────────────────────────

#[pyclass(name = "ParticleSystemActor", module = "atomr_accel._native")]
pub struct PyParticleSystemActor {
    actor_ref: ActorRef<ParticleMsg>,
}

#[pymethods]
impl PyParticleSystemActor {
    /// Spawn a particle system actor with default config.
    #[staticmethod]
    #[pyo3(signature = (system, gravity_y=-9.8, drag=0.0, bounce=0.5, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        gravity_y: f32,
        drag: f32,
        bounce: f32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let cfg = ParticleSystemConfig {
            gravity: Vec3 {
                x: 0.0,
                y: gravity_y,
                z: 0.0,
            },
            drag,
            bounds: None,
            bounce,
        };
        let actor_name = name.unwrap_or_else(|| "particles".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(ParticleSystemActor::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyParticleSystemActor { actor_ref })
    }

    /// Advance the particle system by `dt` seconds. Returns the
    /// number of live particles after the step.
    #[pyo3(signature = (dt, timeout_secs=5.0))]
    fn step(&self, py: Python<'_>, dt: f32, timeout_secs: f64) -> PyResult<usize> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ParticleMsg::Step { dt, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(n))) => Ok(n),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("particle system dropped reply")),
                    Err(_) => Err(errors::map_str("step timed out")),
                }
            })
        })
    }

    /// Async counterpart of `step`.
    #[pyo3(signature = (dt, timeout_secs=5.0))]
    fn step_async<'py>(
        &self,
        py: Python<'py>,
        dt: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(ParticleMsg::Step { dt, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(Ok(n))) => Ok(n),
                Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                Ok(Err(_)) => Err(errors::map_str("particle system dropped reply")),
                Err(_) => Err(errors::map_str("step timed out")),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "ParticleSystemActor(handle)"
    }
}

// ─── SpatialIndexActor ──────────────────────────────────────────

#[pyclass(name = "SpatialIndexActor", module = "atomr_accel._native")]
pub struct PySpatialIndexActor {
    actor_ref: ActorRef<SpatialMsg>,
}

#[pymethods]
impl PySpatialIndexActor {
    /// Spawn a uniform-grid spatial hash with the given cell size.
    #[staticmethod]
    #[pyo3(signature = (system, cell_size, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        cell_size: f32,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let cfg = SpatialIndexConfig { cell_size };
        let actor_name = name.unwrap_or_else(|| "spatial-index".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(SpatialIndexActor::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PySpatialIndexActor { actor_ref })
    }

    /// Query for all point ids in the 3×3×3 cell neighborhood of
    /// `(x, y, z)`.
    #[pyo3(signature = (x, y, z, timeout_secs=2.0))]
    fn query_neighbors(
        &self,
        py: Python<'_>,
        x: f32,
        y: f32,
        z: f32,
        timeout_secs: f64,
    ) -> PyResult<Vec<u64>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SpatialMsg::QueryNeighbors {
                    x,
                    y,
                    z,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(v)) => Ok(v),
                    Ok(Err(_)) => Err(errors::map_str("spatial index dropped reply")),
                    Err(_) => Err(errors::map_str("query_neighbors timed out")),
                }
            })
        })
    }

    /// Async counterpart of `query_neighbors`.
    #[pyo3(signature = (x, y, z, timeout_secs=2.0))]
    fn query_neighbors_async<'py>(
        &self,
        py: Python<'py>,
        x: f32,
        y: f32,
        z: f32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(SpatialMsg::QueryNeighbors { x, y, z, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(_)) => Err(errors::map_str("spatial index dropped reply")),
                Err(_) => Err(errors::map_str("query_neighbors timed out")),
            }
        })
    }

    fn __repr__(&self) -> &'static str {
        "SpatialIndexActor(handle)"
    }
}

// ─── GpuHashMapActor ────────────────────────────────────────────

#[pyclass(name = "GpuHashMapActor", module = "atomr_accel._native")]
pub struct PyGpuHashMapActor {
    actor_ref: ActorRef<GpuHashMapMsg>,
}

#[pymethods]
impl PyGpuHashMapActor {
    /// Spawn an open-addressing hashmap actor.
    #[staticmethod]
    #[pyo3(signature = (system, capacity, key_size_bytes, value_size_bytes, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        capacity: usize,
        key_size_bytes: usize,
        value_size_bytes: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let cfg = GpuHashMapConfig {
            capacity,
            key_size_bytes,
            value_size_bytes,
        };
        let actor_name = name.unwrap_or_else(|| "gpu-hashmap".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(GpuHashMapActor::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyGpuHashMapActor { actor_ref })
    }

    /// Insert one or more `(key, value)` pairs encoded as flat byte
    /// vectors. `keys.len()` must be `n × key_size_bytes`,
    /// `values.len()` must be `n × value_size_bytes`. Returns the
    /// number of entries actually inserted (capped at `capacity`).
    #[pyo3(signature = (keys, values, timeout_secs=5.0))]
    fn insert(
        &self,
        py: Python<'_>,
        keys: Vec<u8>,
        values: Vec<u8>,
        timeout_secs: f64,
    ) -> PyResult<u32> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(GpuHashMapMsg::Insert {
                    keys,
                    values,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(n))) => Ok(n),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("hashmap dropped reply")),
                    Err(_) => Err(errors::map_str("insert timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "GpuHashMapActor(handle)"
    }
}

// ─── ImageFilterPipeline ────────────────────────────────────────

#[pyclass(name = "ImageFilterPipeline", module = "atomr_accel._native")]
pub struct PyImageFilterPipeline {
    actor_ref: ActorRef<ImageFilterMsg>,
}

#[pymethods]
impl PyImageFilterPipeline {
    /// Spawn an image filter pipeline. `kernel_3x3` must be a
    /// length-9 list of f32 weights, row-major.
    #[staticmethod]
    #[pyo3(signature = (system, width, height, channels, kernel_3x3, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        width: u32,
        height: u32,
        channels: u32,
        kernel_3x3: Vec<f32>,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if kernel_3x3.len() != 9 {
            return Err(errors::map_str(format!(
                "kernel_3x3 must be length 9 (got {})",
                kernel_3x3.len()
            )));
        }
        let cfg = ImageFilterConfig {
            width,
            height,
            channels,
            kernel_3x3,
        };
        let actor_name = name.unwrap_or_else(|| "image-filter".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(ImageFilterPipeline::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyImageFilterPipeline { actor_ref })
    }

    /// Host-side per-pixel 3×3 convolve + clamp. Frame must be
    /// `width × height × channels` bytes.
    #[pyo3(signature = (frame, timeout_secs=10.0))]
    fn process(
        &self,
        py: Python<'_>,
        frame: Vec<u8>,
        timeout_secs: f64,
    ) -> PyResult<Vec<u8>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ImageFilterMsg::Process { frame, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(out))) => Ok(out),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("image filter dropped reply")),
                    Err(_) => Err(errors::map_str("process timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "ImageFilterPipeline(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyClothSimulationActor>()?;
    m.add_class::<PyFluidSimulationActor>()?;
    m.add_class::<PyParticleSystemActor>()?;
    m.add_class::<PySpatialIndexActor>()?;
    m.add_class::<PyGpuHashMapActor>()?;
    m.add_class::<PyImageFilterPipeline>()?;
    Ok(())
}
