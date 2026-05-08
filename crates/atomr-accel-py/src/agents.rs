//! `atomr_accel_agents` Python wrappers.
//!
//! Phase 2 ships:
//! - `SharedGpuStateCoordinator` — non-generic; spawn + `acquire_write` /
//!   `release_write` representative methods.
//! - `EmbeddingCache` — non-generic; spawn + `insert` / `get`.
//! - `CpuVectorIndex` — non-generic; spawn + `insert`.
//! - `RagPipeline`, `LangGraphGpuActor` — callback / generic-state, so
//!   structural anchors only.

use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_agents::embedding_cache::{
    EmbeddingCache, EmbeddingCacheConfig, EmbeddingCacheMsg,
};
use atomr_accel_agents::shared_state::{
    SharedGpuStateCoordinator, SharedStateMsg, WriteToken,
};
use atomr_accel_agents::vector_index::{CpuVectorIndex, VectorEntry, VectorIndexMsg};
use atomr_core::actor::ActorRef;

use crate::errors;
use crate::runtime::runtime;
use crate::system::PySystem;

// ─── SharedGpuStateCoordinator ───────────────────────────────────

#[pyclass(name = "SharedGpuStateCoordinator", module = "atomr_accel._native")]
pub struct PySharedGpuStateCoordinator {
    actor_ref: ActorRef<SharedStateMsg>,
}

#[pymethods]
impl PySharedGpuStateCoordinator {
    /// Spawn a coordinator under `system`. Returns a handle wrapping
    /// its `ActorRef<SharedStateMsg>`.
    #[staticmethod]
    #[pyo3(signature = (system, name=None))]
    fn spawn(py: Python<'_>, system: &PySystem, name: Option<String>) -> PyResult<Py<Self>> {
        let actor_name = name.unwrap_or_else(|| "shared-state".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(SharedGpuStateCoordinator::props(), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PySharedGpuStateCoordinator { actor_ref })
    }

    /// Acquire the write token for `agent_id`. Blocks until the token
    /// is available; returns the issued token id.
    #[pyo3(signature = (agent_id, timeout_secs=10.0))]
    fn acquire_write(&self, py: Python<'_>, agent_id: u32, timeout_secs: f64) -> PyResult<u64> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SharedStateMsg::AcquireWrite {
                    agent_id,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(WriteToken(t))) => Ok(t),
                    Ok(Err(_)) => Err(errors::map_str("coordinator dropped reply")),
                    Err(_) => Err(errors::map_str("acquire_write timed out")),
                }
            })
        })
    }

    /// Release a previously-acquired write token.
    #[pyo3(signature = (token))]
    fn release_write(&self, token: u64) {
        self.actor_ref.tell(SharedStateMsg::ReleaseWrite {
            token: WriteToken(token),
        });
    }

    fn __repr__(&self) -> &'static str {
        "SharedGpuStateCoordinator(handle)"
    }
}

// ─── EmbeddingCache ─────────────────────────────────────────────

#[pyclass(name = "EmbeddingCache", module = "atomr_accel._native")]
pub struct PyEmbeddingCache {
    actor_ref: ActorRef<EmbeddingCacheMsg>,
}

#[pymethods]
impl PyEmbeddingCache {
    /// Spawn an `EmbeddingCache` under `system`. `capacity_entries`
    /// is the LRU max; `embedding_dim` is the value vector length.
    #[staticmethod]
    #[pyo3(signature = (system, capacity_entries, embedding_dim, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        capacity_entries: usize,
        embedding_dim: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let actor_name = name.unwrap_or_else(|| "embedding-cache".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(
                    EmbeddingCache::props(EmbeddingCacheConfig {
                        capacity_entries,
                        embedding_dim,
                    }),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyEmbeddingCache { actor_ref })
    }

    /// Try the cache. Returns the cached vector (a list of f32) or
    /// `None` on miss.
    #[pyo3(signature = (key, timeout_secs=5.0))]
    fn get(
        &self,
        py: Python<'_>,
        key: Vec<u8>,
        timeout_secs: f64,
    ) -> PyResult<Option<Vec<f32>>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(EmbeddingCacheMsg::Get { key, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(v)) => Ok(v),
                    Ok(Err(_)) => Err(errors::map_str("embedding cache dropped reply")),
                    Err(_) => Err(errors::map_str("get timed out")),
                }
            })
        })
    }

    /// Insert (or replace) the cache entry for `key`.
    #[pyo3(signature = (key, value, timeout_secs=5.0))]
    fn insert(
        &self,
        py: Python<'_>,
        key: Vec<u8>,
        value: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(EmbeddingCacheMsg::Insert {
                    key,
                    value,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(_)) => Err(errors::map_str("embedding cache dropped reply")),
                    Err(_) => Err(errors::map_str("insert timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "EmbeddingCache(handle)"
    }
}

// ─── CpuVectorIndex ─────────────────────────────────────────────

#[pyclass(name = "CpuVectorIndex", module = "atomr_accel._native")]
pub struct PyCpuVectorIndex {
    actor_ref: ActorRef<VectorIndexMsg>,
}

#[pymethods]
impl PyCpuVectorIndex {
    /// Spawn a CPU-resident linear-scan vector index of dimension
    /// `dim`.
    #[staticmethod]
    #[pyo3(signature = (system, dim, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        dim: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let actor_name = name.unwrap_or_else(|| "vector-index".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(CpuVectorIndex::props(dim), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyCpuVectorIndex { actor_ref })
    }

    /// Insert a `(id, embedding)` entry. The embedding's length must
    /// match the index's `dim` — otherwise raises `Unrecoverable`.
    #[pyo3(signature = (id, embedding, timeout_secs=5.0))]
    fn insert(
        &self,
        py: Python<'_>,
        id: u64,
        embedding: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(VectorIndexMsg::Insert {
                    entry: VectorEntry { id, embedding },
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("vector index dropped reply")),
                    Err(_) => Err(errors::map_str("insert timed out")),
                }
            })
        })
    }

    /// Top-k cosine search. Returns `[(id, score), ...]` sorted by
    /// score descending.
    #[pyo3(signature = (query, top_k, timeout_secs=5.0))]
    fn search(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        top_k: usize,
        timeout_secs: f64,
    ) -> PyResult<Vec<(u64, f32)>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(VectorIndexMsg::Search {
                    query,
                    top_k,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("vector index dropped reply")),
                    Err(_) => Err(errors::map_str("search timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "CpuVectorIndex(handle)"
    }
}

// ─── Structural-anchor classes ──────────────────────────────────

/// `RagPipeline` — query → embedding → vector search → context →
/// LLM. Wires three actor refs plus user-supplied `EmbeddingFn` /
/// `LlmFn` closures.
///
/// TODO Phase 2.5: bridge `EmbeddingFn` / `LlmFn` to Python callables
/// (e.g. via a `PyAny`-callable adapter routed through the GIL) so
/// Python callers can spawn a `RagPipeline` end-to-end.
#[pyclass(name = "RagPipeline", module = "atomr_accel._native")]
pub struct PyRagPipeline {}

#[pymethods]
impl PyRagPipeline {
    fn __repr__(&self) -> &'static str {
        "RagPipeline(handle, structural-anchor)"
    }
}

/// `LangGraphGpuActor<S>` — DAG executor with cycle detection.
/// Generic over the user state type `S`.
///
/// TODO Phase 2.5: bridge `NodeGraph<S>` to a Python-side typed-state
/// builder so Python callers can compose nodes and run them.
#[pyclass(name = "LangGraphGpuActor", module = "atomr_accel._native")]
pub struct PyLangGraphGpuActor {}

#[pymethods]
impl PyLangGraphGpuActor {
    fn __repr__(&self) -> &'static str {
        "LangGraphGpuActor(handle, structural-anchor)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySharedGpuStateCoordinator>()?;
    m.add_class::<PyEmbeddingCache>()?;
    m.add_class::<PyCpuVectorIndex>()?;
    m.add_class::<PyRagPipeline>()?;
    m.add_class::<PyLangGraphGpuActor>()?;
    Ok(())
}
