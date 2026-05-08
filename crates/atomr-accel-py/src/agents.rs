//! `atomr_accel_agents` Python wrappers.
//!
//! Phase 2 ships:
//! - `SharedGpuStateCoordinator` — non-generic; spawn + `acquire_write` /
//!   `release_write` representative methods.
//! - `EmbeddingCache` — non-generic; spawn + `insert` / `get`.
//! - `CpuVectorIndex` — non-generic; spawn + `insert`.
//! - `RagPipeline`, `LangGraphGpuActor` — callback / generic-state, so
//!   structural anchors only.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_agents::embedding_cache::{
    EmbeddingCache, EmbeddingCacheConfig, EmbeddingCacheMsg,
};
use atomr_accel_agents::langgraph_nodes::{
    GraphNode, LangGraphGpuActor, NodeGraph, NodeGraphMsg, NodeId,
};
use atomr_accel_agents::rag::{EmbeddingFn, LlmFn, RagConfig, RagMsg, RagPipeline, RagQuery};
use atomr_accel_agents::shared_state::{SharedGpuStateCoordinator, SharedStateMsg, WriteToken};
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

    /// Async counterpart of `acquire_write`.
    #[pyo3(signature = (agent_id, timeout_secs=10.0))]
    fn acquire_write_async<'py>(
        &self,
        py: Python<'py>,
        agent_id: u32,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    }

    /// Async counterpart of `release_write`. Fire-and-forget; resolves
    /// to `None` once the message is enqueued.
    #[pyo3(signature = (token))]
    fn release_write_async<'py>(&self, py: Python<'py>, token: u64) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            actor.tell(SharedStateMsg::ReleaseWrite {
                token: WriteToken(token),
            });
            Ok::<(), PyErr>(())
        })
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
    fn get(&self, py: Python<'_>, key: Vec<u8>, timeout_secs: f64) -> PyResult<Option<Vec<f32>>> {
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

    /// Async counterpart of `get`.
    #[pyo3(signature = (key, timeout_secs=5.0))]
    fn get_async<'py>(
        &self,
        py: Python<'py>,
        key: Vec<u8>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let (tx, rx) = oneshot::channel();
            actor.tell(EmbeddingCacheMsg::Get { key, reply: tx });
            match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(_)) => Err(errors::map_str("embedding cache dropped reply")),
                Err(_) => Err(errors::map_str("get timed out")),
            }
        })
    }

    /// Async counterpart of `insert`.
    #[pyo3(signature = (key, value, timeout_secs=5.0))]
    fn insert_async<'py>(
        &self,
        py: Python<'py>,
        key: Vec<u8>,
        value: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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

    /// Async counterpart of `insert`.
    #[pyo3(signature = (id, embedding, timeout_secs=5.0))]
    fn insert_async<'py>(
        &self,
        py: Python<'py>,
        id: u64,
        embedding: Vec<f32>,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    }

    /// Async counterpart of `search`.
    #[pyo3(signature = (query, top_k, timeout_secs=5.0))]
    fn search_async<'py>(
        &self,
        py: Python<'py>,
        query: Vec<f32>,
        top_k: usize,
        timeout_secs: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let actor = self.actor_ref.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    }

    fn __repr__(&self) -> &'static str {
        "CpuVectorIndex(handle)"
    }
}

// ─── RagPipeline (Phase 2.5) ────────────────────────────────────

/// `RagPipeline` handle.
///
/// Phase 2.5: wires an internal length-deterministic `EmbeddingFn`
/// + a stub `LlmFn` so Python callers can drive `query(text, k=...)`
/// end-to-end against a host-side `EmbeddingCache` + `CpuVectorIndex`
/// they've already populated. Phase 2.6 will accept Python callables
/// for `EmbeddingFn` / `LlmFn`.
#[pyclass(name = "RagPipeline", module = "atomr_accel._native")]
pub struct PyRagPipeline {
    actor_ref: ActorRef<RagMsg>,
}

#[pymethods]
impl PyRagPipeline {
    /// Spawn a `RagPipeline` wired to an existing `EmbeddingCache`
    /// and `CpuVectorIndex` actor. The internal `EmbeddingFn` returns
    /// a deterministic embedding derived from the query text bytes
    /// (hash truncated/padded to `embedding_dim`); the `LlmFn`
    /// returns a stub answer string referencing the retrieved ids.
    #[staticmethod]
    #[pyo3(signature = (
        system, cache, index, embedding_dim, timeout_secs=5.0, name=None,
    ))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        cache: &PyEmbeddingCache,
        index: &PyCpuVectorIndex,
        embedding_dim: usize,
        timeout_secs: f64,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if embedding_dim == 0 {
            return Err(errors::map_str("embedding_dim must be ≥ 1"));
        }
        // Phase 2.5: typed payloads — internal embedding fn produces
        // a deterministic vector based on the byte sum of the query.
        let dim = embedding_dim;
        let embed_fn: Arc<dyn EmbeddingFn> = Arc::new(move |text: &str| {
            let bytes = text.as_bytes();
            let mut v = vec![0.0f32; dim];
            for (i, &b) in bytes.iter().enumerate() {
                v[i % dim] += b as f32;
            }
            // L2-normalize.
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            Ok(v)
        });
        let llm: Arc<dyn LlmFn> =
            Arc::new(|q: &str, ctx: &[u64]| Ok(format!("rag:'{q}' sources={ctx:?}")));

        let actor_name = name.unwrap_or_else(|| "rag-pipeline".to_string());
        let cache_ref = cache.actor_ref.clone();
        let index_ref = index.actor_ref.clone();
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(
                    RagPipeline::props(RagConfig {
                        embedding: embed_fn,
                        embedding_cache: cache_ref,
                        vector_index: index_ref,
                        llm,
                        timeout: Duration::from_secs_f64(timeout_secs),
                    }),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyRagPipeline { actor_ref })
    }

    /// Query the pipeline. Returns `(answer, source_ids,
    /// embedding_was_cached)`.
    #[pyo3(signature = (text, k=5, timeout_secs=10.0))]
    fn query(
        &self,
        py: Python<'_>,
        text: String,
        k: usize,
        timeout_secs: f64,
    ) -> PyResult<(String, Vec<u64>, bool)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(RagMsg::Query {
                    q: RagQuery {
                        text,
                        top_k: k.max(1),
                    },
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(r))) => Ok((r.answer, r.sources, r.embedding_was_cached)),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("rag pipeline dropped reply")),
                    Err(_) => Err(errors::map_str("query timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "RagPipeline(handle)"
    }
}

// ─── LangGraphGpuActor (Phase 2.5) ──────────────────────────────

/// `LangGraphGpuActor` handle (Phase 2.5: monomorphized over
/// `S = Vec<f32>`; nodes are internal stages that scale the state by
/// a per-node factor).
#[pyclass(name = "LangGraphGpuActor", module = "atomr_accel._native")]
pub struct PyLangGraphGpuActor {
    actor_ref: ActorRef<NodeGraphMsg<Vec<f32>>>,
}

#[pymethods]
impl PyLangGraphGpuActor {
    /// Spawn a `LangGraphGpuActor` over a linear chain of `n_nodes`
    /// internal stages. Node `i` multiplies every element of the
    /// state vector by `i+1`. Phase 2.6 will accept Python-supplied
    /// `GraphNode` callables and an arbitrary topology.
    #[staticmethod]
    #[pyo3(signature = (system, n_nodes=2, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        n_nodes: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if n_nodes == 0 {
            return Err(errors::map_str("n_nodes must be ≥ 1"));
        }
        // Phase 2.5: typed payloads — linear chain of scale-by-(i+1)
        // nodes.
        let mut g = NodeGraph::<Vec<f32>>::new();
        for i in 0..n_nodes {
            let factor = (i + 1) as f32;
            let node: Arc<dyn GraphNode<Vec<f32>>> = Arc::new(move |state: Vec<f32>| {
                Ok(state.into_iter().map(|x| x * factor).collect())
            });
            // SAFETY: `add_node` expects the impl on a value, not Arc.
            // Use a closure adapter that calls the Arc-stored node.
            let node_inner = node.clone();
            g.add_node(NodeId(i as u32), move |s: Vec<f32>| node_inner.run(s));
            if i + 1 < n_nodes {
                g.add_edge(NodeId(i as u32), NodeId((i + 1) as u32));
            }
        }
        g.set_entry(NodeId(0));
        let actor_name = name.unwrap_or_else(|| "langgraph".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(LangGraphGpuActor::<Vec<f32>>::props(g), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PyLangGraphGpuActor { actor_ref })
    }

    /// Run the node graph against `state`. Returns the transformed
    /// state vector.
    // Phase 2.5: typed payloads — `state` is a `Vec<f32>` until the
    // typed-state builder lands.
    #[pyo3(signature = (state, timeout_secs=10.0))]
    fn step(&self, py: Python<'_>, state: Vec<f32>, timeout_secs: f64) -> PyResult<Vec<f32>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(NodeGraphMsg::Run { state, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(s))) => Ok(s),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("langgraph dropped reply")),
                    Err(_) => Err(errors::map_str("step timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "LangGraphGpuActor(handle)"
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
