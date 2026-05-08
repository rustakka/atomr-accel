//! `atomr_accel_patterns` Python wrappers.
//!
//! Phase 2 shipped these classes as **structural anchors** —
//! `__repr__` only — because every actor in `atomr-accel-patterns`
//! is generic over a user-supplied `Req`/`Resp`, expert protocol,
//! backend protocol, or callback closure (`BatchFn`, `DraftFn`,
//! `VerifierFn`, `GateFn`, …).
//!
//! Phase 2.5 (this module) lands a representative `spawn(...)` +
//! one method per actor, using **opaque `Vec<u8>` payloads** for the
//! generic Req/Resp slots (the simplest concrete shape that crosses
//! PyO3 cleanly). Internal echo / mock helpers stand in for the
//! user-supplied closures and per-replica actors. Callers that need
//! typed payloads serialize on the Python side.
//!
//! TODO Phase 2.6: typed payloads via a marshal layer (`bincode`,
//! `arrow`, raw numpy buffers, …) plus user-supplied callbacks via
//! a `PyAny`-callable adapter routed through the GIL.

use std::sync::Arc;
use std::time::Duration;

use atomr_core::prelude::async_trait;
use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;
use atomr_accel_patterns::batching::{
    BatchFn, BatchOverflow, BatchingConfig, BatchingMsg, DynamicBatchingServer,
};
use atomr_accel_patterns::cascade::{
    CascadeConfig, CascadeMsg, CascadeStage, CascadeStageEntry, InferenceCascade,
};
use atomr_accel_patterns::hot_swap::{BackendProtocol, HotSwapMsg, ModelHotSwapServer};
use atomr_accel_patterns::moe::{ExpertProtocol, MoeConfig, MoeMsg, MoeRouter};
use atomr_accel_patterns::replica_pool::{
    ModelReplicaPool, ReplicaMessage, ReplicaPoolConfig, ReplicaPoolMsg, RoutingPolicy,
};
use atomr_accel_patterns::scheduler::{
    FairDispatcher, FairShareConfig, FairShareMsg, FairShareScheduler, TenantConfig, TenantId,
};
use atomr_accel_patterns::speculative::{
    DraftFn, SpecMsg, SpeculativeConfig, SpeculativeDecoder, VerifierFn,
};
use atomr_core::actor::{Actor, ActorRef, Context, Props};

use crate::errors;
use crate::runtime::runtime;
use crate::system::PySystem;

// ─── DynamicBatchingServer ──────────────────────────────────────

/// `DynamicBatchingServer` handle (Phase 2.5: typed payloads pending —
/// shipped with `Req = Resp = Vec<u8>` and an internal echo `BatchFn`).
#[pyclass(name = "DynamicBatchingServer", module = "atomr_accel._native")]
pub struct PyDynamicBatchingServer {
    actor_ref: ActorRef<BatchingMsg<Vec<u8>, Vec<u8>>>,
}

#[pymethods]
impl PyDynamicBatchingServer {
    /// Spawn a `DynamicBatchingServer` with an internal echo
    /// `BatchFn` (`Vec<u8>` in → `Vec<u8>` out, identity). Phase 2.6
    /// will accept a Python-side `BatchFn` callable.
    #[staticmethod]
    #[pyo3(signature = (system, max_batch=8, max_wait_ms=10, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        max_batch: usize,
        max_wait_ms: u64,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        // Phase 2.5: typed payloads — internal echo BatchFn.
        let echo: Arc<dyn BatchFn<Vec<u8>, Vec<u8>>> =
            Arc::new(|reqs: Vec<Vec<u8>>| reqs.into_iter().map(Ok).collect());
        let cfg = BatchingConfig {
            max_batch,
            max_wait: Duration::from_millis(max_wait_ms),
            batch_fn: echo,
            overflow: BatchOverflow::Reject,
        };
        let actor_name = name.unwrap_or_else(|| "batching".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(
                    DynamicBatchingServer::<Vec<u8>, Vec<u8>>::props(cfg),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyDynamicBatchingServer { actor_ref })
    }

    /// Submit an opaque payload through the batcher. With the
    /// default echo `BatchFn` the response is the input bytes.
    // Phase 2.5: typed payloads — `payload` is a raw byte string;
    // callers serialize on the Python side.
    #[pyo3(signature = (payload, timeout_secs=5.0))]
    fn submit(&self, py: Python<'_>, payload: Vec<u8>, timeout_secs: f64) -> PyResult<Vec<u8>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(BatchingMsg::Submit {
                    req: payload,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("batching server dropped reply")),
                    Err(_) => Err(errors::map_str("submit timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "DynamicBatchingServer(handle)"
    }
}

// ─── InferenceCascade ───────────────────────────────────────────

/// `InferenceCascade` handle (Phase 2.5: `Req = Resp = Vec<u8>` with
/// a single internal echo stage at confidence 1.0).
#[pyclass(name = "InferenceCascade", module = "atomr_accel._native")]
pub struct PyInferenceCascade {
    actor_ref: ActorRef<CascadeMsg<Vec<u8>, Vec<u8>>>,
}

#[pymethods]
impl PyInferenceCascade {
    /// Spawn a single-stage cascade whose stage echoes the input
    /// at confidence 1.0. Phase 2.6 will accept Python `CascadeStage`
    /// callables.
    #[staticmethod]
    #[pyo3(signature = (system, name=None))]
    fn spawn(py: Python<'_>, system: &PySystem, name: Option<String>) -> PyResult<Py<Self>> {
        // Phase 2.5: typed payloads — single internal echo stage.
        let stage: Arc<dyn CascadeStage<Vec<u8>, Vec<u8>>> =
            Arc::new(|req: &Vec<u8>| Ok((req.clone(), 1.0)));
        let cfg = CascadeConfig {
            stages: vec![CascadeStageEntry {
                stage,
                confidence_threshold: 0.0,
            }],
        };
        let actor_name = name.unwrap_or_else(|| "cascade".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(
                    InferenceCascade::<Vec<u8>, Vec<u8>>::props(cfg),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyInferenceCascade { actor_ref })
    }

    /// Run the cascade. Returns `(response_bytes, stage_index,
    /// confidence)` — the index of the stage that produced the reply
    /// plus its confidence.
    // Phase 2.5: typed payloads — opaque bytes in/out.
    #[pyo3(signature = (input, timeout_secs=5.0))]
    fn infer(
        &self,
        py: Python<'_>,
        input: Vec<u8>,
        timeout_secs: f64,
    ) -> PyResult<(Vec<u8>, usize, f32)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(CascadeMsg::Predict {
                    req: input,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(r))) => Ok((r.response, r.stage_index, r.confidence)),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cascade dropped reply")),
                    Err(_) => Err(errors::map_str("infer timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "InferenceCascade(handle)"
    }
}

// ─── ModelReplicaPool ───────────────────────────────────────────

/// Internal echo replica: each replica runs an `EchoReplicaActor`
/// that replies to `EchoReplicaMsg::Submit` with the request bytes
/// unchanged. `ReplicaMessage` is implemented directly on the `Msg`
/// type (the `ModelReplicaPool` is generic over the message, not a
/// protocol marker).
pub(crate) enum EchoReplicaMsg {
    Submit {
        req: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, GpuError>>,
    },
}

pub(crate) struct EchoReplicaActor;

#[async_trait]
impl Actor for EchoReplicaActor {
    type Msg = EchoReplicaMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: EchoReplicaMsg) {
        match msg {
            EchoReplicaMsg::Submit { req, reply } => {
                let _ = reply.send(Ok(req));
            }
        }
    }
}

impl ReplicaMessage for EchoReplicaMsg {
    type Req = Vec<u8>;
    type Resp = Vec<u8>;
    fn make_submit(req: Vec<u8>, reply: oneshot::Sender<Result<Vec<u8>, GpuError>>) -> Self {
        EchoReplicaMsg::Submit { req, reply }
    }
}

/// `ModelReplicaPool` handle (Phase 2.5: opaque `Vec<u8>` replicas).
#[pyclass(name = "ModelReplicaPool", module = "atomr_accel._native")]
pub struct PyModelReplicaPool {
    actor_ref: ActorRef<ReplicaPoolMsg<EchoReplicaMsg>>,
}

#[pymethods]
impl PyModelReplicaPool {
    /// Spawn a pool of `n_replicas` internal echo replicas with
    /// round-robin routing. Phase 2.6 will accept Python-supplied
    /// replica actor refs.
    #[staticmethod]
    #[pyo3(signature = (system, n_replicas=2, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        n_replicas: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if n_replicas == 0 {
            return Err(errors::map_str("n_replicas must be ≥ 1"));
        }
        let actor_name = name.unwrap_or_else(|| "replica-pool".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            // Phase 2.5: typed payloads — internal echo replicas.
            let mut replicas = Vec::with_capacity(n_replicas);
            for i in 0..n_replicas {
                let r = system
                    .inner
                    .actor_of(
                        Props::create(|| EchoReplicaActor),
                        &format!("{actor_name}-replica-{i}"),
                    )
                    .map_err(errors::map_str)?;
                replicas.push(r);
            }
            system
                .inner
                .actor_of(
                    ModelReplicaPool::<EchoReplicaMsg>::props(ReplicaPoolConfig {
                        replicas,
                        policy: RoutingPolicy::RoundRobin,
                    }),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyModelReplicaPool { actor_ref })
    }

    /// Submit a payload to the pool; the chosen replica's reply is
    /// returned. Phase 2.6 will add an `acquire`/`release` lease API
    /// once the typed-replica contract lands.
    // Phase 2.5: typed payloads — bytes in/out.
    #[pyo3(signature = (payload, timeout_secs=5.0))]
    fn submit(&self, py: Python<'_>, payload: Vec<u8>, timeout_secs: f64) -> PyResult<Vec<u8>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(ReplicaPoolMsg::Submit {
                    req: payload,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("replica pool dropped reply")),
                    Err(_) => Err(errors::map_str("submit timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "ModelReplicaPool(handle)"
    }
}

// ─── FairShareScheduler ─────────────────────────────────────────

/// `FairShareScheduler` handle (Phase 2.5: `Vec<u8>` payloads,
/// internal echo dispatcher).
#[pyclass(name = "FairShareScheduler", module = "atomr_accel._native")]
pub struct PyFairShareScheduler {
    actor_ref: ActorRef<FairShareMsg<Vec<u8>, Vec<u8>>>,
}

#[pymethods]
impl PyFairShareScheduler {
    /// Spawn a fair-share scheduler. `tenants` is a list of
    /// `(id, weight)`; each tenant gets a per-tenant weight (higher
    /// = more share).
    #[staticmethod]
    #[pyo3(signature = (system, tenants, max_in_flight=4, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        tenants: Vec<(u32, u32)>,
        max_in_flight: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if tenants.is_empty() {
            return Err(errors::map_str("tenants must not be empty"));
        }
        // Phase 2.5: typed payloads — internal echo dispatcher.
        let echo: Arc<dyn FairDispatcher<Vec<u8>, Vec<u8>>> = Arc::new(
            |req: Vec<u8>, reply: oneshot::Sender<Result<Vec<u8>, GpuError>>| {
                tokio::spawn(async move {
                    let _ = reply.send(Ok(req));
                });
            },
        );
        let cfg = FairShareConfig {
            tenants: tenants
                .into_iter()
                .map(|(id, weight)| TenantConfig {
                    id: TenantId(id),
                    weight: weight.max(1),
                })
                .collect(),
            dispatcher: echo,
            max_in_flight,
        };
        let actor_name = name.unwrap_or_else(|| "fair-share".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(
                    FairShareScheduler::<Vec<u8>, Vec<u8>>::props(cfg),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyFairShareScheduler { actor_ref })
    }

    /// Submit a payload tagged with `tenant`. With the default echo
    /// dispatcher the response is the input bytes; the scheduler
    /// orders submissions across tenants by WFQ.
    // Phase 2.5: typed payloads — bytes in/out.
    #[pyo3(signature = (tenant, payload, timeout_secs=5.0))]
    fn submit(
        &self,
        py: Python<'_>,
        tenant: u32,
        payload: Vec<u8>,
        timeout_secs: f64,
    ) -> PyResult<Vec<u8>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(FairShareMsg::Submit {
                    tenant: TenantId(tenant),
                    req: payload,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("scheduler dropped reply")),
                    Err(_) => Err(errors::map_str("submit timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "FairShareScheduler(handle)"
    }
}

// ─── HotSwapServer ──────────────────────────────────────────────

/// Internal echo backend protocol for hot-swap. The actor receives
/// bytes and returns them unchanged, tagged with a `version` byte
/// so callers can prove a swap actually changed the routed actor.
pub(crate) enum BytesBackendMsg {
    Predict {
        req: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, GpuError>>,
    },
}

pub(crate) struct BytesBackendActor {
    version: u8,
}

#[async_trait]
impl Actor for BytesBackendActor {
    type Msg = BytesBackendMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: BytesBackendMsg) {
        match msg {
            BytesBackendMsg::Predict { mut req, reply } => {
                req.push(self.version);
                let _ = reply.send(Ok(req));
            }
        }
    }
}

pub(crate) struct BytesBackendProto;

impl BackendProtocol for BytesBackendProto {
    type Msg = BytesBackendMsg;
    type Req = Vec<u8>;
    type Resp = Vec<u8>;
    fn make_predict(
        req: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, GpuError>>,
    ) -> BytesBackendMsg {
        BytesBackendMsg::Predict { req, reply }
    }
}

/// `HotSwapServer` handle (Phase 2.5: `Vec<u8>` payloads, internal
/// echo backends).
#[pyclass(name = "HotSwapServer", module = "atomr_accel._native")]
pub struct PyHotSwapServer {
    actor_ref: ActorRef<HotSwapMsg<BytesBackendProto>>,
    // We keep a handle to the system so `swap()` can spawn fresh
    // backends without the caller reaching into `system`.
    system: Py<PySystem>,
    /// Monotonic counter — used to name fresh swapped-in backends.
    swap_counter: parking_lot::Mutex<u64>,
}

#[pymethods]
impl PyHotSwapServer {
    /// Spawn a hot-swap server backed by an internal echo backend
    /// (version=0). `swap()` rolls in a fresh backend.
    #[staticmethod]
    #[pyo3(signature = (system, name=None))]
    fn spawn(py: Python<'_>, system: Py<PySystem>, name: Option<String>) -> PyResult<Py<Self>> {
        let actor_name = name.unwrap_or_else(|| "hot-swap".to_string());
        let actor_ref = {
            let sys_ref = system.borrow(py);
            let _guard = runtime().enter();
            // Phase 2.5: typed payloads — internal echo backend.
            let backend = sys_ref
                .inner
                .actor_of(
                    Props::create(|| BytesBackendActor { version: 0 }),
                    &format!("{actor_name}-backend-0"),
                )
                .map_err(errors::map_str)?;
            sys_ref
                .inner
                .actor_of(
                    ModelHotSwapServer::<BytesBackendProto>::props(backend),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(
            py,
            PyHotSwapServer {
                actor_ref,
                system,
                swap_counter: parking_lot::Mutex::new(0),
            },
        )
    }

    /// Send `payload` through the currently-routed backend; returns
    /// `payload` with a single-byte version suffix appended (so
    /// callers can prove a swap routed traffic to a fresh backend).
    // Phase 2.5: typed payloads — bytes in/out.
    #[pyo3(signature = (payload, timeout_secs=5.0))]
    fn serve(&self, py: Python<'_>, payload: Vec<u8>, timeout_secs: f64) -> PyResult<Vec<u8>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(HotSwapMsg::Predict {
                    req: payload,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("hot-swap dropped reply")),
                    Err(_) => Err(errors::map_str("serve timed out")),
                }
            })
        })
    }

    /// Roll in a fresh internal backend; returns the new generation
    /// counter. The next `serve` call routes to the new backend.
    /// Phase 2.6 will accept a Python-supplied backend actor ref.
    #[pyo3(signature = (timeout_secs=5.0))]
    fn swap(&self, py: Python<'_>, timeout_secs: f64) -> PyResult<u64> {
        let v = {
            let mut c = self.swap_counter.lock();
            *c = c.saturating_add(1);
            // version is u8, wrap modulo 256 for the suffix tag.
            (*c % 256) as u8
        };
        let new_backend = {
            let sys_ref = self.system.borrow(py);
            let _guard = runtime().enter();
            sys_ref
                .inner
                .actor_of(
                    Props::create(move || BytesBackendActor { version: v }),
                    &format!("hot-swap-backend-{v}-{}", *self.swap_counter.lock()),
                )
                .map_err(errors::map_str)?
        };
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(HotSwapMsg::SwapIn {
                    new_backend,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(s)) => Ok(s.generation),
                    Ok(Err(_)) => Err(errors::map_str("hot-swap dropped reply")),
                    Err(_) => Err(errors::map_str("swap timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "HotSwapServer(handle)"
    }
}

// ─── SpeculativeDecoder ─────────────────────────────────────────

/// `SpeculativeDecoder` handle. The underlying actor is non-generic
/// (token IDs are `u32`); the only required user input is a draft
/// closure + verifier closure. Phase 2.5 ships a built-in pair: the
/// draft proposes K consecutive integers; the verifier accepts all.
/// Phase 2.6 will accept Python `DraftFn` / `VerifierFn` callables.
#[pyclass(name = "SpeculativeDecoder", module = "atomr_accel._native")]
pub struct PySpeculativeDecoder {
    actor_ref: ActorRef<SpecMsg>,
}

#[pymethods]
impl PySpeculativeDecoder {
    /// Spawn a speculative decoder with built-in echo draft +
    /// accept-all verifier.
    #[staticmethod]
    #[pyo3(signature = (system, k=4, max_total_tokens=64, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        k: usize,
        max_total_tokens: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        let draft: Arc<dyn DraftFn> = Arc::new(|prefix: &[u32], k: usize| {
            let last = prefix.last().copied().unwrap_or(0);
            Ok((1..=k as u32).map(|i| last.saturating_add(i)).collect())
        });
        let verifier: Arc<dyn VerifierFn> =
            Arc::new(|_prefix: &[u32], candidates: &[u32]| Ok((candidates.len(), None)));
        let cfg = SpeculativeConfig {
            draft,
            verifier,
            k: k.max(1),
            max_total_tokens: max_total_tokens.max(1),
        };
        let actor_name = name.unwrap_or_else(|| "spec-decoder".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            system
                .inner
                .actor_of(SpeculativeDecoder::props(cfg), &actor_name)
                .map_err(errors::map_str)?
        };
        Py::new(py, PySpeculativeDecoder { actor_ref })
    }

    /// Run the decode loop on `prompt`. Returns `(tokens,
    /// iterations, accepted_tokens)`. The decoder runs until either
    /// the spawn-time `max_total_tokens` cap is hit or the draft
    /// returns no candidates. `budget` further caps the per-call
    /// token total — the decoder stops once `len(tokens) >= budget`.
    #[pyo3(signature = (prompt, budget=64, timeout_secs=5.0))]
    fn decode(
        &self,
        py: Python<'_>,
        prompt: Vec<u32>,
        budget: usize,
        timeout_secs: f64,
    ) -> PyResult<(Vec<u32>, u32, u32)> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(SpecMsg::Decode {
                    prefix: prompt,
                    reply: tx,
                });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok((mut tokens, stats)))) => {
                        // `budget` is a per-call cap layered above the
                        // actor's spawn-time cap.
                        if tokens.len() > budget {
                            tokens.truncate(budget);
                        }
                        Ok((tokens, stats.iterations, stats.accepted_tokens))
                    }
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("spec decoder dropped reply")),
                    Err(_) => Err(errors::map_str("decode timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "SpeculativeDecoder(handle)"
    }
}

// ─── MoeRouter ──────────────────────────────────────────────────

/// Internal expert protocol: each expert runs a `BiasExpertActor`
/// that adds a per-expert constant to every input element.
pub(crate) enum BiasExpertMsg {
    Run {
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
}

pub(crate) struct BiasExpertActor {
    bias: f32,
}

#[async_trait]
impl Actor for BiasExpertActor {
    type Msg = BiasExpertMsg;
    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: BiasExpertMsg) {
        match msg {
            BiasExpertMsg::Run { input, reply } => {
                let v: Vec<f32> = input.iter().map(|x| x + self.bias).collect();
                let _ = reply.send(Ok(v));
            }
        }
    }
}

pub(crate) struct BiasExpertProto;
impl ExpertProtocol for BiasExpertProto {
    type Msg = BiasExpertMsg;
    fn make_run(
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    ) -> BiasExpertMsg {
        BiasExpertMsg::Run { input, reply }
    }
}

/// `MoeRouter` handle (Phase 2.5: internal `BiasExpertActor` experts;
/// gate scores favor the last expert).
#[pyclass(name = "MoeRouter", module = "atomr_accel._native")]
pub struct PyMoeRouter {
    actor_ref: ActorRef<MoeMsg<BiasExpertProto>>,
}

#[pymethods]
impl PyMoeRouter {
    /// Spawn a MoE router with `n_experts` internal bias experts
    /// (expert `i` adds `i` to every input element). The gate is a
    /// linearly-increasing score so expert `n_experts-1` wins under
    /// `top_k=1`.
    #[staticmethod]
    #[pyo3(signature = (system, n_experts=2, top_k=1, name=None))]
    fn spawn(
        py: Python<'_>,
        system: &PySystem,
        n_experts: usize,
        top_k: usize,
        name: Option<String>,
    ) -> PyResult<Py<Self>> {
        if n_experts == 0 {
            return Err(errors::map_str("n_experts must be ≥ 1"));
        }
        let actor_name = name.unwrap_or_else(|| "moe-router".to_string());
        let actor_ref = {
            let _guard = runtime().enter();
            // Phase 2.5: typed payloads — internal bias experts.
            let mut experts = Vec::with_capacity(n_experts);
            for i in 0..n_experts {
                let e = system
                    .inner
                    .actor_of(
                        Props::create(move || BiasExpertActor { bias: i as f32 }),
                        &format!("{actor_name}-expert-{i}"),
                    )
                    .map_err(errors::map_str)?;
                experts.push(e);
            }
            let n = n_experts;
            let gate: Arc<dyn atomr_accel_patterns::moe::GateFn> =
                Arc::new(move |_input: &[f32]| Ok((0..n).map(|i| i as f32).collect::<Vec<f32>>()));
            system
                .inner
                .actor_of(
                    MoeRouter::<BiasExpertProto>::props(MoeConfig {
                        experts,
                        gate,
                        top_k: top_k.max(1),
                    }),
                    &actor_name,
                )
                .map_err(errors::map_str)?
        };
        Py::new(py, PyMoeRouter { actor_ref })
    }

    /// Route an input vector through the gate to the top-k experts;
    /// returns the softmax-blended output.
    #[pyo3(signature = (input, timeout_secs=5.0))]
    fn route(&self, py: Python<'_>, input: Vec<f32>, timeout_secs: f64) -> PyResult<Vec<f32>> {
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                actor.tell(MoeMsg::Run { input, reply: tx });
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(v))) => Ok(v),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("MoE router dropped reply")),
                    Err(_) => Err(errors::map_str("route timed out")),
                }
            })
        })
    }

    fn __repr__(&self) -> &'static str {
        "MoeRouter(handle)"
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
