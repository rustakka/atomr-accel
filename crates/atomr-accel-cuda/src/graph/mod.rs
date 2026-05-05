//! `GraphActor` — record a CUDA stream-capture once, replay many.
//!
//! Two lifecycle paths:
//!
//! 1. **Caller-driven capture** — user calls `stream.begin_capture()`
//!    directly, performs operations, calls `stream.end_capture()` to
//!    get a `CudaGraph`, then wraps via [`GraphHandle::from_graph`]
//!    and sends `Launch` to replay.
//! 2. **Actor-driven capture** — caller sends `Record { script }`;
//!    actor runs `begin_capture` → loops over `GraphOp`s via
//!    `RecordMode` → `end_capture` → returns a `GraphHandle`.
//!
//! Both paths produce the same `GraphHandle` type; on `Launch` the
//! actor validates `state.generation()` and replays the graph,
//! replying after stream completion.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::cublas::CudaBlas;
use cudarc::driver::sys as driver_sys;
use cudarc::driver::CudaGraph;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::kernel::record::{BlasRecorder, BlasSgemmOp, MemcpyOp, MemcpyRecorder, RecordMode};
#[cfg(feature = "cufft")]
use crate::kernel::record::{FftR2COp, FftRecorder};
#[cfg(feature = "curand")]
use crate::kernel::record::{RngFillUniformOp, RngRecorder};

pub mod child;
#[cfg(feature = "graphs-conditional")]
pub mod conditional;
pub mod dot;
pub mod exec_update;
pub mod record;

pub use child::ChildGraphOp;
pub use dot::{export_dot, DotFlags};
pub use exec_update::{exec_update, GraphExecUpdateOutcome};

const LIB: &str = "graph";

/// Record-side context handed to a [`GraphOpRecord`] impl. Carries
/// the captured stream (so Phase-0.5 variants can keep using
/// `RecordMode::enqueue_record`) plus, when available, the parent
/// graph handle (so Phase-3 variants like `ChildGraphOp` can call
/// `cuGraphAddChildGraphNode` directly).
///
/// Both `stream` and `parent_graph` are optional: tests / mock paths
/// can build a context with neither and still get a typed
/// `Unrecoverable` from any record impl that needs them.
pub struct GraphRecordCtx<'a> {
    stream: Option<&'a Arc<cudarc::driver::CudaStream>>,
    parent_graph: driver_sys::CUgraph,
}

impl<'a> GraphRecordCtx<'a> {
    pub fn new(
        stream: &'a Arc<cudarc::driver::CudaStream>,
        parent_graph: driver_sys::CUgraph,
    ) -> Self {
        Self {
            stream: Some(stream),
            parent_graph,
        }
    }

    /// Mock-mode constructor: parent graph only, no stream.
    pub fn mock(parent_graph: driver_sys::CUgraph) -> Self {
        Self {
            stream: None,
            parent_graph,
        }
    }

    pub fn stream(&self) -> Option<&Arc<cudarc::driver::CudaStream>> {
        self.stream
    }

    pub fn parent_graph(&self) -> driver_sys::CUgraph {
        self.parent_graph
    }
}

/// Test-only mock-context builder. Carries a parent-graph handle and
/// an optional captured stream. Use when verifying the routing of a
/// `GraphOpRecord` impl without a live CUDA driver.
#[doc(hidden)]
pub struct MockGraphRecordCtx {
    parent_graph: driver_sys::CUgraph,
    stream: Option<Arc<cudarc::driver::CudaStream>>,
}

impl MockGraphRecordCtx {
    pub fn new(parent_graph: driver_sys::CUgraph) -> Self {
        Self {
            parent_graph,
            stream: None,
        }
    }

    pub fn with_stream(mut self, stream: Arc<cudarc::driver::CudaStream>) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn as_ctx(&self) -> GraphRecordCtx<'_> {
        GraphRecordCtx {
            stream: self.stream.as_ref(),
            parent_graph: self.parent_graph,
        }
    }
}

/// Phase 3 record-mode trait. Lighter than `RecordMode` (no
/// associated `Op` type) — implementors are typically *one* op carrying
/// the typed request inline.
pub trait GraphOpRecord {
    fn record(&self, ctx: &GraphRecordCtx<'_>) -> Result<(), GpuError>;
}

/// Send/Sync newtype around `Arc<CudaGraph>`. cudarc marks
/// `CudaGraph` `!Sync` because of interior mutability via the CUDA
/// driver. The actor enforces single-threaded access.
pub struct SendGraph(Arc<CudaGraph>);
unsafe impl Send for SendGraph {}
unsafe impl Sync for SendGraph {}

impl Clone for SendGraph {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Clone)]
pub struct GraphHandle {
    graph: Option<SendGraph>,
    generation: u64,
    /// Synthetic-mode raw handles used by no-GPU tests. When `graph`
    /// is `None` and these are non-null, the typed accessors return
    /// these values directly.
    #[doc(hidden)]
    synthetic_cu_graph: driver_sys::CUgraph,
    #[doc(hidden)]
    synthetic_cu_graph_exec: driver_sys::CUgraphExec,
}

impl GraphHandle {
    /// Wrap a manually-captured `CudaGraph` into a `GraphHandle`
    /// with the current `DeviceState` generation.
    pub fn from_graph(graph: Arc<CudaGraph>, state: &Arc<DeviceState>) -> Self {
        Self {
            graph: Some(SendGraph(graph)),
            generation: state.generation(),
            synthetic_cu_graph: std::ptr::null_mut(),
            synthetic_cu_graph_exec: std::ptr::null_mut(),
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Underlying `CUgraph` handle. Used by Phase 3 callers that need
    /// to call sys-level APIs (`cuGraphAddChildGraphNode`,
    /// `cuGraphDebugDotPrint`, etc.).
    ///
    /// # Safety
    /// Returned value must not be destroyed; the handle is owned by
    /// the wrapped `CudaGraph`.
    pub fn cu_graph(&self) -> driver_sys::CUgraph {
        if let Some(g) = self.graph.as_ref() {
            g.0.cu_graph()
        } else {
            self.synthetic_cu_graph
        }
    }

    /// Underlying `CUgraphExec` handle. Used by Phase 3 callers
    /// (`cuGraphExecUpdate_v2`).
    ///
    /// # Safety
    /// Same as [`Self::cu_graph`].
    pub fn cu_graph_exec(&self) -> driver_sys::CUgraphExec {
        if let Some(g) = self.graph.as_ref() {
            g.0.cu_graph_exec()
        } else {
            self.synthetic_cu_graph_exec
        }
    }

    /// Build a synthetic `GraphHandle` with null sys-level handles.
    /// Test-only — the corresponding sys calls return `LibraryError`
    /// (driver present) or `Unrecoverable` (no driver) without
    /// panicking.
    #[doc(hidden)]
    pub fn synthetic_for_tests() -> Self {
        Self {
            graph: None,
            generation: 0,
            synthetic_cu_graph: std::ptr::null_mut(),
            synthetic_cu_graph_exec: std::ptr::null_mut(),
        }
    }
}

/// Operation-script entry. Each variant mirrors a kernel-actor op
/// minus its reply channel.
pub enum GraphOp {
    Sgemm(Box<BlasSgemmOp>),
    /// Device-to-device memcpy on the captured stream.
    Memcpy(Box<MemcpyOp>),
    /// Uniform RNG fill (gated on `curand` feature).
    #[cfg(feature = "curand")]
    RngFillUniform(Box<RngFillUniformOp>),
    /// 1-D R2C FFT (gated on `cufft` feature). The user supplies a
    /// pre-built `CudaFft` plan via `GraphActor::set_fft_plan`
    /// before recording.
    #[cfg(feature = "cufft")]
    FftR2C(Box<FftR2COp>),
}

pub enum GraphMsg {
    /// Record a `Vec<GraphOp>` into a CUDA Graph.
    Record {
        script: Vec<GraphOp>,
        reply: oneshot::Sender<Result<GraphHandle, GpuError>>,
    },
    /// Replay a previously-recorded graph.
    Launch {
        handle: GraphHandle,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Install / replace the cuFFT plan used for `GraphOp::FftR2C`
    /// records. Must be called before recording any FFT op.
    #[cfg(feature = "cufft")]
    SetFftPlan {
        plan: cudarc::cufft::CudaFft,
        reply: oneshot::Sender<()>,
    },
}

struct SendBlas(CudaBlas);
unsafe impl Send for SendBlas {}
unsafe impl Sync for SendBlas {}

#[cfg(feature = "curand")]
struct SendRng(cudarc::curand::CudaRng);
#[cfg(feature = "curand")]
unsafe impl Send for SendRng {}
#[cfg(feature = "curand")]
unsafe impl Sync for SendRng {}

#[cfg(feature = "cufft")]
struct SendFft(cudarc::cufft::CudaFft);
#[cfg(feature = "cufft")]
unsafe impl Send for SendFft {}
#[cfg(feature = "cufft")]
unsafe impl Sync for SendFft {}

pub struct GraphActor {
    inner: GraphInner,
}

#[allow(dead_code)]
enum GraphInner {
    Real {
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        /// Optional cuBLAS handle for recording SGEMM ops. None
        /// disables Sgemm-record entirely.
        blas: Option<Mutex<SendBlas>>,
        #[cfg(feature = "curand")]
        rng: Option<Mutex<SendRng>>,
        #[cfg(feature = "cufft")]
        fft: Mutex<Option<SendFft>>,
    },
    Mock,
}

impl GraphActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            // Try to construct a record-mode CudaBlas on this stream.
            // If the CUDA runtime isn't loadable, leave it as None;
            // Sgemm record will reply Unrecoverable.
            let blas = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                CudaBlas::new(stream.clone())
            })) {
                Ok(Ok(b)) => Some(Mutex::new(SendBlas(b))),
                _ => None,
            };
            #[cfg(feature = "curand")]
            let rng = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cudarc::curand::CudaRng::new(0, stream.clone())
            })) {
                Ok(Ok(r)) => Some(Mutex::new(SendRng(r))),
                _ => None,
            };
            GraphActor {
                inner: GraphInner::Real {
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                    blas,
                    #[cfg(feature = "curand")]
                    rng,
                    #[cfg(feature = "cufft")]
                    fft: Mutex::new(None),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| GraphActor {
            inner: GraphInner::Mock,
        })
    }
}

fn run_record(
    stream: &Arc<cudarc::driver::CudaStream>,
    state: &Arc<DeviceState>,
    blas: &Option<Mutex<SendBlas>>,
    #[cfg(feature = "curand")] rng: &Option<Mutex<SendRng>>,
    #[cfg(feature = "cufft")] fft: &Mutex<Option<SendFft>>,
    script: Vec<GraphOp>,
) -> Result<GraphHandle, GpuError> {
    // Begin capture.
    let begin_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        stream.begin_capture(driver_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
    }));
    match begin_res {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("begin_capture: {e}"),
            });
        }
        Err(_) => {
            return Err(GpuError::Unrecoverable(
                "GraphActor::Record: CUDA driver not loadable".into(),
            ));
        }
    }

    // Helper that ends capture on error before returning.
    let bail = |e: GpuError, stream: &Arc<cudarc::driver::CudaStream>| -> GpuError {
        let _ = stream.end_capture(
            driver_sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        e
    };

    // Drive each op through its recorder.
    for op in script {
        match op {
            GraphOp::Sgemm(b) => {
                let Some(blas_lock) = blas else {
                    return Err(bail(
                        GpuError::Unrecoverable(
                            "GraphActor::Record::Sgemm: cuBLAS not available".into(),
                        ),
                        stream,
                    ));
                };
                let g = blas_lock.lock();
                let mut recorder = BlasRecorder { handle: &g.0 };
                if let Err(e) = recorder.enqueue_record(stream, *b) {
                    return Err(bail(e, stream));
                }
                drop(g);
            }
            GraphOp::Memcpy(m) => {
                let mut recorder = MemcpyRecorder;
                if let Err(e) = recorder.enqueue_record(stream, *m) {
                    return Err(bail(e, stream));
                }
            }
            #[cfg(feature = "curand")]
            GraphOp::RngFillUniform(r) => {
                let Some(rng_lock) = rng else {
                    return Err(bail(
                        GpuError::Unrecoverable(
                            "GraphActor::Record::RngFillUniform: cuRAND not available".into(),
                        ),
                        stream,
                    ));
                };
                let g = rng_lock.lock();
                let mut recorder = RngRecorder { rng: &g.0 };
                if let Err(e) = recorder.enqueue_record(stream, *r) {
                    return Err(bail(e, stream));
                }
                drop(g);
            }
            #[cfg(feature = "cufft")]
            GraphOp::FftR2C(r) => {
                let g = fft.lock();
                let Some(plan) = g.as_ref() else {
                    return Err(bail(
                        GpuError::Unrecoverable(
                            "GraphActor::Record::FftR2C: no plan installed; call \
                             GraphMsg::SetFftPlan first"
                                .into(),
                        ),
                        stream,
                    ));
                };
                let mut recorder = FftRecorder { plan: &plan.0 };
                if let Err(e) = recorder.enqueue_record(stream, *r) {
                    return Err(bail(e, stream));
                }
                drop(g);
            }
        }
    }

    // End capture.
    let end_res = stream.end_capture(
        driver_sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    let cuda_graph = match end_res {
        Ok(Some(g)) => g,
        Ok(None) => {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: "end_capture returned None".into(),
            });
        }
        Err(e) => {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("end_capture: {e}"),
            });
        }
    };
    Ok(GraphHandle::from_graph(Arc::new(cuda_graph), state))
}

#[async_trait]
impl Actor for GraphActor {
    type Msg = GraphMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: GraphMsg) {
        match &self.inner {
            GraphInner::Mock => match msg {
                GraphMsg::Record { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "GraphActor in mock mode".into(),
                    )));
                }
                GraphMsg::Launch { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "GraphActor in mock mode".into(),
                    )));
                }
                #[cfg(feature = "cufft")]
                GraphMsg::SetFftPlan { reply, .. } => {
                    let _ = reply.send(());
                }
            },
            GraphInner::Real {
                stream,
                completion,
                state,
                blas,
                #[cfg(feature = "curand")]
                rng,
                #[cfg(feature = "cufft")]
                fft,
            } => match msg {
                GraphMsg::Record { script, reply } => {
                    let res = run_record(
                        stream,
                        state,
                        blas,
                        #[cfg(feature = "curand")]
                        rng,
                        #[cfg(feature = "cufft")]
                        fft,
                        script,
                    );
                    let _ = reply.send(res);
                }
                #[cfg(feature = "cufft")]
                GraphMsg::SetFftPlan { plan, reply } => {
                    *fft.lock() = Some(SendFft(plan));
                    let _ = reply.send(());
                }
                GraphMsg::Launch { handle, reply } => {
                    if handle.generation != state.generation() {
                        let _ = reply.send(Err(GpuError::GpuRefStale(
                            "graph captured against rebuilt context",
                        )));
                        return;
                    }
                    let Some(graph) = handle.graph.as_ref() else {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "GraphActor::Launch: synthetic GraphHandle has no captured graph"
                                .into(),
                        )));
                        return;
                    };
                    let res = graph.0.launch().map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("launch: {e}"),
                    });
                    if let Err(e) = res {
                        let _ = reply.send(Err(e));
                        return;
                    }
                    let stream = stream.clone();
                    let completion = completion.clone();
                    tokio::spawn(async move {
                        let r = completion.await_completion(&stream).await;
                        let _ = reply.send(r);
                    });
                }
            },
        }
    }
}
