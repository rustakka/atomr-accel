//! `BlasLtActor` — wraps [`cudarc::cublaslt::CudaBlasLT`] for
//! transformer-shaped fused matmul (matmul + bias + activation +
//! aux-store + bias-grad reduction) across the full dtype matrix
//! cuBLASLt accepts.
//!
//! See [`epilogue`] for the curated `Epilogue` enum, [`heuristic`]
//! for the algorithm cache, [`workspace`] for the workspace pool,
//! [`scaling`] for the fp8 scale-pointer wiring, and [`matmul`] for
//! the typed `MatmulRequest<T>` plus its `BlasLtDispatch` impl.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
pub use cudarc::cublaslt::Activation;
use cudarc::cublaslt::{CudaBlasLT, MatmulConfig};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{BlasLtDispatch, BlasLtDispatchCtx};
use crate::stream::StreamAllocator;

pub mod epilogue;
pub mod heuristic;
pub mod matmul;
pub mod scaling;
pub mod workspace;

pub use epilogue::Epilogue;
pub use heuristic::{HeuristicCacheRef, HeuristicEntry, HeuristicKey, DEFAULT_HEURISTIC_CAPACITY};
pub use matmul::MatmulRequest;
pub use scaling::ScaleSet;
pub use workspace::{WorkspaceLease, WorkspacePool};

const LIB: &str = "cublaslt";

/// Public message surface.
pub enum BlasLtMsg {
    /// Generic matmul over any dtype that implements
    /// [`crate::dtype::GemmSupported`]. Boxed-erased so `BlasLtActor`
    /// has a single mailbox type.
    Matmul(Box<dyn BlasLtDispatch>),

    /// Legacy f32-only constructor preserved for back-compat.
    /// New callers should use [`BlasLtMsg::matmul`] /
    /// [`BlasLtMsg::Matmul`] with a typed [`MatmulRequest<f32>`].
    #[deprecated(
        since = "0.2.0",
        note = "use BlasLtMsg::Matmul(Box::new(MatmulRequest::<f32> { … }))"
    )]
    MatmulF32 {
        cfg: MatmulConfig,
        a: GpuRef<f32>,
        b: GpuRef<f32>,
        c: GpuRef<f32>,
        bias: Option<GpuRef<f32>>,
        activation: Option<Activation>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

impl BlasLtMsg {
    /// Convenience constructor — `BlasLtMsg::matmul::<f32>(req)` is a
    /// drop-in for callers migrating off the deprecated `MatmulF32`.
    pub fn matmul<T>(req: MatmulRequest<T>) -> Self
    where
        T: crate::dtype::GemmSupported,
        MatmulRequest<T>: BlasLtDispatch,
    {
        Self::Matmul(Box::new(req))
    }
}

pub struct BlasLtActor {
    inner: BlasLtInner,
}

enum BlasLtInner {
    Real {
        blas_lt: Arc<CudaBlasLT>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        #[allow(dead_code)]
        state: Arc<DeviceState>,
        workspace_pool: WorkspacePool,
        heuristic_cache: HeuristicCacheRef,
        sm_arch: u32,
    },
    Mock,
}

impl BlasLtActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
    ) -> Props<Self> {
        Props::create(move || {
            let blas_lt = match CudaBlasLT::new(stream.clone()) {
                Ok(b) => b,
                Err(e) => panic!("ContextPoisoned: CudaBlasLT::new failed: {e}"),
            };
            // SM arch detection — best-effort. cudarc exposes the
            // device-attribute query through the stream's context, but
            // it's a fallible runtime call. Default to 0 if we can't
            // resolve it; the heuristic cache simply won't share
            // entries across arches in that case (correct, just
            // slightly less hit-rate).
            let sm_arch = stream
                .context()
                .attribute(
                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .ok()
                .map(|m| m as u32 * 10)
                .unwrap_or(0);
            BlasLtActor {
                inner: BlasLtInner::Real {
                    blas_lt: Arc::new(blas_lt),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                    workspace_pool: WorkspacePool::new(),
                    heuristic_cache: HeuristicCacheRef::default_size(),
                    sm_arch,
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| BlasLtActor {
            inner: BlasLtInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for BlasLtActor {
    type Msg = BlasLtMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: BlasLtMsg) {
        match &self.inner {
            BlasLtInner::Mock => match msg {
                BlasLtMsg::Matmul(req) => {
                    // Synthesize a minimal context that lets the
                    // dispatch's mock-mode reply path fire. We don't
                    // touch a CUDA handle, so the typed request must
                    // reply with `Unrecoverable("mock mode")` itself
                    // — every `BlasLtDispatch` impl owns its reply.
                    drop(req);
                }
                #[allow(deprecated)]
                BlasLtMsg::MatmulF32 { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "BlasLtActor in mock mode".into(),
                    )));
                }
            },
            BlasLtInner::Real {
                blas_lt,
                stream,
                completion,
                workspace_pool,
                heuristic_cache,
                sm_arch,
                ..
            } => match msg {
                BlasLtMsg::Matmul(req) => {
                    let dctx = BlasLtDispatchCtx {
                        blas_lt: blas_lt.clone(),
                        stream,
                        completion,
                        workspace: workspace_pool,
                        heuristic: heuristic_cache.clone(),
                        sm_arch: *sm_arch,
                    };
                    req.dispatch(&dctx);
                }
                #[allow(deprecated)]
                BlasLtMsg::MatmulF32 {
                    cfg,
                    a,
                    b,
                    c,
                    bias,
                    activation,
                    reply,
                } => {
                    enqueue_matmul_f32_legacy(
                        blas_lt.clone(),
                        stream,
                        completion,
                        cfg,
                        a,
                        b,
                        c,
                        bias,
                        activation,
                        reply,
                    );
                }
            },
        }
    }
}

/// Legacy-path enqueue identical to the pre-Phase-1 implementation
/// (kept for back-compat with the deprecated `BlasLtMsg::MatmulF32`).
fn enqueue_matmul_f32_legacy(
    blas_lt: Arc<CudaBlasLT>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    cfg: MatmulConfig,
    a: GpuRef<f32>,
    b: GpuRef<f32>,
    c: GpuRef<f32>,
    bias: Option<GpuRef<f32>>,
    activation: Option<Activation>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    use crate::kernel::envelope;
    use cudarc::cublaslt::Matmul;

    let (a_slice, b_slice, c_slice) = match envelope::access_all_3(&a, &b, &c) {
        Ok(t) => t,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let bias_slice = match bias.as_ref() {
        None => None,
        Some(g) => match g.access() {
            Ok(s) => Some(s.clone()),
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        },
    };
    let mut c_owned = match Arc::try_unwrap(c_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "BlasLt C has multiple live references".into(),
            )));
            return;
        }
    };
    c.record_write(stream);
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let bias_ref = bias_slice.as_ref().map(|s| &**s);
        let act_ref = activation.as_ref();
        // SAFETY: matmul is unsafe due to dim-validity contract.
        let res =
            unsafe { blas_lt.matmul(cfg, &*a_slice, &*b_slice, &mut c_owned, bias_ref, act_ref) };
        match res {
            Ok(()) => Ok((a_slice, b_slice, c_owned, bias_slice, blas_lt)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("matmul: {e}"),
            }),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blas_lt_msg_matmul_constructor() {
        // Compile-time check: BlasLtMsg::matmul::<f32> resolves and
        // produces a `BlasLtMsg::Matmul` variant.
        let (tx, _rx) = oneshot::channel::<Result<(), GpuError>>();
        // We need to drop tx without sending; constructing a real
        // MatmulRequest requires a GpuRef which needs a device, so
        // we only verify the constructor type.
        let _f: fn(MatmulRequest<f32>) -> BlasLtMsg = BlasLtMsg::matmul::<f32>;
        drop(tx);
    }
}
