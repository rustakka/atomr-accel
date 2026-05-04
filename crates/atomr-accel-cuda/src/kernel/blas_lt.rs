//! `BlasLtActor` — wraps [`cudarc::cublaslt::CudaBlasLT`] for fused
//! matmul + activation (Relu/Gelu) + optional bias. Primary use case:
//! transformer FFN/MLP layers.
//!
//! cuBLASLt allocates an internal workspace at construction (32 MiB
//! on Hopper / SM 9.x, 4 MiB else, auto-detected by cudarc).

use std::sync::Arc;

use async_trait::async_trait;
pub use cudarc::cublaslt::Activation;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use atomr_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "cublaslt";

pub enum BlasLtMsg {
    /// f32 matmul: C = alpha * op(A) * op(B) + beta * C, optionally
    /// followed by `+ bias` and an activation.
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
            BlasLtActor {
                inner: BlasLtInner::Real {
                    blas_lt: Arc::new(blas_lt),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
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
                ..
            } => match msg {
                BlasLtMsg::MatmulF32 {
                    cfg,
                    a,
                    b,
                    c,
                    bias,
                    activation,
                    reply,
                } => {
                    enqueue_matmul_f32(
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

fn enqueue_matmul_f32(
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
