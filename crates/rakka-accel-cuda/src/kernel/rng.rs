//! `RngActor` — wraps a [`cudarc::curand::CudaRng`] handle and fills
//! device buffers with uniform / normal / log-normal distributions.
//!
//! Reseed model: `Reseed { seed }` calls `CudaRng::set_seed` in
//! place — no panic-restart. Reasoning: reseed is a control-plane
//! operation; restart-on-reseed would tear down all in-flight work
//! and is too heavy. The seed is journaled by `ReplayHarness` (F5)
//! so deterministic replay works.

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::curand::{
    result::{LogNormalFill, NormalFill, UniformFill},
    CudaRng,
};
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "curand";

pub enum RngMsg {
    FillUniformF32 {
        dst: GpuRef<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    FillUniformF64 {
        dst: GpuRef<f64>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    FillUniformU32 {
        dst: GpuRef<u32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    FillNormalF32 {
        dst: GpuRef<f32>,
        mean: f32,
        std: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    FillNormalF64 {
        dst: GpuRef<f64>,
        mean: f64,
        std: f64,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    FillLogNormalF32 {
        dst: GpuRef<f32>,
        mean: f32,
        std: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Reseed {
        seed: u64,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct RngActor {
    inner: RngInner,
}

/// `CudaRng` holds a raw `*mut curandGenerator_st` and so is `!Send`
/// + `!Sync`. RngActor runs exclusively on the [`crate::dispatcher::GpuDispatcher`]
/// single thread; we assert Send + Sync via this newtype so rakka's
/// `Actor: Send + 'static` bound is satisfied.
struct SendCudaRng(CudaRng);

// SAFETY: the underlying handle is bound to a single CUDA stream; it
// is only ever accessed from the GpuDispatcher's pinned thread. The
// outer parking_lot::Mutex ensures exclusive access from the actor.
unsafe impl Send for SendCudaRng {}
unsafe impl Sync for SendCudaRng {}

enum RngInner {
    Real {
        rng: Mutex<SendCudaRng>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        #[allow(dead_code)]
        state: Arc<DeviceState>,
    },
    Mock,
}

impl RngActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        seed: u64,
    ) -> Props<Self> {
        Props::create(move || {
            let rng = match CudaRng::new(seed, stream.clone()) {
                Ok(r) => r,
                Err(e) => panic!("ContextPoisoned: CudaRng::new failed: {e}"),
            };
            RngActor {
                inner: RngInner::Real {
                    rng: Mutex::new(SendCudaRng(rng)),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| RngActor { inner: RngInner::Mock })
    }
}

#[async_trait]
impl Actor for RngActor {
    type Msg = RngMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: RngMsg) {
        let (rng_lock, stream, completion) = match &self.inner {
            RngInner::Mock => {
                reply_mock(msg);
                return;
            }
            RngInner::Real { rng, stream, completion, .. } => (rng, stream, completion),
        };

        match msg {
            RngMsg::FillUniformF32 { dst, reply } => {
                fill_uniform::<f32>(rng_lock, stream, completion, dst, reply);
            }
            RngMsg::FillUniformF64 { dst, reply } => {
                fill_uniform::<f64>(rng_lock, stream, completion, dst, reply);
            }
            RngMsg::FillUniformU32 { dst, reply } => {
                fill_uniform::<u32>(rng_lock, stream, completion, dst, reply);
            }
            RngMsg::FillNormalF32 { dst, mean, std, reply } => {
                fill_normal::<f32>(rng_lock, stream, completion, dst, mean, std, reply);
            }
            RngMsg::FillNormalF64 { dst, mean, std, reply } => {
                fill_normal::<f64>(rng_lock, stream, completion, dst, mean, std, reply);
            }
            RngMsg::FillLogNormalF32 { dst, mean, std, reply } => {
                fill_log_normal::<f32>(rng_lock, stream, completion, dst, mean, std, reply);
            }
            RngMsg::Reseed { seed, reply } => {
                let mut g = rng_lock.lock();
                let _ = reply.send(g.0.set_seed(seed).map_err(|e| {
                    GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("set_seed: {e}"),
                    }
                }));
            }
        }
    }
}

fn reply_mock(msg: RngMsg) {
    let err = || GpuError::Unrecoverable("RngActor in mock mode".into());
    match msg {
        RngMsg::FillUniformF32 { reply, .. }
        | RngMsg::FillNormalF32 { reply, .. }
        | RngMsg::FillLogNormalF32 { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        RngMsg::FillUniformF64 { reply, .. } | RngMsg::FillNormalF64 { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        RngMsg::FillUniformU32 { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        RngMsg::Reseed { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
    }
}

fn fill_uniform<T>(
    rng_lock: &Mutex<SendCudaRng>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    dst: GpuRef<T>,
    reply: oneshot::Sender<Result<(), GpuError>>,
) where
    T: Send + 'static,
    cudarc::curand::sys::curandGenerator_t: UniformFill<T>,
{
    let dst_slice = match dst.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut owned = match Arc::try_unwrap(dst_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "RNG dst has multiple live references".into(),
            )));
            return;
        }
    };
    let lib_lock = rng_lock;
    dst.record_write(stream);
    envelope::run_kernel(
        LIB,
        stream,
        completion,
        (),
        reply,
        move || {
            let g = lib_lock.lock();
            g.0.fill_with_uniform(&mut owned)
                .map(|_| (owned,))
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("fill_uniform: {e}"),
                })
        },
    );
}

fn fill_normal<T>(
    rng_lock: &Mutex<SendCudaRng>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    dst: GpuRef<T>,
    mean: T,
    std: T,
    reply: oneshot::Sender<Result<(), GpuError>>,
) where
    T: Send + 'static + Copy,
    cudarc::curand::sys::curandGenerator_t: NormalFill<T>,
{
    let dst_slice = match dst.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut owned = match Arc::try_unwrap(dst_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "RNG dst has multiple live references".into(),
            )));
            return;
        }
    };
    let lib_lock = rng_lock;
    dst.record_write(stream);
    envelope::run_kernel(
        LIB,
        stream,
        completion,
        (),
        reply,
        move || {
            let g = lib_lock.lock();
            g.0.fill_with_normal(&mut owned, mean, std)
                .map(|_| (owned,))
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("fill_normal: {e}"),
                })
        },
    );
}

fn fill_log_normal<T>(
    rng_lock: &Mutex<SendCudaRng>,
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    dst: GpuRef<T>,
    mean: T,
    std: T,
    reply: oneshot::Sender<Result<(), GpuError>>,
) where
    T: Send + 'static + Copy,
    cudarc::curand::sys::curandGenerator_t: LogNormalFill<T>,
{
    let dst_slice = match dst.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut owned = match Arc::try_unwrap(dst_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "RNG dst has multiple live references".into(),
            )));
            return;
        }
    };
    let lib_lock = rng_lock;
    dst.record_write(stream);
    envelope::run_kernel(
        LIB,
        stream,
        completion,
        (),
        reply,
        move || {
            let g = lib_lock.lock();
            g.0.fill_with_log_normal(&mut owned, mean, std)
                .map(|_| (owned,))
                .map_err(|e| GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("fill_log_normal: {e}"),
                })
        },
    );
}
