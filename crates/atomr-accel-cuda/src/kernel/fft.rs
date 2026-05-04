//! `FftActor` — wraps [`cudarc::cufft::CudaFft`] with an LRU cache of
//! plans keyed by shape + transform kind + batch.
//!
//! cuFFT's plan creation can take milliseconds; the cache amortizes
//! that across many transforms of the same shape.
//!
//! Each transform variant is a `Msg` enum entry. `Msg` arity is kept
//! deliberately small in F2: 1D real↔complex (R2C/C2R) f32, 1D
//! complex↔complex f32 (forward + inverse), 2D R2C f32. The full
//! cudarc surface (D2Z/Z2D/Z2Z f64, batched f64, plan_many) lands
//! incrementally as patterns demand it.
//!
//! **Inverse normalization:** cuFFT does NOT normalize inverse
//! transforms by 1/N — caller's responsibility (typically folded
//! into a downstream kernel).

use std::num::NonZeroUsize;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::cufft::sys as cufft_sys;
use cudarc::cufft::{CudaFft, FftDirection};
use lru::LruCache;
use parking_lot::Mutex;
use atomr_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "cufft";

/// Transform kind. Mirrors a useful subset of `cufftType` and the
/// direction needed for in-place complex transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FftKind {
    /// f32 real → complex
    R2cF32,
    /// f32 complex → real
    C2rF32,
    /// f32 complex ↔ complex (direction picked at exec time)
    C2cF32,
}

impl FftKind {
    fn cufft_type(self) -> cufft_sys::cufftType {
        match self {
            FftKind::R2cF32 => cufft_sys::cufftType::CUFFT_R2C,
            FftKind::C2rF32 => cufft_sys::cufftType::CUFFT_C2R,
            FftKind::C2cF32 => cufft_sys::cufftType::CUFFT_C2C,
        }
    }
}

/// Plan-cache key. Captures everything cuFFT cares about for
/// `cufftPlan{1,2,3}d` plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanKey {
    Plan1d {
        n: i32,
        kind: FftKind,
        batch: i32,
    },
    Plan2d {
        nx: i32,
        ny: i32,
        kind: FftKind,
    },
    Plan3d {
        nx: i32,
        ny: i32,
        nz: i32,
        kind: FftKind,
    },
}

pub enum FftMsg {
    /// 1D real → complex forward transform (f32 → complex32).
    Forward1dR2C {
        n: i32,
        batch: i32,
        src: GpuRef<f32>,
        dst: GpuRef<cufft_sys::float2>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// 1D complex → real inverse transform (complex32 → f32).
    /// Caller is responsible for 1/N normalization.
    Inverse1dC2R {
        n: i32,
        batch: i32,
        src: GpuRef<cufft_sys::float2>,
        dst: GpuRef<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// 1D complex ↔ complex transform.
    Exec1dC2C {
        n: i32,
        batch: i32,
        direction: FftDirection,
        src: GpuRef<cufft_sys::float2>,
        dst: GpuRef<cufft_sys::float2>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// 2D R2C transform.
    Forward2dR2C {
        nx: i32,
        ny: i32,
        src: GpuRef<f32>,
        dst: GpuRef<cufft_sys::float2>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct FftActor {
    inner: FftInner,
}

/// `CudaFft` is `Send + Sync` (it has explicit `unsafe impl`s).
/// The plan LRU must serialize access from the actor task; we use a
/// parking_lot mutex which is fast and uncontended on the
/// dispatcher thread.
struct PlanCache {
    cache: LruCache<PlanKey, Arc<CudaFft>>,
}

impl PlanCache {
    fn new(cap: NonZeroUsize) -> Self {
        Self {
            cache: LruCache::new(cap),
        }
    }
}

enum FftInner {
    Real {
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        plans: Mutex<PlanCache>,
        #[allow(dead_code)]
        state: Arc<DeviceState>,
    },
    Mock,
}

const DEFAULT_CACHE_SIZE: usize = 64;

impl FftActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        _ctx: Arc<cudarc::driver::CudaContext>,
    ) -> Props<Self> {
        Props::create(move || FftActor {
            inner: FftInner::Real {
                stream: stream.clone(),
                completion: completion.clone(),
                plans: Mutex::new(PlanCache::new(
                    NonZeroUsize::new(DEFAULT_CACHE_SIZE).unwrap(),
                )),
                state: state.clone(),
            },
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| FftActor {
            inner: FftInner::Mock,
        })
    }
}

impl FftActor {
    fn get_or_create_plan(&self, key: PlanKey) -> Result<Arc<CudaFft>, GpuError> {
        let FftInner::Real { stream, plans, .. } = &self.inner else {
            return Err(GpuError::Unrecoverable("fft mock".into()));
        };
        let mut g = plans.lock();
        if let Some(plan) = g.cache.get(&key) {
            return Ok(plan.clone());
        }
        let plan = match key {
            PlanKey::Plan1d { n, kind, batch } => {
                CudaFft::plan_1d(n, kind.cufft_type(), batch, stream.clone())
            }
            PlanKey::Plan2d { nx, ny, kind } => {
                CudaFft::plan_2d(nx, ny, kind.cufft_type(), stream.clone())
            }
            PlanKey::Plan3d { nx, ny, nz, kind } => {
                CudaFft::plan_3d(nx, ny, nz, kind.cufft_type(), stream.clone())
            }
        };
        let plan = plan.map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("plan {key:?}: {e}"),
        })?;
        let plan = Arc::new(plan);
        g.cache.put(key, plan.clone());
        Ok(plan)
    }
}

#[async_trait]
impl Actor for FftActor {
    type Msg = FftMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: FftMsg) {
        let (stream, completion) = match &self.inner {
            FftInner::Mock => {
                reply_mock(msg);
                return;
            }
            FftInner::Real {
                stream, completion, ..
            } => (stream.clone(), completion.clone()),
        };

        match msg {
            FftMsg::Forward1dR2C {
                n,
                batch,
                src,
                dst,
                reply,
            } => {
                let plan = match self.get_or_create_plan(PlanKey::Plan1d {
                    n,
                    kind: FftKind::R2cF32,
                    batch,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let (src_slice, dst_slice) = match envelope::access_all_2(&src, &dst) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let mut dst_owned = match Arc::try_unwrap(dst_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "FFT dst has multiple live references".into(),
                        )));
                        return;
                    }
                };
                dst.record_write(&stream);
                envelope::run_kernel(LIB, &stream, &completion, (), reply, move || {
                    plan.exec_r2c(&*src_slice, &mut dst_owned)
                        .map(|_| (src_slice, dst_owned, plan))
                        .map_err(|e| GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("exec_r2c: {e}"),
                        })
                });
            }
            FftMsg::Inverse1dC2R {
                n,
                batch,
                src,
                dst,
                reply,
            } => {
                let plan = match self.get_or_create_plan(PlanKey::Plan1d {
                    n,
                    kind: FftKind::C2rF32,
                    batch,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let (src_slice, dst_slice) = match envelope::access_all_2(&src, &dst) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let mut src_owned = match Arc::try_unwrap(src_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "FFT C2R src has multiple live references".into(),
                        )));
                        return;
                    }
                };
                let mut dst_owned = match Arc::try_unwrap(dst_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "FFT C2R dst has multiple live references".into(),
                        )));
                        return;
                    }
                };
                dst.record_write(&stream);
                envelope::run_kernel(LIB, &stream, &completion, (), reply, move || {
                    plan.exec_c2r(&mut src_owned, &mut dst_owned)
                        .map(|_| (src_owned, dst_owned, plan))
                        .map_err(|e| GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("exec_c2r: {e}"),
                        })
                });
            }
            FftMsg::Exec1dC2C {
                n,
                batch,
                direction,
                src,
                dst,
                reply,
            } => {
                let plan = match self.get_or_create_plan(PlanKey::Plan1d {
                    n,
                    kind: FftKind::C2cF32,
                    batch,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let (src_slice, dst_slice) = match envelope::access_all_2(&src, &dst) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let mut src_owned = match Arc::try_unwrap(src_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "FFT C2C src has multiple live references".into(),
                        )));
                        return;
                    }
                };
                let mut dst_owned = match Arc::try_unwrap(dst_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "FFT C2C dst has multiple live references".into(),
                        )));
                        return;
                    }
                };
                dst.record_write(&stream);
                envelope::run_kernel(LIB, &stream, &completion, (), reply, move || {
                    plan.exec_c2c(&mut src_owned, &mut dst_owned, direction)
                        .map(|_| (src_owned, dst_owned, plan))
                        .map_err(|e| GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("exec_c2c: {e}"),
                        })
                });
            }
            FftMsg::Forward2dR2C {
                nx,
                ny,
                src,
                dst,
                reply,
            } => {
                let plan = match self.get_or_create_plan(PlanKey::Plan2d {
                    nx,
                    ny,
                    kind: FftKind::R2cF32,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let (src_slice, dst_slice) = match envelope::access_all_2(&src, &dst) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        return;
                    }
                };
                let mut dst_owned = match Arc::try_unwrap(dst_slice) {
                    Ok(s) => s,
                    Err(_) => {
                        let _ = reply.send(Err(GpuError::Unrecoverable(
                            "FFT 2D dst has multiple live references".into(),
                        )));
                        return;
                    }
                };
                dst.record_write(&stream);
                envelope::run_kernel(LIB, &stream, &completion, (), reply, move || {
                    plan.exec_r2c(&*src_slice, &mut dst_owned)
                        .map(|_| (src_slice, dst_owned, plan))
                        .map_err(|e| GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("exec_r2c (2d): {e}"),
                        })
                });
            }
        }
    }
}

fn reply_mock(msg: FftMsg) {
    let err = || GpuError::Unrecoverable("FftActor in mock mode".into());
    match msg {
        FftMsg::Forward1dR2C { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        FftMsg::Inverse1dC2R { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        FftMsg::Exec1dC2C { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
        FftMsg::Forward2dR2C { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
    }
}
