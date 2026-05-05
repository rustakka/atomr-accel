//! `RngActor` — wraps a cuRAND `curandGenerator_t` handle and fills
//! device buffers with the full distribution matrix.
//!
//! Phase 1 cuRAND surface (vs. F2):
//!
//! * **Explicit generator selection** via [`RngGeneratorKind`]
//!   (Philox4_32_10, XORWOW, MTGP32, MRG32K3A, plus all four Sobol
//!   variants). [`RngMsg::SetGenerator`] reconstructs the handle in
//!   place so callers can switch families at runtime.
//! * **Distribution matrix** (`Uniform`, `Normal`, `LogNormal`,
//!   `Poisson`, `Exponential`, `Beta`, `Cauchy`, `Gamma`,
//!   `Discrete`) routed through [`Distribution<T>`] →
//!   [`FillRequest<T>`] → `RngDispatch::fill`.
//! * **Quasi-random Sobol** parallel to pseudo-random: see
//!   [`sobol`] for dimension configuration.
//! * **Host API parallel to device API** under the `curand-host`
//!   feature: see [`host`].
//!
//! Reseed model: `SetSeed { seed }` calls
//! [`crate::sys::curand::set_seed`] in place — no panic-restart.
//! Reseed is a control-plane op; restart-on-reseed would tear down
//! all in-flight work. The seed is journaled by `ReplayHarness` (F5)
//! so deterministic replay still works.

use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::dtype::RngFloatSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::RngDispatch;
use crate::stream::StreamAllocator;
use crate::sys::curand as csys;

pub mod dist;
#[cfg(feature = "curand-host")]
pub mod host;
#[cfg(feature = "curand-quasirandom")]
pub mod sobol;

pub use crate::sys::curand::RngGeneratorKind;
pub use dist::{Distribution, FillRequest};

pub(crate) const LIB: &str = "curand";

/// Public messages for [`RngActor`].
///
/// Two-track API:
///
/// * **Modern** — [`RngMsg::Fill`], [`RngMsg::SetSeed`],
///   [`RngMsg::SetGenerator`]. Callers build a [`FillRequest<T>`],
///   wrap it in `Box<dyn RngDispatch>`, and send it as
///   `RngMsg::Fill(Box::new(req))`.
/// * **Legacy** — `Fill{Uniform,Normal,LogNormal}*` plus `Reseed`,
///   preserved for F2 callers. Marked `#[deprecated]`.
#[non_exhaustive]
pub enum RngMsg {
    /// Type-erased dispatch: see [`RngDispatch`].
    Fill(Box<dyn RngDispatch>),
    /// Re-seed the **active** generator (no-op for quasi generators).
    SetSeed {
        seed: u64,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Tear down the current generator and reconstruct it as `kind`.
    /// Pseudo→quasi (or vice-versa) is supported. Quasi generators
    /// take effect with the default 1-dimensional Sobol; use
    /// [`sobol::SetDimensions`] to widen.
    SetGenerator {
        kind: RngGeneratorKind,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::Fill(Box::new(FillRequest { ... })) instead")]
    FillUniformF32 {
        dst: GpuRef<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::Fill(Box::new(FillRequest { ... })) instead")]
    FillUniformF64 {
        dst: GpuRef<f64>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::Fill(Box::new(FillRequest { ... })) instead")]
    FillUniformU32 {
        dst: GpuRef<u32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::Fill(Box::new(FillRequest { ... })) instead")]
    FillNormalF32 {
        dst: GpuRef<f32>,
        mean: f32,
        std: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::Fill(Box::new(FillRequest { ... })) instead")]
    FillNormalF64 {
        dst: GpuRef<f64>,
        mean: f64,
        std: f64,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::Fill(Box::new(FillRequest { ... })) instead")]
    FillLogNormalF32 {
        dst: GpuRef<f32>,
        mean: f32,
        std: f32,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    #[deprecated(note = "use RngMsg::SetSeed { seed, reply } instead")]
    Reseed {
        seed: u64,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

/// `curandGenerator_t` is a raw `*mut curandGenerator_st` and so is
/// `!Send + !Sync`. The actor runs exclusively on
/// [`crate::dispatcher::GpuDispatcher`]'s pinned thread; we assert
/// `Send + Sync` via this newtype so atomr's `Actor: Send + 'static`
/// bound is satisfied.
pub(crate) struct SendGen(pub(crate) cudarc::curand::sys::curandGenerator_t);

// SAFETY: the generator is only ever touched from the GpuDispatcher's
// pinned OS thread; the outer parking_lot::Mutex enforces exclusion
// against any actor handler running there.
unsafe impl Send for SendGen {}
unsafe impl Sync for SendGen {}

pub struct RngActor {
    inner: RngInner,
}

pub(crate) enum RngInner {
    Real {
        gen: Mutex<SendGen>,
        kind: Mutex<RngGeneratorKind>,
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
        Self::props_with_kind(
            stream,
            _allocator,
            completion,
            state,
            seed,
            RngGeneratorKind::default(),
        )
    }

    /// Same as [`Self::props`] but lets the caller pick the cuRAND
    /// generator family upfront.
    pub fn props_with_kind(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        seed: u64,
        kind: RngGeneratorKind,
    ) -> Props<Self> {
        Props::create(move || {
            let g = unsafe {
                construct_generator(kind, &stream, seed).unwrap_or_else(|e| {
                    panic!("ContextPoisoned: cuRAND generator init failed ({kind:?}): {e}")
                })
            };
            RngActor {
                inner: RngInner::Real {
                    gen: Mutex::new(SendGen(g)),
                    kind: Mutex::new(kind),
                    stream: stream.clone(),
                    completion: completion.clone(),
                    state: state.clone(),
                },
            }
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| RngActor {
            inner: RngInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for RngActor {
    type Msg = RngMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: RngMsg) {
        let (gen_lock, kind_lock, stream, completion) = match &self.inner {
            RngInner::Mock => {
                reply_mock(msg);
                return;
            }
            RngInner::Real {
                gen,
                kind,
                stream,
                completion,
                ..
            } => (gen, kind, stream, completion),
        };

        #[allow(deprecated)]
        match msg {
            RngMsg::Fill(req) => {
                let gen_handle = gen_lock.lock().0;
                if let Err(e) = req.fill(gen_handle, stream, completion) {
                    // RngDispatch::fill is responsible for sending its
                    // own reply on success; on Err the reply is also
                    // expected to have been sent by the impl. The
                    // returned error is therefore advisory — log it
                    // for tracing parity with other actors.
                    tracing::warn!(lib = LIB, error = %e, "RngActor::Fill pre-launch error");
                }
            }
            RngMsg::SetSeed { seed, reply } | RngMsg::Reseed { seed, reply } => {
                let g = gen_lock.lock();
                let active = *kind_lock.lock();
                let r = if active.is_quasi() {
                    // Quasi generators don't accept a pseudo seed —
                    // cuRAND returns CURAND_STATUS_TYPE_ERROR. Treat
                    // SetSeed on a quasi RNG as a no-op so callers
                    // can journal a single seed regardless of family.
                    Ok(())
                } else {
                    unsafe { csys::set_seed(g.0, seed) }.map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("set_seed: {e}"),
                    })
                };
                let _ = reply.send(r);
            }
            RngMsg::SetGenerator { kind, reply } => {
                let mut g = gen_lock.lock();
                let mut active = kind_lock.lock();
                let r = unsafe {
                    let _ = csys::destroy_generator(g.0);
                    match construct_generator(kind, stream, 0) {
                        Ok(new_g) => {
                            g.0 = new_g;
                            *active = kind;
                            Ok(())
                        }
                        Err(e) => Err(GpuError::LibraryError {
                            lib: LIB,
                            msg: format!("set_generator({kind:?}): {e}"),
                        }),
                    }
                };
                let _ = reply.send(r);
            }
            // Legacy variants — translate into the modern path.
            RngMsg::FillUniformF32 { dst, reply } => {
                let req = FillRequest::<f32> {
                    buf: dst,
                    dist: Distribution::Uniform { lo: 0.0, hi: 1.0 },
                    reply,
                };
                let gen_handle = gen_lock.lock().0;
                let _ = Box::new(req).fill(gen_handle, stream, completion);
            }
            RngMsg::FillUniformF64 { dst, reply } => {
                let req = FillRequest::<f64> {
                    buf: dst,
                    dist: Distribution::Uniform { lo: 0.0, hi: 1.0 },
                    reply,
                };
                let gen_handle = gen_lock.lock().0;
                let _ = Box::new(req).fill(gen_handle, stream, completion);
            }
            RngMsg::FillUniformU32 { dst, reply } => {
                let gen_handle = gen_lock.lock().0;
                dist::fill_uniform_u32(gen_handle, stream, completion, dst, reply);
            }
            RngMsg::FillNormalF32 {
                dst,
                mean,
                std,
                reply,
            } => {
                let req = FillRequest::<f32> {
                    buf: dst,
                    dist: Distribution::Normal { mean, std },
                    reply,
                };
                let gen_handle = gen_lock.lock().0;
                let _ = Box::new(req).fill(gen_handle, stream, completion);
            }
            RngMsg::FillNormalF64 {
                dst,
                mean,
                std,
                reply,
            } => {
                let req = FillRequest::<f64> {
                    buf: dst,
                    dist: Distribution::Normal { mean, std },
                    reply,
                };
                let gen_handle = gen_lock.lock().0;
                let _ = Box::new(req).fill(gen_handle, stream, completion);
            }
            RngMsg::FillLogNormalF32 {
                dst,
                mean,
                std,
                reply,
            } => {
                let req = FillRequest::<f32> {
                    buf: dst,
                    dist: Distribution::LogNormal { mean, std },
                    reply,
                };
                let gen_handle = gen_lock.lock().0;
                let _ = Box::new(req).fill(gen_handle, stream, completion);
            }
        }
    }
}

/// Build a fresh cuRAND generator of `kind`, bind it to `stream`, and
/// (for pseudo families) seed it. Used by both [`RngActor::props`] and
/// [`RngMsg::SetGenerator`].
///
/// # Safety
/// The returned handle is owned by the caller. It must be released
/// through [`csys::destroy_generator`].
pub(crate) unsafe fn construct_generator(
    kind: RngGeneratorKind,
    stream: &Arc<cudarc::driver::CudaStream>,
    seed: u64,
) -> Result<cudarc::curand::sys::curandGenerator_t, cudarc::curand::result::CurandError> {
    let g = csys::create_generator(kind)?;
    csys::set_stream(g, stream.cu_stream() as _)?;
    if !kind.is_quasi() {
        csys::set_seed(g, seed)?;
    }
    Ok(g)
}

impl Drop for RngActor {
    fn drop(&mut self) {
        if let RngInner::Real { gen, .. } = &self.inner {
            let g = gen.lock();
            if !g.0.is_null() {
                let _ = unsafe { csys::destroy_generator(g.0) };
            }
        }
    }
}

#[allow(deprecated)]
fn reply_mock(msg: RngMsg) {
    let err = || GpuError::Unrecoverable("RngActor in mock mode".into());
    match msg {
        RngMsg::Fill(req) => {
            // Drop the boxed dispatch. We can't fish out the reply
            // sender generically, so the caller observes the channel
            // close as a "Cancelled" / send error — same as the F2
            // mock semantics for any unsupported variant.
            drop(req);
        }
        RngMsg::SetSeed { reply, .. }
        | RngMsg::SetGenerator { reply, .. }
        | RngMsg::Reseed { reply, .. } => {
            let _ = reply.send(Err(err()));
        }
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
    }
}

/// Re-exported here so callers can `use atomr_accel_cuda::kernel::rng::props::*`
/// for the public surface without remembering which module defines
/// what.
pub mod props {
    pub use super::dist::{Distribution, FillRequest};
    pub use super::{RngActor, RngGeneratorKind, RngMsg};
}

// ---------------------------------------------------------------------
// Capability-marker compile-fail check.
//
// `FillRequest<T>` is parameterised by `T: RngFloatSupported`, which
// is implemented for `f32` and `f64` only. A call site that tries to
// instantiate `FillRequest<u32>` must fail to compile. We can't run
// `compile_fail` doctests on the test target (no `pub use` of GpuRef
// inside this module path), so the check lives in a docstring under
// the publicly-reachable [`FillRequest`] re-export below.
// ---------------------------------------------------------------------

/// Compile-fail proof that [`FillRequest`] rejects non-float dtypes.
///
/// ```compile_fail
/// use atomr_accel_cuda::kernel::{Distribution, FillRequest};
/// fn _bad(b: atomr_accel_cuda::gpu_ref::GpuRef<u32>) {
///     let (tx, _rx) = tokio::sync::oneshot::channel();
///     let _r: FillRequest<u32> = FillRequest {
///         buf: b,
///         dist: Distribution::Uniform { lo: 0u32, hi: 1u32 },
///         reply: tx,
///     };
/// }
/// ```
pub fn _capability_marker_compile_fail_doc<T: RngFloatSupported>(_: T::Scalar) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_msg_legacy_variants_present() {
        // Ensure deprecated names remain in the API surface so older
        // callers compile (they only get a deprecation warning).
        #[allow(deprecated)]
        fn _accept(m: RngMsg) {
            match m {
                RngMsg::FillUniformF32 { .. } => {}
                RngMsg::FillUniformF64 { .. } => {}
                RngMsg::FillUniformU32 { .. } => {}
                RngMsg::FillNormalF32 { .. } => {}
                RngMsg::FillNormalF64 { .. } => {}
                RngMsg::FillLogNormalF32 { .. } => {}
                RngMsg::Reseed { .. } => {}
                RngMsg::Fill(_) | RngMsg::SetSeed { .. } | RngMsg::SetGenerator { .. } => {}
            }
        }
    }

    #[test]
    fn set_generator_kind_round_trip() {
        // Round-trips every variant through `to_sys` to make sure no
        // arm panics or returns a stale numeric value. (Real handle
        // creation requires a CUDA context; that's covered by the
        // GPU-runtime e2e suite.)
        let all = [
            RngGeneratorKind::PseudoDefault,
            RngGeneratorKind::Philox4_32_10,
            RngGeneratorKind::XorWow,
            RngGeneratorKind::Mrg32K3A,
            RngGeneratorKind::Mtgp32,
            RngGeneratorKind::Sobol32,
            RngGeneratorKind::ScrambledSobol32,
            RngGeneratorKind::Sobol64,
            RngGeneratorKind::ScrambledSobol64,
        ];
        let mut seen = std::collections::HashSet::new();
        for k in all {
            let v = k.to_sys() as u32;
            assert!(seen.insert(v), "duplicate sys value for {k:?}");
            assert_eq!(k.is_quasi(), (v as i32) >= 200);
        }
    }
}
