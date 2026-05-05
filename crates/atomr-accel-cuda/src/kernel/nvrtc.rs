//! `NvrtcActor` — JIT-compile and launch user-supplied CUDA C++
//! kernels at runtime.
//!
//! Two-step lifecycle:
//! 1. `Compile { src, kernel_name, opts, reply }` → returns a
//!    [`KernelHandle`] tied to the current `DeviceState` generation.
//! 2. `Launch { kernel, args, cfg, reply }` → enqueues a kernel call
//!    on the actor's stream. Replies after stream completion.
//!
//! `KernelHandle` is `Send + Sync + 'static` and survives across actor
//! boundaries. It carries a generation token; if the underlying
//! context is rebuilt, [`KernelHandle::launch_check`] returns
//! `GpuError::GpuRefStale` and the launch fails fast.
//!
//! ## Phase 0.3 — boxed-dispatch arg types
//!
//! `KernelArg` previously had eleven explicit variants (one per dtype
//! for each of slice / scalar) and `handle_launch` matched on each
//! twice (once to validate, once to push). Phase 0.3 collapses the
//! typed pairs into two boxed-dyn variants plus a `Usize` fallback:
//!
//! * [`KernelArg::DevSlice`] — wraps a `Box<dyn DevSliceArg>`. The
//!   blanket impl `impl<T: CudaDtype> DevSliceArg for GpuRef<T>` covers
//!   every dtype the runtime understands.
//! * [`KernelArg::Scalar`] — wraps a `Box<dyn ScalarArg>`. Blanket
//!   impl `impl<T: CudaDtype> ScalarArg for T`.
//! * [`KernelArg::Usize`] — `usize` is not a `CudaDtype` (its size is
//!   platform-dependent) so the dedicated variant keeps it on the
//!   non-allocating path callers use most often.
//!
//! The pre-Phase-0.3 typed variants (`DevSliceF32`, `ScalarI32`, …)
//! are preserved as `#[deprecated]` aliases so existing callers
//! compile unchanged. They are normalised to the canonical
//! `DevSlice` / `Scalar` form via [`KernelArg::canonicalize`] inside
//! `handle_launch` before the launch loop.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::dispatch::{DevSliceArg, ScalarArg};
use crate::kernel::envelope;
use crate::stream::StreamAllocator;

const LIB: &str = "nvrtc";

/// Subset of cudarc's [`CompileOptions`] exposed at our message
/// surface. The full struct is large; F3 ships the common knobs.
#[derive(Debug, Clone, Default)]
pub struct NvrtcOpts {
    pub ftz: Option<bool>,
    pub maxrregcount: Option<usize>,
    pub name: Option<String>,
    pub use_fast_math: Option<bool>,
}

impl NvrtcOpts {
    fn into_cudarc(self) -> CompileOptions {
        CompileOptions {
            ftz: self.ftz,
            maxrregcount: self.maxrregcount,
            name: self.name,
            use_fast_math: self.use_fast_math,
            ..Default::default()
        }
    }
}

/// Handle to a JIT-compiled, loaded kernel function. Validity is
/// gated by [`crate::device::DeviceState::generation`].
#[derive(Clone)]
pub struct KernelHandle {
    func: Arc<CudaFunction>,
    /// `DeviceState.generation` at compile time.
    generation: u64,
    /// Source hash — used by the actor's module cache to dedupe.
    #[allow(dead_code)]
    src_hash: u64,
    pub name: String,
}

impl KernelHandle {
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// A single argument to an NVRTC kernel launch.
///
/// The two boxed variants ([`KernelArg::DevSlice`] and
/// [`KernelArg::Scalar`]) are the canonical Phase-0.3+ form; every
/// dtype the runtime understands routes through them via the
/// [`DevSliceArg`] / [`ScalarArg`] blanket impls. The remaining
/// typed-variant aliases are `#[deprecated]` and exist so pre-Phase-0.3
/// callers still compile.
pub enum KernelArg {
    /// Canonical: a typed device slice as `Box<dyn DevSliceArg>`.
    /// Construct as `KernelArg::DevSlice(Box::new(my_gpu_ref))` for any
    /// `GpuRef<T: CudaDtype>` (which is every supported dtype, including
    /// `u8` raw byte buffers).
    DevSlice(Box<dyn DevSliceArg>),
    /// Canonical: a typed scalar as `Box<dyn ScalarArg>`. Construct as
    /// `KernelArg::Scalar(Box::new(2.0_f32))`.
    Scalar(Box<dyn ScalarArg>),
    /// `usize` is not a `CudaDtype` (its size is platform-dependent)
    /// so it has its own variant.
    Usize(usize),

    // ----- Phase-0.2 typed-variant aliases (deprecated) ----------------
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceF32(GpuRef<f32>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceF64(GpuRef<f64>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceI32(GpuRef<i32>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceU32(GpuRef<u32>),
    #[deprecated(note = "use KernelArg::DevSlice with GpuRef directly")]
    DevSliceU8(GpuRef<u8>),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarF32(f32),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarF64(f64),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarI32(i32),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarU32(u32),
    #[deprecated(note = "use KernelArg::Scalar with the scalar value directly")]
    ScalarU64(u64),
}

impl KernelArg {
    /// Normalise any pre-Phase-0.3 typed-variant alias to the
    /// canonical [`KernelArg::DevSlice`] / [`KernelArg::Scalar`] /
    /// [`KernelArg::Usize`] form.
    ///
    /// Used by the actor to fold the ten deprecated typed variants
    /// into the two boxed-dyn variants before the launch loop. After
    /// canonicalisation the launch loop has exactly three arms
    /// (`DevSlice`, `Scalar`, `Usize`) instead of eleven.
    #[allow(deprecated)]
    pub fn canonicalize(self) -> KernelArg {
        match self {
            KernelArg::DevSlice(_) | KernelArg::Scalar(_) | KernelArg::Usize(_) => self,

            KernelArg::DevSliceF32(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceF64(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceI32(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceU32(g) => KernelArg::DevSlice(Box::new(g)),
            KernelArg::DevSliceU8(g) => KernelArg::DevSlice(Box::new(g)),

            KernelArg::ScalarF32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarF64(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarI32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarU32(v) => KernelArg::Scalar(Box::new(v)),
            KernelArg::ScalarU64(v) => KernelArg::Scalar(Box::new(v)),
        }
    }
}

pub enum NvrtcMsg {
    Compile {
        src: String,
        kernel_name: String,
        opts: NvrtcOpts,
        reply: oneshot::Sender<Result<KernelHandle, GpuError>>,
    },
    Launch {
        kernel: KernelHandle,
        args: Vec<KernelArg>,
        cfg: LaunchConfig,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct NvrtcActor {
    inner: NvrtcInner,
}

struct SendModule(Arc<CudaModule>);
unsafe impl Send for SendModule {}
unsafe impl Sync for SendModule {}
impl Clone for SendModule {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

enum NvrtcInner {
    Real {
        ctx: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        modules: Mutex<HashMap<u64, SendModule>>,
    },
    Mock,
}

impl NvrtcActor {
    pub fn props(
        stream: Arc<cudarc::driver::CudaStream>,
        _allocator: Arc<dyn StreamAllocator>,
        completion: Arc<dyn CompletionStrategy>,
        state: Arc<DeviceState>,
        ctx: Arc<cudarc::driver::CudaContext>,
    ) -> Props<Self> {
        Props::create(move || NvrtcActor {
            inner: NvrtcInner::Real {
                ctx: ctx.clone(),
                stream: stream.clone(),
                completion: completion.clone(),
                state: state.clone(),
                modules: Mutex::new(HashMap::new()),
            },
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| NvrtcActor {
            inner: NvrtcInner::Mock,
        })
    }
}

#[async_trait]
impl Actor for NvrtcActor {
    type Msg = NvrtcMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: NvrtcMsg) {
        match &self.inner {
            NvrtcInner::Mock => match msg {
                NvrtcMsg::Compile { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "NvrtcActor in mock mode".into(),
                    )));
                }
                NvrtcMsg::Launch { reply, .. } => {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "NvrtcActor in mock mode".into(),
                    )));
                }
            },
            NvrtcInner::Real {
                ctx,
                stream,
                completion,
                state,
                modules,
            } => match msg {
                NvrtcMsg::Compile {
                    src,
                    kernel_name,
                    opts,
                    reply,
                } => {
                    let _ = reply.send(handle_compile(ctx, state, modules, src, kernel_name, opts));
                }
                NvrtcMsg::Launch {
                    kernel,
                    args,
                    cfg,
                    reply,
                } => {
                    handle_launch(stream, completion, state, kernel, args, cfg, reply);
                }
            },
        }
    }
}

fn hash_src(src: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut h);
    h.finish()
}

fn handle_compile(
    ctx: &Arc<cudarc::driver::CudaContext>,
    state: &Arc<DeviceState>,
    modules: &Mutex<HashMap<u64, SendModule>>,
    src: String,
    kernel_name: String,
    opts: NvrtcOpts,
) -> Result<KernelHandle, GpuError> {
    let src_hash = hash_src(&src);
    // Module cache (per actor lifetime).
    let module = {
        let mut g = modules.lock();
        if let Some(m) = g.get(&src_hash) {
            m.clone()
        } else {
            let ptx: Ptx = compile_ptx_with_opts(&src, opts.into_cudarc()).map_err(|e| {
                GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("compile_ptx: {e}"),
                }
            })?;
            let module = ctx.load_module(ptx).map_err(|e| GpuError::LibraryError {
                lib: LIB,
                msg: format!("load_module: {e}"),
            })?;
            let sm = SendModule(module);
            g.insert(src_hash, sm.clone());
            sm
        }
    };
    let func = module
        .0
        .load_function(&kernel_name)
        .map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("load_function {kernel_name}: {e}"),
        })?;
    Ok(KernelHandle {
        func: Arc::new(func),
        generation: state.generation(),
        src_hash,
        name: kernel_name,
    })
}

fn handle_launch(
    stream: &Arc<cudarc::driver::CudaStream>,
    completion: &Arc<dyn CompletionStrategy>,
    state: &Arc<DeviceState>,
    kernel: KernelHandle,
    args: Vec<KernelArg>,
    cfg: LaunchConfig,
    reply: oneshot::Sender<Result<(), GpuError>>,
) {
    if kernel.generation != state.generation() {
        let _ = reply.send(Err(GpuError::GpuRefStale(
            "nvrtc kernel from prior context generation",
        )));
        return;
    }

    // Collapse all deprecated typed variants into the canonical
    // boxed-dyn form so the loops below have a uniform 3-arm match
    // instead of one arm per (slice|scalar) × dtype.
    let args: Vec<KernelArg> = args.into_iter().map(KernelArg::canonicalize).collect();

    // Validate every device-slice arg first; abort on stale.
    let mut gpu_owners: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    for arg in &args {
        if let KernelArg::DevSlice(b) = arg {
            match b.validate() {
                Ok(owner) => gpu_owners.push(owner),
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            }
        }
    }

    let func = kernel.func.clone();
    let stream_clone = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let mut builder = stream_clone.launch_builder(&func);
        // Push args. Two boxed-dyn calls (DevSlice / Scalar) plus
        // the literal Usize variant — versus the previous 11-arm
        // explicit match. The `gpu_owners` Vec already holds keep-
        // alive `Arc<CudaSlice<T>>` clones so the buffers cannot be
        // deallocated under the kernel.
        // SAFETY: kernel signature must match args; user contract.
        for arg in args.iter() {
            match arg {
                KernelArg::DevSlice(b) => {
                    if let Err(e) = b.push(&mut builder) {
                        return Err(e);
                    }
                }
                KernelArg::Scalar(b) => {
                    b.push(&mut builder);
                }
                KernelArg::Usize(v) => {
                    builder.arg(v);
                }
                // Unreachable: every deprecated variant was folded
                // into one of the three canonical forms above by
                // `canonicalize()`. The `unreachable!()` arm guards
                // against future enum additions that bypass the
                // canonicaliser.
                #[allow(deprecated)]
                KernelArg::DevSliceF32(_)
                | KernelArg::DevSliceF64(_)
                | KernelArg::DevSliceI32(_)
                | KernelArg::DevSliceU32(_)
                | KernelArg::DevSliceU8(_)
                | KernelArg::ScalarF32(_)
                | KernelArg::ScalarF64(_)
                | KernelArg::ScalarI32(_)
                | KernelArg::ScalarU32(_)
                | KernelArg::ScalarU64(_) => unreachable!("canonicalize() folds these arms"),
            }
        }
        let res = unsafe { builder.launch(cfg) };
        match res {
            Ok(_) => Ok((gpu_owners, func, args)),
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("launch: {e}"),
            }),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `Vec<KernelArg>` mixing scalar f32, scalar i32, and a
    /// (host-fake) `GpuRef<u8>` slice. We can't construct a real
    /// `GpuRef` without a CUDA context, so this test asserts only the
    /// *compile* side: the canonical variants accept the right types
    /// and the canonicaliser produces the expected variant counts.
    #[test]
    fn launch_args_collapse_compile() {
        let args: Vec<KernelArg> = vec![
            KernelArg::Scalar(Box::new(1.0f32)),
            KernelArg::Scalar(Box::new(42i32)),
            KernelArg::Usize(128),
        ];
        assert_eq!(args.len(), 3);
        // Canonicalisation is a no-op on already-canonical forms.
        let canon: Vec<KernelArg> = args.into_iter().map(KernelArg::canonicalize).collect();
        assert_eq!(canon.len(), 3);
        // Variant identity check.
        let mut n_scalar = 0;
        let mut n_usize = 0;
        for a in &canon {
            match a {
                KernelArg::Scalar(_) => n_scalar += 1,
                KernelArg::Usize(_) => n_usize += 1,
                _ => panic!("unexpected variant"),
            }
        }
        assert_eq!((n_scalar, n_usize), (2, 1));
    }

    /// Each `#[deprecated]` constructor canonicalises to the matching
    /// boxed variant.
    #[test]
    fn deprecated_aliases_still_construct() {
        #[allow(deprecated)]
        let aliases = vec![
            KernelArg::ScalarF32(1.0),
            KernelArg::ScalarF64(2.0),
            KernelArg::ScalarI32(3),
            KernelArg::ScalarU32(4),
            KernelArg::ScalarU64(5),
        ];
        for a in aliases {
            let c = a.canonicalize();
            assert!(matches!(c, KernelArg::Scalar(_)));
        }
    }
}
