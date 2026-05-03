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

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions, Ptx};
use parking_lot::Mutex;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::device::DeviceState;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
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

/// Subset of CUDA scalar argument types accepted by `Launch`. The
/// FFI-safety contract for user-supplied kernels lives entirely on
/// the caller; this enum just enumerates what we know how to push
/// through cudarc's `launch_builder().arg(...)` chain.
pub enum KernelArg {
    DevSliceF32(GpuRef<f32>),
    DevSliceF64(GpuRef<f64>),
    DevSliceI32(GpuRef<i32>),
    DevSliceU32(GpuRef<u32>),
    DevSliceU8(GpuRef<u8>),
    ScalarF32(f32),
    ScalarF64(f64),
    ScalarI32(i32),
    ScalarU32(u32),
    ScalarU64(u64),
    Usize(usize),
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
    fn clone(&self) -> Self { Self(self.0.clone()) }
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
        Props::create(|| NvrtcActor { inner: NvrtcInner::Mock })
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
            NvrtcInner::Real { ctx, stream, completion, state, modules } => match msg {
                NvrtcMsg::Compile { src, kernel_name, opts, reply } => {
                    let _ = reply.send(handle_compile(ctx, state, modules, src, kernel_name, opts));
                }
                NvrtcMsg::Launch { kernel, args, cfg, reply } => {
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
    // Validate every GpuRef arg first; abort on stale.
    let mut gpu_owners: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    for arg in &args {
        match arg {
            KernelArg::DevSliceF32(g) => match g.access() { Ok(s) => gpu_owners.push(Box::new(s.clone())), Err(e) => { let _ = reply.send(Err(e)); return; } }
            KernelArg::DevSliceF64(g) => match g.access() { Ok(s) => gpu_owners.push(Box::new(s.clone())), Err(e) => { let _ = reply.send(Err(e)); return; } }
            KernelArg::DevSliceI32(g) => match g.access() { Ok(s) => gpu_owners.push(Box::new(s.clone())), Err(e) => { let _ = reply.send(Err(e)); return; } }
            KernelArg::DevSliceU32(g) => match g.access() { Ok(s) => gpu_owners.push(Box::new(s.clone())), Err(e) => { let _ = reply.send(Err(e)); return; } }
            KernelArg::DevSliceU8(g) => match g.access() { Ok(s) => gpu_owners.push(Box::new(s.clone())), Err(e) => { let _ = reply.send(Err(e)); return; } }
            _ => {}
        }
    }

    let func = kernel.func.clone();
    let stream_clone = stream.clone();
    envelope::run_kernel(LIB, stream, completion, (), reply, move || {
        let mut builder = stream_clone.launch_builder(&func);
        // Push args. Device-pointer args use the owners we already
        // validated; scalars push by reference into the LaunchArgs.
        // We hold scalars in local variables so their references
        // outlive the .arg() call.
        // SAFETY: kernel signature must match args; user contract.
        for arg in args.iter() {
            match arg {
                KernelArg::DevSliceF32(g) => {
                    let s = g.access().expect("re-validated above");
                    builder.arg(&**s);
                }
                KernelArg::DevSliceF64(g) => {
                    let s = g.access().expect("re-validated above");
                    builder.arg(&**s);
                }
                KernelArg::DevSliceI32(g) => {
                    let s = g.access().expect("re-validated above");
                    builder.arg(&**s);
                }
                KernelArg::DevSliceU32(g) => {
                    let s = g.access().expect("re-validated above");
                    builder.arg(&**s);
                }
                KernelArg::DevSliceU8(g) => {
                    let s = g.access().expect("re-validated above");
                    builder.arg(&**s);
                }
                KernelArg::ScalarF32(v) => { builder.arg(v); }
                KernelArg::ScalarF64(v) => { builder.arg(v); }
                KernelArg::ScalarI32(v) => { builder.arg(v); }
                KernelArg::ScalarU32(v) => { builder.arg(v); }
                KernelArg::ScalarU64(v) => { builder.arg(v); }
                KernelArg::Usize(v) => { builder.arg(v); }
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
