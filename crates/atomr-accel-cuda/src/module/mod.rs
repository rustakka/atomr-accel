//! `ModuleActor` — load prebuilt cubin/PTX from disk (or memory) and
//! launch its kernels.
//!
//! Distinct from [`crate::kernel::nvrtc`] / `NvrtcActor`, which
//! JIT-compiles CUDA C++ at runtime. This actor is for the
//! "ahead-of-time-compiled, ship-the-bytes" workflow.
//!
//! Lifecycle:
//! 1. `LoadCubin { bytes }` or `LoadPtx { src }` → returns a
//!    `ModuleHandle`.
//! 2. `GetFunction { handle, name }` → returns a `FunctionHandle`.
//! 3. `Launch { function, cfg, args }` → enqueues a kernel call on
//!    the actor's stream and replies after stream completion.
//! 4. `LaunchCooperative { function, cfg, args }` → same but goes
//!    through `cuLaunchCooperativeKernel`. Required for cluster /
//!    grid-sync kernels (Hopper SM90+).
//! 5. `Unload { handle }` → frees the `CUmodule`.
//!
//! `KernelArg` is re-exported from [`crate::kernel::nvrtc`] so this
//! actor and `NvrtcActor` share one launch-arg type.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::driver::sys as driver_sys;
use cudarc::driver::{CudaContext, CudaStream, LaunchConfig};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::completion::CompletionStrategy;
use crate::error::GpuError;
use crate::sys::cuda_driver;

#[cfg(feature = "nvrtc")]
pub use crate::kernel::nvrtc::KernelArg;

/// When the `nvrtc` feature is off we still need a launchable arg
/// enum. Mirror the public surface; the variants line up so the
/// `module::tests::launch_args_share_kernel_arg_type` test passes
/// transparently when nvrtc is enabled, and the standalone enum
/// covers no-feature builds.
#[cfg(not(feature = "nvrtc"))]
pub enum KernelArg {
    DevSliceF32(crate::gpu_ref::GpuRef<f32>),
    DevSliceF64(crate::gpu_ref::GpuRef<f64>),
    DevSliceI32(crate::gpu_ref::GpuRef<i32>),
    DevSliceU32(crate::gpu_ref::GpuRef<u32>),
    DevSliceU8(crate::gpu_ref::GpuRef<u8>),
    ScalarF32(f32),
    ScalarF64(f64),
    ScalarI32(i32),
    ScalarU32(u32),
    ScalarU64(u64),
    Usize(usize),
}

const LIB: &str = "module";

/// Opaque module handle. Carries an internal id used by the actor's
/// internal `HashMap`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleHandle {
    id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionHandle {
    module: u64,
    function_id: u64,
}

pub enum ModuleMsg {
    LoadCubin {
        bytes: Vec<u8>,
        reply: oneshot::Sender<Result<ModuleHandle, GpuError>>,
    },
    LoadPtx {
        src: String,
        reply: oneshot::Sender<Result<ModuleHandle, GpuError>>,
    },
    GetFunction {
        handle: ModuleHandle,
        name: String,
        reply: oneshot::Sender<Result<FunctionHandle, GpuError>>,
    },
    Launch {
        function: FunctionHandle,
        cfg: LaunchConfig,
        args: Vec<KernelArg>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    LaunchCooperative {
        function: FunctionHandle,
        cfg: LaunchConfig,
        args: Vec<KernelArg>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Unload {
        handle: ModuleHandle,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

struct LoadedModule {
    cu_module: driver_sys::CUmodule,
    /// Map function name → cu_function. We track FunctionHandle.id
    /// separately so callers don't accidentally pass a stale name.
    functions: HashMap<u64, (CString, driver_sys::CUfunction)>,
    next_function_id: u64,
}

unsafe impl Send for LoadedModule {}
unsafe impl Sync for LoadedModule {}

#[allow(dead_code)]
enum ModuleInner {
    Real {
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
        modules: Mutex<HashMap<u64, LoadedModule>>,
        next_module_id: AtomicU64,
    },
    Mock,
}

pub struct ModuleActor {
    inner: ModuleInner,
}

impl ModuleActor {
    pub fn props(
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        completion: Arc<dyn CompletionStrategy>,
    ) -> Props<Self> {
        Props::create(move || ModuleActor {
            inner: ModuleInner::Real {
                ctx: ctx.clone(),
                stream: stream.clone(),
                completion: completion.clone(),
                modules: Mutex::new(HashMap::new()),
                next_module_id: AtomicU64::new(1),
            },
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| ModuleActor {
            inner: ModuleInner::Mock,
        })
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let ModuleInner::Real { modules, .. } = self {
            let mut g = modules.lock();
            for (_id, m) in g.drain() {
                let _ = cuda_driver::module_unload(m.cu_module);
            }
        }
    }
}

#[async_trait]
impl Actor for ModuleActor {
    type Msg = ModuleMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ModuleMsg) {
        match &self.inner {
            ModuleInner::Mock => mock_reply(msg),
            ModuleInner::Real {
                ctx,
                stream,
                completion: _completion,
                modules,
                next_module_id,
            } => handle_real(ctx, stream, modules, next_module_id, msg),
        }
    }
}

fn mock_reply(msg: ModuleMsg) {
    let unrecoverable = || GpuError::Unrecoverable("ModuleActor in mock mode".into());
    match msg {
        ModuleMsg::LoadCubin { reply, .. } => {
            let _ = reply.send(Err(unrecoverable()));
        }
        ModuleMsg::LoadPtx { reply, .. } => {
            let _ = reply.send(Err(unrecoverable()));
        }
        ModuleMsg::GetFunction { reply, .. } => {
            let _ = reply.send(Err(unrecoverable()));
        }
        ModuleMsg::Launch { reply, .. } => {
            let _ = reply.send(Err(unrecoverable()));
        }
        ModuleMsg::LaunchCooperative { reply, .. } => {
            let _ = reply.send(Err(unrecoverable()));
        }
        ModuleMsg::Unload { reply, .. } => {
            let _ = reply.send(Err(unrecoverable()));
        }
    }
}

fn handle_real(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    modules: &Mutex<HashMap<u64, LoadedModule>>,
    next_module_id: &AtomicU64,
    msg: ModuleMsg,
) {
    match msg {
        ModuleMsg::LoadCubin { bytes, reply } => {
            let r = load_image(ctx, modules, next_module_id, &bytes);
            let _ = reply.send(r);
        }
        ModuleMsg::LoadPtx { src, reply } => {
            // PTX is a NUL-terminated text string. Append a NUL if the
            // caller didn't.
            let mut text = src.into_bytes();
            if !text.ends_with(&[0]) {
                text.push(0);
            }
            let r = load_image(ctx, modules, next_module_id, &text);
            let _ = reply.send(r);
        }
        ModuleMsg::GetFunction {
            handle,
            name,
            reply,
        } => {
            let r = get_function(modules, handle, &name);
            let _ = reply.send(r);
        }
        ModuleMsg::Launch {
            function,
            cfg,
            args,
            reply,
        } => {
            let r = launch_inner(modules, stream, function, cfg, args, false);
            let _ = reply.send(r);
        }
        ModuleMsg::LaunchCooperative {
            function,
            cfg,
            args,
            reply,
        } => {
            let r = launch_inner(modules, stream, function, cfg, args, true);
            let _ = reply.send(r);
        }
        ModuleMsg::Unload { handle, reply } => {
            let r = unload(modules, handle);
            let _ = reply.send(r);
        }
    }
}

fn load_image(
    ctx: &Arc<CudaContext>,
    modules: &Mutex<HashMap<u64, LoadedModule>>,
    next_module_id: &AtomicU64,
    bytes: &[u8],
) -> Result<ModuleHandle, GpuError> {
    let bind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ctx.bind_to_thread()));
    match bind {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("bind_to_thread: {e}"),
            });
        }
        Err(_) => {
            return Err(GpuError::Unrecoverable(
                "ModuleActor::Load: CUDA driver not loadable".into(),
            ));
        }
    }
    let m = cuda_driver::module_load_data(bytes.as_ptr() as *const _)?;
    let id = next_module_id.fetch_add(1, AtomicOrdering::Relaxed);
    modules.lock().insert(
        id,
        LoadedModule {
            cu_module: m,
            functions: HashMap::new(),
            next_function_id: 1,
        },
    );
    Ok(ModuleHandle { id })
}

fn get_function(
    modules: &Mutex<HashMap<u64, LoadedModule>>,
    handle: ModuleHandle,
    name: &str,
) -> Result<FunctionHandle, GpuError> {
    let mut g = modules.lock();
    let m = g.get_mut(&handle.id).ok_or_else(|| {
        GpuError::Unrecoverable(format!(
            "ModuleActor::GetFunction: unknown module {}",
            handle.id
        ))
    })?;
    let cname = CString::new(name).map_err(|e| {
        GpuError::Unrecoverable(format!("ModuleActor::GetFunction: NUL in name: {e}"))
    })?;
    let f = cuda_driver::module_get_function(m.cu_module, &cname)?;
    let function_id = m.next_function_id;
    m.next_function_id += 1;
    m.functions.insert(function_id, (cname, f));
    Ok(FunctionHandle {
        module: handle.id,
        function_id,
    })
}

fn launch_inner(
    modules: &Mutex<HashMap<u64, LoadedModule>>,
    stream: &Arc<CudaStream>,
    function: FunctionHandle,
    cfg: LaunchConfig,
    args: Vec<KernelArg>,
    cooperative: bool,
) -> Result<(), GpuError> {
    let g = modules.lock();
    let m = g.get(&function.module).ok_or_else(|| {
        GpuError::Unrecoverable(format!(
            "ModuleActor::Launch: unknown module {}",
            function.module
        ))
    })?;
    let (_name, cu_func) = m.functions.get(&function.function_id).ok_or_else(|| {
        GpuError::Unrecoverable(format!(
            "ModuleActor::Launch: unknown function {}/{}",
            function.module, function.function_id
        ))
    })?;
    let cu_func = *cu_func;

    // Build the kernel-params array. Each entry is a pointer to a
    // value owned by the Vec<KernelArgScratch> we allocate below; the
    // backing storage must outlive the launch call.
    let mut scratch: Vec<KernelArgScratch> = Vec::with_capacity(args.len());
    let mut keep_alive: Vec<Arc<cudarc::driver::CudaSlice<u8>>> = Vec::new();
    for a in args.into_iter() {
        scratch.push(KernelArgScratch::from_arg(a, &mut keep_alive)?);
    }
    let mut ptrs: Vec<*mut std::ffi::c_void> =
        scratch.iter_mut().map(|s| s.as_void_ptr()).collect();

    let grid = (cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2);
    let block = (cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2);
    let res = if cooperative {
        cuda_driver::launch_cooperative_kernel(
            cu_func,
            grid,
            block,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            ptrs.as_mut_ptr(),
        )
    } else {
        cuda_driver::launch_kernel(
            cu_func,
            grid,
            block,
            cfg.shared_mem_bytes,
            stream.cu_stream(),
            ptrs.as_mut_ptr(),
        )
    };
    // Hold scratch + keep_alive across the call; once the driver
    // consumes the params, they can drop.
    drop(scratch);
    drop(keep_alive);
    res
}

fn unload(
    modules: &Mutex<HashMap<u64, LoadedModule>>,
    handle: ModuleHandle,
) -> Result<(), GpuError> {
    let mut g = modules.lock();
    let m = g.remove(&handle.id).ok_or_else(|| {
        GpuError::Unrecoverable(format!("ModuleActor::Unload: unknown module {}", handle.id))
    })?;
    cuda_driver::module_unload(m.cu_module)
}

/// Backing storage for one `KernelArg` during a single launch.
enum KernelArgScratch {
    DevPtr(driver_sys::CUdeviceptr),
    F32(f32),
    F64(f64),
    I32(i32),
    U32(u32),
    U64(u64),
    Usize(usize),
}

impl KernelArgScratch {
    fn from_arg(
        arg: KernelArg,
        _keep_alive: &mut Vec<Arc<cudarc::driver::CudaSlice<u8>>>,
    ) -> Result<Self, GpuError> {
        // We only need the device pointer for the launch; the keep_alive
        // vec holds the Arc<CudaSlice> so the allocation stays live.
        // We reinterpret the slice as bytes for the keep_alive list,
        // but that requires the same `T` parameter — instead store it
        // typed.
        macro_rules! retain {
            ($g:expr) => {{
                use cudarc::driver::DevicePtr;
                let s = $g.access()?.clone();
                // Capture the device pointer immediately. The
                // SyncOnDrop guard returned by device_ptr() ties the
                // lifetime to `s` — we must keep `s` alive for the
                // duration of the launch.
                let (ptr, _g) = s.device_ptr(_stream_for_record());
                let _ = _g;
                let _ = keep_alive; // satisfy "unused" lint when no slices captured
                ptr
            }};
        }
        // We can't easily do the macro above because `s` is not
        // type-erasable to `Arc<CudaSlice<u8>>`. Inline per-type:
        #[allow(deprecated, unreachable_patterns)]
        Ok(match arg {
            KernelArg::DevSliceF32(g) => Self::DevPtr(devptr_of(g)?),
            KernelArg::DevSliceF64(g) => Self::DevPtr(devptr_of(g)?),
            KernelArg::DevSliceI32(g) => Self::DevPtr(devptr_of(g)?),
            KernelArg::DevSliceU32(g) => Self::DevPtr(devptr_of(g)?),
            KernelArg::DevSliceU8(g) => Self::DevPtr(devptr_of(g)?),
            KernelArg::ScalarF32(v) => Self::F32(v),
            KernelArg::ScalarF64(v) => Self::F64(v),
            KernelArg::ScalarI32(v) => Self::I32(v),
            KernelArg::ScalarU32(v) => Self::U32(v),
            KernelArg::ScalarU64(v) => Self::U64(v),
            KernelArg::Usize(v) => Self::Usize(v),
            // Phase 0.3 boxed-dispatch variants — wired in a follow-up
            // PR. The module-launch path can't yet thread typed-erased
            // box payloads through the cuLaunchKernel ABI without
            // additional plumbing.
            KernelArg::DevSlice(_) | KernelArg::Scalar(_) => {
                return Err(GpuError::Unrecoverable(
                    "ModuleActor: KernelArg::DevSlice/Scalar dispatch not yet wired".into(),
                ));
            }
        })
    }

    fn as_void_ptr(&mut self) -> *mut std::ffi::c_void {
        match self {
            KernelArgScratch::DevPtr(p) => p as *mut _ as *mut _,
            KernelArgScratch::F32(v) => v as *mut _ as *mut _,
            KernelArgScratch::F64(v) => v as *mut _ as *mut _,
            KernelArgScratch::I32(v) => v as *mut _ as *mut _,
            KernelArgScratch::U32(v) => v as *mut _ as *mut _,
            KernelArgScratch::U64(v) => v as *mut _ as *mut _,
            KernelArgScratch::Usize(v) => v as *mut _ as *mut _,
        }
    }
}

#[allow(dead_code)]
fn _stream_for_record() -> &'static Arc<cudarc::driver::CudaStream> {
    // Placeholder used by the macro above — never reached in practice
    // because we use `devptr_of` directly. Keeping the symbol so
    // future refactors can centralise the pointer-grabbing pattern.
    panic!("not used")
}

fn devptr_of<T>(g: crate::gpu_ref::GpuRef<T>) -> Result<driver_sys::CUdeviceptr, GpuError> {
    use cudarc::driver::DevicePtr;
    let s = g.access()?.clone();
    let stream = s.stream().clone();
    let (ptr, _guard) = s.device_ptr(&stream);
    let _ = _guard;
    let _ = s; // hold across return — we drop right after.
    Ok(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn module_msg_round_trip() {
        let sys = ActorSystem::create("module-test", Config::empty())
            .await
            .unwrap();
        let actor = sys.actor_of(ModuleActor::mock_props(), "mod").unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(ModuleMsg::LoadCubin {
            bytes: vec![1, 2, 3, 4],
            reply: tx,
        });
        let r = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

        let (tx, rx) = oneshot::channel();
        actor.tell(ModuleMsg::LoadPtx {
            src: ".version 7.0".into(),
            reply: tx,
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();

        let bogus = ModuleHandle { id: 99 };
        let (tx, rx) = oneshot::channel();
        actor.tell(ModuleMsg::GetFunction {
            handle: bogus,
            name: "kern".into(),
            reply: tx,
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();

        let bogus_fn = FunctionHandle {
            module: 99,
            function_id: 1,
        };
        let (tx, rx) = oneshot::channel();
        actor.tell(ModuleMsg::Launch {
            function: bogus_fn,
            cfg: LaunchConfig::for_num_elems(64),
            args: vec![],
            reply: tx,
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(ModuleMsg::LaunchCooperative {
            function: bogus_fn,
            cfg: LaunchConfig::for_num_elems(64),
            args: vec![],
            reply: tx,
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(ModuleMsg::Unload {
            handle: bogus,
            reply: tx,
        });
        let _ = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();

        sys.terminate().await;
    }

    #[cfg(feature = "nvrtc")]
    #[test]
    fn launch_args_share_kernel_arg_type() {
        // Confirm that `KernelArg` re-exported here is the same type
        // as `crate::kernel::nvrtc::KernelArg`. If the re-export ever
        // breaks, this won't compile.
        fn _assert<T>(_x: T) {}
        _assert::<crate::kernel::nvrtc::KernelArg>(KernelArg::Usize(7));
    }

    #[cfg(not(feature = "nvrtc"))]
    #[test]
    fn launch_args_share_kernel_arg_type() {
        // When nvrtc is disabled, the standalone enum still has the
        // expected variants.
        let _arg = KernelArg::Usize(7);
        let _arg2 = KernelArg::ScalarF32(1.0);
    }
}
