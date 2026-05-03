//! `ContextActor` — the inner tier of the §5.11 supervision tree.
//!
//! Owns the `Arc<CudaContext>`. On `Init`, this actor:
//!
//! 1. Builds (or rebuilds) the `CudaContext` for `device_id`.
//! 2. Bumps `DeviceState.generation` and installs the new context, so
//!    any surviving `GpuRef<T>` from a previous incarnation will fail
//!    validation (§5.8).
//! 3. Constructs a per-actor `CudaStream` (one per kernel actor) via
//!    [`PerActorAllocator`] and spawns the configured library
//!    children.
//! 4. Notifies the parent `DeviceActor` via
//!    [`DeviceMsg::ContextReady { children }`].

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::completion::{CompletionStrategy, HostFnCompletion};
use crate::error::{device_supervisor_strategy, GpuError, CONTEXT_POISONED_TAG};
use crate::gpu_ref::GpuRef;
use crate::kernel::{envelope, BlasActor};
use crate::stream::{PerActorAllocator, StreamAllocator};

use super::alloc_msg::HostBuf;
use super::device_actor::{DeviceConfig, DeviceMsg, EnabledLibraries, KernelChildren};
use super::state::DeviceState;

pub enum ContextMsg {
    /// Self-sent from `pre_start` and `post_restart`. Builds (or
    /// rebuilds) the `CudaContext` and spawns library children.
    Init,

    // --- Per-dtype allocations (forwarded from DeviceMsg::Allocate*).
    AllocateF32 { len: usize, reply: oneshot::Sender<Result<GpuRef<f32>, GpuError>> },
    AllocateF64 { len: usize, reply: oneshot::Sender<Result<GpuRef<f64>, GpuError>> },
    AllocateI8  { len: usize, reply: oneshot::Sender<Result<GpuRef<i8>,  GpuError>> },
    AllocateI32 { len: usize, reply: oneshot::Sender<Result<GpuRef<i32>, GpuError>> },
    AllocateI64 { len: usize, reply: oneshot::Sender<Result<GpuRef<i64>, GpuError>> },
    AllocateU8  { len: usize, reply: oneshot::Sender<Result<GpuRef<u8>,  GpuError>> },
    AllocateU32 { len: usize, reply: oneshot::Sender<Result<GpuRef<u32>, GpuError>> },
    AllocateU64 { len: usize, reply: oneshot::Sender<Result<GpuRef<u64>, GpuError>> },
    #[cfg(feature = "f16")]
    AllocateF16  { len: usize, reply: oneshot::Sender<Result<GpuRef<half::f16>, GpuError>> },
    #[cfg(feature = "f16")]
    AllocateBf16 { len: usize, reply: oneshot::Sender<Result<GpuRef<half::bf16>, GpuError>> },

    // --- Memcpy variants. The `dst` round-trips back via the reply
    // so a pinned buffer can return to its pool.
    CopyToHostF32   { src: GpuRef<f32>, dst: HostBuf<f32>, reply: oneshot::Sender<Result<HostBuf<f32>, GpuError>> },
    CopyFromHostF32 { src: HostBuf<f32>, dst: GpuRef<f32>, reply: oneshot::Sender<Result<HostBuf<f32>, GpuError>> },
    CopyToHostF64   { src: GpuRef<f64>, dst: HostBuf<f64>, reply: oneshot::Sender<Result<HostBuf<f64>, GpuError>> },
    CopyFromHostF64 { src: HostBuf<f64>, dst: GpuRef<f64>, reply: oneshot::Sender<Result<HostBuf<f64>, GpuError>> },
    CopyToHostI32   { src: GpuRef<i32>, dst: HostBuf<i32>, reply: oneshot::Sender<Result<HostBuf<i32>, GpuError>> },
    CopyFromHostI32 { src: HostBuf<i32>, dst: GpuRef<i32>, reply: oneshot::Sender<Result<HostBuf<i32>, GpuError>> },
    CopyToHostU32   { src: GpuRef<u32>, dst: HostBuf<u32>, reply: oneshot::Sender<Result<HostBuf<u32>, GpuError>> },
    CopyFromHostU32 { src: HostBuf<u32>, dst: GpuRef<u32>, reply: oneshot::Sender<Result<HostBuf<u32>, GpuError>> },
    CopyToHostU8    { src: GpuRef<u8>,  dst: HostBuf<u8>,  reply: oneshot::Sender<Result<HostBuf<u8>,  GpuError>> },
    CopyFromHostU8  { src: HostBuf<u8>, dst: GpuRef<u8>,   reply: oneshot::Sender<Result<HostBuf<u8>,  GpuError>> },
}

pub struct ContextActor {
    state: Arc<DeviceState>,
    config: DeviceConfig,
    parent: ActorRef<DeviceMsg>,
    /// Primary CUDA stream owned by ContextActor for its own
    /// allocation work; library children get fresh streams via the
    /// allocator.
    stream: Option<Arc<cudarc::driver::CudaStream>>,
    /// Allocator handing out fresh streams to each kernel-actor
    /// child. None until Init succeeds.
    allocator: Option<Arc<dyn StreamAllocator>>,
    /// Default completion strategy injected into every kernel actor.
    completion: Arc<dyn CompletionStrategy>,
    children: Option<KernelChildren>,
}

impl ContextActor {
    pub fn props(
        state: Arc<DeviceState>,
        config: DeviceConfig,
        parent: ActorRef<DeviceMsg>,
    ) -> Props<Self> {
        let s = state.clone();
        let c = config.clone();
        let p = parent.clone();
        let completion: Arc<dyn CompletionStrategy> = Arc::new(HostFnCompletion::new());
        Props::create(move || ContextActor {
            state: s.clone(),
            config: c.clone(),
            parent: p.clone(),
            stream: None,
            allocator: None,
            completion: completion.clone(),
            children: None,
        })
        .with_supervisor_strategy(device_supervisor_strategy())
    }

    /// Bring up the CUDA context, install it in shared state, spawn
    /// the configured library children, and notify the parent.
    async fn run_init(&mut self, ctx: &mut Context<Self>) {
        let device_id = self.config.device_id;

        if self.config.mock_mode {
            self.state.bump_generation();
            let stub = ctx
                .spawn::<BlasActor>(BlasActor::mock_props(), "blas")
                .unwrap_or_else(|e| panic!("Unrecoverable: spawn mock BlasActor: {e}"));
            let children = KernelChildren {
                blas: stub,
                #[cfg(feature = "cudnn")]
                cudnn: None,
                #[cfg(feature = "cufft")]
                fft: None,
                #[cfg(feature = "curand")]
                rng: None,
            };
            self.children = Some(children.clone());
            self.parent.tell(DeviceMsg::ContextReady { children });
            info!(device_id, "ContextActor (mock) ready");
            return;
        }

        let cuda_ctx = match cudarc::driver::CudaContext::new(device_id as usize) {
            Ok(c) => c,
            Err(e) => {
                panic!("{CONTEXT_POISONED_TAG}: CudaContext::new({device_id}) failed: {e}");
            }
        };
        let stream = match cuda_ctx.new_stream() {
            Ok(s) => s,
            Err(e) => {
                panic!("{CONTEXT_POISONED_TAG}: new_stream failed on device {device_id}: {e}");
            }
        };

        self.state.bump_generation();
        self.state.install_context(cuda_ctx.clone());
        self.stream = Some(stream.clone());

        // Fresh-stream allocator: each kernel-actor child gets its own
        // stream for max kernel concurrency.
        let allocator: Arc<dyn StreamAllocator> =
            Arc::new(PerActorAllocator::with_context(cuda_ctx.clone()));
        self.allocator = Some(allocator.clone());

        let libs = self.config.enabled_libraries;

        // BlasActor is always spawned (BLAS is the F1 default).
        let blas_stream = if libs.contains(EnabledLibraries::BLAS) {
            allocator.acquire(Default::default())
        } else {
            stream.clone()
        };
        let blas_props = BlasActor::props(
            blas_stream.clone(),
            allocator.clone(),
            self.completion.clone(),
            self.state.clone(),
        );
        // Acquire returned a fresh stream not equal to `blas_stream`
        // because PerActorAllocator with_context mints fresh on every
        // call. We pass blas_stream for now; the BlasActor::props
        // debug_assert checks ptr_eq with what allocator.acquire
        // returns inside the closure, which mints again. To keep that
        // assert satisfied, use SingleStreamAllocator-style wrapper.
        let _ = blas_props; // construct-only check

        // Cleaner: always use the legacy props (single-stream binding)
        // for BlasActor in this phase since BlasActor itself enforces
        // ptr_eq. Future phases that fork can drop the assert.
        let blas_alloc = crate::stream::PerActorAllocator::new(blas_stream.clone());
        let blas_props = BlasActor::props_legacy(
            blas_stream.clone(),
            blas_alloc,
            HostFnCompletion::new(),
            self.state.clone(),
        );
        let blas_ref = ctx
            .spawn::<BlasActor>(blas_props, "blas")
            .unwrap_or_else(|e| panic!("Unrecoverable: spawn BlasActor: {e}"));

        #[cfg(feature = "cudnn")]
        let cudnn_ref = if libs.contains(EnabledLibraries::CUDNN) {
            let s = allocator.acquire(Default::default());
            let props = crate::kernel::CudnnActor::props(
                s,
                allocator.clone(),
                self.completion.clone(),
                self.state.clone(),
            );
            Some(
                ctx.spawn::<crate::kernel::CudnnActor>(props, "cudnn")
                    .unwrap_or_else(|e| panic!("Unrecoverable: spawn CudnnActor: {e}")),
            )
        } else {
            None
        };

        #[cfg(feature = "cufft")]
        let fft_ref = if libs.contains(EnabledLibraries::CUFFT) {
            let s = allocator.acquire(Default::default());
            let props = crate::kernel::FftActor::props(
                s,
                allocator.clone(),
                self.completion.clone(),
                self.state.clone(),
                cuda_ctx.clone(),
            );
            Some(
                ctx.spawn::<crate::kernel::FftActor>(props, "fft")
                    .unwrap_or_else(|e| panic!("Unrecoverable: spawn FftActor: {e}")),
            )
        } else {
            None
        };

        #[cfg(feature = "curand")]
        let rng_ref = if libs.contains(EnabledLibraries::CURAND) {
            let s = allocator.acquire(Default::default());
            let props = crate::kernel::RngActor::props(
                s,
                allocator.clone(),
                self.completion.clone(),
                self.state.clone(),
                /* seed */ 0,
            );
            Some(
                ctx.spawn::<crate::kernel::RngActor>(props, "rng")
                    .unwrap_or_else(|e| panic!("Unrecoverable: spawn RngActor: {e}")),
            )
        } else {
            None
        };

        let children = KernelChildren {
            blas: blas_ref,
            #[cfg(feature = "cudnn")]
            cudnn: cudnn_ref,
            #[cfg(feature = "cufft")]
            fft: fft_ref,
            #[cfg(feature = "curand")]
            rng: rng_ref,
        };
        self.children = Some(children.clone());
        self.parent.tell(DeviceMsg::ContextReady { children });
        info!(device_id, generation = self.state.generation(), "ContextActor ready");
    }

    /// Allocate a typed buffer on the actor's stream. Bound to the
    /// `Allocate*` ContextMsg variants via the macro below.
    fn alloc<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<GpuRef<T>, GpuError> {
        if self.config.mock_mode {
            return Err(GpuError::Unrecoverable("alloc not supported in mock mode".into()));
        }
        let Some(stream) = self.stream.clone() else {
            return Err(GpuError::GpuRefStale("context not ready"));
        };
        match stream.alloc_zeros::<T>(len) {
            Ok(slice) => Ok(GpuRef::<T>::new(Arc::new(slice), &self.state)),
            Err(e) => Err(GpuError::OutOfMemory(format!("alloc {len}: {e}"))),
        }
    }
}

/// Helper: do an async D2H copy via cudarc's `memcpy_dtoh` and
/// schedule completion via the shared envelope.
fn run_copy_to_host<T: DeviceRepr + 'static>(
    src: GpuRef<T>,
    mut dst: HostBuf<T>,
    stream: Arc<cudarc::driver::CudaStream>,
    completion: Arc<dyn CompletionStrategy>,
    reply: oneshot::Sender<Result<HostBuf<T>, GpuError>>,
) where
    T: Send,
{
    let src_slice = match src.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    if src_slice.len() != dst.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
            "memcpy len mismatch: src={} dst={}",
            src_slice.len(),
            dst.len()
        ))));
        return;
    }

    // Synchronous-enqueue path: call cudarc's memcpy_dtoh which
    // dispatches an async copy on the stream.
    let res = match &mut dst {
        HostBuf::Owned(v) => stream.memcpy_dtoh(&*src_slice, v.as_mut_slice()),
        HostBuf::Pinned(p) => stream.memcpy_dtoh(&*src_slice, p.as_mut_slice()),
    };
    if let Err(e) = res {
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: "driver",
            msg: format!("memcpy_dtoh: {e}"),
        }));
        return;
    }

    // Spawn completion-await; on success return dst back to caller.
    envelope::run_kernel(
        "driver",
        &stream,
        &completion,
        dst,
        reply,
        move || Ok::<_, GpuError>((src_slice,)),
    );
}

fn run_copy_from_host<T: DeviceRepr + 'static>(
    src: HostBuf<T>,
    dst: GpuRef<T>,
    stream: Arc<cudarc::driver::CudaStream>,
    completion: Arc<dyn CompletionStrategy>,
    reply: oneshot::Sender<Result<HostBuf<T>, GpuError>>,
) where
    T: Send,
{
    let dst_slice = match dst.access() {
        Ok(s) => s.clone(),
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    if dst_slice.len() != src.len() {
        let _ = reply.send(Err(GpuError::Unrecoverable(format!(
            "memcpy len mismatch: src={} dst={}",
            src.len(),
            dst_slice.len()
        ))));
        return;
    }
    let mut dst_owned = match Arc::try_unwrap(dst_slice) {
        Ok(s) => s,
        Err(_) => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "H2D destination has multiple live references".into(),
            )));
            return;
        }
    };
    let res = match &src {
        HostBuf::Owned(v) => stream.memcpy_htod(v.as_slice(), &mut dst_owned),
        HostBuf::Pinned(p) => stream.memcpy_htod(p.as_slice(), &mut dst_owned),
    };
    if let Err(e) = res {
        let _ = reply.send(Err(GpuError::LibraryError {
            lib: "driver",
            msg: format!("memcpy_htod: {e}"),
        }));
        return;
    }
    dst.record_write(&stream);
    envelope::run_kernel(
        "driver",
        &stream,
        &completion,
        src,
        reply,
        move || Ok::<_, GpuError>((dst_owned,)),
    );
}

#[async_trait]
impl Actor for ContextActor {
    type Msg = ContextMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        ctx.self_ref().tell(ContextMsg::Init);
    }

    async fn handle(&mut self, ctx: &mut Context<Self>, msg: ContextMsg) {
        match msg {
            ContextMsg::Init => self.run_init(ctx).await,

            ContextMsg::AllocateF32 { len, reply } => { let _ = reply.send(self.alloc::<f32>(len)); }
            ContextMsg::AllocateF64 { len, reply } => { let _ = reply.send(self.alloc::<f64>(len)); }
            ContextMsg::AllocateI8  { len, reply } => { let _ = reply.send(self.alloc::<i8>(len)); }
            ContextMsg::AllocateI32 { len, reply } => { let _ = reply.send(self.alloc::<i32>(len)); }
            ContextMsg::AllocateI64 { len, reply } => { let _ = reply.send(self.alloc::<i64>(len)); }
            ContextMsg::AllocateU8  { len, reply } => { let _ = reply.send(self.alloc::<u8>(len)); }
            ContextMsg::AllocateU32 { len, reply } => { let _ = reply.send(self.alloc::<u32>(len)); }
            ContextMsg::AllocateU64 { len, reply } => { let _ = reply.send(self.alloc::<u64>(len)); }
            #[cfg(feature = "f16")]
            ContextMsg::AllocateF16  { len, reply } => { let _ = reply.send(self.alloc::<half::f16>(len)); }
            #[cfg(feature = "f16")]
            ContextMsg::AllocateBf16 { len, reply } => { let _ = reply.send(self.alloc::<half::bf16>(len)); }

            ContextMsg::CopyToHostF32 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_to_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyFromHostF32 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_from_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyToHostF64 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_to_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyFromHostF64 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_from_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyToHostI32 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_to_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyFromHostI32 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_from_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyToHostU32 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_to_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyFromHostU32 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_from_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyToHostU8 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_to_host(src, dst, stream, self.completion.clone(), reply);
            }
            ContextMsg::CopyFromHostU8 { src, dst, reply } => {
                let stream = self.stream.clone().expect("ctx not ready");
                run_copy_from_host(src, dst, stream, self.completion.clone(), reply);
            }
        }
    }

    async fn post_restart(&mut self, ctx: &mut Context<Self>, err: &str) {
        warn!(device_id = self.config.device_id, %err, "ContextActor post_restart");
        self.parent.tell(DeviceMsg::ContextLost);
        ctx.self_ref().tell(ContextMsg::Init);
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        debug!(device_id = self.config.device_id, "ContextActor post_stop");
        self.stream = None;
        self.allocator = None;
        self.children = None;
        self.state.clear_context();
        self.parent.tell(DeviceMsg::ContextLost);
    }
}
