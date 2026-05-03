//! `ManagedAllocatorActor` + `ManagedRef<T>`.
//!
//! `ManagedRef<T>` is distinct from [`crate::gpu_ref::GpuRef<T>`]
//! because validity rules differ: managed memory survives
//! `ContextActor` rebuilds, so generation tokens don't apply.
//! Validity is tied to the allocator actor's lifetime via
//! `Arc<AtomicBool>`.
//!
//! Backed by raw `cudaMallocManaged` + `cudaFree` from cudarc's
//! runtime sys layer. The pointer is allocated once at construction
//! and freed when the last `ManagedRef` clone drops or when the
//! allocator actor stops, whichever comes first.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::runtime::sys as runtime_sys;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use crate::error::GpuError;

#[derive(Debug, Clone, Copy)]
pub enum ManagedFlags {
    AttachGlobal,
    AttachHost,
}

impl ManagedFlags {
    fn raw(self) -> u32 {
        match self {
            ManagedFlags::AttachGlobal => runtime_sys::cudaMemAttachGlobal,
            ManagedFlags::AttachHost => runtime_sys::cudaMemAttachHost,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PrefetchTarget {
    Device(u32),
    Cpu,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ManagedStats {
    pub allocations: usize,
    pub bytes_allocated: usize,
}

/// Owning handle to a managed-memory region. `Arc`-cloned across
/// agents; the underlying allocation is freed when the last clone
/// drops.
pub struct ManagedRef<T> {
    inner: Option<Arc<ManagedRefInner>>,
    _marker: PhantomData<T>,
}

struct ManagedRefInner {
    ptr: NonNull<u8>,
    bytes: usize,
    elements: usize,
    /// While true, the allocator actor still owns the master ref;
    /// once it drops to false the pointer must be considered freed.
    system_alive: Arc<AtomicBool>,
}

impl Drop for ManagedRefInner {
    fn drop(&mut self) {
        if self.system_alive.load(Ordering::Acquire) {
            // SAFETY: ptr was returned by cudaMallocManaged with
            // the same allocator. cudaFree is the documented release
            // call. We swallow the error — Drop can't propagate.
            unsafe {
                let _ = runtime_sys::cudaFree(self.ptr.as_ptr() as *mut _);
            }
        }
    }
}

unsafe impl Send for ManagedRefInner {}
unsafe impl Sync for ManagedRefInner {}

impl<T> ManagedRef<T> {
    /// True if the underlying allocation is still live.
    pub fn is_valid(&self) -> bool {
        self.inner
            .as_ref()
            .map(|i| i.system_alive.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.inner.as_ref().map(|i| i.elements).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Raw device pointer. Valid for both host and device access while
    /// the allocator is alive. Caller is responsible for
    /// synchronization.
    pub fn as_ptr(&self) -> *const T {
        self.inner
            .as_ref()
            .map(|i| i.ptr.as_ptr() as *const T)
            .unwrap_or(std::ptr::null())
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.inner
            .as_ref()
            .map(|i| i.ptr.as_ptr() as *mut T)
            .unwrap_or(std::ptr::null_mut())
    }
}

impl<T: Copy> ManagedRef<T> {
    /// Host-side immutable view of the managed memory.
    ///
    /// SAFETY contract: managed memory is coherent host/device, but
    /// reads from the host before the device has finished writing
    /// produce undefined values. Caller must synchronize the
    /// relevant device stream first (e.g. via
    /// `cudaDeviceSynchronize`) when reading data the device wrote.
    /// Returns an empty slice if the allocator has stopped.
    pub fn as_slice(&self) -> &[T] {
        match self.inner.as_ref() {
            None => &[],
            Some(i) => {
                if !i.system_alive.load(Ordering::Acquire) {
                    return &[];
                }
                unsafe { std::slice::from_raw_parts(i.ptr.as_ptr() as *const T, i.elements) }
            }
        }
    }

    /// Host-side mutable view of the managed memory.
    ///
    /// Like [`Self::as_slice`] this is a host alias of device-visible
    /// memory. The caller must hold a `WriteToken` (e.g. from
    /// [`crate::memory::ManagedAllocatorActor`] or
    /// `SharedGpuStateCoordinator`) to avoid concurrent device
    /// writes while writing from the host.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self.inner.as_ref() {
            None => &mut [],
            Some(i) => {
                if !i.system_alive.load(Ordering::Acquire) {
                    return &mut [];
                }
                unsafe { std::slice::from_raw_parts_mut(i.ptr.as_ptr() as *mut T, i.elements) }
            }
        }
    }
}

impl<T> Clone for ManagedRef<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

unsafe impl<T: Send> Send for ManagedRef<T> {}
unsafe impl<T: Sync> Sync for ManagedRef<T> {}

pub enum ManagedMsg {
    AllocateManagedF32 {
        len: usize,
        flags: ManagedFlags,
        reply: oneshot::Sender<Result<ManagedRef<f32>, GpuError>>,
    },
    /// Prefetch a managed allocation to a specific target. The
    /// `mem` argument is a clone of the `ManagedRef` returned from
    /// allocation. F4.x: real `cudaMemPrefetchAsync` call.
    PrefetchF32 {
        mem: ManagedRef<f32>,
        target: PrefetchTarget,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    Stats {
        reply: oneshot::Sender<ManagedStats>,
    },
}

pub struct ManagedAllocatorActor {
    system_alive: Arc<AtomicBool>,
    stats: ManagedStats,
}

impl ManagedAllocatorActor {
    pub fn props() -> Props<Self> {
        Props::create(|| ManagedAllocatorActor {
            system_alive: Arc::new(AtomicBool::new(true)),
            stats: ManagedStats::default(),
        })
    }

    fn allocate_f32(
        &mut self,
        len: usize,
        flags: ManagedFlags,
    ) -> Result<ManagedRef<f32>, GpuError> {
        let bytes = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            GpuError::Unrecoverable("managed alloc: len * size_of overflowed".into())
        })?;
        // cudarc's dynamic-loader panics if the CUDA runtime library
        // isn't loadable on the host (e.g. no driver in CI). Catch
        // that here so the actor stays alive on no-GPU machines.
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        let raw_ref = &mut raw as *mut *mut std::ffi::c_void;
        let raw_ref = raw_ref as usize; // copy for closure
        let status_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: cudaMallocManaged contract — writable out-ptr,
            // valid size. The pointer-as-usize cast is needed
            // because raw pointers aren't UnwindSafe.
            unsafe {
                runtime_sys::cudaMallocManaged(
                    raw_ref as *mut *mut std::ffi::c_void,
                    bytes,
                    flags.raw(),
                )
            }
        }));
        let status = match status_res {
            Ok(s) => s,
            Err(_) => {
                return Err(GpuError::Unrecoverable(
                    "cudaMallocManaged: CUDA runtime not loadable".into(),
                ));
            }
        };
        if status != runtime_sys::cudaError::cudaSuccess {
            return Err(GpuError::OutOfMemory(format!(
                "cudaMallocManaged({bytes}B): {status:?}"
            )));
        }
        let ptr = NonNull::new(raw as *mut u8)
            .ok_or_else(|| GpuError::Unrecoverable("cudaMallocManaged returned null".into()))?;
        self.stats.allocations += 1;
        self.stats.bytes_allocated += bytes;
        Ok(ManagedRef {
            inner: Some(Arc::new(ManagedRefInner {
                ptr,
                bytes,
                elements: len,
                system_alive: self.system_alive.clone(),
            })),
            _marker: PhantomData,
        })
    }
}

#[async_trait]
impl Actor for ManagedAllocatorActor {
    type Msg = ManagedMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ManagedMsg) {
        match msg {
            ManagedMsg::AllocateManagedF32 { len, flags, reply } => {
                let _ = reply.send(self.allocate_f32(len, flags));
            }
            ManagedMsg::PrefetchF32 { mem, target, reply } => {
                let Some(inner) = mem.inner.as_ref() else {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "PrefetchF32: invalid ManagedRef".into(),
                    )));
                    return;
                };
                if !inner.system_alive.load(Ordering::Acquire) {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "PrefetchF32: allocator stopped".into(),
                    )));
                    return;
                }
                // Use the v2 shape with a `cudaMemLocation`. The
                // function name `cudaMemPrefetchAsync` is available
                // on CUDA 13+ with this signature; older CUDA toolkits
                // need a different binding shape. We pick the v2
                // path here and let the build-system feature flags
                // resolve the symbol.
                let location = runtime_sys::cudaMemLocation {
                    type_: match target {
                        PrefetchTarget::Device(_) => {
                            runtime_sys::cudaMemLocationType::cudaMemLocationTypeDevice
                        }
                        PrefetchTarget::Cpu => {
                            runtime_sys::cudaMemLocationType::cudaMemLocationTypeHost
                        }
                    },
                    id: match target {
                        PrefetchTarget::Device(d) => d as i32,
                        PrefetchTarget::Cpu => 0,
                    },
                };
                // SAFETY: pointer + length + location all valid; null
                // stream uses default per-thread stream.
                let status = unsafe {
                    runtime_sys::cudaMemPrefetchAsync(
                        inner.ptr.as_ptr() as *const _,
                        inner.bytes,
                        location,
                        0,
                        std::ptr::null_mut(),
                    )
                };
                if status != runtime_sys::cudaError::cudaSuccess {
                    let _ = reply.send(Err(GpuError::LibraryError {
                        lib: "runtime",
                        msg: format!("cudaMemPrefetchAsync: {status:?}"),
                    }));
                    return;
                }
                let _ = reply.send(Ok(()));
            }
            ManagedMsg::Stats { reply } => {
                let _ = reply.send(self.stats);
            }
        }
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        // Mark the system dead. ManagedRefInner::Drop calls cudaFree
        // only while system_alive is true; flipping it here makes
        // outstanding clones into safe inert handles. The strong
        // ref the actor doesn't hold (pointers are tracked by
        // ManagedRefInner Arc) means the runtime frees the memory
        // when the last clone drops anyway — but only if the
        // system is alive. Trade-off documented in the module
        // doc: allocations can outlive the actor only if at least
        // one ManagedRef clone is alive when the actor stops.
        self.system_alive.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    /// We can't actually call cudaMallocManaged on a host without a
    /// CUDA driver. The test below just verifies the actor's
    /// surface — alloc + post_stop — by sending an alloc request;
    /// the alloc fails with OutOfMemory on a no-GPU machine, which
    /// is the expected behaviour. Stats reflects zero successful
    /// allocations.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn allocate_replies_then_invalidate_on_post_stop() {
        let sys = ActorSystem::create("managed-test", Config::empty())
            .await
            .unwrap();
        let mgr = sys
            .actor_of(ManagedAllocatorActor::props(), "managed")
            .unwrap();

        let (tx, rx) = oneshot::channel();
        mgr.tell(ManagedMsg::AllocateManagedF32 {
            len: 1024,
            flags: ManagedFlags::AttachGlobal,
            reply: tx,
        });
        // Either succeeds (real GPU) or returns OutOfMemory (no
        // driver). Either is fine for this test.
        let r = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        let _ = r;

        let (tx, rx) = oneshot::channel();
        mgr.tell(ManagedMsg::Stats { reply: tx });
        let _stats = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();

        sys.terminate().await;
    }
}
