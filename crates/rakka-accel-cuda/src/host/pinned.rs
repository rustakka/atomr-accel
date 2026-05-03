//! `PinnedBufferPool` actor + `PinnedBuf<T>` handle.
//!
//! Lifecycle:
//! 1. Construct the pool actor with [`PinnedBufferPool::props`].
//! 2. Send [`PinnedPoolMsg::Acquire { len_bytes, reply }`] — actor
//!    pops from the free-list (or grows up to `max_buffers`) and
//!    replies with a [`PinnedBuf<T>`] handle whose Drop sends an
//!    `InternalReturn` back to the pool.
//! 3. Use the buffer: `as_mut_slice()` for fill, then move into a
//!    `DeviceMsg::CopyToDeviceT` / `CopyToHostT` call.
//!
//! cudarc 0.19 only exposes raw [`cudarc::driver::result::malloc_host`]
//! and [`cudarc::driver::result::free_host`]; no safe wrapper. We
//! build [`PinnedSlot`] around the raw pointer + capacity + Drop and
//! treat it as the moral equivalent of `Box<[u8]>` over pinned
//! memory.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::marker::PhantomData;

use async_trait::async_trait;
use rakka_core::actor::{Actor, ActorRef, Context, Props};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, warn};

use crate::error::GpuError;

#[derive(Debug, Clone, Copy)]
pub struct PinnedBufferPoolConfig {
    pub initial_buffers: usize,
    pub max_buffers: usize,
    pub buffer_capacity_bytes: usize,
    /// If true, requests larger than `buffer_capacity_bytes` get a
    /// one-shot oversize allocation that's freed (not pooled) on
    /// release. If false, oversize requests fail.
    pub allow_oversize: bool,
}

impl Default for PinnedBufferPoolConfig {
    fn default() -> Self {
        Self {
            initial_buffers: 4,
            max_buffers: 32,
            buffer_capacity_bytes: 4 * 1024 * 1024,
            allow_oversize: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PinnedPoolStats {
    pub in_use: usize,
    pub free: usize,
    pub total: usize,
    pub bytes_allocated: usize,
}

/// Internal slot — owns the raw `cuMemHostAlloc`'d region.
pub struct PinnedSlot {
    ptr: *mut c_void,
    capacity_bytes: usize,
    /// True if this slot was minted as an oversize one-shot; freed
    /// rather than returned to the pool on release.
    oversize: bool,
}

// SAFETY: pinned memory is host RAM. Sending the raw pointer between
// threads is safe; dereferencing requires care which we constrain to
// the actor + the holder of `PinnedBuf<T>`.
unsafe impl Send for PinnedSlot {}
unsafe impl Sync for PinnedSlot {}

impl PinnedSlot {
    fn new(capacity_bytes: usize, oversize: bool) -> Result<Self, GpuError> {
        // `cuMemHostAlloc` flags: 0 = default (portable across devices
        // off, write-combined off, mapped off). Sufficient for
        // standard async memcpy.
        let ptr = unsafe { cudarc::driver::result::malloc_host(capacity_bytes, 0) }
            .map_err(|e| GpuError::OutOfMemory(format!("pinned alloc {capacity_bytes}B: {e}")))?;
        Ok(Self { ptr, capacity_bytes, oversize })
    }

    fn free(self) {
        // Drop the slot; the Drop impl below frees the pinned memory.
        drop(self);
    }
}

impl Drop for PinnedSlot {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cudarc::driver::result::free_host(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// Generation token for pool-level invalidation. Bumped on actor
/// restart so that surviving `PinnedBuf<T>` instances cannot
/// accidentally return into a fresh pool.
type PinnedGeneration = u64;

/// Public messages.
pub enum PinnedPoolMsg {
    Acquire {
        len_bytes: usize,
        reply: oneshot::Sender<Result<PinnedBufHandle, GpuError>>,
    },
    /// Drop-driven return path used by `PinnedBuf::Drop`. Not part of
    /// the public API for callers.
    InternalReturn {
        slot: PinnedSlot,
        generation: PinnedGeneration,
    },
    Stats {
        reply: oneshot::Sender<PinnedPoolStats>,
    },
}

/// Untyped handle returned by `Acquire`. Convert to a typed
/// [`PinnedBuf<T>`] via [`PinnedBufHandle::into_typed`].
pub struct PinnedBufHandle {
    slot: Option<PinnedSlot>,
    generation: PinnedGeneration,
    return_tx: mpsc::UnboundedSender<PinnedPoolMsg>,
}

impl PinnedBufHandle {
    pub fn capacity_bytes(&self) -> usize {
        self.slot.as_ref().map(|s| s.capacity_bytes).unwrap_or(0)
    }

    /// Convert to a typed buffer. `len` is the number of `T`s the
    /// caller intends to use; must satisfy
    /// `len * size_of::<T>() <= capacity_bytes`.
    pub fn into_typed<T>(mut self, len: usize) -> Result<PinnedBuf<T>, GpuError> {
        let needed = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            GpuError::Unrecoverable("pinned buf: len * size_of overflowed".into())
        })?;
        if needed > self.capacity_bytes() {
            return Err(GpuError::Unrecoverable(format!(
                "pinned buf: requested {len} elements ({needed} B) exceeds capacity {} B",
                self.capacity_bytes()
            )));
        }
        let slot = self.slot.take().expect("PinnedBufHandle slot was None");
        let ptr = slot.ptr as *mut T;
        Ok(PinnedBuf {
            inner: Some(PinnedBufInner { slot, len, return_tx: self.return_tx.clone(), generation: self.generation }),
            ptr,
            len,
            _marker: PhantomData,
        })
    }
}

impl Drop for PinnedBufHandle {
    fn drop(&mut self) {
        // If into_typed wasn't called, return the slot ourselves.
        if let Some(slot) = self.slot.take() {
            let _ = self.return_tx.send(PinnedPoolMsg::InternalReturn {
                slot,
                generation: self.generation,
            });
        }
    }
}

/// Typed pinned buffer.
///
/// Send + Sync — the raw pointer is page-locked host memory; the
/// actor + this handle are the only writers, and the pool's
/// generation gate prevents post-restart aliasing.
pub struct PinnedBuf<T> {
    inner: Option<PinnedBufInner>,
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

struct PinnedBufInner {
    slot: PinnedSlot,
    #[allow(dead_code)]
    len: usize,
    return_tx: mpsc::UnboundedSender<PinnedPoolMsg>,
    generation: PinnedGeneration,
}

unsafe impl<T: Send> Send for PinnedBuf<T> {}
unsafe impl<T: Sync> Sync for PinnedBuf<T> {}

impl<T> PinnedBuf<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// View as a host slice. SAFETY: relies on the buffer being
    /// initialized for the requested length. For zero-init you must
    /// fill before reading.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: ptr was returned by cuMemHostAlloc and the slot
        // owns it for at least the lifetime of `self`. len matches
        // the typed view we requested in into_typed.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for PinnedBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuf").field("len", &self.len).finish()
    }
}

impl<T> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let _ = inner.return_tx.send(PinnedPoolMsg::InternalReturn {
                slot: inner.slot,
                generation: inner.generation,
            });
        }
    }
}

/// The pool actor.
pub struct PinnedBufferPool {
    config: PinnedBufferPoolConfig,
    free: VecDeque<PinnedSlot>,
    in_use: usize,
    total_minted: usize,
    bytes_allocated: usize,
    /// Bumped on `pre_restart`. Slots returning with an old
    /// generation are dropped instead of pooled.
    generation: PinnedGeneration,
    /// Sender used by `PinnedBuf::Drop` to post `InternalReturn`. We
    /// keep one mpsc that funnels into `handle()` — the actor's own
    /// mailbox is async-trait `tell`-only, but mpsc gives us
    /// non-blocking sync sends from `Drop`.
    return_tx: mpsc::UnboundedSender<PinnedPoolMsg>,
    return_rx_observer: Option<ActorRef<PinnedPoolMsg>>,
}

impl PinnedBufferPool {
    pub fn props(config: PinnedBufferPoolConfig) -> Props<Self> {
        Props::create(move || {
            let (tx, _rx) = mpsc::unbounded_channel();
            // Initial buffers are minted lazily on first acquire; in
            // F2 we trade a tiny first-acquire latency for simpler
            // post_restart handling.
            PinnedBufferPool {
                config,
                free: VecDeque::new(),
                in_use: 0,
                total_minted: 0,
                bytes_allocated: 0,
                generation: 0,
                return_tx: tx,
                return_rx_observer: None,
            }
        })
    }

    /// Test/diagnostic: snapshot stats without going through the
    /// mailbox.
    pub fn stats(&self) -> PinnedPoolStats {
        PinnedPoolStats {
            in_use: self.in_use,
            free: self.free.len(),
            total: self.total_minted,
            bytes_allocated: self.bytes_allocated,
        }
    }

    fn try_acquire(
        &mut self,
        len_bytes: usize,
    ) -> Result<PinnedBufHandle, GpuError> {
        let cap = self.config.buffer_capacity_bytes;
        let oversize = len_bytes > cap;

        let slot = if oversize {
            if !self.config.allow_oversize {
                return Err(GpuError::OutOfMemory(format!(
                    "pinned pool: oversize request {len_bytes}B exceeds slot capacity {cap}B"
                )));
            }
            // One-shot allocation, freed on release.
            self.bytes_allocated += len_bytes;
            self.total_minted += 1;
            PinnedSlot::new(len_bytes, true)?
        } else if let Some(slot) = self.free.pop_front() {
            slot
        } else {
            if self.total_minted >= self.config.max_buffers {
                return Err(GpuError::OutOfMemory(format!(
                    "pinned pool: max_buffers={} reached",
                    self.config.max_buffers
                )));
            }
            self.bytes_allocated += cap;
            self.total_minted += 1;
            PinnedSlot::new(cap, false)?
        };

        self.in_use += 1;
        Ok(PinnedBufHandle {
            slot: Some(slot),
            generation: self.generation,
            return_tx: self.return_tx.clone(),
        })
    }

    fn return_slot(&mut self, slot: PinnedSlot, generation: PinnedGeneration) {
        if generation != self.generation {
            // Cross-generation return. Drop instead of pool to avoid
            // mixing. The Drop impl on PinnedSlot frees the memory.
            self.bytes_allocated = self.bytes_allocated.saturating_sub(slot.capacity_bytes);
            self.total_minted = self.total_minted.saturating_sub(1);
            slot.free();
            return;
        }
        self.in_use = self.in_use.saturating_sub(1);
        if slot.oversize {
            // Don't pool oversize allocations.
            self.bytes_allocated = self.bytes_allocated.saturating_sub(slot.capacity_bytes);
            self.total_minted = self.total_minted.saturating_sub(1);
            slot.free();
        } else {
            self.free.push_back(slot);
        }
    }
}

#[async_trait]
impl Actor for PinnedBufferPool {
    type Msg = PinnedPoolMsg;

    async fn pre_start(&mut self, ctx: &mut Context<Self>) {
        // Wire the mpsc → actor pump so Drop-driven returns funnel
        // into `handle()`. We replace the sender with a fresh one and
        // start a forwarder task that bridges the receiver into the
        // actor's mailbox.
        let (tx, mut rx) = mpsc::unbounded_channel::<PinnedPoolMsg>();
        self.return_tx = tx;
        let self_ref = ctx.self_ref().clone();
        self.return_rx_observer = Some(self_ref.clone());
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                self_ref.tell(msg);
            }
        });
        debug!(
            initial = self.config.initial_buffers,
            max = self.config.max_buffers,
            cap = self.config.buffer_capacity_bytes,
            "PinnedBufferPool started"
        );
    }

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: PinnedPoolMsg) {
        match msg {
            PinnedPoolMsg::Acquire { len_bytes, reply } => {
                let r = self.try_acquire(len_bytes);
                let _ = reply.send(r);
            }
            PinnedPoolMsg::InternalReturn { slot, generation } => {
                self.return_slot(slot, generation);
            }
            PinnedPoolMsg::Stats { reply } => {
                let _ = reply.send(self.stats());
            }
        }
    }

    async fn pre_restart(&mut self, _ctx: &mut Context<Self>, err: &str) {
        warn!(%err, "PinnedBufferPool pre_restart — dropping all in-flight buffers");
        // Drop the free-list. Slots that haven't returned yet will
        // arrive with the pre-restart generation and be dropped (not
        // pooled) by `return_slot`.
        self.free.clear();
        self.generation += 1;
        self.in_use = 0;
        self.total_minted = 0;
        self.bytes_allocated = 0;
    }

    async fn post_stop(&mut self, _ctx: &mut Context<Self>) {
        debug!("PinnedBufferPool post_stop");
        // Slots in `self.free` drop here, freeing pinned memory via
        // PinnedSlot::Drop.
    }
}
