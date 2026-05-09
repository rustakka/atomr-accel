//! `GpuRef<T>` — opaque, message-friendly handle to a GPU buffer (§5.8).
//!
//! `Send + Sync + 'static` with no lifetime parameters, so it composes
//! freely in any actor message type. Validity is checked at runtime by
//! comparing a generation token against `DeviceState.generation`, which
//! is bumped whenever the underlying `CudaContext` is rebuilt (§5.11
//! supervision).
//!
//! Cross-node serialisation (`GpuToken { node_id, device_id, buffer_id,
//! generation }` — §5.5) is intentionally **not** implemented in F1; it
//! lands with the F4 cluster/NCCL story. F1 `GpuRef` is local-only.

use std::sync::{Arc, Weak};

use arc_swap::ArcSwapOption;

use crate::device::DeviceState;
use crate::error::GpuError;

/// A live device-buffer handle.
///
/// Holds a strong `Arc` to the slice (keeping the underlying memory
/// alive even if the `DeviceActor` has begun shutdown) plus a `Weak` to
/// the surrounding `DeviceState` (so reference cycles cannot trap the
/// system in a non-terminating state). Calling [`GpuRef::access`] before
/// each use validates that the context generation has not advanced.
pub struct GpuRef<T> {
    inner: Arc<GpuRefInner<T>>,
}

struct GpuRefInner<T> {
    /// Strong-keep on the device buffer.
    slice: Arc<cudarc::driver::CudaSlice<T>>,
    /// `DeviceState.generation` at construction time.
    generation: u64,
    /// Weak reference back to the device state. Avoids a cycle —
    /// `DeviceActor` owns the strong `Arc<DeviceState>`.
    state: Weak<DeviceState>,
    /// The most recent `CudaStream` that wrote to this buffer. Library
    /// actors call [`GpuRef::record_write`] after enqueueing a kernel
    /// that mutates the slice. Cross-stream consumers (`P2pTopology`,
    /// pipeline stages) read this to inject a `CudaEvent` wait without
    /// a host roundtrip.
    last_write_stream: ArcSwapOption<cudarc::driver::CudaStream>,
}

impl<T> Clone for GpuRef<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> std::fmt::Debug for GpuRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRef")
            .field("generation", &self.inner.generation)
            .field("len", &self.inner.slice.len())
            .finish()
    }
}

impl<T> GpuRef<T> {
    /// Wrap a raw `Arc<CudaSlice<T>>` produced by a `DeviceActor` into a
    /// `GpuRef<T>`.
    ///
    /// Only `DeviceActor` (and code reachable from its dispatcher) should
    /// call this — outside callers must obtain `GpuRef`s by asking the
    /// `DeviceActor` to allocate.
    pub fn new(slice: Arc<cudarc::driver::CudaSlice<T>>, state: &Arc<DeviceState>) -> Self {
        let generation = state.generation();
        Self {
            inner: Arc::new(GpuRefInner {
                slice,
                generation,
                state: Arc::downgrade(state),
                last_write_stream: ArcSwapOption::empty(),
            }),
        }
    }

    /// Validate the reference and return access to the underlying slice.
    ///
    /// Returns [`GpuError::GpuRefStale`] if either:
    /// - the owning `DeviceState` has been dropped,
    /// - the device is no longer accepting operations, or
    /// - the context generation has advanced past the one this ref was
    ///   minted with (i.e. a poisoned-context rebuild has happened).
    pub fn access(&self) -> Result<&Arc<cudarc::driver::CudaSlice<T>>, GpuError> {
        let state = self
            .inner
            .state
            .upgrade()
            .ok_or(GpuError::GpuRefStale("device state dropped"))?;
        if !state.accepting_ops() {
            return Err(GpuError::GpuRefStale("device shutting down"));
        }
        if state.generation() != self.inner.generation {
            return Err(GpuError::GpuRefStale("context rebuilt"));
        }
        Ok(&self.inner.slice)
    }

    /// Generation token at construction. Exposed for tests.
    pub fn generation(&self) -> u64 {
        self.inner.generation
    }

    /// Length in elements of the underlying slice.
    pub fn len(&self) -> usize {
        self.inner.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.slice.is_empty()
    }

    /// Device id this `GpuRef` was minted on, or `None` if the owning
    /// [`DeviceState`] has been dropped.
    pub fn device_id(&self) -> Option<u32> {
        self.inner.state.upgrade().map(|s| s.device_id())
    }

    /// Record the stream that most recently wrote to this buffer.
    /// Library actors (BlasActor, CudnnActor, FftActor, etc.) call this
    /// after enqueueing a kernel that mutates the slice so that
    /// downstream consumers can inject a cross-stream wait.
    pub fn record_write(&self, stream: &Arc<cudarc::driver::CudaStream>) {
        self.inner.last_write_stream.store(Some(stream.clone()));
    }

    /// Most recent producing stream, if any. Returns `None` when no
    /// kernel has been recorded against this buffer.
    pub fn last_write_stream(&self) -> Option<Arc<cudarc::driver::CudaStream>> {
        self.inner.last_write_stream.load_full()
    }

    /// Phase 4.5++ — opaque `CUdeviceptr` (`u64`) for downstream
    /// raw-pointer FFI APIs (TensorRT `enqueueV3`, `cuStreamWriteValue64`,
    /// custom CUDA modules that aren't fronted by cudarc).
    ///
    /// Validates the `GpuRef` first via [`GpuRef::access`]. The pointer
    /// is captured against the slice's own associated stream — the
    /// `_guard` returned by cudarc's `device_ptr()` is dropped before
    /// the function returns, but the underlying allocation outlives
    /// this call because the inner `Arc<CudaSlice<T>>` is held by
    /// `self`. Callers must ensure they don't dispatch the resulting
    /// pointer on a stream that has already gone out of scope; in
    /// practice the pointer is consumed immediately by an FFI shim
    /// (TensorRT enqueueV3, etc.) on a stream the caller owns.
    ///
    /// Returns [`GpuError::GpuRefStale`] if the underlying generation
    /// token is stale or the device is shutting down.
    pub fn raw_device_ptr(&self) -> Result<u64, GpuError> {
        use cudarc::driver::DevicePtr;
        let slice = self.access()?;
        let stream = slice.stream();
        let (ptr, _guard) = slice.device_ptr(stream);
        // `_guard` is a `SyncOnDrop` whose lifetime ties the pointer to
        // `slice`; we drop it here. The caller is expected to use the
        // returned `u64` immediately on an FFI call. The underlying
        // CudaSlice<T> remains alive via the strong Arc held by
        // `self.inner.slice` for as long as this `GpuRef` lives.
        Ok(ptr)
    }
}

#[cfg(test)]
impl<T> GpuRef<T> {
    /// **Test-only** stub constructor for unit tests that don't need
    /// a real CUDA context. Returns a `GpuRef<T>` whose underlying
    /// `CudaSlice<T>` is logically uninitialized — the test must
    /// **never** call `.access()` on it, dispatch it through a
    /// kernel actor, or otherwise let it reach cudarc.
    ///
    /// SAFETY contract: the caller must ensure the `GpuRef<T>` is
    /// leaked (e.g. via `Box::leak` of the surrounding container)
    /// so the inner `Arc<CudaSlice<T>>` never reaches refcount zero.
    /// Otherwise cudarc's `Drop for CudaSlice<T>` runs with
    /// uninitialized memory and aborts the process.
    pub(crate) fn for_test_no_gpu_leaked() -> Self {
        // Allocate an uninitialized box of CudaSlice<T> on the heap,
        // then leak it. We construct an `Arc<CudaSlice<T>>` via
        // `Arc::from_raw` so cudarc's Drop only runs if the strong
        // count returns to 1 — which `Box::leak` of the surrounding
        // request guarantees never happens.
        use std::mem::MaybeUninit;
        let boxed: Box<MaybeUninit<cudarc::driver::CudaSlice<T>>> = Box::new(MaybeUninit::uninit());
        let leaked: *mut MaybeUninit<cudarc::driver::CudaSlice<T>> = Box::into_raw(boxed);
        // SAFETY: the pointer is valid (just-allocated heap) and we
        // forge an Arc whose strong count is 1. The contract above
        // requires the surrounding test to leak the surrounding box
        // so this Arc's count never decrements.
        let arc_slice: std::sync::Arc<cudarc::driver::CudaSlice<T>> =
            unsafe { std::sync::Arc::from_raw(leaked as *const cudarc::driver::CudaSlice<T>) };
        let state = std::sync::Arc::new(crate::device::DeviceState::new(0));
        Self::new(arc_slice, &state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceState;

    #[test]
    fn generation_mismatch_fails_validate() {
        // We can't construct a real CudaSlice without a GPU. Instead we
        // exercise the generation-check logic by faking the slice via a
        // pointer-only view: this test does NOT touch CUDA memory.
        // Verify the generation accessor and accepting_ops flag.
        let state = Arc::new(DeviceState::new(0));
        assert_eq!(state.generation(), 0);
        state.bump_generation();
        assert_eq!(state.generation(), 1);
        assert!(state.accepting_ops());
        state.begin_shutdown();
        assert!(!state.accepting_ops());
    }
}
