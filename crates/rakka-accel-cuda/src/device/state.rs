//! `DeviceState` — the shared state that survives `ContextActor` restarts
//! (§5.11 outer/inner-tier split, §5.8 `GpuRef` validity).
//!
//! `Arc<DeviceState>` is held by:
//! - the outer `DeviceActor` (lifetime: ActorSystem)
//! - each `ContextActor` incarnation (replaced on restart)
//! - every live `GpuRef<T>` (via `Weak`)
//!
//! On context rebuild, [`DeviceState::install_context`] swaps in the new
//! `Arc<CudaContext>` and bumps the generation atomically. `GpuRef`s
//! minted against the old generation will fail their next `access()`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use tokio::sync::watch;

#[cfg(feature = "cuda-runtime-tests")]
type ContextHandle = cudarc::driver::CudaContext;

#[cfg(not(feature = "cuda-runtime-tests"))]
type ContextHandle = cudarc::driver::CudaContext;

pub struct DeviceState {
    device_id: u32,
    /// Bumped on every context rebuild (§5.8). Acquire/Release to pair
    /// with the `ArcSwap` of `current_ctx`.
    generation: AtomicU64,
    /// Set false at the start of `DeviceActor::post_stop`. Outstanding
    /// `GpuRef::access()` calls will fail fast with `GpuRefStale`.
    accepting_ops: AtomicBool,
    /// The current live context, swapped in by `ContextActor::pre_start`.
    /// `None` between rebuilds and at startup.
    current_ctx: ArcSwapOption<ContextHandle>,
    /// Watch channel that publishes the new generation each time
    /// [`DeviceState::bump_generation`] is called. Allows top-level
    /// observers (`P2pTopology`, `NcclWorldActor`, `PlacementActor`,
    /// `ReplayHarness`) to subscribe to context rebuilds without
    /// polling.
    generation_tx: watch::Sender<u64>,
}

impl DeviceState {
    pub fn new(device_id: u32) -> Self {
        let (tx, _rx) = watch::channel(0u64);
        Self {
            device_id,
            generation: AtomicU64::new(0),
            accepting_ops: AtomicBool::new(true),
            current_ctx: ArcSwapOption::empty(),
            generation_tx: tx,
        }
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Bump the generation. Called by `ContextActor` after building a
    /// new `CudaContext` and before spawning library children.
    pub fn bump_generation(&self) -> u64 {
        // fetch_add returns the previous value; the new generation is
        // that+1.
        let new = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        // Best-effort publish to subscribers. If no receivers are
        // attached the send is a no-op.
        let _ = self.generation_tx.send(new);
        new
    }

    /// Subscribe to generation changes. Receivers see every
    /// [`DeviceState::bump_generation`] call. Used by top-level
    /// observers that need to react to context rebuilds.
    pub fn generation_watch(&self) -> watch::Receiver<u64> {
        self.generation_tx.subscribe()
    }

    pub fn accepting_ops(&self) -> bool {
        self.accepting_ops.load(Ordering::Acquire)
    }

    /// Mark that the `DeviceActor` is winding down. Any subsequent
    /// `GpuRef::access()` returns `GpuRefStale`.
    pub fn begin_shutdown(&self) {
        self.accepting_ops.store(false, Ordering::Release);
    }

    /// Install a freshly built CUDA context into the shared state.
    /// Called from `ContextActor::pre_start` (and the post-restart path).
    pub fn install_context(&self, ctx: Arc<ContextHandle>) {
        self.current_ctx.store(Some(ctx));
    }

    /// Drop the current context reference held by the shared state.
    /// Called from `ContextActor::post_stop` so a poisoned context can be
    /// torn down before the new incarnation builds its replacement.
    pub fn clear_context(&self) {
        self.current_ctx.store(None);
    }

    /// Snapshot of the current `CudaContext`, if any. `KernelActor`s use
    /// this in their own `pre_start` to acquire the handle they need.
    pub fn current_context(&self) -> Option<Arc<ContextHandle>> {
        self.current_ctx.load_full()
    }
}

impl std::fmt::Debug for DeviceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceState")
            .field("device_id", &self.device_id)
            .field("generation", &self.generation())
            .field("accepting_ops", &self.accepting_ops())
            .field("has_context", &self.current_ctx.load().is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_starts_zero_and_bumps_monotonically() {
        let s = DeviceState::new(0);
        assert_eq!(s.generation(), 0);
        assert_eq!(s.bump_generation(), 1);
        assert_eq!(s.bump_generation(), 2);
        assert_eq!(s.generation(), 2);
    }

    #[test]
    fn shutdown_flips_accepting_ops() {
        let s = DeviceState::new(0);
        assert!(s.accepting_ops());
        s.begin_shutdown();
        assert!(!s.accepting_ops());
    }
}
