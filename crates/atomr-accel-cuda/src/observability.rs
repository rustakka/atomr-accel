//! Observability glue: install [`atomr_telemetry::TelemetryExtension`]
//! on a host `ActorSystem` and expose a small set of GPU-specific
//! probes that callers feed from kernel actors / placement actors /
//! stream allocators.
//!
//! Probes are designed to be **callable from anywhere**: the helpers
//! here just look up the installed extension via
//! `TelemetryExtension::from_system(...)` and update an internal
//! counter. When telemetry isn't installed the call short-circuits.
//!
//! The dashboard at `atomr-dashboard/` (in the atomr workspace)
//! consumes the resulting [`atomr_telemetry::dto::NodeSnapshot`] over
//! WebSocket — point it at any atomr-accel-cuda host and the GPU probes
//! show up automatically alongside the standard actor / cluster /
//! sharding panels.

use std::sync::Arc;

use atomr_core::actor::ActorSystem;
use atomr_telemetry::TelemetryExtension;
use parking_lot::Mutex;

/// Convenience helper: construct + install a telemetry extension with
/// sensible defaults (1024-deep broadcast bus). Returns the shared
/// `Arc` so the caller can register exporters or read snapshots.
///
/// ```ignore
/// let sys = ActorSystem::create("gpu-host", Config::empty()).await?;
/// let _telemetry = atomr_accel_cuda::observability::install(&sys, "gpu-host-1");
/// ```
pub fn install(system: &ActorSystem, node_name: impl Into<String>) -> Arc<TelemetryExtension> {
    TelemetryExtension::new(node_name, 1024).install(system)
}

/// In-memory counters for the GPU-specific probes. The probes are
/// passive: kernel actors / stream allocators bump the counters
/// directly; the dashboard polls via [`GpuProbes::snapshot`].
///
/// Construct one per host (or per device if you want per-device
/// breakdowns), share via `Arc`, and pass to whichever code paths
/// produce the events.
#[derive(Default)]
pub struct GpuProbes {
    inner: Mutex<GpuProbeState>,
}

#[derive(Default, Debug, Clone)]
pub struct GpuProbeState {
    /// Cumulative number of `GpuRef` allocations that succeeded.
    pub allocations_total: u64,
    /// Cumulative number of `GpuRef` allocations that returned
    /// `OutOfMemory`.
    pub oom_total: u64,
    /// Highest observed `DeviceState::generation` (bumps on context
    /// rebuild).
    pub max_generation_observed: u64,
    /// Currently in-flight kernel launches across all actors.
    pub kernels_in_flight: u32,
    /// Cumulative kernel-launch count.
    pub kernels_total: u64,
    /// Free / total VRAM as last reported by a `Stats` poll. 0 when
    /// no poll has occurred yet.
    pub vram_free_bytes: u64,
    pub vram_total_bytes: u64,
}

impl GpuProbes {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record a successful allocation.
    pub fn record_alloc_ok(&self) {
        let mut g = self.inner.lock();
        g.allocations_total = g.allocations_total.saturating_add(1);
    }

    /// Record an allocation that hit OOM.
    pub fn record_alloc_oom(&self) {
        let mut g = self.inner.lock();
        g.oom_total = g.oom_total.saturating_add(1);
    }

    /// Record observation of a new `DeviceState::generation`. The
    /// stored value is monotonically max'd.
    pub fn record_generation(&self, gen: u64) {
        let mut g = self.inner.lock();
        if gen > g.max_generation_observed {
            g.max_generation_observed = gen;
        }
    }

    /// Bump the in-flight kernel count when a kernel is enqueued.
    pub fn kernel_enter(&self) {
        let mut g = self.inner.lock();
        g.kernels_in_flight = g.kernels_in_flight.saturating_add(1);
        g.kernels_total = g.kernels_total.saturating_add(1);
    }

    /// Decrement the in-flight kernel count when a kernel completes.
    pub fn kernel_exit(&self) {
        let mut g = self.inner.lock();
        g.kernels_in_flight = g.kernels_in_flight.saturating_sub(1);
    }

    /// Record the latest VRAM snapshot (typically from
    /// `cuMemGetInfo` once per `PlacementActor::PollStats` tick).
    pub fn record_vram(&self, free_bytes: u64, total_bytes: u64) {
        let mut g = self.inner.lock();
        g.vram_free_bytes = free_bytes;
        g.vram_total_bytes = total_bytes;
    }

    /// Snapshot the current counter state.
    pub fn snapshot(&self) -> GpuProbeState {
        self.inner.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_config::Config;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn install_returns_handle_and_from_system_finds_it() {
        let sys = ActorSystem::create("obs-test", Config::empty())
            .await
            .unwrap();
        let handle = install(&sys, "obs-test");
        assert_eq!(handle.node, "obs-test");
        let from_sys = TelemetryExtension::from_system(&sys).expect("installed");
        assert_eq!(from_sys.node, "obs-test");
        sys.terminate().await;
    }

    #[test]
    fn gpu_probes_record_lifecycle() {
        let p = GpuProbes::new();
        p.record_alloc_ok();
        p.record_alloc_ok();
        p.record_alloc_oom();
        p.record_generation(1);
        p.record_generation(3);
        p.record_generation(2); // out-of-order — max stays at 3
        p.kernel_enter();
        p.kernel_enter();
        p.kernel_exit();
        p.record_vram(2048, 4096);
        let s = p.snapshot();
        assert_eq!(s.allocations_total, 2);
        assert_eq!(s.oom_total, 1);
        assert_eq!(s.max_generation_observed, 3);
        assert_eq!(s.kernels_in_flight, 1);
        assert_eq!(s.kernels_total, 2);
        assert_eq!(s.vram_free_bytes, 2048);
        assert_eq!(s.vram_total_bytes, 4096);
    }
}
