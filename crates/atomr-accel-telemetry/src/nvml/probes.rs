//! atomr-telemetry probe registration for NVML metrics.
//!
//! Each metric exposed by [`super::actor::NvmlSnapshot`] is mirrored
//! into a string-keyed registration so the dashboard can enumerate
//! the GPU-specific probes alongside the existing actor / cluster /
//! sharding probes.
//!
//! Registration is idempotent: calling [`register_all`] twice on the
//! same [`ProbeRegistration`] does not duplicate metrics. This makes
//! the probe set safe to register from `main` without worrying about
//! a re-entry.
//!
//! The registration shim is intentionally light. atomr-telemetry's
//! existing extension model is event-bus + snapshot, which doesn't
//! map cleanly onto NVML's pull cadence — so probes here record
//! their *names* and let the caller decide how to forward them
//! (Prometheus exporter, custom snapshot, etc.).

use std::collections::BTreeSet;

use parking_lot::Mutex;

/// Names of the metrics registered by [`register_all`]. Held inside
/// [`ProbeRegistration`] so duplicate calls become no-ops.
pub const METRIC_NAMES: &[&str] = &[
    "nvml.power.instant_milliwatts",
    "nvml.power.average_milliwatts",
    "nvml.temp.gpu_celsius",
    "nvml.temp.memory_celsius",
    "nvml.ecc.sbe_volatile",
    "nvml.ecc.dbe_volatile",
    "nvml.ecc.sbe_aggregate",
    "nvml.ecc.dbe_aggregate",
    "nvml.clock.sm_mhz",
    "nvml.clock.mem_mhz",
    "nvml.clock.video_mhz",
    "nvml.throttle.reasons_bitmask",
    "nvml.pcie.tx_kib_per_s",
    "nvml.pcie.rx_kib_per_s",
    "nvml.memory.total_bytes",
    "nvml.memory.used_bytes",
    "nvml.memory.free_bytes",
    "nvml.processes.count",
    "nvml.mig.mode_current",
    "nvml.mig.mode_pending",
    "nvml.mig.instances_count",
];

/// Probe-registration handle. Construct once per process; share
/// freely.
#[derive(Default)]
pub struct ProbeRegistration {
    inner: Mutex<BTreeSet<&'static str>>,
}

impl ProbeRegistration {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of metrics currently registered.
    pub fn metric_count(&self) -> usize {
        self.inner.lock().len()
    }

    /// Iterate over the registered metric names. Returns a snapshot
    /// to avoid holding the lock.
    pub fn metric_names(&self) -> Vec<&'static str> {
        self.inner.lock().iter().copied().collect()
    }

    fn add(&self, name: &'static str) -> bool {
        self.inner.lock().insert(name)
    }
}

/// Register the full NVML metric catalogue against the given
/// registration. Returns the number of *new* metrics registered;
/// idempotent calls return 0.
pub fn register_all(reg: &ProbeRegistration) -> usize {
    let mut new = 0usize;
    for name in METRIC_NAMES.iter() {
        if reg.add(*name) {
            new += 1;
        }
    }
    new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_all_idempotent_self_test() {
        let reg = ProbeRegistration::new();
        let n1 = register_all(&reg);
        assert_eq!(n1, METRIC_NAMES.len());
        let n2 = register_all(&reg);
        assert_eq!(n2, 0);
    }
}
