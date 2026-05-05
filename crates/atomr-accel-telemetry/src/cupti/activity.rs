//! CUPTI activity-API tracing primitives.
//!
//! `ActivityCategory` is the public-facing categorisation; each
//! variant maps onto one or more `CUpti_ActivityKind` enums from
//! `cudarc::cupti::sys`. The translation lives here so the
//! actor / session layer doesn't need to know about cudarc's enum
//! shape.
//!
//! `Activity` is the minimal-fidelity decoded record we push into
//! the mpsc channel. Full CUPTI records carry a wide union of
//! per-kind fields; we project the ones every consumer wants
//! (timestamps, duration, optional kernel name, optional bytes).

use cudarc::cupti::sys as cu_sys;

/// Logical activity categories the public API exposes. Each variant
/// is mapped to one or more underlying CUPTI kinds via
/// [`ActivityCategory::cupti_kinds`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ActivityCategory {
    /// Concurrent kernel launches. Maps to
    /// `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL` (preferred over
    /// `_KERNEL` because it's the modern, lockless variant).
    KernelLaunch,
    /// Async memcpy timing.
    Memcpy,
    /// Driver-API entry / exit timing.
    DriverApi,
    /// Runtime-API entry / exit timing.
    RuntimeApi,
    /// Range profiler — wraps CUPTI's profiling-API metric collection.
    RangeProfiler,
}

impl ActivityCategory {
    /// Stable bit value for each category. The bits are namespaced
    /// to this crate; consumers of the public API can persist and
    /// re-load category sets through the `bit` / `from_bitmask`
    /// pair.
    pub const fn bit(self) -> u32 {
        match self {
            ActivityCategory::KernelLaunch => 1 << 0,
            ActivityCategory::Memcpy => 1 << 1,
            ActivityCategory::DriverApi => 1 << 2,
            ActivityCategory::RuntimeApi => 1 << 3,
            ActivityCategory::RangeProfiler => 1 << 4,
        }
    }

    /// Decode a bitmask back to a vector of categories. Unknown
    /// bits are silently ignored.
    pub fn from_bitmask(mask: u32) -> Vec<Self> {
        let mut out = Vec::new();
        let candidates = [
            ActivityCategory::KernelLaunch,
            ActivityCategory::Memcpy,
            ActivityCategory::DriverApi,
            ActivityCategory::RuntimeApi,
            ActivityCategory::RangeProfiler,
        ];
        for c in candidates {
            if mask & c.bit() != 0 {
                out.push(c);
            }
        }
        out
    }

    /// Underlying `CUpti_ActivityKind`(s) this category enables.
    /// The session `Start` handler iterates this list and calls
    /// `cuptiActivityEnable` per kind.
    pub fn cupti_kinds(self) -> &'static [cu_sys::CUpti_ActivityKind] {
        match self {
            ActivityCategory::KernelLaunch => {
                &[cu_sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL]
            }
            ActivityCategory::Memcpy => &[cu_sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY],
            ActivityCategory::DriverApi => {
                &[cu_sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER]
            }
            ActivityCategory::RuntimeApi => {
                &[cu_sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME]
            }
            ActivityCategory::RangeProfiler => {
                // Range-profiler records arrive through a different
                // CUPTI-API; the kind list is empty for the activity
                // path, and the range_profiler module wires the
                // collector instead.
                &[]
            }
        }
    }
}

/// Minimal-fidelity activity record. Each variant carries the bits
/// of the underlying CUPTI record that downstream consumers
/// (Chrome-trace exporter, tracing collector, dashboards) need.
#[derive(Debug, Clone)]
pub enum Activity {
    Kernel {
        name: String,
        device_id: u32,
        stream_id: u32,
        start_ns: u64,
        end_ns: u64,
        correlation_id: u32,
    },
    Memcpy {
        kind: u8,
        bytes: u64,
        device_id: u32,
        stream_id: u32,
        start_ns: u64,
        end_ns: u64,
        correlation_id: u32,
    },
    DriverApi {
        cbid: u32,
        thread_id: u32,
        start_ns: u64,
        end_ns: u64,
        correlation_id: u32,
    },
    RuntimeApi {
        cbid: u32,
        thread_id: u32,
        start_ns: u64,
        end_ns: u64,
        correlation_id: u32,
    },
    /// One entry from the range profiler. The metric / unit pair
    /// depends on the metrics requested at session start.
    RangeProfilerSample {
        range_name: String,
        metric_name: String,
        value: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cupti_kinds_lookup() {
        assert!(!ActivityCategory::KernelLaunch.cupti_kinds().is_empty());
        assert!(!ActivityCategory::Memcpy.cupti_kinds().is_empty());
        // RangeProfiler intentionally returns an empty slice.
        assert!(ActivityCategory::RangeProfiler.cupti_kinds().is_empty());
    }
}
