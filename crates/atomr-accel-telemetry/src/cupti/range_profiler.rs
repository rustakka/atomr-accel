//! CUPTI range profiler API plumbing.
//!
//! The range profiler is the modern (CUDA ≥ 11.x) replacement for
//! the deprecated `cuptiMetricCreateFromName` flow. It targets a
//! `CUcontext`, scopes a measurement window via push/pop, and
//! delivers one record per metric per range.
//!
//! cudarc 0.19 doesn't expose the range-profiler API in its safe
//! layer; the function pointers come from `cudarc::cupti::sys`. We
//! resolve them lazily so the module doesn't fail to compile on
//! older CUDA SDKs.

use std::sync::Arc;

use parking_lot::RwLock;
use tracing::warn;

use super::activity::Activity;

/// Configuration for one profiling pass. A pass is the unit of work
/// that CUPTI recommends batching metric collection into.
#[derive(Default, Clone, Debug)]
pub struct RangeProfilerPass {
    /// Names of the metrics to collect (e.g. `"sm__cycles_active.avg"`).
    pub metrics: Vec<String>,
    /// Maximum number of ranges per pass. CUPTI allocates buffers
    /// proportional to this number, so callers should bound it.
    pub max_ranges_per_pass: u32,
}

/// In-memory collector that the session actor pushes range
/// profiler samples into. Consumers `Drain` to read accumulated
/// records out.
#[derive(Default)]
pub struct RangeProfilerCollector {
    samples: RwLock<Vec<Activity>>,
}

impl RangeProfilerCollector {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Push a sample. Called from the session actor on every
    /// `cuptiProfilerEndPass` decode.
    pub fn push(&self, sample: Activity) {
        match &sample {
            Activity::RangeProfilerSample { .. } => {
                self.samples.write().push(sample);
            }
            other => {
                warn!(?other, "non-range-profiler activity dropped on collector");
            }
        }
    }

    /// Atomically take ownership of every sample collected so far.
    /// The collector becomes empty after this call.
    pub fn drain(&self) -> Vec<Activity> {
        std::mem::take(&mut *self.samples.write())
    }

    /// Current sample count.
    pub fn len(&self) -> usize {
        self.samples.read().len()
    }

    /// Empty check.
    pub fn is_empty(&self) -> bool {
        self.samples.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collector_round_trip() {
        let c = RangeProfilerCollector::new();
        c.push(Activity::RangeProfilerSample {
            range_name: "fwd".into(),
            metric_name: "sm__cycles_active.avg".into(),
            value: 12345.0,
        });
        assert_eq!(c.len(), 1);
        let drained = c.drain();
        assert_eq!(drained.len(), 1);
        assert!(c.is_empty());
    }
}
