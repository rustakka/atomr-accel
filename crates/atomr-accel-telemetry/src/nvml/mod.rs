//! NVML actor: polls `libnvidia-ml.so.1` via `libloading` for
//! GPU-wide metrics that aren't exposed by the CUDA driver / runtime
//! APIs themselves.
//!
//! Why NVML and not CUDA: NVML is the only library that surfaces
//! power, ECC counts, throttle reasons, PCIe bandwidth, MIG
//! configuration, and the per-process GPU-memory list. NVIDIA ships
//! it inside the driver package — `libnvidia-ml.so.1` lives
//! alongside `libcuda.so.1` on every system that runs CUDA.
//!
//! On consumer / WSL setups NVML may be missing (the Windows driver
//! exposes a different shim). The actor degrades gracefully:
//! `NvmlActor::try_new` returns `Err(NvmlError::LibraryUnavailable)`
//! and the caller skips installation.

mod actor;
pub mod probes;

pub use actor::{NvmlActor, NvmlConfig, NvmlError, NvmlMsg, NvmlReply, NvmlSnapshot};
pub use probes::{register_all, ProbeRegistration};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::oneshot;

    /// `NvmlMsg` constructs cleanly across all variants, including
    /// the reply-channel ones. We can't actually drive the actor on
    /// a no-GPU host, but the message wiring must round-trip.
    #[test]
    fn nvml_msg_constructs() {
        let (tx, _rx) = oneshot::channel::<NvmlReply<NvmlSnapshot>>();
        let msg = NvmlMsg::Snapshot { reply: tx };
        match msg {
            NvmlMsg::Snapshot { .. } => {}
            _ => panic!("Snapshot variant didn't round-trip"),
        }

        let (tx, _rx) = oneshot::channel::<NvmlReply<()>>();
        let msg = NvmlMsg::SetInterval {
            interval: Duration::from_millis(500),
            reply: tx,
        };
        match msg {
            NvmlMsg::SetInterval { interval, .. } => {
                assert_eq!(interval, Duration::from_millis(500));
            }
            _ => panic!("SetInterval variant didn't round-trip"),
        }

        let (tx, _rx) = oneshot::channel::<NvmlReply<()>>();
        let msg = NvmlMsg::Shutdown { reply: tx };
        assert!(matches!(msg, NvmlMsg::Shutdown { .. }));
    }

    /// Registering the probe set twice on the same registration
    /// must not panic and the second call must be a no-op (return
    /// the same handle).
    #[test]
    fn probe_registration_is_idempotent() {
        let reg = ProbeRegistration::new();
        assert_eq!(reg.metric_count(), 0);
        let _ = register_all(&reg);
        let count_after_first = reg.metric_count();
        assert!(count_after_first > 0);
        let _ = register_all(&reg);
        let count_after_second = reg.metric_count();
        // Idempotent: same metrics, no duplicates.
        assert_eq!(count_after_first, count_after_second);
    }
}
