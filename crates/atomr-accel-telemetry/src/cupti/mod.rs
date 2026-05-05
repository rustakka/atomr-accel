//! CUPTI integration: an opt-in actor that drives the CUPTI activity
//! API + range profiler.
//!
//! Activity records are streamed into a Tokio `mpsc` channel; the
//! [`session::CuptiSession`] actor exposes `Start` / `Stop` / `Drain`
//! messages so applications can scope a tracing session to a
//! specific request.
//!
//! ## CUPTI bootstrap
//!
//! CUPTI must be initialised **before** the first `cuInit`. The
//! [`session::CuptiBootstrap::install`] helper opens
//! `libcupti.so.<major>` and registers the activity buffer
//! callbacks. Call it from your `main` before constructing the
//! atomr `ActorSystem`.

pub mod activity;
pub mod range_profiler;
pub mod session;

pub use activity::{Activity, ActivityCategory};
pub use session::{CuptiBootstrap, CuptiError, CuptiMsg, CuptiReply, CuptiSession};

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    /// Every `CuptiMsg` variant constructs and round-trips through a
    /// match. The reply channels stay live (we drop the receiver
    /// immediately after construction).
    #[test]
    fn cupti_session_msg_round_trip() {
        let (tx, _rx) = oneshot::channel::<CuptiReply<()>>();
        let msg = CuptiMsg::Start {
            categories: vec![
                ActivityCategory::KernelLaunch,
                ActivityCategory::Memcpy,
                ActivityCategory::DriverApi,
                ActivityCategory::RuntimeApi,
                ActivityCategory::RangeProfiler,
            ],
            reply: tx,
        };
        match msg {
            CuptiMsg::Start { categories, .. } => {
                assert_eq!(categories.len(), 5);
            }
            _ => panic!("Start variant didn't round-trip"),
        }

        let (tx, _rx) = oneshot::channel::<CuptiReply<()>>();
        let msg = CuptiMsg::Stop { reply: tx };
        assert!(matches!(msg, CuptiMsg::Stop { .. }));

        let (tx, _rx) = oneshot::channel::<CuptiReply<Vec<Activity>>>();
        let msg = CuptiMsg::Drain { reply: tx };
        assert!(matches!(msg, CuptiMsg::Drain { .. }));
    }

    /// `ActivityCategory` is convertible into / out of a `u32`
    /// bitmask so callers can negotiate categories with downstream
    /// systems (RPC, on-disk profile config, etc.). The conversion
    /// must round-trip.
    #[test]
    fn activity_category_bitmask_round_trip() {
        let all = [
            ActivityCategory::KernelLaunch,
            ActivityCategory::Memcpy,
            ActivityCategory::DriverApi,
            ActivityCategory::RuntimeApi,
            ActivityCategory::RangeProfiler,
        ];
        let mut mask = 0u32;
        for c in &all {
            mask |= c.bit();
        }
        let recovered = ActivityCategory::from_bitmask(mask);
        let mut sorted_in: Vec<u32> = all.iter().map(|c| c.bit()).collect();
        sorted_in.sort_unstable();
        let mut sorted_out: Vec<u32> = recovered.iter().map(|c| c.bit()).collect();
        sorted_out.sort_unstable();
        assert_eq!(sorted_in, sorted_out);

        // Empty mask -> empty set.
        assert!(ActivityCategory::from_bitmask(0).is_empty());

        // Idempotent: converting back to a mask matches the input.
        let mask2 = recovered.iter().map(|c| c.bit()).fold(0u32, |a, b| a | b);
        assert_eq!(mask2, mask);
    }

    /// `CuptiBootstrap::install` returns an `Err` when libcupti.so
    /// isn't available on the host. We force the failure path with
    /// a non-existent library path.
    #[test]
    fn bootstrap_install_returns_err_on_missing_lib() {
        let res = CuptiBootstrap::install_with_library_path("/nonexistent/libcupti.so.999");
        assert!(res.is_err(), "expected Err on missing library");
    }
}
