//! Top-k contraction autotune (gated on `cutensor-autotune`).
//!
//! `cutensor` exposes a fixed set of `cutensorAlgo_t` values
//! (`CUTENSOR_ALGO_GETT`, `CUTENSOR_ALGO_TGETT`, `CUTENSOR_ALGO_TTGT`,
//! plus the default-patient meta-algo). For workloads whose shapes
//! repeat — most training inner loops — we probe the top `k` algos,
//! measure their runtime, and cache the winner.
//!
//! The probing primitive [`autotune_pick`] is dependency-injected
//! over a `Measure` trait so the unit test can mock cuTENSOR
//! execution without a GPU. Production callers pass a
//! [`HandleMeasure`] that times `cutensorContract` calls via
//! `CudaEvent`s.

use cudarc::cutensor::sys as ct_sys;

/// Algorithms the autotune iterates over. Order matters: we keep the
/// default first so a tied measurement leaves it the winner.
pub const TOP_K_ALGOS: &[ct_sys::cutensorAlgo_t] = &[
    ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT,
    ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_GETT,
    ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_TGETT,
    ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_TTGT,
];

/// Measurement callback. Real impl times a single cuTENSOR contract
/// launch; the mock used in tests just returns a precomputed value
/// keyed by algo.
pub trait Measure {
    /// Run the contraction with `algo` and return its elapsed time
    /// (lower is better). Returning `None` skips this algo (e.g. it
    /// failed to plan).
    fn measure(&mut self, algo: ct_sys::cutensorAlgo_t) -> Option<f64>;
}

/// Execute `measure` over each algo in [`TOP_K_ALGOS`] and return the
/// fastest. Returns `None` if every measurement failed.
pub fn autotune_pick<M: Measure>(measure: &mut M) -> Option<ct_sys::cutensorAlgo_t> {
    let mut best: Option<(ct_sys::cutensorAlgo_t, f64)> = None;
    for algo in TOP_K_ALGOS.iter().copied() {
        if let Some(t) = measure.measure(algo) {
            match best {
                Some((_, bt)) if t < bt => best = Some((algo, t)),
                None => best = Some((algo, t)),
                _ => {}
            }
        }
    }
    best.map(|(a, _)| a)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mock<F: FnMut(ct_sys::cutensorAlgo_t) -> Option<f64>>(F);
    impl<F: FnMut(ct_sys::cutensorAlgo_t) -> Option<f64>> Measure for Mock<F> {
        fn measure(&mut self, a: ct_sys::cutensorAlgo_t) -> Option<f64> {
            (self.0)(a)
        }
    }

    #[test]
    fn autotune_picks_best_of_topk() {
        // GETT wins (lowest time).
        let mut m = Mock(|a: ct_sys::cutensorAlgo_t| match a {
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT => Some(10.0),
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_GETT => Some(2.5),
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_TGETT => Some(5.0),
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_TTGT => Some(7.5),
            _ => None,
        });
        let pick = autotune_pick(&mut m).expect("a winner");
        assert_eq!(pick, ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_GETT);

        // All-fail → no pick.
        let mut m = Mock(|_| None);
        assert!(autotune_pick(&mut m).is_none());

        // Default ties with GETT — default keeps the lead by virtue
        // of being measured first.
        let mut m = Mock(|a: ct_sys::cutensorAlgo_t| match a {
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT => Some(3.0),
            ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_GETT => Some(3.0),
            _ => None,
        });
        let pick = autotune_pick(&mut m).expect("a winner");
        assert_eq!(pick, ct_sys::cutensorAlgo_t::CUTENSOR_ALGO_DEFAULT);
    }
}
