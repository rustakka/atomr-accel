//! `cp.async` pipeline macro shim.
//!
//! `cp.async` (sm_80+) and the Hopper-introduced `cp.async.bulk`
//! variants live entirely on the device side. This module provides
//! Rust constants for the macro names defined in `atomr_hopper.cuh`
//! plus host-side helpers that compute the right `mbarrier` arrival
//! count for a given pipeline stage count.

/// Number of mbarrier arrival slots needed to fence a pipelined
/// producer/consumer with the given `stages`. Matches the formula
/// `2 * stages` used by stage-balanced double-buffer kernels.
pub const fn mbarrier_arrival_count(stages: u32) -> u32 {
    stages * 2
}

/// Pipeline stage policy for `cp.async`-driven shared-memory
/// double/triple/quad buffering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStages {
    Double,
    Triple,
    Quad,
}

impl PipelineStages {
    pub fn count(self) -> u32 {
        match self {
            PipelineStages::Double => 2,
            PipelineStages::Triple => 3,
            PipelineStages::Quad => 4,
        }
    }
}

/// Macro names exposed by `atomr_hopper.cuh` for callers to reference
/// in their NVRTC sources.
pub mod macro_names {
    /// Issue a `cp.async.cg.shared.global` (16B-aligned). 4-arg macro:
    /// `(dst_smem_addr, src_global_addr, bytes, predicate)`.
    pub const CP_ASYNC_CG_16: &str = "ATOMR_CP_ASYNC_CG_16";
    /// Issue a `cp.async.ca.shared.global` (4B-aligned, cache-all).
    pub const CP_ASYNC_CA_4: &str = "ATOMR_CP_ASYNC_CA_4";
    /// Commit-group barrier (`cp.async.commit_group`).
    pub const CP_ASYNC_COMMIT_GROUP: &str = "ATOMR_CP_ASYNC_COMMIT_GROUP";
    /// Wait-group barrier (`cp.async.wait_group <N>`).
    pub const CP_ASYNC_WAIT_GROUP: &str = "ATOMR_CP_ASYNC_WAIT_GROUP";
    /// Bulk async copy (`cp.async.bulk.shared::cluster.global`) for
    /// TMA-driven loads.
    pub const CP_ASYNC_BULK: &str = "ATOMR_CP_ASYNC_BULK";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrival_counts_balanced() {
        assert_eq!(mbarrier_arrival_count(2), 4);
        assert_eq!(mbarrier_arrival_count(3), 6);
        assert_eq!(mbarrier_arrival_count(4), 8);
        assert_eq!(PipelineStages::Double.count(), 2);
        assert_eq!(PipelineStages::Triple.count(), 3);
        assert_eq!(PipelineStages::Quad.count(), 4);
    }
}
