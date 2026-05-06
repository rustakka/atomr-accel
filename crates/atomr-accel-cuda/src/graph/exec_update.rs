//! `cuGraphExecUpdate` — in-place parameter update for an instantiated
//! graph. Lets callers re-bind `GpuRef` pointers without re-running
//! `cuGraphInstantiate`.
//!
//! The driver returns a [`GraphExecUpdateOutcome`] indicating whether
//! the update succeeded as-is or whether the topology changed and the
//! caller must re-instantiate.

use cudarc::driver::sys as driver_sys;

use crate::error::GpuError;
use crate::graph::GraphHandle;

const LIB: &str = "graph";

/// Result of an update attempt. Values mirror the
/// `CUgraphExecUpdateResult` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExecUpdateOutcome {
    Success,
    /// Topology mismatch: the new graph's nodes don't line up. The
    /// caller must rebuild from scratch.
    TopologyChanged,
    /// Other (driver-classified) failure.
    Other,
}

impl From<driver_sys::CUgraphExecUpdateResult> for GraphExecUpdateOutcome {
    fn from(r: driver_sys::CUgraphExecUpdateResult) -> Self {
        // CU_GRAPH_EXEC_UPDATE_SUCCESS = 0
        match r as u32 {
            0 => GraphExecUpdateOutcome::Success,
            // CUDA_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2 (older drivers)
            // CUDA_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3
            // We classify everything except success as TopologyChanged
            // for the conservative "must reinstantiate" path; the Other
            // bucket is reserved for future granularity.
            2..=8 => GraphExecUpdateOutcome::TopologyChanged,
            _ => GraphExecUpdateOutcome::Other,
        }
    }
}

/// Try to apply `new_graph`'s parameters to `exec`'s instantiated
/// state. Wraps `cuGraphExecUpdate_v2` (CUDA 12+); returns an
/// `Unrecoverable` on hosts where the symbol isn't loadable.
pub fn exec_update(
    exec: &GraphHandle,
    new_graph_cu: driver_sys::CUgraph,
) -> Result<GraphExecUpdateOutcome, GpuError> {
    let exec_handle = exec.cu_graph_exec();
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut info = driver_sys::CUgraphExecUpdateResultInfo_st {
            result: driver_sys::CUgraphExecUpdateResult_enum::CU_GRAPH_EXEC_UPDATE_SUCCESS,
            errorNode: std::ptr::null_mut(),
            errorFromNode: std::ptr::null_mut(),
        };
        // SAFETY: exec_handle and new_graph_cu are caller-owned;
        // info is a local out-pointer.
        let s = unsafe {
            driver_sys::cuGraphExecUpdate_v2(exec_handle, new_graph_cu, &mut info as *mut _)
        };
        (s, info.result)
    }));
    match probe {
        Ok((s, result)) => {
            if s == driver_sys::cudaError_enum::CUDA_SUCCESS {
                Ok(GraphExecUpdateOutcome::from(result))
            } else if s == driver_sys::cudaError_enum::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE {
                // Treat as a topology-class error — the caller should
                // reinstantiate.
                Ok(GraphExecUpdateOutcome::TopologyChanged)
            } else {
                Err(GpuError::LibraryError {
                    lib: LIB,
                    msg: format!("cuGraphExecUpdate_v2: {s:?}"),
                })
            }
        }
        Err(_) => Err(GpuError::Unrecoverable(
            "exec_update: CUDA driver not loadable".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_classification_round_trip() {
        // Synthesize each variant via the From impl using cudarc's
        // enum where possible.
        use driver_sys::CUgraphExecUpdateResult_enum::*;
        assert_eq!(
            GraphExecUpdateOutcome::from(CU_GRAPH_EXEC_UPDATE_SUCCESS),
            GraphExecUpdateOutcome::Success
        );
        // Topology-changed bucket — any non-zero, non-other value.
        let topology_value: driver_sys::CUgraphExecUpdateResult =
            unsafe { std::mem::transmute::<u32, _>(2) };
        assert_eq!(
            GraphExecUpdateOutcome::from(topology_value),
            GraphExecUpdateOutcome::TopologyChanged
        );
    }

    #[test]
    fn param_rebind_round_trip() {
        // Mock-mode: with a synthetic GraphHandle and null new graph,
        // the call surfaces Unrecoverable on no-GPU hosts and a
        // LibraryError on real ones. No panic.
        let exec = GraphHandle::synthetic_for_tests();
        let r = exec_update(&exec, std::ptr::null_mut());
        match r {
            Ok(_) => {}
            Err(GpuError::Unrecoverable(_)) => {}
            Err(GpuError::LibraryError { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
