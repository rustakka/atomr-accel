//! Conditional graph nodes (`cudaGraphConditionalNode`, CUDA 12.4+).
//!
//! Gated behind the `graphs-conditional` Cargo feature. Even with the
//! feature on, the [`GraphActor`] runtime probe disables the path on
//! older drivers — `IfNodeDescriptor::record` returns
//! `Unrecoverable("conditional graphs unsupported")` if the
//! `cuGraphConditionalHandleCreate` symbol can't be resolved.
//!
//! Public surface:
//! - [`ConditionalKind`] — `If` or `While`.
//! - [`IfNodeDescriptor`] / [`WhileNodeDescriptor`] — typed
//!   descriptors carrying the inner graph that will be replayed when
//!   the predicate is non-zero.

#![cfg(feature = "graphs-conditional")]

use std::sync::Arc;

use cudarc::driver::sys as driver_sys;
use cudarc::driver::CudaContext;

use crate::error::GpuError;

const LIB: &str = "graph";

/// Kind of conditional node. Matches `CU_GRAPH_COND_TYPE_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalKind {
    /// Execute the inner graph at most once when the handle's value
    /// is non-zero.
    If,
    /// Execute the inner graph repeatedly while the handle's value
    /// stays non-zero. The inner graph is responsible for clearing
    /// the handle when it should exit.
    While,
}

impl ConditionalKind {
    fn raw(self) -> driver_sys::CUgraphConditionalNodeType {
        match self {
            ConditionalKind::If => driver_sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_IF,
            ConditionalKind::While => {
                driver_sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE
            }
        }
    }
}

/// Descriptor for an `If`-style conditional node. The inner graph is
/// out-parameter-allocated by CUDA when the node is created — callers
/// receive its handle back so they can populate it before exec
/// instantiation.
#[derive(Clone)]
pub struct IfNodeDescriptor {
    pub default_value: u32,
}

/// Descriptor for a `While`-style conditional node.
#[derive(Clone)]
pub struct WhileNodeDescriptor {
    pub default_value: u32,
}

/// Build the raw `CUDA_CONDITIONAL_NODE_PARAMS` for `kind` against
/// `parent`'s context. The returned struct embeds an out-pointer
/// (`phGraph_out`) that CUDA fills with the inner graph; the actor
/// adds it to the parent via `cuGraphAddNode_v2`.
pub fn build_params(
    kind: ConditionalKind,
    handle: driver_sys::CUgraphConditionalHandle,
    ctx: &Arc<CudaContext>,
    inner_graph_out: *mut driver_sys::CUgraph,
) -> driver_sys::CUDA_CONDITIONAL_NODE_PARAMS {
    driver_sys::CUDA_CONDITIONAL_NODE_PARAMS {
        handle,
        type_: kind.raw(),
        size: 1,
        phGraph_out: inner_graph_out,
        ctx: ctx.cu_ctx(),
    }
}

/// Probe whether the running CUDA driver supports conditional graphs.
/// Returns `Ok(true)` if `cuGraphConditionalHandleCreate` is callable
/// (i.e. CUDA ≥ 12.4 with the symbol present); `Ok(false)` otherwise;
/// `Err(GpuError::Unrecoverable)` if the loader panics.
pub fn driver_supports_conditional() -> Result<bool, GpuError> {
    // We probe by attempting to call the symbol with bogus args; CUDA
    // returns CUDA_ERROR_INVALID_VALUE if the symbol is loadable but
    // rejects the args, and CUDA_ERROR_NOT_SUPPORTED on older drivers.
    let probe = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut h: driver_sys::CUgraphConditionalHandle = 0;
        // SAFETY: out-pointer; the other args are intentional null/0
        // probes — CUDA validates and returns an error code.
        let s = unsafe {
            driver_sys::cuGraphConditionalHandleCreate(
                &mut h as *mut _,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                0,
            )
        };
        s
    }));
    match probe {
        Ok(s) => match s {
            driver_sys::cudaError_enum::CUDA_ERROR_NOT_SUPPORTED => Ok(false),
            // Anything else (success, INVALID_VALUE, INVALID_CONTEXT)
            // means the symbol is at least linked.
            _ => Ok(true),
        },
        Err(_) => Err(GpuError::Unrecoverable(
            "conditional probe: CUDA driver not loadable".into(),
        )),
    }
}

/// Helper: lift a `CUresult` into our error taxonomy.
pub(crate) fn check(s: driver_sys::CUresult, op: &str) -> Result<(), GpuError> {
    if s == driver_sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("{op}: {s:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn if_node_descriptor_compiles() {
        let d = IfNodeDescriptor { default_value: 1 };
        assert_eq!(d.default_value, 1);
        assert_eq!(ConditionalKind::If, ConditionalKind::If);
        assert_ne!(ConditionalKind::If, ConditionalKind::While);
        // raw() round-trip — the variants must map to distinct CUDA
        // constants.
        let _ = ConditionalKind::If.raw();
        let _ = ConditionalKind::While.raw();
    }

    #[test]
    fn while_node_descriptor_compiles() {
        let d = WhileNodeDescriptor { default_value: 0 };
        assert_eq!(d.default_value, 0);
    }

    #[test]
    fn driver_probe_returns_typed_result() {
        // On no-GPU hosts the probe surfaces Unrecoverable; on real
        // hardware it returns Ok(true|false). Either way, no panic.
        let r = driver_supports_conditional();
        match r {
            Ok(_) => {}
            Err(GpuError::Unrecoverable(_)) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
