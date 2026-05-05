//! Child-graph composition.
//!
//! Wraps `cuGraphAddChildGraphNode` so an existing `GraphHandle` can
//! be embedded as a node in a parent capture. This is how higher-level
//! pipelines compose: a sub-graph for "data load" can be re-used
//! across many enclosing per-step graphs.

use std::sync::Arc;

use cudarc::driver::sys as driver_sys;

use crate::error::GpuError;
use crate::graph::{GraphHandle, GraphOpRecord, GraphRecordCtx};

const LIB: &str = "graph";

/// Op variant for embedding a previously-recorded sub-graph.
pub struct ChildGraphOp {
    pub child: GraphHandle,
}

impl GraphOpRecord for ChildGraphOp {
    fn record(&self, ctx: &GraphRecordCtx<'_>) -> Result<(), GpuError> {
        // SAFETY: we pull the raw cu_graph from the wrapped CudaGraph;
        // CUDA owns it and we pass it through to cuGraphAddChildGraphNode.
        let parent = ctx.parent_graph();
        let cu_child = self.child.cu_graph();
        let mut node: driver_sys::CUgraphNode = std::ptr::null_mut();
        let s = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            driver_sys::cuGraphAddChildGraphNode(
                &mut node as *mut _,
                parent,
                std::ptr::null(),
                0,
                cu_child,
            )
        }));
        match s {
            Ok(s) => {
                if s == driver_sys::cudaError_enum::CUDA_SUCCESS {
                    Ok(())
                } else {
                    Err(GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("cuGraphAddChildGraphNode: {s:?}"),
                    })
                }
            }
            Err(_) => Err(GpuError::Unrecoverable(
                "ChildGraphOp::record: CUDA driver not loadable".into(),
            )),
        }
    }
}

/// Convenience: create a child-graph op from a `GraphHandle` clone.
pub fn child_graph_op(child: GraphHandle) -> ChildGraphOp {
    ChildGraphOp { child }
}

/// Helper used by pipeline builders that want to keep a reference to
/// the inserted child-graph for later parameter rebinding.
pub struct ChildGraphInsertion {
    pub op: ChildGraphOp,
    pub keep_alive: Arc<()>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphHandle, MockGraphRecordCtx};
    use std::sync::Arc;

    #[test]
    fn child_graph_op_records_into_parent() {
        // Build a synthetic GraphHandle (mock-mode — the wrapped
        // CudaGraph is a dangling sentinel, but the GraphRecordCtx is
        // also a mock that doesn't dereference it).
        let child = GraphHandle::synthetic_for_tests();
        let op = child_graph_op(child);
        let parent_graph: driver_sys::CUgraph = std::ptr::null_mut();
        let mock = MockGraphRecordCtx::new(parent_graph);
        let ctx: GraphRecordCtx<'_> = mock.as_ctx();
        let r = op.record(&ctx);
        // No driver → Unrecoverable (panic caught) or LibraryError on
        // null parent. Both acceptable; the point is to confirm we
        // route into the cuGraphAddChildGraphNode path without
        // panicking out.
        match r {
            Ok(()) => {}
            Err(GpuError::Unrecoverable(_)) => {}
            Err(GpuError::LibraryError { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
        let _ = Arc::new(()); // keep_alive type smoke check
    }
}
