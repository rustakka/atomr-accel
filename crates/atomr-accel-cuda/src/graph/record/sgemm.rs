//! SGEMM op for [`super::super::GraphOp`].
//!
//! Wraps the lower-level [`crate::kernel::record::BlasSgemmOp`] /
//! [`crate::kernel::record::BlasRecorder`] pair into a single
//! `GraphOp`-implementing type.

use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::graph::{GraphOp, GraphRecordCtx};
use crate::kernel::record::{BlasRecorder, BlasSgemmOp, RecordMode};

/// SGEMM op for graph capture: `C := alpha · A·B + beta · C`,
/// column-major, no transpose.
///
/// The op needs a cuBLAS handle on the captured stream, supplied
/// by `GraphRecordCtx::blas`. If absent the op fails with
/// [`GpuError::Unrecoverable`].
///
/// `record` consumes the held `GpuRef`s on first invocation
/// (matching the pre-trait closed-enum semantics where the boxed
/// op was destructured by-move). A second `record` call on the
/// same op returns [`GpuError::Unrecoverable`].
pub struct SgemmOp {
    inner: Option<BlasSgemmOp>,
}

impl SgemmOp {
    pub fn new(
        a: GpuRef<f32>,
        b: GpuRef<f32>,
        c: GpuRef<f32>,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        beta: f32,
    ) -> Self {
        Self {
            inner: Some(BlasSgemmOp {
                a,
                b,
                c,
                m,
                n,
                k,
                alpha,
                beta,
            }),
        }
    }
}

impl GraphOp for SgemmOp {
    fn record(&mut self, ctx: &mut GraphRecordCtx<'_>) -> Result<(), GpuError> {
        let stream = ctx.require_stream()?;
        let blas = ctx.blas.ok_or_else(|| {
            GpuError::Unrecoverable("SgemmOp::record: cuBLAS handle not available in ctx".into())
        })?;
        let op = self
            .inner
            .take()
            .ok_or_else(|| GpuError::Unrecoverable("SgemmOp::record: already consumed".into()))?;
        let mut recorder = BlasRecorder { handle: blas };
        recorder.enqueue_record(stream, op)
    }

    fn op_name(&self) -> &'static str {
        "graph::sgemm"
    }
}
