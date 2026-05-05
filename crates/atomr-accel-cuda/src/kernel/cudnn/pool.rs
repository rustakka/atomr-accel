//! Pooling requests (max / avg, fwd + bwd) for the cuDNN actor.
//!
//! Routes through `CUDNN_BACKEND_OPERATION_RESAMPLE_FWD/BWD_DESCRIPTOR`
//! in the v9 frontend graph.

#![allow(dead_code)]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dtype::CudnnSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::cudnn::conv::dtype_tag;
use crate::kernel::cudnn::graph::{
    DtypeTag, OpSpec, OperationGraphSpec, PoolKind, TensorLayout, TensorSpec,
};
use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};

/// Pooling op kind — `kind == PoolKind::*Bwd` selects the backward
/// pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolMode {
    Max,
    Avg,
    AvgExcludePadding,
}

impl PoolMode {
    pub fn fwd(self) -> PoolKind {
        match self {
            PoolMode::Max => PoolKind::MaxFwd,
            PoolMode::Avg | PoolMode::AvgExcludePadding => PoolKind::AvgFwd,
        }
    }
    pub fn bwd(self) -> PoolKind {
        match self {
            PoolMode::Max => PoolKind::MaxBwd,
            PoolMode::Avg | PoolMode::AvgExcludePadding => PoolKind::AvgBwd,
        }
    }
}

/// Pooling parameter struct, dim-generic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PoolParams {
    pub mode: PoolMode,
    /// Window per spatial dim.
    pub window: Vec<i64>,
    pub pre_padding: Vec<i64>,
    pub post_padding: Vec<i64>,
    pub stride: Vec<i64>,
}

impl PoolParams {
    /// 2D pooling helper.
    pub fn pool_2d(mode: PoolMode, kernel: i64, stride: i64, padding: i64) -> Self {
        Self {
            mode,
            window: vec![kernel, kernel],
            pre_padding: vec![padding, padding],
            post_padding: vec![padding, padding],
            stride: vec![stride, stride],
        }
    }
}

/// Forward pooling.
pub struct PoolFwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub x_dims: Vec<i64>,
    pub y_dims: Vec<i64>,
    pub layout: TensorLayout,
    pub params: PoolParams,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> PoolFwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_pool_fwd_graph(
            dtype_tag::<T>(),
            &self.x_dims,
            &self.y_dims,
            self.layout,
            &self.params,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for PoolFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "pool_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "PoolFwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Backward pooling.
pub struct PoolBwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub dy: GpuRef<T>,
    pub dx: GpuRef<T>,
    pub x_dims: Vec<i64>,
    pub y_dims: Vec<i64>,
    pub layout: TensorLayout,
    pub params: PoolParams,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> PoolBwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_pool_bwd_graph(
            dtype_tag::<T>(),
            &self.x_dims,
            &self.y_dims,
            self.layout,
            &self.params,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for PoolBwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "pool_bwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "PoolBwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

pub fn build_pool_fwd_graph(
    dtype: DtypeTag,
    x_dims: &[i64],
    y_dims: &[i64],
    layout: TensorLayout,
    p: &PoolParams,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("pool_fwd");
    let x_uid = g.add_tensor(TensorSpec::new(1, dtype, x_dims.to_vec(), layout));
    let y_uid = g.add_tensor(TensorSpec::new(2, dtype, y_dims.to_vec(), layout));
    g.add_op(OpSpec::PoolFwd {
        kind: p.mode.fwd(),
        x: x_uid,
        y: y_uid,
        window: p.window.clone(),
        pre_padding: p.pre_padding.clone(),
        post_padding: p.post_padding.clone(),
        stride: p.stride.clone(),
        compute_dtype: dtype,
    });
    g
}

pub fn build_pool_bwd_graph(
    dtype: DtypeTag,
    x_dims: &[i64],
    y_dims: &[i64],
    layout: TensorLayout,
    p: &PoolParams,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("pool_bwd");
    let x_uid = g.add_tensor(TensorSpec::new(1, dtype, x_dims.to_vec(), layout));
    let y_uid = g.add_tensor(TensorSpec::new(2, dtype, y_dims.to_vec(), layout));
    let dy_uid = g.add_tensor(TensorSpec::new(3, dtype, y_dims.to_vec(), layout));
    let dx_uid = g.add_tensor(TensorSpec::new(4, dtype, x_dims.to_vec(), layout));
    g.add_op(OpSpec::PoolBwd {
        kind: p.mode.bwd(),
        dy: dy_uid,
        x: x_uid,
        y: y_uid,
        dx: dx_uid,
        window: p.window.clone(),
        pre_padding: p.pre_padding.clone(),
        post_padding: p.post_padding.clone(),
        stride: p.stride.clone(),
        compute_dtype: dtype,
    });
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_fwd_bwd_round_trip() {
        let p = PoolParams::pool_2d(PoolMode::Max, 2, 2, 0);
        let g_fwd = build_pool_fwd_graph(
            DtypeTag::F32,
            &[1, 16, 8, 8],
            &[1, 16, 4, 4],
            TensorLayout::NchwPacked,
            &p,
        );
        match &g_fwd.ops[0] {
            OpSpec::PoolFwd { kind, .. } => assert_eq!(*kind, PoolKind::MaxFwd),
            _ => panic!("wrong op"),
        }
        let g_bwd = build_pool_bwd_graph(
            DtypeTag::F32,
            &[1, 16, 8, 8],
            &[1, 16, 4, 4],
            TensorLayout::NchwPacked,
            &p,
        );
        assert_eq!(g_bwd.tensors.len(), 4);
        match &g_bwd.ops[0] {
            OpSpec::PoolBwd { kind, .. } => assert_eq!(*kind, PoolKind::MaxBwd),
            _ => panic!("wrong op"),
        }

        // Avg pool round-trip.
        let avg = PoolParams::pool_2d(PoolMode::Avg, 2, 2, 0);
        let g = build_pool_fwd_graph(
            DtypeTag::F32,
            &[1, 16, 8, 8],
            &[1, 16, 4, 4],
            TensorLayout::NchwPacked,
            &avg,
        );
        match &g.ops[0] {
            OpSpec::PoolFwd { kind, .. } => assert_eq!(*kind, PoolKind::AvgFwd),
            _ => panic!("wrong op"),
        }
    }
}
