//! Normalisation requests for the cuDNN actor: BatchNorm (training +
//! inference + persistent), LayerNorm, InstanceNorm, GroupNorm,
//! RMSNorm.

#![allow(dead_code)]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dtype::CudnnSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::cudnn::conv::dtype_tag;
use crate::kernel::cudnn::graph::{
    DtypeTag, NormMode, NormPhase, OpSpec, OperationGraphSpec, TensorLayout, TensorSpec,
};
use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};

/// BatchNorm — training-mode running-mean/var update + per-channel
/// scale + bias.
pub struct BatchNormRequest<T: CudnnSupported> {
    pub phase: NormPhase,
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub scale: GpuRef<T>,
    pub bias: GpuRef<T>,
    pub running_mean: Option<GpuRef<T>>,
    pub running_var: Option<GpuRef<T>>,
    pub saved_mean: Option<GpuRef<T>>,
    pub saved_var: Option<GpuRef<T>>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub epsilon: f64,
    pub exp_avg_factor: f64,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> BatchNormRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_norm_fwd_graph(
            NormMode::BatchNorm,
            self.phase,
            dtype_tag::<T>(),
            &self.dims,
            self.layout,
            self.epsilon,
            self.exp_avg_factor,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for BatchNormRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "batchnorm"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "BatchNormRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// LayerNorm — normalises across the trailing axes, scale + bias
/// applied per-feature.
pub struct LayerNormRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub scale: GpuRef<T>,
    pub bias: GpuRef<T>,
    pub saved_mean: Option<GpuRef<T>>,
    pub saved_var: Option<GpuRef<T>>,
    pub dims: Vec<i64>,
    pub norm_axes: Vec<i64>,
    pub layout: TensorLayout,
    pub epsilon: f64,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> LayerNormRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_norm_fwd_graph(
            NormMode::LayerNorm,
            NormPhase::Training,
            dtype_tag::<T>(),
            &self.dims,
            self.layout,
            self.epsilon,
            0.0,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for LayerNormRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "layernorm"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "LayerNormRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// InstanceNorm — normalises per-sample, per-channel.
pub struct InstanceNormRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub scale: GpuRef<T>,
    pub bias: GpuRef<T>,
    pub saved_mean: Option<GpuRef<T>>,
    pub saved_var: Option<GpuRef<T>>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub epsilon: f64,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> InstanceNormRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_norm_fwd_graph(
            NormMode::InstanceNorm,
            NormPhase::Training,
            dtype_tag::<T>(),
            &self.dims,
            self.layout,
            self.epsilon,
            0.0,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for InstanceNormRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "instancenorm"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "InstanceNormRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// GroupNorm — generalisation of BatchNorm/LayerNorm/InstanceNorm
/// with `groups` channel partitions.
pub struct GroupNormRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub scale: GpuRef<T>,
    pub bias: GpuRef<T>,
    pub saved_mean: Option<GpuRef<T>>,
    pub saved_var: Option<GpuRef<T>>,
    pub dims: Vec<i64>,
    pub groups: u32,
    pub layout: TensorLayout,
    pub epsilon: f64,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> CudnnDispatch for GroupNormRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "groupnorm"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "GroupNormRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Norm backward request, applies to BatchNorm / LayerNorm /
/// InstanceNorm / GroupNorm uniformly via `mode`.
pub struct NormBwdRequest<T: CudnnSupported> {
    pub mode: NormMode,
    pub x: GpuRef<T>,
    pub dy: GpuRef<T>,
    pub scale: GpuRef<T>,
    pub mean: GpuRef<T>,
    pub var: GpuRef<T>,
    pub dx: GpuRef<T>,
    pub dscale: GpuRef<T>,
    pub dbias: GpuRef<T>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub epsilon: f64,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> CudnnDispatch for NormBwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        match self.mode {
            NormMode::BatchNorm => "batchnorm_bwd",
            NormMode::LayerNorm => "layernorm_bwd",
            NormMode::InstanceNorm => "instancenorm_bwd",
            NormMode::GroupNorm => "groupnorm_bwd",
            NormMode::RmsNorm => "rmsnorm_bwd",
        }
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "NormBwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Build the spec-side norm-fwd op graph.
pub fn build_norm_fwd_graph(
    mode: NormMode,
    phase: NormPhase,
    dtype: DtypeTag,
    dims: &[i64],
    layout: TensorLayout,
    epsilon: f64,
    exp_avg_factor: f64,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("norm_fwd");
    let x_uid = g.add_tensor(TensorSpec::new(1, dtype, dims.to_vec(), layout));
    let scale_uid = g.add_tensor(TensorSpec::new(2, dtype, vec![1, dims[1], 1, 1], layout));
    let bias_uid = g.add_tensor(TensorSpec::new(3, dtype, vec![1, dims[1], 1, 1], layout));
    let y_uid = g.add_tensor(TensorSpec::new(4, dtype, dims.to_vec(), layout));
    g.add_op(OpSpec::NormFwd {
        mode,
        phase,
        x: x_uid,
        scale: scale_uid,
        bias: bias_uid,
        mean: None,
        var: None,
        y: y_uid,
        compute_dtype: dtype,
        epsilon,
        exp_avg_factor,
    });
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm_layernorm_instancenorm_round_trip() {
        let bn = build_norm_fwd_graph(
            NormMode::BatchNorm,
            NormPhase::Training,
            DtypeTag::F32,
            &[2, 3, 4, 4],
            TensorLayout::NchwPacked,
            1e-5,
            0.1,
        );
        assert_eq!(bn.ops.len(), 1);
        match &bn.ops[0] {
            OpSpec::NormFwd { mode, phase, .. } => {
                assert_eq!(*mode, NormMode::BatchNorm);
                assert_eq!(*phase, NormPhase::Training);
            }
            _ => panic!("wrong op"),
        }

        let ln = build_norm_fwd_graph(
            NormMode::LayerNorm,
            NormPhase::Training,
            DtypeTag::F32,
            &[2, 3, 4, 4],
            TensorLayout::NchwPacked,
            1e-5,
            0.0,
        );
        assert_ne!(bn.signature(), ln.signature());

        let in_ = build_norm_fwd_graph(
            NormMode::InstanceNorm,
            NormPhase::Training,
            DtypeTag::F32,
            &[2, 3, 4, 4],
            TensorLayout::NchwPacked,
            1e-5,
            0.0,
        );
        assert_ne!(ln.signature(), in_.signature());

        // Persistent batchnorm has its own phase signature.
        let bn_persist = build_norm_fwd_graph(
            NormMode::BatchNorm,
            NormPhase::PersistentTraining,
            DtypeTag::F32,
            &[2, 3, 4, 4],
            TensorLayout::NchwPacked,
            1e-5,
            0.1,
        );
        assert_ne!(bn.signature(), bn_persist.signature());
    }
}
