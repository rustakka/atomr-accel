//! Activation, dropout, and LRN requests for the cuDNN actor.
//!
//! Activation set: `Relu`, `Sigmoid`, `Tanh` (existing) plus `Gelu`,
//! `GeluApprox`, `Swish`, `Elu`, `Softplus`, `Identity`. cuDNN routes
//! these through pointwise descriptor ops in the v9 frontend graph.

#![allow(dead_code)]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dtype::CudnnSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::cudnn::conv::dtype_tag;
use crate::kernel::cudnn::graph::{
    DtypeTag, OpSpec, OperationGraphSpec, PointwiseMode, TensorLayout, TensorSpec,
};
use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};

/// Activation function tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationKind {
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    GeluApprox,
    Swish,
    Elu,
    Softplus,
    Identity,
}

impl ActivationKind {
    /// Map to the v9 frontend `PointwiseMode`.
    pub fn pointwise_mode(self) -> PointwiseMode {
        match self {
            ActivationKind::Relu => PointwiseMode::Relu,
            ActivationKind::Sigmoid => PointwiseMode::Sigmoid,
            ActivationKind::Tanh => PointwiseMode::Tanh,
            ActivationKind::Gelu => PointwiseMode::Gelu,
            ActivationKind::GeluApprox => PointwiseMode::GeluApprox,
            ActivationKind::Swish => PointwiseMode::Swish,
            ActivationKind::Elu => PointwiseMode::Elu,
            ActivationKind::Softplus => PointwiseMode::Softplus,
            ActivationKind::Identity => PointwiseMode::Identity,
        }
    }

    /// Map to the legacy v7 `cudnnActivationMode_t` for the back-compat
    /// dispatch path. Approximate / parametric activations fall back
    /// to the plain `Relu`/`Sigmoid` etc. equivalent.
    #[cfg(feature = "cudnn")]
    pub fn cudnn_legacy_mode(self) -> cudarc::cudnn::sys::cudnnActivationMode_t {
        use cudarc::cudnn::sys::cudnnActivationMode_t::*;
        match self {
            ActivationKind::Relu | ActivationKind::Identity => CUDNN_ACTIVATION_RELU,
            ActivationKind::Sigmoid => CUDNN_ACTIVATION_SIGMOID,
            ActivationKind::Tanh => CUDNN_ACTIVATION_TANH,
            ActivationKind::Elu => CUDNN_ACTIVATION_ELU,
            ActivationKind::Swish => CUDNN_ACTIVATION_SWISH,
            ActivationKind::Gelu | ActivationKind::GeluApprox => CUDNN_ACTIVATION_RELU,
            ActivationKind::Softplus => CUDNN_ACTIVATION_RELU,
        }
    }
}

/// Activation forward request. dims are the raw tensor dims.
pub struct ActivationFwdRequest<T: CudnnSupported> {
    pub kind: ActivationKind,
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> ActivationFwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        let dt = dtype_tag::<T>();
        let mut g = OperationGraphSpec::new("activation_fwd");
        let x_uid = g.add_tensor(TensorSpec::new(1, dt, self.dims.clone(), self.layout));
        let y_uid = g.add_tensor(TensorSpec::new(2, dt, self.dims.clone(), self.layout));
        g.add_op(OpSpec::Pointwise {
            mode: self.kind.pointwise_mode(),
            x: x_uid,
            b: None,
            y: y_uid,
            compute_dtype: dt,
            alpha1: 1.0,
            alpha2: 0.0,
        });
        g
    }
}

impl<T: CudnnSupported> CudnnDispatch for ActivationFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "activation_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "ActivationFwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Softmax mode (instance vs channel-wise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SoftmaxMode {
    Instance,
    Channel,
}

/// Softmax forward request.
pub struct SoftmaxFwdRequest<T: CudnnSupported> {
    pub mode: SoftmaxMode,
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> CudnnDispatch for SoftmaxFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "softmax_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "SoftmaxFwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Dropout forward request: produces `y = x * mask / (1 - p)` and
/// records the mask state for backward.
pub struct DropoutFwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub mask: GpuRef<u8>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub probability: f32,
    pub seed: u64,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> CudnnDispatch for DropoutFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "dropout_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "DropoutFwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Local-response-normalisation parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LrnParams {
    pub n: u32,
    pub alpha_milli: i64,
    pub beta_milli: i64,
    pub k_milli: i64,
}

impl LrnParams {
    pub fn new(n: u32, alpha: f64, beta: f64, k: f64) -> Self {
        Self {
            n,
            alpha_milli: (alpha * 1_000_000.0) as i64,
            beta_milli: (beta * 1_000_000.0) as i64,
            k_milli: (k * 1_000_000.0) as i64,
        }
    }
}

/// LRN forward request.
pub struct LrnFwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub dims: Vec<i64>,
    pub layout: TensorLayout,
    pub params: LrnParams,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> CudnnDispatch for LrnFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "lrn_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "LrnFwdRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Build the spec-side activation-fwd op graph.
pub fn build_activation_fwd_graph(
    dtype: DtypeTag,
    dims: &[i64],
    layout: TensorLayout,
    kind: ActivationKind,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("activation_fwd");
    let x_uid = g.add_tensor(TensorSpec::new(1, dtype, dims.to_vec(), layout));
    let y_uid = g.add_tensor(TensorSpec::new(2, dtype, dims.to_vec(), layout));
    g.add_op(OpSpec::Pointwise {
        mode: kind.pointwise_mode(),
        x: x_uid,
        b: None,
        y: y_uid,
        compute_dtype: dtype,
        alpha1: 1.0,
        alpha2: 0.0,
    });
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_kinds_have_pointwise_mode() {
        assert_eq!(ActivationKind::Relu.pointwise_mode(), PointwiseMode::Relu);
        assert_eq!(ActivationKind::Gelu.pointwise_mode(), PointwiseMode::Gelu);
        assert_eq!(ActivationKind::Swish.pointwise_mode(), PointwiseMode::Swish);
        assert_eq!(
            ActivationKind::Softplus.pointwise_mode(),
            PointwiseMode::Softplus
        );
        assert_eq!(ActivationKind::Elu.pointwise_mode(), PointwiseMode::Elu);
        assert_eq!(
            ActivationKind::Identity.pointwise_mode(),
            PointwiseMode::Identity
        );
    }

    #[test]
    fn activation_fwd_graph_builds() {
        let g = build_activation_fwd_graph(
            DtypeTag::F32,
            &[1, 3, 8, 8],
            TensorLayout::NchwPacked,
            ActivationKind::Gelu,
        );
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.ops.len(), 1);
    }

    #[test]
    fn lrn_params_quantization() {
        let p = LrnParams::new(5, 0.0001, 0.75, 1.0);
        assert_eq!(p.n, 5);
        assert_eq!(p.alpha_milli, 100);
        assert_eq!(p.beta_milli, 750_000);
    }
}
