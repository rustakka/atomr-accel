//! `Cudnn` — Python handle wrapping `ActorRef<CudnnMsg>`.
//!
//! Obtained via `Device.cudnn()` (only when the `cudnn` feature is
//! compiled in *and* the device's `EnabledLibraries::CUDNN` flag is
//! set). Phase 1.5 ships method-level breadth across the typed
//! `CudnnMsg::Op` request types — conv / pool / norm / activation /
//! softmax / dropout / lrn forward (plus selected backward + RNN +
//! attention surfaces). All calls work in mock mode (the underlying
//! actor drops the boxed reply senders, surfacing as
//! `cudnn dropped reply`).

#![cfg(feature = "cudnn")]

use std::marker::PhantomData;
use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{
    ActivationFwdRequest, ActivationKind, AttentionMask, AttentionParams, BatchNormRequest,
    ConvBwdDataRequest, ConvBwdFilterRequest, ConvDescParams, ConvFwdRequest, CudnnMsg,
    DropoutFwdRequest, EpilogueKind, GroupNormRequest, InstanceNormRequest, LayerNormRequest,
    LrnFwdRequest, LrnParams, MultiHeadAttnFwdRequest, NormBwdRequest, NormMode, NormPhase,
    PoolBwdRequest, PoolFwdRequest, PoolMode, PoolParams, RnnDirection, RnnFwdRequest, RnnMode,
    RnnParams, SoftmaxFwdRequest, SoftmaxMode, TensorLayout,
};
use atomr_core::actor::ActorRef;

use crate::buffer::{PyGpuBufferF32, PyGpuBufferU8};
use crate::errors;
use crate::runtime::runtime;

#[pyclass(name = "Cudnn", module = "atomr_accel._native")]
pub struct PyCudnn {
    actor_ref: ActorRef<CudnnMsg>,
}

impl PyCudnn {
    pub fn new(actor_ref: ActorRef<CudnnMsg>) -> Self {
        Self { actor_ref }
    }
}

// ----- helpers: string-keyed enum mappings -----------------------------

fn pool_mode_from_str(s: &str) -> PyResult<PoolMode> {
    match s {
        "max" | "Max" => Ok(PoolMode::Max),
        "avg" | "average" | "Avg" | "AverageInclude" | "average_include" => Ok(PoolMode::Avg),
        "avg_exclude_padding" | "AvgExcludePadding" | "AverageExclude" | "average_exclude" => {
            Ok(PoolMode::AvgExcludePadding)
        }
        other => Err(errors::map_str(format!("unknown pool mode: {other}"))),
    }
}

fn softmax_mode_from_str(s: &str) -> PyResult<SoftmaxMode> {
    match s {
        "channel" | "Channel" => Ok(SoftmaxMode::Channel),
        "instance" | "Instance" => Ok(SoftmaxMode::Instance),
        other => Err(errors::map_str(format!("unknown softmax mode: {other}"))),
    }
}

fn activation_kind_from_str(s: &str) -> PyResult<ActivationKind> {
    match s {
        "relu" | "Relu" => Ok(ActivationKind::Relu),
        "sigmoid" | "Sigmoid" => Ok(ActivationKind::Sigmoid),
        "tanh" | "Tanh" => Ok(ActivationKind::Tanh),
        "gelu" | "Gelu" => Ok(ActivationKind::Gelu),
        "gelu_approx" | "GeluApprox" => Ok(ActivationKind::GeluApprox),
        "swish" | "Swish" | "silu" => Ok(ActivationKind::Swish),
        "elu" | "Elu" => Ok(ActivationKind::Elu),
        "softplus" | "Softplus" => Ok(ActivationKind::Softplus),
        "identity" | "Identity" => Ok(ActivationKind::Identity),
        other => Err(errors::map_str(format!("unknown activation kind: {other}"))),
    }
}

fn norm_phase_from_str(s: &str) -> PyResult<NormPhase> {
    match s {
        "train" | "training" | "Training" => Ok(NormPhase::Training),
        "inference" | "Inference" | "eval" => Ok(NormPhase::Inference),
        "persistent" | "PersistentTraining" | "persistent_training" => {
            Ok(NormPhase::PersistentTraining)
        }
        other => Err(errors::map_str(format!("unknown norm phase: {other}"))),
    }
}

fn norm_mode_from_str(s: &str) -> PyResult<NormMode> {
    match s {
        "batch" | "BatchNorm" | "batch_norm" => Ok(NormMode::BatchNorm),
        "layer" | "LayerNorm" | "layer_norm" => Ok(NormMode::LayerNorm),
        "instance" | "InstanceNorm" | "instance_norm" => Ok(NormMode::InstanceNorm),
        "group" | "GroupNorm" | "group_norm" => Ok(NormMode::GroupNorm),
        "rms" | "RmsNorm" | "rms_norm" => Ok(NormMode::RmsNorm),
        other => Err(errors::map_str(format!("unknown norm mode: {other}"))),
    }
}

fn rnn_mode_from_str(s: &str) -> PyResult<RnnMode> {
    match s {
        "rnn" | "Rnn" | "rnn_relu" => Ok(RnnMode::Rnn),
        "rnn_tanh" | "RnnTanh" => Ok(RnnMode::RnnTanh),
        "lstm" | "Lstm" | "LSTM" => Ok(RnnMode::Lstm),
        "gru" | "Gru" | "GRU" => Ok(RnnMode::Gru),
        other => Err(errors::map_str(format!("unknown rnn mode: {other}"))),
    }
}

fn rnn_direction_from_str(s: &str) -> PyResult<RnnDirection> {
    match s {
        "uni" | "unidirectional" | "Unidirectional" => Ok(RnnDirection::Unidirectional),
        "bi" | "bidirectional" | "Bidirectional" => Ok(RnnDirection::Bidirectional),
        other => Err(errors::map_str(format!("unknown rnn direction: {other}"))),
    }
}

fn attention_mask_from_str(s: &str, window: u32) -> PyResult<AttentionMask> {
    match s {
        "none" | "None" => Ok(AttentionMask::None),
        "causal" | "Causal" => Ok(AttentionMask::Causal),
        "sliding_window" | "SlidingWindow" => Ok(AttentionMask::SlidingWindow(window)),
        "causal_sliding_window" | "CausalSlidingWindow" => {
            Ok(AttentionMask::CausalSlidingWindow(window))
        }
        other => Err(errors::map_str(format!("unknown attention mask: {other}"))),
    }
}

#[pymethods]
impl PyCudnn {
    /// 2-D forward convolution, f32 NCHW. Shapes are `(N, C, H, W)`
    /// for `x` and `y`, `(K, C, R, S)` for `w`. `pad`, `stride`,
    /// `dilation` are `(h, w)` pairs. Returns nothing on success;
    /// the result is in `y`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, w, y,
        x_shape, w_shape, y_shape,
        pad=(0, 0), stride=(1, 1), dilation=(1, 1),
        groups=1,
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn conv2d_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        w: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        x_shape: (i64, i64, i64, i64),
        w_shape: (i64, i64, i64, i64),
        y_shape: (i64, i64, i64, i64),
        pad: (i64, i64),
        stride: (i64, i64),
        dilation: (i64, i64),
        groups: i64,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let w = w
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("w consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;

        let conv = ConvDescParams {
            spatial_dims: 2,
            pre_padding: vec![pad.0, pad.1],
            post_padding: vec![pad.0, pad.1],
            stride: vec![stride.0, stride.1],
            dilation: vec![dilation.0, dilation.1],
            groups,
        };
        let x_dims = vec![x_shape.0, x_shape.1, x_shape.2, x_shape.3];
        let w_dims = vec![w_shape.0, w_shape.1, w_shape.2, w_shape.3];
        let y_dims = vec![y_shape.0, y_shape.1, y_shape.2, y_shape.3];
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ConvFwdRequest::<f32> {
                    x,
                    x_dims,
                    w,
                    w_dims,
                    y,
                    y_dims,
                    bias: None,
                    conv,
                    layout: TensorLayout::NchwPacked,
                    epilogue: EpilogueKind::None,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("conv2d_fwd_f32 timed out")),
                }
            })
        })
    }

    /// 2-D backward-data convolution (`dx = conv_bwd_data(w, dy)`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        dy, w, dx,
        dy_shape, w_shape, dx_shape,
        pad=(0, 0), stride=(1, 1), dilation=(1, 1),
        groups=1,
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn conv2d_bwd_data_f32(
        &self,
        py: Python<'_>,
        dy: Py<PyGpuBufferF32>,
        w: Py<PyGpuBufferF32>,
        dx: Py<PyGpuBufferF32>,
        dy_shape: (i64, i64, i64, i64),
        w_shape: (i64, i64, i64, i64),
        dx_shape: (i64, i64, i64, i64),
        pad: (i64, i64),
        stride: (i64, i64),
        dilation: (i64, i64),
        groups: i64,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let dy = dy
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dy consumed"))?;
        let w = w
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("w consumed"))?;
        let dx = dx
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dx consumed"))?;

        let conv = ConvDescParams {
            spatial_dims: 2,
            pre_padding: vec![pad.0, pad.1],
            post_padding: vec![pad.0, pad.1],
            stride: vec![stride.0, stride.1],
            dilation: vec![dilation.0, dilation.1],
            groups,
        };
        let dy_dims = vec![dy_shape.0, dy_shape.1, dy_shape.2, dy_shape.3];
        let w_dims = vec![w_shape.0, w_shape.1, w_shape.2, w_shape.3];
        let dx_dims = vec![dx_shape.0, dx_shape.1, dx_shape.2, dx_shape.3];
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ConvBwdDataRequest::<f32> {
                    dy,
                    dy_dims,
                    w,
                    w_dims,
                    dx,
                    dx_dims,
                    conv,
                    layout: TensorLayout::NchwPacked,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("conv2d_bwd_data_f32 timed out")),
                }
            })
        })
    }

    /// 2-D backward-filter convolution (`dw = conv_bwd_filter(x, dy)`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, dy, dw,
        x_shape, dy_shape, dw_shape,
        pad=(0, 0), stride=(1, 1), dilation=(1, 1),
        groups=1,
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn conv2d_bwd_filter_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        dy: Py<PyGpuBufferF32>,
        dw: Py<PyGpuBufferF32>,
        x_shape: (i64, i64, i64, i64),
        dy_shape: (i64, i64, i64, i64),
        dw_shape: (i64, i64, i64, i64),
        pad: (i64, i64),
        stride: (i64, i64),
        dilation: (i64, i64),
        groups: i64,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let dy = dy
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dy consumed"))?;
        let dw = dw
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dw consumed"))?;

        let conv = ConvDescParams {
            spatial_dims: 2,
            pre_padding: vec![pad.0, pad.1],
            post_padding: vec![pad.0, pad.1],
            stride: vec![stride.0, stride.1],
            dilation: vec![dilation.0, dilation.1],
            groups,
        };
        let x_dims = vec![x_shape.0, x_shape.1, x_shape.2, x_shape.3];
        let dy_dims = vec![dy_shape.0, dy_shape.1, dy_shape.2, dy_shape.3];
        let dw_dims = vec![dw_shape.0, dw_shape.1, dw_shape.2, dw_shape.3];
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ConvBwdFilterRequest::<f32> {
                    x,
                    x_dims,
                    dy,
                    dy_dims,
                    dw,
                    dw_dims,
                    conv,
                    layout: TensorLayout::NchwPacked,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("conv2d_bwd_filter_f32 timed out")),
                }
            })
        })
    }

    /// 2-D forward pooling (max / avg), f32 NCHW.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y,
        x_shape, y_shape,
        kernel=(2, 2), stride=(2, 2), pad=(0, 0),
        mode="max",
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn pool2d_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        x_shape: (i64, i64, i64, i64),
        y_shape: (i64, i64, i64, i64),
        kernel: (i64, i64),
        stride: (i64, i64),
        pad: (i64, i64),
        mode: &str,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let mode = pool_mode_from_str(mode)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;

        let params = PoolParams {
            mode,
            window: vec![kernel.0, kernel.1],
            pre_padding: vec![pad.0, pad.1],
            post_padding: vec![pad.0, pad.1],
            stride: vec![stride.0, stride.1],
        };
        let x_dims = vec![x_shape.0, x_shape.1, x_shape.2, x_shape.3];
        let y_dims = vec![y_shape.0, y_shape.1, y_shape.2, y_shape.3];
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = PoolFwdRequest::<f32> {
                    x,
                    y,
                    x_dims,
                    y_dims,
                    layout: TensorLayout::NchwPacked,
                    params,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("pool2d_fwd_f32 timed out")),
                }
            })
        })
    }

    /// 2-D backward pooling.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, dy, dx,
        x_shape, y_shape,
        kernel=(2, 2), stride=(2, 2), pad=(0, 0),
        mode="max",
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn pool2d_bwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        dy: Py<PyGpuBufferF32>,
        dx: Py<PyGpuBufferF32>,
        x_shape: (i64, i64, i64, i64),
        y_shape: (i64, i64, i64, i64),
        kernel: (i64, i64),
        stride: (i64, i64),
        pad: (i64, i64),
        mode: &str,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let mode = pool_mode_from_str(mode)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let dy = dy
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dy consumed"))?;
        let dx = dx
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dx consumed"))?;

        let params = PoolParams {
            mode,
            window: vec![kernel.0, kernel.1],
            pre_padding: vec![pad.0, pad.1],
            post_padding: vec![pad.0, pad.1],
            stride: vec![stride.0, stride.1],
        };
        let x_dims = vec![x_shape.0, x_shape.1, x_shape.2, x_shape.3];
        let y_dims = vec![y_shape.0, y_shape.1, y_shape.2, y_shape.3];
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = PoolBwdRequest::<f32> {
                    x,
                    y,
                    dy,
                    dx,
                    x_dims,
                    y_dims,
                    layout: TensorLayout::NchwPacked,
                    params,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("pool2d_bwd_f32 timed out")),
                }
            })
        })
    }

    /// Forward softmax. `mode='channel'` normalises across the channel
    /// dim (NCHW); `mode='instance'` normalises across all non-batch
    /// dims.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y,
        dims,
        mode="channel",
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn softmax_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        mode: &str,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let mode = softmax_mode_from_str(mode)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = SoftmaxFwdRequest::<f32> {
                    mode,
                    x,
                    y,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("softmax_fwd_f32 timed out")),
                }
            })
        })
    }

    /// Forward activation: relu / sigmoid / tanh / gelu / gelu_approx
    /// / swish (silu) / elu / softplus / identity.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y,
        dims,
        kind="relu",
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn activation_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        kind: &str,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let kind = activation_kind_from_str(kind)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = ActivationFwdRequest::<f32> {
                    kind,
                    x,
                    y,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("activation_fwd_f32 timed out")),
                }
            })
        })
    }

    /// Forward batch normalisation (training or inference). Running
    /// mean/var and saved mean/var buffers are optional; pass `None`
    /// to skip them.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, scale, bias,
        dims,
        running_mean=None, running_var=None,
        saved_mean=None, saved_var=None,
        phase="train",
        epsilon=1e-5, exp_avg_factor=0.1,
        timeout_secs=60.0,
    ))]
    fn batch_norm_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        scale: Py<PyGpuBufferF32>,
        bias: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        running_mean: Option<Py<PyGpuBufferF32>>,
        running_var: Option<Py<PyGpuBufferF32>>,
        saved_mean: Option<Py<PyGpuBufferF32>>,
        saved_var: Option<Py<PyGpuBufferF32>>,
        phase: &str,
        epsilon: f64,
        exp_avg_factor: f64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let phase = norm_phase_from_str(phase)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let scale = scale
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("scale consumed"))?;
        let bias = bias
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("bias consumed"))?;
        let running_mean = running_mean
            .map(|b| b.borrow(py).clone_ref().ok_or("running_mean consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let running_var = running_var
            .map(|b| b.borrow(py).clone_ref().ok_or("running_var consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let saved_mean = saved_mean
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_mean consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let saved_var = saved_var
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_var consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = BatchNormRequest::<f32> {
                    phase,
                    x,
                    y,
                    scale,
                    bias,
                    running_mean,
                    running_var,
                    saved_mean,
                    saved_var,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    epsilon,
                    exp_avg_factor,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("batch_norm_f32 timed out")),
                }
            })
        })
    }

    /// Forward layer normalisation. `norm_axes` lists the trailing
    /// dim indices to normalise over.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, scale, bias,
        dims,
        norm_axes,
        saved_mean=None, saved_var=None,
        epsilon=1e-5,
        timeout_secs=60.0,
    ))]
    fn layer_norm_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        scale: Py<PyGpuBufferF32>,
        bias: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        norm_axes: Vec<i64>,
        saved_mean: Option<Py<PyGpuBufferF32>>,
        saved_var: Option<Py<PyGpuBufferF32>>,
        epsilon: f64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let scale = scale
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("scale consumed"))?;
        let bias = bias
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("bias consumed"))?;
        let saved_mean = saved_mean
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_mean consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let saved_var = saved_var
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_var consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = LayerNormRequest::<f32> {
                    x,
                    y,
                    scale,
                    bias,
                    saved_mean,
                    saved_var,
                    dims,
                    norm_axes,
                    layout: TensorLayout::NchwPacked,
                    epsilon,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("layer_norm_f32 timed out")),
                }
            })
        })
    }

    /// Forward instance normalisation (per-sample, per-channel).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, scale, bias,
        dims,
        saved_mean=None, saved_var=None,
        epsilon=1e-5,
        timeout_secs=60.0,
    ))]
    fn instance_norm_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        scale: Py<PyGpuBufferF32>,
        bias: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        saved_mean: Option<Py<PyGpuBufferF32>>,
        saved_var: Option<Py<PyGpuBufferF32>>,
        epsilon: f64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let scale = scale
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("scale consumed"))?;
        let bias = bias
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("bias consumed"))?;
        let saved_mean = saved_mean
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_mean consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let saved_var = saved_var
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_var consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = InstanceNormRequest::<f32> {
                    x,
                    y,
                    scale,
                    bias,
                    saved_mean,
                    saved_var,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    epsilon,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("instance_norm_f32 timed out")),
                }
            })
        })
    }

    /// Forward group normalisation (channel-group partitioning).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, scale, bias,
        dims,
        groups,
        saved_mean=None, saved_var=None,
        epsilon=1e-5,
        timeout_secs=60.0,
    ))]
    fn group_norm_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        scale: Py<PyGpuBufferF32>,
        bias: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        groups: u32,
        saved_mean: Option<Py<PyGpuBufferF32>>,
        saved_var: Option<Py<PyGpuBufferF32>>,
        epsilon: f64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let scale = scale
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("scale consumed"))?;
        let bias = bias
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("bias consumed"))?;
        let saved_mean = saved_mean
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_mean consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let saved_var = saved_var
            .map(|b| b.borrow(py).clone_ref().ok_or("saved_var consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = GroupNormRequest::<f32> {
                    x,
                    y,
                    scale,
                    bias,
                    saved_mean,
                    saved_var,
                    dims,
                    groups,
                    layout: TensorLayout::NchwPacked,
                    epsilon,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("group_norm_f32 timed out")),
                }
            })
        })
    }

    /// Backward normalisation (BatchNorm / LayerNorm / InstanceNorm /
    /// GroupNorm / RmsNorm) selected by `mode`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, dy, scale, mean, var, dx, dscale, dbias,
        dims,
        mode="batch",
        epsilon=1e-5,
        timeout_secs=60.0,
    ))]
    fn norm_bwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        dy: Py<PyGpuBufferF32>,
        scale: Py<PyGpuBufferF32>,
        mean: Py<PyGpuBufferF32>,
        var: Py<PyGpuBufferF32>,
        dx: Py<PyGpuBufferF32>,
        dscale: Py<PyGpuBufferF32>,
        dbias: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        mode: &str,
        epsilon: f64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let mode = norm_mode_from_str(mode)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let dy = dy
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dy consumed"))?;
        let scale = scale
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("scale consumed"))?;
        let mean = mean
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("mean consumed"))?;
        let var = var
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("var consumed"))?;
        let dx = dx
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dx consumed"))?;
        let dscale = dscale
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dscale consumed"))?;
        let dbias = dbias
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("dbias consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = NormBwdRequest::<f32> {
                    mode,
                    x,
                    dy,
                    scale,
                    mean,
                    var,
                    dx,
                    dscale,
                    dbias,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    epsilon,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("norm_bwd_f32 timed out")),
                }
            })
        })
    }

    /// Forward dropout: `y = x * mask / (1 - p)`. `mask` is a `u8`
    /// buffer recording the surviving entries.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y, mask,
        dims,
        probability=0.5,
        seed=0,
        timeout_secs=60.0,
    ))]
    fn dropout_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        mask: Py<PyGpuBufferU8>,
        dims: Vec<i64>,
        probability: f32,
        seed: u64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let mask = mask
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("mask consumed"))?;
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = DropoutFwdRequest::<f32> {
                    x,
                    y,
                    mask,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    probability,
                    seed,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("dropout_fwd_f32 timed out")),
                }
            })
        })
    }

    /// Forward local-response normalisation.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, y,
        dims,
        n=5, lrn_alpha=1e-4, lrn_beta=0.75, lrn_k=2.0,
        alpha=1.0, beta=0.0,
        timeout_secs=60.0,
    ))]
    fn lrn_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        dims: Vec<i64>,
        n: u32,
        lrn_alpha: f64,
        lrn_beta: f64,
        lrn_k: f64,
        alpha: f32,
        beta: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let params = LrnParams::new(n, lrn_alpha, lrn_beta, lrn_k);
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = LrnFwdRequest::<f32> {
                    x,
                    y,
                    dims,
                    layout: TensorLayout::NchwPacked,
                    params,
                    alpha,
                    beta,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("lrn_fwd_f32 timed out")),
                }
            })
        })
    }

    /// Forward RNN / LSTM / GRU. `c_in` / `c_out` only used for LSTM.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        x, h_in, weights, y, h_out,
        c_in=None, c_out=None,
        mode="lstm", direction="uni",
        num_layers=1,
        input_size=0, hidden_size=0,
        seq_length=0, batch_size=0,
        dropout=0.0,
        timeout_secs=60.0,
    ))]
    fn rnn_fwd_f32(
        &self,
        py: Python<'_>,
        x: Py<PyGpuBufferF32>,
        h_in: Py<PyGpuBufferF32>,
        weights: Py<PyGpuBufferF32>,
        y: Py<PyGpuBufferF32>,
        h_out: Py<PyGpuBufferF32>,
        c_in: Option<Py<PyGpuBufferF32>>,
        c_out: Option<Py<PyGpuBufferF32>>,
        mode: &str,
        direction: &str,
        num_layers: u32,
        input_size: i64,
        hidden_size: i64,
        seq_length: i64,
        batch_size: i64,
        dropout: f32,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let mode = rnn_mode_from_str(mode)?;
        let direction = rnn_direction_from_str(direction)?;
        let x = x
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("x consumed"))?;
        let h_in = h_in
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("h_in consumed"))?;
        let weights = weights
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("weights consumed"))?;
        let y = y
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("y consumed"))?;
        let h_out = h_out
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("h_out consumed"))?;
        let c_in = c_in
            .map(|b| b.borrow(py).clone_ref().ok_or("c_in consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let c_out = c_out
            .map(|b| b.borrow(py).clone_ref().ok_or("c_out consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let params = RnnParams {
            mode,
            direction,
            num_layers,
            input_size,
            hidden_size,
            seq_length,
            batch_size,
            dropout,
        };
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = RnnFwdRequest::<f32> {
                    x,
                    h_in,
                    c_in,
                    weights,
                    y,
                    h_out,
                    c_out,
                    layout: TensorLayout::NchwPacked,
                    params,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("rnn_fwd_f32 timed out")),
                }
            })
        })
    }

    // TODO Phase 1.5 deeper coverage — `rnn_bwd_f32` requires 14
    // distinct GpuRef args (x, y, dy, h_in, c_in, h_out, c_out, dh_out,
    // dc_out, weights, dx, dh_in, dc_in, dweights). Surface it once
    // pyo3's signature length proves stable; for now callers can
    // reach the same actor via the typed Rust API.

    /// Forward multi-head attention (fused). `q`, `k`, `v`, `o` are
    /// 4-D `[batch, heads, seq, head_dim]` tensors.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        q, k, v, o,
        batch, seq_q, seq_kv,
        heads_q, heads_kv, head_dim,
        stats=None, bias=None,
        mask="none", window=0,
        scale=None,
        dropout=0.0, dropout_seed=0,
        timeout_secs=60.0,
    ))]
    fn multihead_attn_fwd_f32(
        &self,
        py: Python<'_>,
        q: Py<PyGpuBufferF32>,
        k: Py<PyGpuBufferF32>,
        v: Py<PyGpuBufferF32>,
        o: Py<PyGpuBufferF32>,
        batch: i64,
        seq_q: i64,
        seq_kv: i64,
        heads_q: i64,
        heads_kv: i64,
        head_dim: i64,
        stats: Option<Py<PyGpuBufferF32>>,
        bias: Option<Py<PyGpuBufferF32>>,
        mask: &str,
        window: u32,
        scale: Option<f64>,
        dropout: f32,
        dropout_seed: u64,
        timeout_secs: f64,
    ) -> PyResult<()> {
        let mask = attention_mask_from_str(mask, window)?;
        let q = q
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("q consumed"))?;
        let k = k
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("k consumed"))?;
        let v = v
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("v consumed"))?;
        let o = o
            .borrow(py)
            .clone_ref()
            .ok_or_else(|| errors::map_str("o consumed"))?;
        let stats = stats
            .map(|b| b.borrow(py).clone_ref().ok_or("stats consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let bias = bias
            .map(|b| b.borrow(py).clone_ref().ok_or("bias consumed"))
            .transpose()
            .map_err(errors::map_str)?;
        let scale = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());
        let params = AttentionParams {
            batch,
            seq_q,
            seq_kv,
            heads_q,
            heads_kv,
            head_dim,
            mask,
            scale,
            dropout,
            dropout_seed,
        };
        let actor = self.actor_ref.clone();
        let rt = runtime();
        py.allow_threads(|| {
            rt.block_on(async move {
                let (tx, rx) = oneshot::channel();
                let req = MultiHeadAttnFwdRequest::<f32> {
                    q,
                    k,
                    v,
                    o,
                    stats,
                    bias,
                    layout: TensorLayout::NchwPacked,
                    params,
                    reply: tx,
                    _ty: PhantomData,
                };
                actor.tell(CudnnMsg::Op(Box::new(req)));
                match tokio::time::timeout(Duration::from_secs_f64(timeout_secs), rx).await {
                    Ok(Ok(Ok(()))) => Ok(()),
                    Ok(Ok(Err(e))) => Err(errors::map_gpu(e)),
                    Ok(Err(_)) => Err(errors::map_str("cudnn dropped reply")),
                    Err(_) => Err(errors::map_str("multihead_attn_fwd_f32 timed out")),
                }
            })
        })
    }

    // TODO Phase 1.5 deeper coverage — `multihead_attn_bwd_f32` needs
    // 9 GpuRef args (q, k, v, o, do_, dq, dk, dv, stats) plus the
    // params struct. Surface it once we add a small `AttnBwdInputs`
    // pyclass to keep the call ergonomic.

    fn __repr__(&self) -> &'static str {
        "Cudnn(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCudnn>()?;
    Ok(())
}
