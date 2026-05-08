//! `Cudnn` — Python handle wrapping `ActorRef<CudnnMsg>`.
//!
//! Obtained via `Device.cudnn()` (only when the `cudnn` feature is
//! compiled in *and* the device's `EnabledLibraries::CUDNN` flag is
//! set). Phase 1 ships a single representative method, `conv2d_fwd_f32`,
//! that exercises the typed `ConvFwdRequest::<f32>` dispatch end-to-end
//! (mock-mode replies surface as `Unrecoverable`).
//!
//! Pool / batch_norm / layer_norm / RNN / attention / dropout follow in
//! the Phase 1.5 cuDNN-coverage tracking issue.

#![cfg(feature = "cudnn")]

use std::marker::PhantomData;
use std::time::Duration;

use pyo3::prelude::*;
use tokio::sync::oneshot;

use atomr_accel_cuda::kernel::{
    ConvDescParams, ConvFwdRequest, CudnnMsg, EpilogueKind, TensorLayout,
};
use atomr_core::actor::ActorRef;

use crate::buffer::PyGpuBufferF32;
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

    fn __repr__(&self) -> &'static str {
        "Cudnn(handle)"
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCudnn>()?;
    Ok(())
}
