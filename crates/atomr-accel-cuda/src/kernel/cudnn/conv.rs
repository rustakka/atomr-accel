//! Convolution requests for the cuDNN actor (Phase 2 frontend graph
//! API).
//!
//! Three op families:
//!
//! * [`ConvFwdRequest<T>`] — y = conv(x, w) [+ optional bias + optional
//!   activation, when fused via `epilogue`].
//! * [`ConvBwdDataRequest<T>`] — dx = conv_bwd_data(w, dy).
//! * [`ConvBwdFilterRequest<T>`] — dw = conv_bwd_filter(x, dy).
//!
//! Each supports 1D / 2D / 3D, NCHW + NHWC packed layouts (or fully
//! strided), arbitrary group count, and arbitrary dilation.

#![allow(dead_code)]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dtype::CudnnSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::cudnn::activation::ActivationKind;
use crate::kernel::cudnn::graph::{
    DtypeTag, OpSpec, OperationGraphSpec, PointwiseMode, TensorLayout, TensorSpec,
};
use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};

/// Convolution descriptor parameters, dimension-generic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvDescParams {
    /// Number of spatial dimensions (1, 2, or 3).
    pub spatial_dims: usize,
    /// Per-dim leading padding.
    pub pre_padding: Vec<i64>,
    /// Per-dim trailing padding.
    pub post_padding: Vec<i64>,
    /// Per-dim filter stride.
    pub stride: Vec<i64>,
    /// Per-dim dilation.
    pub dilation: Vec<i64>,
    /// Group count (≥ 1).
    pub groups: i64,
}

impl ConvDescParams {
    /// Symmetric same-padding helper for 2D conv.
    pub fn symmetric_2d(pad: i64, stride: i64, dilation: i64) -> Self {
        Self {
            spatial_dims: 2,
            pre_padding: vec![pad, pad],
            post_padding: vec![pad, pad],
            stride: vec![stride, stride],
            dilation: vec![dilation, dilation],
            groups: 1,
        }
    }

    /// Symmetric helper for 1D conv.
    pub fn symmetric_1d(pad: i64, stride: i64, dilation: i64) -> Self {
        Self {
            spatial_dims: 1,
            pre_padding: vec![pad],
            post_padding: vec![pad],
            stride: vec![stride],
            dilation: vec![dilation],
            groups: 1,
        }
    }

    /// Symmetric helper for 3D conv.
    pub fn symmetric_3d(pad: i64, stride: i64, dilation: i64) -> Self {
        Self {
            spatial_dims: 3,
            pre_padding: vec![pad, pad, pad],
            post_padding: vec![pad, pad, pad],
            stride: vec![stride, stride, stride],
            dilation: vec![dilation, dilation, dilation],
            groups: 1,
        }
    }

    pub fn with_groups(mut self, g: i64) -> Self {
        self.groups = g;
        self
    }
}

/// Optional fused epilogue tail attached to conv-fwd. The bias is
/// represented as an opaque marker on the spec layer (the graph
/// builder records "there is a bias of this dtype + shape"); the
/// concrete `GpuRef<T>` lives on [`ConvFwdRequest`] proper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpilogueKind {
    /// No epilogue.
    None,
    /// Add bias broadcast across spatial dims.
    Bias,
    /// Bias + activation.
    BiasActivation(ActivationKind),
}

pub(crate) fn dtype_tag<T: CudnnSupported>() -> DtypeTag {
    match T::NAME {
        "f32" => DtypeTag::F32,
        "f64" => DtypeTag::F64,
        "f16" => DtypeTag::F16,
        "bf16" => DtypeTag::Bf16,
        "i8" => DtypeTag::I8,
        other => panic!("unsupported cuDNN dtype name: {other}"),
    }
}

/// Build the spec-side conv-fwd op graph, parameterised by dtype +
/// dims. Independent of `GpuRef` so callers (tests, plan-cache lookup)
/// can build it without owning device buffers.
pub fn build_conv_fwd_graph(
    dtype: DtypeTag,
    x_dims: &[i64],
    w_dims: &[i64],
    y_dims: &[i64],
    conv: &ConvDescParams,
    layout: TensorLayout,
    epilogue: EpilogueKind,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("conv_fwd");
    let x_uid = g.add_tensor(TensorSpec::new(1, dtype, x_dims.to_vec(), layout));
    let w_uid = g.add_tensor(TensorSpec::new(2, dtype, w_dims.to_vec(), layout));
    let y_uid = g.add_tensor(TensorSpec::new(3, dtype, y_dims.to_vec(), layout));
    g.add_op(OpSpec::ConvFwd {
        x: x_uid,
        w: w_uid,
        y: y_uid,
        spatial_dims: conv.spatial_dims,
        pre_padding: conv.pre_padding.clone(),
        post_padding: conv.post_padding.clone(),
        stride: conv.stride.clone(),
        dilation: conv.dilation.clone(),
        compute_dtype: dtype,
        alpha: 1.0,
        beta: 0.0,
    });
    match epilogue {
        EpilogueKind::None => {}
        EpilogueKind::Bias => {
            let b_uid =
                g.add_tensor(TensorSpec::new(4, dtype, bias_dims(y_dims), layout));
            let yb_uid = g.add_tensor(TensorSpec::new(5, dtype, y_dims.to_vec(), layout));
            g.add_op(OpSpec::Pointwise {
                mode: PointwiseMode::Add,
                x: y_uid,
                b: Some(b_uid),
                y: yb_uid,
                compute_dtype: dtype,
                alpha1: 1.0,
                alpha2: 1.0,
            });
        }
        EpilogueKind::BiasActivation(act) => {
            let b_uid =
                g.add_tensor(TensorSpec::new(4, dtype, bias_dims(y_dims), layout));
            let yb_uid = g.add_tensor(TensorSpec::new(5, dtype, y_dims.to_vec(), layout));
            g.add_op(OpSpec::Pointwise {
                mode: PointwiseMode::Add,
                x: y_uid,
                b: Some(b_uid),
                y: yb_uid,
                compute_dtype: dtype,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            let act_out = g.add_tensor(TensorSpec::new(6, dtype, y_dims.to_vec(), layout));
            g.add_op(OpSpec::Pointwise {
                mode: act.pointwise_mode(),
                x: yb_uid,
                b: None,
                y: act_out,
                compute_dtype: dtype,
                alpha1: 1.0,
                alpha2: 0.0,
            });
        }
    }
    g
}

/// Build the spec-side conv-bwd-data op graph.
pub fn build_conv_bwd_data_graph(
    dtype: DtypeTag,
    dy_dims: &[i64],
    w_dims: &[i64],
    dx_dims: &[i64],
    conv: &ConvDescParams,
    layout: TensorLayout,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("conv_bwd_data");
    let dy_uid = g.add_tensor(TensorSpec::new(1, dtype, dy_dims.to_vec(), layout));
    let w_uid = g.add_tensor(TensorSpec::new(2, dtype, w_dims.to_vec(), layout));
    let dx_uid = g.add_tensor(TensorSpec::new(3, dtype, dx_dims.to_vec(), layout));
    g.add_op(OpSpec::ConvBwdData {
        dy: dy_uid,
        w: w_uid,
        dx: dx_uid,
        spatial_dims: conv.spatial_dims,
        pre_padding: conv.pre_padding.clone(),
        post_padding: conv.post_padding.clone(),
        stride: conv.stride.clone(),
        dilation: conv.dilation.clone(),
        compute_dtype: dtype,
        alpha: 1.0,
        beta: 0.0,
    });
    g
}

/// Build the spec-side conv-bwd-filter op graph.
pub fn build_conv_bwd_filter_graph(
    dtype: DtypeTag,
    x_dims: &[i64],
    dy_dims: &[i64],
    dw_dims: &[i64],
    conv: &ConvDescParams,
    layout: TensorLayout,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("conv_bwd_filter");
    let x_uid = g.add_tensor(TensorSpec::new(1, dtype, x_dims.to_vec(), layout));
    let dy_uid = g.add_tensor(TensorSpec::new(2, dtype, dy_dims.to_vec(), layout));
    let dw_uid = g.add_tensor(TensorSpec::new(3, dtype, dw_dims.to_vec(), layout));
    g.add_op(OpSpec::ConvBwdFilter {
        x: x_uid,
        dy: dy_uid,
        dw: dw_uid,
        spatial_dims: conv.spatial_dims,
        pre_padding: conv.pre_padding.clone(),
        post_padding: conv.post_padding.clone(),
        stride: conv.stride.clone(),
        dilation: conv.dilation.clone(),
        compute_dtype: dtype,
        alpha: 1.0,
        beta: 0.0,
    });
    g
}

/// Bias broadcast dim-vector matching `y_dims`. cuDNN bias tensors
/// are `[1, C, 1, 1...]` regardless of channel-first vs channel-last
/// layout — the layout is captured in strides, not dims.
fn bias_dims(y_dims: &[i64]) -> Vec<i64> {
    let mut out = vec![1i64; y_dims.len()];
    if y_dims.len() >= 2 {
        out[1] = y_dims[1];
    }
    out
}

// ----- Request types -------------------------------------------------

/// Forward convolution: `y = alpha * conv(x, w) + beta * y`,
/// optionally with a fused bias / activation tail.
pub struct ConvFwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub x_dims: Vec<i64>,
    pub w: GpuRef<T>,
    pub w_dims: Vec<i64>,
    pub y: GpuRef<T>,
    pub y_dims: Vec<i64>,
    pub bias: Option<GpuRef<T>>,
    pub conv: ConvDescParams,
    pub layout: TensorLayout,
    pub epilogue: EpilogueKind,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> ConvFwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_conv_fwd_graph(
            dtype_tag::<T>(),
            &self.x_dims,
            &self.w_dims,
            &self.y_dims,
            &self.conv,
            self.layout,
            self.epilogue,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for ConvFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "conv_fwd"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "ConvFwdRequest dispatch requires the v9 frontend graph builder \
                  (cudnnBackendCreateDescriptor path); skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Backward-data convolution: `dx = alpha * conv_bwd_data(w, dy) + beta * dx`.
pub struct ConvBwdDataRequest<T: CudnnSupported> {
    pub dy: GpuRef<T>,
    pub dy_dims: Vec<i64>,
    pub w: GpuRef<T>,
    pub w_dims: Vec<i64>,
    pub dx: GpuRef<T>,
    pub dx_dims: Vec<i64>,
    pub conv: ConvDescParams,
    pub layout: TensorLayout,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> ConvBwdDataRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_conv_bwd_data_graph(
            dtype_tag::<T>(),
            &self.dy_dims,
            &self.w_dims,
            &self.dx_dims,
            &self.conv,
            self.layout,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for ConvBwdDataRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "conv_bwd_data"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "ConvBwdDataRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Backward-filter convolution: `dw = alpha * conv_bwd_filter(x, dy) + beta * dw`.
pub struct ConvBwdFilterRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub x_dims: Vec<i64>,
    pub dy: GpuRef<T>,
    pub dy_dims: Vec<i64>,
    pub dw: GpuRef<T>,
    pub dw_dims: Vec<i64>,
    pub conv: ConvDescParams,
    pub layout: TensorLayout,
    pub alpha: T::Scalar,
    pub beta: T::Scalar,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> ConvBwdFilterRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_conv_bwd_filter_graph(
            dtype_tag::<T>(),
            &self.x_dims,
            &self.dy_dims,
            &self.dw_dims,
            &self.conv,
            self.layout,
        )
    }
}

impl<T: CudnnSupported> CudnnDispatch for ConvBwdFilterRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        "conv_bwd_filter"
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "ConvBwdFilterRequest dispatch requires the v9 frontend graph builder; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::cudnn::graph::cache_key;

    fn round_trip_fwd(dt: DtypeTag, dt_name: &'static str, layout: TensorLayout) {
        let g = build_conv_fwd_graph(
            dt,
            &[1, 3, 8, 8],
            &[16, 3, 3, 3],
            &[1, 16, 6, 6],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            layout,
            EpilogueKind::None,
        );
        assert_eq!(g.tensors.len(), 3);
        assert_eq!(g.ops.len(), 1);
        let key = cache_key("conv_fwd", dt, &g);
        assert_eq!(key.op_kind, "conv_fwd");
        assert_eq!(key.dtype, dt);
        // Re-building from the same inputs yields the same signature.
        let g2 = build_conv_fwd_graph(
            dt,
            &[1, 3, 8, 8],
            &[16, 3, 3, 3],
            &[1, 16, 6, 6],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            layout,
            EpilogueKind::None,
        );
        assert_eq!(g.signature(), g2.signature());
        assert_eq!(dt.name(), dt_name);
    }

    #[test]
    fn conv_fwd_request_round_trip_f32_f64_f16_bf16() {
        round_trip_fwd(DtypeTag::F32, "f32", TensorLayout::NchwPacked);
        round_trip_fwd(DtypeTag::F64, "f64", TensorLayout::NchwPacked);
        round_trip_fwd(DtypeTag::F16, "f16", TensorLayout::NchwPacked);
        round_trip_fwd(DtypeTag::Bf16, "bf16", TensorLayout::NchwPacked);
        // Also run NHWC for f32 to exercise the layout path.
        round_trip_fwd(DtypeTag::F32, "f32", TensorLayout::NhwcPacked);
    }

    #[test]
    fn conv_bwd_data_filter_request_round_trip() {
        let g = build_conv_bwd_data_graph(
            DtypeTag::F32,
            &[1, 16, 6, 6],
            &[16, 3, 3, 3],
            &[1, 3, 8, 8],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            TensorLayout::NchwPacked,
        );
        assert_eq!(g.ops.len(), 1);
        match &g.ops[0] {
            OpSpec::ConvBwdData { spatial_dims, .. } => assert_eq!(*spatial_dims, 2),
            _ => panic!("wrong op"),
        }

        let g = build_conv_bwd_filter_graph(
            DtypeTag::F32,
            &[1, 3, 8, 8],
            &[1, 16, 6, 6],
            &[16, 3, 3, 3],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            TensorLayout::NchwPacked,
        );
        match &g.ops[0] {
            OpSpec::ConvBwdFilter { spatial_dims, .. } => assert_eq!(*spatial_dims, 2),
            _ => panic!("wrong op"),
        }
    }

    #[test]
    fn nchw_vs_nhwc_layout_handled() {
        let g_nchw = build_conv_fwd_graph(
            DtypeTag::F32,
            &[1, 3, 8, 8],
            &[16, 3, 3, 3],
            &[1, 16, 6, 6],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            TensorLayout::NchwPacked,
            EpilogueKind::None,
        );
        let g_nhwc = build_conv_fwd_graph(
            DtypeTag::F32,
            &[1, 3, 8, 8],
            &[16, 3, 3, 3],
            &[1, 16, 6, 6],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            TensorLayout::NhwcPacked,
            EpilogueKind::None,
        );
        assert_ne!(g_nchw.signature(), g_nhwc.signature());
        assert_eq!(g_nchw.tensors[0].strides, vec![192, 64, 8, 1]);
        assert_ne!(g_nhwc.tensors[0].strides, g_nchw.tensors[0].strides);
    }

    #[test]
    fn conv_fwd_with_bias_activation_epilogue() {
        let g = build_conv_fwd_graph(
            DtypeTag::F32,
            &[1, 3, 8, 8],
            &[16, 3, 3, 3],
            &[1, 16, 6, 6],
            &ConvDescParams::symmetric_2d(0, 1, 1),
            TensorLayout::NhwcPacked,
            EpilogueKind::BiasActivation(ActivationKind::Relu),
        );
        // conv + bias-add + activation
        assert_eq!(g.ops.len(), 3);
        assert_eq!(g.tensors.len(), 6);
    }

    #[test]
    fn conv_1d_and_3d_descriptor_params() {
        let p1 = ConvDescParams::symmetric_1d(1, 1, 1);
        assert_eq!(p1.spatial_dims, 1);
        assert_eq!(p1.stride.len(), 1);
        let p3 = ConvDescParams::symmetric_3d(1, 2, 1);
        assert_eq!(p3.spatial_dims, 3);
        assert_eq!(p3.stride, vec![2, 2, 2]);
    }

    #[test]
    fn conv_grouped() {
        let p = ConvDescParams::symmetric_2d(0, 1, 1).with_groups(8);
        assert_eq!(p.groups, 8);
    }
}
