//! RNN / LSTM / GRU forward + backward training requests.
//!
//! Routes through cuDNN's RNN v8 API (`cudnnRNNForward` /
//! `cudnnRNNBackwardData_v8` / `cudnnRNNBackwardWeights_v8`).

#![allow(dead_code)]

use std::marker::PhantomData;

use tokio::sync::oneshot;

use crate::dtype::CudnnSupported;
use crate::error::GpuError;
use crate::gpu_ref::GpuRef;
use crate::kernel::cudnn::conv::dtype_tag;
use crate::kernel::cudnn::graph::{DtypeTag, OperationGraphSpec, TensorLayout, TensorSpec};
use crate::kernel::dispatch::{CudnnDispatch, CudnnDispatchCtx};

/// RNN cell mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnMode {
    Rnn,
    RnnTanh,
    Lstm,
    Gru,
}

impl RnnMode {
    pub fn op_kind(self) -> &'static str {
        match self {
            RnnMode::Rnn => "rnn_relu",
            RnnMode::RnnTanh => "rnn_tanh",
            RnnMode::Lstm => "lstm",
            RnnMode::Gru => "gru",
        }
    }

    /// Number of gates / linear projections internal to one cell.
    pub fn num_gates(self) -> u32 {
        match self {
            RnnMode::Rnn | RnnMode::RnnTanh => 1,
            RnnMode::Lstm => 4,
            RnnMode::Gru => 3,
        }
    }

    /// Whether the cell carries a separate cell state (LSTM only).
    pub fn has_cell_state(self) -> bool {
        matches!(self, RnnMode::Lstm)
    }
}

/// Direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RnnDirection {
    Unidirectional,
    Bidirectional,
}

impl RnnDirection {
    pub fn factor(self) -> u32 {
        match self {
            RnnDirection::Unidirectional => 1,
            RnnDirection::Bidirectional => 2,
        }
    }
}

/// RNN parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnParams {
    pub mode: RnnMode,
    pub direction: RnnDirection,
    pub num_layers: u32,
    pub input_size: i64,
    pub hidden_size: i64,
    pub seq_length: i64,
    pub batch_size: i64,
    pub dropout: f32,
}

impl RnnParams {
    pub fn new(
        mode: RnnMode,
        direction: RnnDirection,
        num_layers: u32,
        input_size: i64,
        hidden_size: i64,
        seq_length: i64,
        batch_size: i64,
    ) -> Self {
        Self {
            mode,
            direction,
            num_layers,
            input_size,
            hidden_size,
            seq_length,
            batch_size,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(mut self, d: f32) -> Self {
        self.dropout = d;
        self
    }

    pub fn output_size(&self) -> i64 {
        self.hidden_size * self.direction.factor() as i64
    }
}

/// RNN forward request.
pub struct RnnFwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub h_in: GpuRef<T>,
    pub c_in: Option<GpuRef<T>>,
    pub weights: GpuRef<T>,
    pub y: GpuRef<T>,
    pub h_out: GpuRef<T>,
    pub c_out: Option<GpuRef<T>>,
    pub layout: TensorLayout,
    pub params: RnnParams,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> RnnFwdRequest<T> {
    pub fn graph_spec(&self) -> OperationGraphSpec {
        build_rnn_fwd_spec(dtype_tag::<T>(), &self.params, self.layout)
    }
}

impl<T: CudnnSupported> CudnnDispatch for RnnFwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        match self.params.mode {
            RnnMode::Rnn | RnnMode::RnnTanh => "rnn_fwd",
            RnnMode::Lstm => "lstm_fwd",
            RnnMode::Gru => "gru_fwd",
        }
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "RnnFwdRequest dispatch requires the v8 RNN API path; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// RNN backward (data + weights) request.
pub struct RnnBwdRequest<T: CudnnSupported> {
    pub x: GpuRef<T>,
    pub y: GpuRef<T>,
    pub dy: GpuRef<T>,
    pub h_in: GpuRef<T>,
    pub c_in: Option<GpuRef<T>>,
    pub h_out: GpuRef<T>,
    pub c_out: Option<GpuRef<T>>,
    pub dh_out: GpuRef<T>,
    pub dc_out: Option<GpuRef<T>>,
    pub weights: GpuRef<T>,
    pub dx: GpuRef<T>,
    pub dh_in: GpuRef<T>,
    pub dc_in: Option<GpuRef<T>>,
    pub dweights: GpuRef<T>,
    pub layout: TensorLayout,
    pub params: RnnParams,
    pub reply: oneshot::Sender<Result<(), GpuError>>,
    pub _ty: PhantomData<T>,
}

impl<T: CudnnSupported> CudnnDispatch for RnnBwdRequest<T> {
    fn dtype_name(&self) -> &'static str {
        T::NAME
    }
    fn op_kind(&self) -> &'static str {
        match self.params.mode {
            RnnMode::Rnn | RnnMode::RnnTanh => "rnn_bwd",
            RnnMode::Lstm => "lstm_bwd",
            RnnMode::Gru => "gru_bwd",
        }
    }
    fn dispatch(self: Box<Self>, _ctx: &CudnnDispatchCtx<'_>) {
        let _ = self.reply.send(Err(GpuError::LibraryError {
            lib: "cudnn",
            msg: "RnnBwdRequest dispatch requires the v8 RNN API path; \
                  skeleton entry point only"
                .to_string(),
        }));
    }
}

/// Build a spec-side `OperationGraphSpec` for plan-cache keying.
/// The actual RNN API does not use the v9 backend graph descriptor
/// builder — but we keep a stable signature surface so the cache key
/// machinery is shared.
pub fn build_rnn_fwd_spec(
    dtype: DtypeTag,
    p: &RnnParams,
    layout: TensorLayout,
) -> OperationGraphSpec {
    let mut g = OperationGraphSpec::new("rnn_fwd");
    let x_dims = vec![p.seq_length, p.batch_size, p.input_size];
    let h_dims = vec![
        p.num_layers as i64 * p.direction.factor() as i64,
        p.batch_size,
        p.hidden_size,
    ];
    let y_dims = vec![p.seq_length, p.batch_size, p.output_size()];
    g.add_tensor(TensorSpec::new(1, dtype, x_dims, layout));
    g.add_tensor(TensorSpec::new(2, dtype, h_dims.clone(), layout));
    g.add_tensor(TensorSpec::new(3, dtype, y_dims, layout));
    g.add_tensor(TensorSpec::new(4, dtype, h_dims.clone(), layout));
    if p.mode.has_cell_state() {
        g.add_tensor(TensorSpec::new(5, dtype, h_dims.clone(), layout));
        g.add_tensor(TensorSpec::new(6, dtype, h_dims, layout));
    }
    // Weights tensor — packed dim depends on (input_size, hidden_size,
    // num_gates, num_layers, direction). Use a single placeholder
    // dim; the plan-cache key digests this via the rest of the spec.
    let weight_dim = p.mode.num_gates() as i64
        * p.hidden_size
        * (p.input_size + p.hidden_size + 2)
        * p.num_layers as i64
        * p.direction.factor() as i64;
    g.add_tensor(TensorSpec::new(99, dtype, vec![weight_dim], layout));
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(mode: RnnMode) {
        let p = RnnParams::new(mode, RnnDirection::Bidirectional, 2, 128, 256, 32, 8);
        let g = build_rnn_fwd_spec(DtypeTag::F32, &p, TensorLayout::NchwPacked);
        // x, h_in, y, h_out [, c_in, c_out], weights
        let expected = if mode.has_cell_state() { 7 } else { 5 };
        assert_eq!(g.tensors.len(), expected);
        assert_eq!(p.output_size(), 512);
    }

    #[test]
    fn rnn_lstm_gru_request_round_trip() {
        round_trip(RnnMode::Rnn);
        round_trip(RnnMode::RnnTanh);
        round_trip(RnnMode::Lstm);
        round_trip(RnnMode::Gru);
    }

    #[test]
    fn cell_state_only_for_lstm() {
        assert!(!RnnMode::Rnn.has_cell_state());
        assert!(!RnnMode::Gru.has_cell_state());
        assert!(RnnMode::Lstm.has_cell_state());
    }

    #[test]
    fn gate_counts() {
        assert_eq!(RnnMode::Rnn.num_gates(), 1);
        assert_eq!(RnnMode::Lstm.num_gates(), 4);
        assert_eq!(RnnMode::Gru.num_gates(), 3);
    }
}
