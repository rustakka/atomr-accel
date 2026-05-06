//! Epilogue Visitor Tree (EVT) builder.
//!
//! Gated behind the `evt` cargo feature. CUTLASS's EVT API lets the
//! caller compose fused activation, bias, residual-add, quantize, and
//! reduce ops as a tree of "visitors" applied to each output tile.
//! Here we mirror that as a small DSL: [`EvtBuilder`] accumulates
//! [`EpilogueOp`]s into an [`EpilogueVisitorTree`] that the kernel
//! emitter renders into CUTLASS template arguments.

use crate::dtype::CutlassDtype;

/// One node in the visitor tree. Order is significant: ops are
/// applied in insertion order, with the previous result threaded as
/// the input to the next.
#[derive(Debug, Clone, PartialEq)]
pub enum EpilogueOp {
    /// `out = alpha * gemm + beta * c`.
    LinearCombination { alpha: f32, beta: f32 },
    /// Add a per-row bias vector.
    BiasAdd { dtype: CutlassDtype },
    /// `out = max(0, in)`.
    Relu,
    /// Approximate gelu (CUTLASS `epilogue::thread::GELU_taylor`).
    Gelu,
    /// `out = in * sigmoid(in)` (silu / swish).
    Silu,
    /// `out = tanh(in)`.
    Tanh,
    /// Per-tensor / per-channel quantize. CUTLASS template parameter
    /// `cutlass::epilogue::thread::Quantize`.
    Quantize {
        out_dtype: CutlassDtype,
        per_channel: bool,
    },
    /// Add a residual tensor read from device memory.
    ResidualAdd { dtype: CutlassDtype },
    /// Reduce per-row (sum / max). Used by softmax fusions.
    Reduce { kind: ReduceKind },
}

/// Reduction discriminator for the `Reduce` epilogue op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Sum,
    Max,
    Min,
    Mean,
}

impl ReduceKind {
    pub fn short_name(self) -> &'static str {
        match self {
            ReduceKind::Sum => "sum",
            ReduceKind::Max => "max",
            ReduceKind::Min => "min",
            ReduceKind::Mean => "mean",
        }
    }
}

impl EpilogueOp {
    /// Stable short name used in plan-cache keys.
    pub fn short_name(&self) -> &'static str {
        match self {
            EpilogueOp::LinearCombination { .. } => "linear",
            EpilogueOp::BiasAdd { .. } => "bias_add",
            EpilogueOp::Relu => "relu",
            EpilogueOp::Gelu => "gelu",
            EpilogueOp::Silu => "silu",
            EpilogueOp::Tanh => "tanh",
            EpilogueOp::Quantize { .. } => "quantize",
            EpilogueOp::ResidualAdd { .. } => "residual_add",
            EpilogueOp::Reduce { .. } => "reduce",
        }
    }

    /// Render the op as a CUTLASS visitor template type.
    pub fn render(&self, prev: &str) -> String {
        match self {
            EpilogueOp::LinearCombination { alpha: _, beta: _ } => {
                format!("cutlass::epilogue::fusion::Sm90LinearCombination<{prev}>")
            }
            EpilogueOp::BiasAdd { dtype } => {
                format!(
                    "cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, {ty}, {prev}>",
                    ty = dtype.as_cutlass_type(),
                    prev = prev,
                )
            }
            EpilogueOp::Relu => format!(
                "cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::ReLu, float, {prev}>",
            ),
            EpilogueOp::Gelu => format!(
                "cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::GELU_taylor, float, {prev}>",
            ),
            EpilogueOp::Silu => format!(
                "cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::SiLu, float, {prev}>",
            ),
            EpilogueOp::Tanh => format!(
                "cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Tanh, float, {prev}>",
            ),
            EpilogueOp::Quantize { out_dtype, per_channel } => {
                let scope = if *per_channel { "PerChannel" } else { "PerTensor" };
                format!(
                    "cutlass::epilogue::fusion::Sm90{scope}Quantize<{ty}, {prev}>",
                    ty = out_dtype.as_cutlass_type(),
                )
            }
            EpilogueOp::ResidualAdd { dtype } => format!(
                "cutlass::epilogue::fusion::Sm90Residual<{ty}, {prev}>",
                ty = dtype.as_cutlass_type(),
            ),
            EpilogueOp::Reduce { kind } => format!(
                "cutlass::epilogue::fusion::Sm90Reduce<cutlass::reduce::{kind}, float, {prev}>",
                kind = match kind {
                    ReduceKind::Sum => "sum",
                    ReduceKind::Max => "maximum",
                    ReduceKind::Min => "minimum",
                    ReduceKind::Mean => "mean",
                }
            ),
        }
    }
}

/// A built epilogue visitor tree. Stored as a flat insertion-ordered
/// vector so the plan-cache key is deterministic and the render path
/// can be a single fold.
#[derive(Debug, Clone, PartialEq)]
pub struct EpilogueVisitorTree {
    ops: Vec<EpilogueOp>,
}

impl EpilogueVisitorTree {
    pub fn ops(&self) -> &[EpilogueOp] {
        &self.ops
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Render the full tree as a nested CUTLASS template type. The
    /// innermost node is the GEMM accumulator.
    pub fn render(&self) -> String {
        let mut s = String::from("cutlass::epilogue::fusion::Sm90AccFetch");
        for op in &self.ops {
            s = op.render(&s);
        }
        s
    }

    /// Stable opaque id for plan-cache keys. `u64` because the tree
    /// can grow large and we want hashes to compose cheaply.
    pub fn id(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for op in &self.ops {
            op.short_name().hash(&mut h);
            // Carry the dtype bytes into the hash so e.g. fp8 vs bf16
            // bias-add keys diverge.
            match op {
                EpilogueOp::BiasAdd { dtype }
                | EpilogueOp::ResidualAdd { dtype }
                | EpilogueOp::Quantize {
                    out_dtype: dtype, ..
                } => {
                    dtype.short_name().hash(&mut h);
                }
                EpilogueOp::Reduce { kind } => kind.short_name().hash(&mut h),
                EpilogueOp::LinearCombination { alpha, beta } => {
                    alpha.to_bits().hash(&mut h);
                    beta.to_bits().hash(&mut h);
                }
                _ => {}
            }
        }
        h.finish()
    }
}

/// Builder for [`EpilogueVisitorTree`]. Chainable.
#[derive(Debug, Default, Clone)]
pub struct EvtBuilder {
    ops: Vec<EpilogueOp>,
}

impl EvtBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn linear(mut self, alpha: f32, beta: f32) -> Self {
        self.ops.push(EpilogueOp::LinearCombination { alpha, beta });
        self
    }

    pub fn bias_add(mut self, dtype: CutlassDtype) -> Self {
        self.ops.push(EpilogueOp::BiasAdd { dtype });
        self
    }

    pub fn relu(mut self) -> Self {
        self.ops.push(EpilogueOp::Relu);
        self
    }

    pub fn gelu(mut self) -> Self {
        self.ops.push(EpilogueOp::Gelu);
        self
    }

    pub fn silu(mut self) -> Self {
        self.ops.push(EpilogueOp::Silu);
        self
    }

    pub fn tanh(mut self) -> Self {
        self.ops.push(EpilogueOp::Tanh);
        self
    }

    pub fn quantize(mut self, out_dtype: CutlassDtype, per_channel: bool) -> Self {
        self.ops.push(EpilogueOp::Quantize {
            out_dtype,
            per_channel,
        });
        self
    }

    pub fn residual_add(mut self, dtype: CutlassDtype) -> Self {
        self.ops.push(EpilogueOp::ResidualAdd { dtype });
        self
    }

    pub fn reduce(mut self, kind: ReduceKind) -> Self {
        self.ops.push(EpilogueOp::Reduce { kind });
        self
    }

    pub fn push(mut self, op: EpilogueOp) -> Self {
        self.ops.push(op);
        self
    }

    pub fn build(self) -> EpilogueVisitorTree {
        EpilogueVisitorTree { ops: self.ops }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epilogue_visitor_tree_builder_round_trip() {
        let tree = EvtBuilder::new()
            .linear(1.0, 0.5)
            .bias_add(CutlassDtype::F16)
            .relu()
            .quantize(CutlassDtype::F8E4m3, true)
            .build();

        assert_eq!(tree.len(), 4);
        assert!(!tree.is_empty());
        assert_eq!(
            tree.ops()[1],
            EpilogueOp::BiasAdd {
                dtype: CutlassDtype::F16
            }
        );

        let rendered = tree.render();
        // Innermost (GEMM accumulator) must be present and the relu
        // and quantize must wrap around it.
        assert!(rendered.contains("Sm90AccFetch"));
        assert!(rendered.contains("ReLu"));
        assert!(rendered.contains("Quantize"));

        // ID stability: same sequence -> same id; different sequence
        // -> different id.
        let same = EvtBuilder::new()
            .linear(1.0, 0.5)
            .bias_add(CutlassDtype::F16)
            .relu()
            .quantize(CutlassDtype::F8E4m3, true)
            .build();
        assert_eq!(tree.id(), same.id());

        let different = EvtBuilder::new()
            .linear(2.0, 0.0)
            .bias_add(CutlassDtype::Bf16)
            .gelu()
            .build();
        assert_ne!(tree.id(), different.id());

        // Empty tree builds and renders as a bare AccFetch.
        let empty = EvtBuilder::new().build();
        assert!(empty.is_empty());
        assert_eq!(empty.render(), "cutlass::epilogue::fusion::Sm90AccFetch");

        // Push API parity.
        let pushed = EvtBuilder::new()
            .push(EpilogueOp::Reduce {
                kind: ReduceKind::Sum,
            })
            .build();
        assert_eq!(
            pushed.ops()[0],
            EpilogueOp::Reduce {
                kind: ReduceKind::Sum
            }
        );
    }
}
