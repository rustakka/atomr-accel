//! Optimizer kinds. F4 ships SGD and AdamW configs; the actual
//! parameter-update kernels live in F4.x once the gradient
//! buffers are flowing through NCCL.

#[derive(Debug, Clone, Copy)]
pub enum OptimizerKind {
    Sgd {
        lr: f32,
        momentum: f32,
        weight_decay: f32,
    },
    AdamW {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
}

impl OptimizerKind {
    pub fn lr(&self) -> f32 {
        match self {
            OptimizerKind::Sgd { lr, .. } => *lr,
            OptimizerKind::AdamW { lr, .. } => *lr,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StepStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub step_micros: u64,
}
