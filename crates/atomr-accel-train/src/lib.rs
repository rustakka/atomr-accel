//! Distributed training blueprints on atomr-accel-cuda.
//!
//! ```ignore
//! use atomr_accel_train::prelude::*;
//! ```
//!
//! - [`data_parallel::DataParallelTrainer`] — N-replica trainer
//!   wired to NCCL all-reduce.
//! - [`pipeline_parallel::PipelineParallelTrainer`] — staged
//!   forward/backward across pipeline ranks.
//! - [`tensor_parallel::TensorParallelTrainer`] — sharded matmul
//!   coordinator.
//! - [`parameter_server::AsyncParameterServer`] — async PS protocol.
//! - [`optimizer`] / [`loss`] — typed enums for the common choices.

pub mod data_parallel;
pub mod loss;
pub mod optimizer;
pub mod parameter_server;
pub mod pipeline_parallel;
pub mod tensor_parallel;

pub mod prelude {
    //! Canonical re-exports. `use atomr_accel_train::prelude::*;`.
    pub use crate::data_parallel::{
        DataParallelTrainer, ReplicaStepResult, TrainSample, TrainerConfig, TrainerMsg,
    };
    pub use crate::loss::LossKind;
    pub use crate::optimizer::{OptimizerKind, StepStats};
    pub use crate::parameter_server::{
        AsyncParameterServer, ParameterServerMsg, ParameterServerStats, WorkerId,
    };
    pub use crate::pipeline_parallel::{
        PipelineConfig, PipelineParallelTrainer, PipelineTrainerMsg,
    };
    pub use crate::tensor_parallel::{
        ShardStepResult, TensorParallelConfig, TensorParallelMsg, TensorParallelTrainer,
    };
}
