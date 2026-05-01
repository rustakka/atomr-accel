//! Distributed training blueprints (§7.3).
//!
//! F4 ships:
//! - [`optimizer`] — Optimizer enum (SGD, AdamW).
//! - [`loss`] — Loss kinds (MSE, CrossEntropy).
//! - [`data_parallel::DataParallelTrainer`] — N-replica trainer
//!   skeleton wired to NCCL all-reduce.
//!
//! F5 fleshes out: pipeline_parallel, tensor_parallel,
//! parameter_server.

pub mod data_parallel;
pub mod loss;
pub mod optimizer;
pub mod parameter_server;
pub mod pipeline_parallel;
pub mod tensor_parallel;
