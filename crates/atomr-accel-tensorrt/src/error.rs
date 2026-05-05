//! Error types for the TensorRT actor surface.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrtError {
    #[error("libnvinfer not available: {0}")]
    NotLinked(&'static str),

    #[error("TensorRT builder failed: {0}")]
    Build(String),

    #[error("TensorRT runtime failed: {0}")]
    Runtime(String),

    #[error("TensorRT engine pointer was null")]
    NullEngine,

    #[error("TensorRT execution context error: {0}")]
    Execution(String),

    #[error("ONNX parser error: {0}")]
    Onnx(String),

    #[error("INT8 calibrator error: {0}")]
    Calibration(String),

    #[error("plugin error: {0}")]
    Plugin(String),

    #[error("refit error: {0}")]
    Refit(String),

    #[error("invalid argument: {0}")]
    InvalidArg(String),
}
