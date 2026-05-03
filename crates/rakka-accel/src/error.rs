//! Backend-agnostic error taxonomy.
//!
//! Every backend's error type implements `From<AccelError>`, so
//! generic code can return [`AccelError`] directly while
//! backend-specific code keeps richer typed variants. The enum is
//! `#[non_exhaustive]` — backends are free to add `LibraryError`
//! tags (`"cublas"`, `"cudnn"`, `"hipblas"`, `"mps"`, etc.) without
//! breaking core consumers.

use thiserror::Error;

pub type AccelResult<T> = Result<T, AccelError>;

/// Marker prefix used in panic messages to signal a poisoned-context
/// error. Backends panic with messages containing these tags so the
/// supervisor decider can route to `Restart` / `Resume` / `Stop` /
/// `Escalate` without parsing the typed enum from a panic payload.
pub const CONTEXT_POISONED_TAG: &str = "ContextPoisoned";
pub const OUT_OF_MEMORY_TAG: &str = "OutOfMemory";
pub const UNRECOVERABLE_TAG: &str = "Unrecoverable";

/// Typed error enum surfaced through every actor reply channel.
///
/// Mirrors the original `GpuError` from `rakka-accel-cuda` but lives
/// in the backend-agnostic core. Backends wrap or re-export this as
/// their public `Error` associated type.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AccelError {
    /// Device context is in a sticky-error state. Triggers
    /// `ContextActor` restart and a generation bump.
    #[error("ContextPoisoned: {0}")]
    ContextPoisoned(String),

    /// Allocation failed but the context is still usable. Supervisor
    /// `Resume`s the actor.
    #[error("OutOfMemory: {0}")]
    OutOfMemory(String),

    /// Hardware fault or repeated poisoning past the retry budget.
    #[error("Unrecoverable: {0}")]
    Unrecoverable(String),

    /// `AccelRef::access()` was called on a buffer whose context was
    /// rebuilt or whose `DeviceActor` is shutting down.
    #[error("AccelRef stale: {0}")]
    AccelRefStale(&'static str),

    /// Driver-level error (e.g. `cuInit`, `hipInit`, `MTLDevice`
    /// setup) before any specific library got involved.
    #[error("driver error: {0}")]
    Driver(String),

    /// Library error tagged with the originating component name —
    /// e.g. `"cublas"`, `"cudnn"`, `"cufft"`, `"curand"`,
    /// `"cusolver"`, `"cublaslt"`, `"nvrtc"`, `"nccl"`, `"hipblas"`,
    /// `"rocfft"`, `"mps"`. Callers that need to discriminate match
    /// on `lib`.
    #[error("{lib} error: {msg}")]
    LibraryError { lib: &'static str, msg: String },

    #[error("ask timed out before completion")]
    Timeout,
}

impl AccelError {
    /// Construct a tagged library error.
    pub fn lib(lib: &'static str, msg: impl Into<String>) -> Self {
        Self::LibraryError {
            lib,
            msg: msg.into(),
        }
    }

    /// Format suitable for panicking out of an actor handler so
    /// that the rakka supervisor's decider can route it to a
    /// directive based on the tagged prefix.
    pub fn panic_message(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_error_constructor() {
        let e = AccelError::lib("cudnn", "create_handle failed");
        match e {
            AccelError::LibraryError { lib, msg } => {
                assert_eq!(lib, "cudnn");
                assert!(msg.contains("create_handle"));
            }
            _ => panic!("expected LibraryError"),
        }
    }

    #[test]
    fn panic_message_carries_tag() {
        let e = AccelError::ContextPoisoned("cuInit failed".into());
        let m = e.panic_message();
        assert!(m.contains(CONTEXT_POISONED_TAG));
    }
}
