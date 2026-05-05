//! Safe wrapper around `nvinfer1::ICudaEngine`.
//!
//! The C++ object is `*mut sys::ICudaEngine`; we wrap it in a newtype
//! that owns the pointer and is `Send + Sync` because the engine is
//! immutable post-build (multiple `IExecutionContext`s share it
//! safely). The `Drop` impl calls the FFI destroy shim under the
//! `tensorrt-link` feature; without the feature the pointer is always
//! null and `Drop` is a no-op (so unit tests construct engines without
//! libnvinfer).

use std::sync::Arc;

use crate::error::TrtError;
use crate::sys;

/// Owned, immutable TensorRT engine.
///
/// Built either from a serialised plan via [`TrtRuntime::deserialize`]
/// or from a fresh build via [`crate::builder::IBuilderConfig`] +
/// `TrtActor::Build`.
pub struct TrtEngine {
    raw: *mut sys::ICudaEngine,
    /// Cached number of I/O tensors; populated under the link feature.
    num_io: usize,
}

// SAFETY: post-build engines are immutable and the C++ runtime is
// thread-safe for concurrent reads / `IExecutionContext` creation.
unsafe impl Send for TrtEngine {}
unsafe impl Sync for TrtEngine {}

impl TrtEngine {
    /// Construct a wrapper from a raw pointer obtained from the FFI
    /// shim. Returns `Err` if the pointer is null.
    ///
    /// # Safety
    /// Caller must ensure `raw` was returned by a TensorRT runtime /
    /// builder shim and has not been destroyed.
    pub unsafe fn from_raw(raw: *mut sys::ICudaEngine, num_io: usize) -> Result<Self, TrtError> {
        if raw.is_null() {
            Err(TrtError::NullEngine)
        } else {
            Ok(Self { raw, num_io })
        }
    }

    /// Test-only constructor (no FFI). Used by the unit tests to
    /// exercise the Send/Sync newtype on hosts without libnvinfer.
    #[allow(dead_code)]
    pub(crate) fn for_test() -> Self {
        Self {
            raw: std::ptr::null_mut(),
            num_io: 0,
        }
    }

    pub fn raw(&self) -> *mut sys::ICudaEngine {
        self.raw
    }

    pub fn num_io_tensors(&self) -> usize {
        self.num_io
    }

    /// Wrap the engine in an `Arc<TrtEngine>` so multiple
    /// `ExecutionContext`s can share it.
    pub fn into_shared(self) -> Arc<TrtEngine> {
        Arc::new(self)
    }
}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        #[cfg(feature = "tensorrt-link")]
        unsafe {
            if !self.raw.is_null() {
                sys::atomr_trt_engine_destroy(self.raw);
            }
        }
        // Without `tensorrt-link`: pointer is null (test-only path),
        // nothing to free.
    }
}

/// Owned plan blob (serialised engine).
///
/// Stored as a `Vec<u8>` rather than the TensorRT `IHostMemory*` so
/// it survives shim teardown and can be journaled / written to disk.
#[derive(Debug, Clone)]
pub struct EnginePlan(pub Vec<u8>);

impl EnginePlan {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

/// Refit handle — holds an `IRefitter*` for in-place engine weight
/// updates.
pub struct TrtRefitter {
    raw: *mut sys::IRefitter,
}

unsafe impl Send for TrtRefitter {}
unsafe impl Sync for TrtRefitter {}

impl TrtRefitter {
    /// # Safety
    /// `raw` must be a valid pointer returned by the refitter shim.
    pub unsafe fn from_raw(raw: *mut sys::IRefitter) -> Result<Self, TrtError> {
        if raw.is_null() {
            Err(TrtError::Refit("null refitter".into()))
        } else {
            Ok(Self { raw })
        }
    }

    #[allow(dead_code)]
    pub(crate) fn for_test() -> Self {
        Self {
            raw: std::ptr::null_mut(),
        }
    }

    pub fn raw(&self) -> *mut sys::IRefitter {
        self.raw
    }
}

impl Drop for TrtRefitter {
    fn drop(&mut self) {
        #[cfg(feature = "tensorrt-link")]
        unsafe {
            if !self.raw.is_null() {
                sys::atomr_trt_refitter_destroy(self.raw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn engine_handle_send_sync() {
        // Newtype must be Send + Sync so it can live inside Arc<...>
        // and ride actor messages across tokio threads.
        assert_send_sync::<TrtEngine>();
        assert_send_sync::<Arc<TrtEngine>>();
        assert_send_sync::<TrtRefitter>();

        let e = TrtEngine::for_test();
        assert_eq!(e.num_io_tensors(), 0);
        let shared: Arc<TrtEngine> = e.into_shared();
        assert!(Arc::strong_count(&shared) >= 1);
    }

    #[test]
    fn engine_plan_round_trip() {
        let plan = EnginePlan::new(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(plan.as_slice(), &[0xDE, 0xAD, 0xBE, 0xEF]);
    }
}
