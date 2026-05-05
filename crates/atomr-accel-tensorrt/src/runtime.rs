//! Safe wrapper around `nvinfer1::IExecutionContext` and the
//! `IRuntime` deserialiser. This is the inference-time hot path.
//!
//! The runtime actor (`TrtActor`, see `actor.rs`) drives an
//! `ExecutionContext` per inference, calling `enqueueV3` on a CUDA
//! stream provided by `atomr-accel-cuda::DeviceActor`. The actor
//! never blocks on the GPU; completion is signalled via the same
//! host-fn-completion mechanism the BLAS/cuDNN actors use.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::oneshot;

use crate::engine::TrtEngine;
use crate::error::TrtError;
use crate::sys;

/// Shape of a dynamic tensor input. Exactly mirrors `nvinfer1::Dims`
/// (max 8 dims).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorShape {
    pub nb_dims: usize,
    pub dims: [i32; 8],
}

impl TensorShape {
    pub fn new(dims: &[i32]) -> Self {
        assert!(dims.len() <= 8, "TensorRT supports at most 8 dimensions");
        let mut out = [0i32; 8];
        out[..dims.len()].copy_from_slice(dims);
        Self {
            nb_dims: dims.len(),
            dims: out,
        }
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.dims[..self.nb_dims]
    }
}

/// Per-call inputs/outputs: tensor name → device pointer.
/// Pointers are raw `u64`s (CUDA device addresses) so the message is
/// `Send + Sync` without lifetimes from `Arc<CudaSlice<T>>`.
#[derive(Debug, Clone, Default)]
pub struct ExecutionBindings {
    pub addresses: HashMap<String, u64>,
    pub input_shapes: HashMap<String, TensorShape>,
}

impl ExecutionBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(&mut self, name: impl Into<String>, device_ptr: u64) -> &mut Self {
        self.addresses.insert(name.into(), device_ptr);
        self
    }

    pub fn set_shape(&mut self, name: impl Into<String>, shape: TensorShape) -> &mut Self {
        self.input_shapes.insert(name.into(), shape);
        self
    }
}

/// Wrapper around an owned `IExecutionContext*`. Held inside the
/// `TrtActor` and is **not** `Send` to outside callers — the actor
/// owns it for life and serialises access.
pub struct ExecutionContext {
    raw: *mut sys::IExecutionContext,
    engine: Arc<TrtEngine>,
}

// SAFETY: The `IExecutionContext` is only ever touched from inside the
// owning actor (single-thread access serialised by the mailbox). The
// underlying TensorRT runtime is thread-safe for concurrent
// `enqueueV3` calls *across distinct* contexts that share an engine.
unsafe impl Send for ExecutionContext {}
unsafe impl Sync for ExecutionContext {}

impl ExecutionContext {
    /// # Safety
    /// `raw` must be a valid pointer returned by
    /// `IcudaEngine::createExecutionContext`.
    pub unsafe fn from_raw(
        raw: *mut sys::IExecutionContext,
        engine: Arc<TrtEngine>,
    ) -> Result<Self, TrtError> {
        if raw.is_null() {
            Err(TrtError::Execution("null execution context".into()))
        } else {
            Ok(Self { raw, engine })
        }
    }

    pub(crate) fn for_test(engine: Arc<TrtEngine>) -> Self {
        Self {
            raw: std::ptr::null_mut(),
            engine,
        }
    }

    pub fn raw(&self) -> *mut sys::IExecutionContext {
        self.raw
    }

    pub fn engine(&self) -> &Arc<TrtEngine> {
        &self.engine
    }
}

impl Drop for ExecutionContext {
    fn drop(&mut self) {
        #[cfg(feature = "tensorrt-link")]
        unsafe {
            if !self.raw.is_null() {
                sys::atomr_trt_context_destroy(self.raw);
            }
        }
    }
}

/// Owned wrapper for `IRuntime`, used to deserialise plan blobs.
pub struct TrtRuntime {
    raw: *mut sys::IRuntime,
}

unsafe impl Send for TrtRuntime {}
unsafe impl Sync for TrtRuntime {}

impl TrtRuntime {
    /// Construct a runtime. Without the `tensorrt-link` feature this
    /// returns `Err(NotLinked)`.
    pub fn new() -> Result<Self, TrtError> {
        #[cfg(feature = "tensorrt-link")]
        {
            let raw = unsafe { sys::atomr_trt_runtime_create(0) };
            if raw.is_null() {
                Err(TrtError::Runtime("runtime create returned null".into()))
            } else {
                Ok(Self { raw })
            }
        }
        #[cfg(not(feature = "tensorrt-link"))]
        {
            Err(TrtError::NotLinked(
                "TrtRuntime requires the `tensorrt-link` feature",
            ))
        }
    }

    pub(crate) fn for_test() -> Self {
        Self {
            raw: std::ptr::null_mut(),
        }
    }

    /// Deserialise a plan blob. Without the link feature this is an
    /// error.
    pub fn deserialize(&self, _plan: &[u8]) -> Result<TrtEngine, TrtError> {
        #[cfg(feature = "tensorrt-link")]
        {
            let raw = unsafe {
                sys::atomr_trt_runtime_deserialize(self.raw, _plan.as_ptr(), _plan.len())
            };
            if raw.is_null() {
                Err(TrtError::Runtime("deserialize returned null".into()))
            } else {
                let num_io = unsafe { sys::atomr_trt_engine_num_io_tensors(raw) } as usize;
                unsafe { TrtEngine::from_raw(raw, num_io) }
            }
        }
        #[cfg(not(feature = "tensorrt-link"))]
        {
            Err(TrtError::NotLinked(
                "TrtRuntime::deserialize requires the `tensorrt-link` feature",
            ))
        }
    }
}

impl Drop for TrtRuntime {
    fn drop(&mut self) {
        #[cfg(feature = "tensorrt-link")]
        unsafe {
            if !self.raw.is_null() {
                sys::atomr_trt_runtime_destroy(self.raw);
            }
        }
    }
}

/// Reply payload for an enqueue request. Ok = stream submission
/// succeeded (kernel still running on the GPU); the caller awaits
/// real completion via the shared CUDA stream completion sentinel.
pub type EnqueueReply = Result<(), TrtError>;

/// Standalone enqueue request type — embedded into the `TrtActor`'s
/// message enum but exposed here so the message dispatcher in
/// `actor.rs` and tests can construct it without crossing module
/// boundaries.
pub struct EnqueueRequest {
    pub bindings: ExecutionBindings,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub reply: oneshot::Sender<EnqueueReply>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn execution_context_msg_round_trip() {
        let engine = Arc::new(TrtEngine::for_test());
        let ctx = ExecutionContext::for_test(engine.clone());
        assert!(Arc::ptr_eq(ctx.engine(), &engine));

        let mut bindings = ExecutionBindings::new();
        bindings
            .bind("input", 0xDEADBEEF)
            .set_shape("input", TensorShape::new(&[1, 3, 224, 224]));
        assert_eq!(bindings.addresses.get("input").copied(), Some(0xDEADBEEF));
        assert_eq!(
            bindings.input_shapes.get("input").map(|s| s.as_slice()),
            Some(&[1i32, 3, 224, 224][..])
        );

        assert_send_sync::<ExecutionBindings>();
        assert_send_sync::<TrtRuntime>();
        assert_send_sync::<ExecutionContext>();
    }

    #[test]
    fn shape_round_trip() {
        let s = TensorShape::new(&[2, 4, 8]);
        assert_eq!(s.nb_dims, 3);
        assert_eq!(s.as_slice(), &[2, 4, 8]);
    }

    #[test]
    fn runtime_unlinked_returns_not_linked() {
        // Without the link feature, TrtRuntime::new must surface a
        // clean error instead of panicking.
        #[cfg(not(feature = "tensorrt-link"))]
        {
            let r = TrtRuntime::new();
            assert!(matches!(r, Err(TrtError::NotLinked(_))));
        }
    }
}
