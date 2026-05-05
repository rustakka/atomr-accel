//! `TrtActor` — sibling of `atomr_accel_cuda::DeviceActor`.
//!
//! Lifecycle:
//! - On `Build` it consumes a network builder (or ONNX bytes when
//!   `tensorrt-onnx` is enabled) plus an [`IBuilderConfig`], drives
//!   `IBuilder::buildSerializedNetwork` and returns an
//!   [`EnginePlan`].
//! - On `Deserialize` it loads a previously built plan into an
//!   [`TrtEngine`].
//! - On `CreateContext` it creates a fresh [`ExecutionContext`].
//! - On `EnqueueOnStream { stream, context, reply }` it submits the
//!   inference on the supplied `Arc<cudarc::driver::CudaStream>` —
//!   the same stream type carried by `DeviceActor` so the two actors
//!   share one CUDA execution timeline.
//! - On `Refit` it patches engine weights via [`TrtRefitter`].
//!
//! The actor keeps the `TrtEngine` alive in an `Arc` so multiple
//! `ExecutionContext`s can share it.

use std::sync::Arc;

use tokio::sync::oneshot;

use crate::builder::IBuilderConfig;
use crate::engine::{EnginePlan, TrtEngine};
use crate::error::TrtError;
use crate::runtime::{ExecutionBindings, ExecutionContext};

/// Network description for `TrtMsg::Build`. The builder API has
/// many entry points; for now we accept either a serialised ONNX blob
/// (under `tensorrt-onnx`) or a precompiled TensorRT plan to import.
#[derive(Debug, Clone)]
pub enum NetworkSource {
    /// Raw ONNX bytes. Requires the `tensorrt-onnx` feature.
    Onnx(Vec<u8>),
    /// A previously serialised TensorRT plan; just deserialise.
    SerializedPlan(Vec<u8>),
}

/// Descriptor of a single weight blob to push into the engine via
/// the refitter. The pointer / device pointer is **not** held inside
/// the message; instead callers pass a host-side blob (refitter
/// stages it). Future variants can add a `DevicePtr` tag if direct
/// device-to-device refit is desired.
pub struct RefitWeights {
    pub name: String,
    pub bytes: Vec<u8>,
    pub dtype: crate::sys::DataType,
}

/// Reply types for each `TrtMsg` variant. Each is a `oneshot::Sender`
/// so the actor never blocks on IO.
pub type BuildReply = oneshot::Sender<Result<EnginePlan, TrtError>>;
pub type DeserializeReply = oneshot::Sender<Result<Arc<TrtEngine>, TrtError>>;
pub type CreateContextReply = oneshot::Sender<Result<ExecutionContext, TrtError>>;
pub type EnqueueReply = oneshot::Sender<Result<(), TrtError>>;
pub type RefitReply = oneshot::Sender<Result<(), TrtError>>;

/// Public message surface for `TrtActor`.
///
/// The variant `EnqueueOnStream` accepts the `Arc<CudaStream>` from
/// `atomr-accel-cuda::DeviceActor` so the TensorRT runtime shares
/// the device's stream timeline (no cross-stream synchronisation,
/// no extra event hops).
pub enum TrtMsg {
    /// Build a TensorRT engine from a network source + config.
    /// Returns the serialised plan on success.
    Build {
        source: NetworkSource,
        config: Box<IBuilderConfig>,
        reply: BuildReply,
    },

    /// Deserialise a plan blob into a shared engine handle.
    Deserialize {
        plan: EnginePlan,
        reply: DeserializeReply,
    },

    /// Create a fresh `IExecutionContext` from an existing engine.
    /// Returns the new context (caller owns it).
    CreateContext {
        engine: Arc<TrtEngine>,
        reply: CreateContextReply,
    },

    /// Submit `enqueueV3` on the supplied CUDA stream. The actor
    /// returns immediately after submission; real GPU completion is
    /// observed by `atomr-accel-cuda`'s completion strategy on the
    /// shared stream.
    EnqueueOnStream {
        stream: Arc<cudarc::driver::CudaStream>,
        context: ExecutionContext,
        bindings: ExecutionBindings,
        reply: EnqueueReply,
    },

    /// Refit a built engine in-place with new weights. Requires the
    /// engine to have been built with `RefitPolicy::OnDemand` or
    /// `WeightsStreaming`.
    Refit {
        engine: Arc<TrtEngine>,
        weights: Vec<RefitWeights>,
        reply: RefitReply,
    },
}

/// `TrtActor` — owns nothing across messages besides the FFI
/// runtime/builder handles, all engines/contexts ride the messages.
///
/// The actor itself is intentionally minimal: most of the heavy
/// state lives in `Arc<TrtEngine>` values that the caller threads
/// through. This mirrors `DeviceActor`'s design where per-context
/// state lives in the `ContextActor` but engines live with the
/// caller.
pub struct TrtActor {
    /// Cached runtime; lazily created on first `Deserialize`. Held
    /// behind a `parking_lot::Mutex` because the actor mailbox
    /// already serialises but interior mutability avoids a redundant
    /// `&mut self` thread through every method.
    runtime: parking_lot::Mutex<Option<crate::runtime::TrtRuntime>>,
}

impl TrtActor {
    pub fn new() -> Self {
        Self {
            runtime: parking_lot::Mutex::new(None),
        }
    }

    /// Get-or-create the cached runtime. Without `tensorrt-link` the
    /// inner constructor returns `NotLinked`.
    pub fn ensure_runtime(&self) -> Result<(), TrtError> {
        let mut guard = self.runtime.lock();
        if guard.is_none() {
            *guard = Some(crate::runtime::TrtRuntime::new()?);
        }
        Ok(())
    }
}

impl Default for TrtActor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Precision;

    #[test]
    fn trt_msg_constructs() {
        // Walk every variant — confirms the message enum builds and
        // is `Send`-clean (oneshot::Sender is Send for any T).
        let (b_tx, _b_rx) = oneshot::channel();
        let _build = TrtMsg::Build {
            source: NetworkSource::SerializedPlan(vec![1, 2, 3]),
            config: Box::new(IBuilderConfig::new().with_precision(Precision::Fp16)),
            reply: b_tx,
        };

        let (d_tx, _d_rx) = oneshot::channel();
        let _deser = TrtMsg::Deserialize {
            plan: EnginePlan::new(vec![0xAA; 8]),
            reply: d_tx,
        };

        let engine = Arc::new(TrtEngine::for_test());
        let (c_tx, _c_rx) = oneshot::channel();
        let _ctx = TrtMsg::CreateContext {
            engine: engine.clone(),
            reply: c_tx,
        };

        let (r_tx, _r_rx) = oneshot::channel();
        let _refit = TrtMsg::Refit {
            engine: engine.clone(),
            weights: vec![RefitWeights {
                name: "fc.weight".into(),
                bytes: vec![0; 16],
                dtype: crate::sys::DataType::kHALF,
            }],
            reply: r_tx,
        };

        // Verify the actor itself is Send so it can live inside an
        // `atomr_core::actor::Actor`.
        fn assert_send<T: Send>() {}
        assert_send::<TrtActor>();
    }

    #[test]
    fn actor_runtime_lazy_init() {
        let actor = TrtActor::new();
        // Without the link feature this should error cleanly, never
        // panic.
        #[cfg(not(feature = "tensorrt-link"))]
        {
            let r = actor.ensure_runtime();
            assert!(matches!(r, Err(TrtError::NotLinked(_))));
        }
        #[cfg(feature = "tensorrt-link")]
        {
            // Real link path is exercised by integration tests with a
            // GPU host.
            let _ = actor;
        }
    }
}
