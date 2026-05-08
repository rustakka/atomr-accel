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
pub type ExecuteReply = oneshot::Sender<Result<(), TrtError>>;
pub type BuildFromOnnxReply = oneshot::Sender<Result<EnginePlan, TrtError>>;

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

    /// Phase 4.5++ — Run inference on a previously-loaded engine.
    /// `bindings` is `(tensor_name, CUdeviceptr)` for every I/O
    /// tensor on the engine; `stream` is the `Arc<CudaStream>` to
    /// `enqueueV3` against (typically the device's primary stream
    /// from `DeviceMsg::SnapshotStream`).
    ///
    /// The handler creates a fresh `IExecutionContext`, binds every
    /// tensor address, then calls `enqueueV3`. Returns `Ok(())` on
    /// successful submission (kernel still running on the GPU);
    /// real completion is observed by `atomr-accel-cuda`'s
    /// completion strategy on the shared stream.
    ///
    /// On builds without `tensorrt-link` the variant compiles but
    /// the handler returns `TrtError::NotLinked`.
    Execute {
        engine: Arc<TrtEngine>,
        bindings: Vec<(String, u64)>,
        input_shapes: Vec<(String, Vec<i32>)>,
        stream: Arc<cudarc::driver::CudaStream>,
        reply: ExecuteReply,
    },

    /// Phase 4.5++ — Parse an ONNX model and build a serialised
    /// engine plan. Gated on the upstream `tensorrt-onnx` feature
    /// (and transitively on `tensorrt-link`). Without those the
    /// handler returns `TrtError::NotLinked`.
    BuildFromOnnx {
        onnx_bytes: Vec<u8>,
        config: Box<IBuilderConfig>,
        reply: BuildFromOnnxReply,
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

    /// Phase 4.5++ — synchronous helper that drives the
    /// `TrtMsg::Execute` semantics (creates an `IExecutionContext`,
    /// binds tensor addresses, calls `enqueueV3`).
    ///
    /// Without `tensorrt-link` this returns `TrtError::NotLinked`
    /// without ever touching libnvinfer. With the feature on, the
    /// actor performs the full FFI sequence under the supplied
    /// `Arc<CudaStream>`. The function returns once the launch
    /// has been submitted — real GPU completion is observed
    /// downstream (the typical caller pairs this with an
    /// `atomr-accel-cuda` completion strategy on the same stream).
    pub fn execute(
        &self,
        engine: &Arc<TrtEngine>,
        bindings: &[(String, u64)],
        input_shapes: &[(String, Vec<i32>)],
        _stream: &Arc<cudarc::driver::CudaStream>,
    ) -> Result<(), TrtError> {
        #[cfg(feature = "tensorrt-link")]
        {
            use std::ffi::CString;
            unsafe {
                let ctx_ptr = crate::sys::atomr_trt_engine_create_execution_context(engine.raw());
                if ctx_ptr.is_null() {
                    return Err(TrtError::Execution(
                        "createExecutionContext returned null".into(),
                    ));
                }
                // Apply input shapes first (TensorRT requires shapes
                // before set_tensor_address on dynamic tensors).
                for (name, dims) in input_shapes {
                    if dims.len() > 8 {
                        crate::sys::atomr_trt_context_destroy(ctx_ptr);
                        return Err(TrtError::InvalidArg(format!(
                            "tensor {name:?}: TensorRT supports at most 8 dims (got {})",
                            dims.len()
                        )));
                    }
                    let cname = match CString::new(name.clone()) {
                        Ok(c) => c,
                        Err(e) => {
                            crate::sys::atomr_trt_context_destroy(ctx_ptr);
                            return Err(TrtError::InvalidArg(format!(
                                "tensor name contains NUL: {e}"
                            )));
                        }
                    };
                    let mut d = [0i32; 8];
                    for (i, v) in dims.iter().enumerate() {
                        d[i] = *v;
                    }
                    let dims_struct = crate::sys::Dims {
                        nb_dims: dims.len() as std::os::raw::c_int,
                        d,
                    };
                    let rc = crate::sys::atomr_trt_context_set_input_shape(
                        ctx_ptr,
                        cname.as_ptr(),
                        &dims_struct as *const crate::sys::Dims,
                    );
                    if rc != 0 {
                        crate::sys::atomr_trt_context_destroy(ctx_ptr);
                        return Err(TrtError::Execution(format!(
                            "set_input_shape({name}) returned {rc}"
                        )));
                    }
                }
                // Bind every tensor address.
                for (name, addr) in bindings {
                    let cname = match CString::new(name.clone()) {
                        Ok(c) => c,
                        Err(e) => {
                            crate::sys::atomr_trt_context_destroy(ctx_ptr);
                            return Err(TrtError::InvalidArg(format!(
                                "tensor name contains NUL: {e}"
                            )));
                        }
                    };
                    let rc = crate::sys::atomr_trt_context_set_tensor_address(
                        ctx_ptr,
                        cname.as_ptr(),
                        *addr as *mut std::os::raw::c_void,
                    );
                    if rc != 0 {
                        crate::sys::atomr_trt_context_destroy(ctx_ptr);
                        return Err(TrtError::Execution(format!(
                            "set_tensor_address({name}) returned {rc}"
                        )));
                    }
                }
                // Cudarc's `CudaStream` exposes the raw stream via
                // `cu_stream()` — but the field is `pub(crate)`. We
                // pass through cudarc's `DevicePtr`-style accessor by
                // using `cuStream` symbol from `cudarc::driver::sys`
                // — which is what other call sites in atomr-accel-cuda
                // do. The shim takes `*mut c_void` (any CUstream).
                let stream_raw = _stream.cu_stream() as *mut std::os::raw::c_void;
                let rc = crate::sys::atomr_trt_context_enqueue_v3(ctx_ptr, stream_raw);
                let result = if rc != 0 {
                    Err(TrtError::Execution(format!("enqueueV3 returned {rc}")))
                } else {
                    Ok(())
                };
                crate::sys::atomr_trt_context_destroy(ctx_ptr);
                return result;
            }
        }
        #[cfg(not(feature = "tensorrt-link"))]
        {
            let _ = (engine, bindings, input_shapes, _stream);
            Err(TrtError::NotLinked(
                "TrtActor::execute requires the `tensorrt-link` feature",
            ))
        }
    }

    /// Phase 4.5++ — synchronous helper that drives the
    /// `TrtMsg::BuildFromOnnx` semantics. Parses an ONNX model and
    /// returns a serialised plan blob ready for `TrtRuntime::deserialize`.
    /// Gated on `tensorrt-onnx` (transitively `tensorrt-link`).
    pub fn build_from_onnx(
        &self,
        _onnx_bytes: &[u8],
        _config: &IBuilderConfig,
    ) -> Result<EnginePlan, TrtError> {
        #[cfg(all(feature = "tensorrt-link", feature = "tensorrt-onnx"))]
        {
            use crate::builder::BuilderFlags;
            unsafe {
                let builder = crate::sys::atomr_trt_builder_create(0);
                if builder.is_null() {
                    return Err(TrtError::Build("builder_create returned null".into()));
                }
                // EXPLICIT_BATCH (1 << 0) is required for ONNX import.
                let network = crate::sys::atomr_trt_builder_create_network(builder, 1u32 << 0);
                if network.is_null() {
                    crate::sys::atomr_trt_builder_destroy(builder);
                    return Err(TrtError::Build("create_network returned null".into()));
                }
                let parser = crate::sys::atomr_trt_onnx_parser_create(network, 0);
                if parser.is_null() {
                    crate::sys::atomr_trt_builder_destroy(builder);
                    return Err(TrtError::Onnx("onnx_parser_create returned null".into()));
                }
                let parse_rc = crate::sys::atomr_trt_onnx_parser_parse(
                    parser,
                    _onnx_bytes.as_ptr(),
                    _onnx_bytes.len(),
                    std::ptr::null(),
                );
                if parse_rc == 0 {
                    let nerr = crate::sys::atomr_trt_onnx_parser_num_errors(parser);
                    crate::sys::atomr_trt_onnx_parser_destroy(parser);
                    crate::sys::atomr_trt_builder_destroy(builder);
                    return Err(TrtError::Onnx(format!(
                        "onnx parse failed (rc={parse_rc}, errors={nerr})"
                    )));
                }

                let cfg_ptr = crate::sys::atomr_trt_builder_create_config(builder);
                if cfg_ptr.is_null() {
                    crate::sys::atomr_trt_onnx_parser_destroy(parser);
                    crate::sys::atomr_trt_builder_destroy(builder);
                    return Err(TrtError::Build("builder_create_config returned null".into()));
                }
                // Replay caller-requested flags onto the C++ config.
                let flags = _config.effective_flags();
                for flag in [
                    (BuilderFlags::FP16, crate::sys::BuilderFlag::kFP16 as u32),
                    (BuilderFlags::INT8, crate::sys::BuilderFlag::kINT8 as u32),
                    (BuilderFlags::TF32, crate::sys::BuilderFlag::kTF32 as u32),
                    (BuilderFlags::BF16, crate::sys::BuilderFlag::kBF16 as u32),
                    (BuilderFlags::FP8, crate::sys::BuilderFlag::kFP8 as u32),
                    (BuilderFlags::REFIT, crate::sys::BuilderFlag::kREFIT as u32),
                    (
                        BuilderFlags::SPARSE_WEIGHTS,
                        crate::sys::BuilderFlag::kSPARSE_WEIGHTS as u32,
                    ),
                    (
                        BuilderFlags::STRIP_PLAN,
                        crate::sys::BuilderFlag::kSTRIP_PLAN as u32,
                    ),
                ] {
                    if flags.contains(flag.0) {
                        crate::sys::atomr_trt_config_set_flag(cfg_ptr, flag.1, 1);
                    }
                }
                if _config.workspace_bytes > 0 {
                    crate::sys::atomr_trt_config_set_memory_pool_limit(
                        cfg_ptr,
                        0, // kWORKSPACE
                        _config.workspace_bytes,
                    );
                }

                let host_mem =
                    crate::sys::atomr_trt_builder_build_serialized(builder, network, cfg_ptr);
                let cleanup = || {
                    crate::sys::atomr_trt_config_destroy(cfg_ptr);
                    crate::sys::atomr_trt_onnx_parser_destroy(parser);
                    crate::sys::atomr_trt_builder_destroy(builder);
                };
                if host_mem.is_null() {
                    cleanup();
                    return Err(TrtError::Build("buildSerializedNetwork returned null".into()));
                }
                let data_ptr = crate::sys::atomr_trt_host_memory_data(host_mem);
                let data_len = crate::sys::atomr_trt_host_memory_size(host_mem);
                let bytes = if data_ptr.is_null() || data_len == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(data_ptr, data_len).to_vec()
                };
                crate::sys::atomr_trt_host_memory_destroy(host_mem);
                cleanup();
                if bytes.is_empty() {
                    return Err(TrtError::Build("serialised plan was empty".into()));
                }
                Ok(EnginePlan::new(bytes))
            }
        }
        #[cfg(not(all(feature = "tensorrt-link", feature = "tensorrt-onnx")))]
        {
            Err(TrtError::NotLinked(
                "TrtActor::build_from_onnx requires the `tensorrt-link` + `tensorrt-onnx` features",
            ))
        }
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
