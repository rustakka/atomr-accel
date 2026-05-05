//! ONNX import via `nvonnxparser`.
//!
//! Gated by the `tensorrt-onnx` feature so the base crate compiles
//! without `libnvonnxparser.so`.

#![cfg(feature = "tensorrt-onnx")]
#![allow(dead_code)]

use std::sync::Arc;
use tokio::sync::oneshot;

use crate::builder::IBuilderConfig;
use crate::engine::EnginePlan;
use crate::error::TrtError;
use crate::sys;

/// Owned `IOnnxParser*` wrapper. The parser is paired with an
/// `INetworkDefinition*`, so under the link feature both pointers
/// must be passed in together.
pub struct OnnxParser {
    raw: *mut sys::IOnnxParser,
}

unsafe impl Send for OnnxParser {}
unsafe impl Sync for OnnxParser {}

impl OnnxParser {
    /// # Safety
    /// `raw` must be a valid pointer returned by
    /// `nvonnxparser::createParser`.
    pub unsafe fn from_raw(raw: *mut sys::IOnnxParser) -> Result<Self, TrtError> {
        if raw.is_null() {
            Err(TrtError::Onnx("null parser".into()))
        } else {
            Ok(Self { raw })
        }
    }

    pub(crate) fn for_test() -> Self {
        Self {
            raw: std::ptr::null_mut(),
        }
    }

    pub fn raw(&self) -> *mut sys::IOnnxParser {
        self.raw
    }
}

impl Drop for OnnxParser {
    fn drop(&mut self) {
        #[cfg(feature = "tensorrt-link")]
        unsafe {
            if !self.raw.is_null() {
                sys::atomr_trt_onnx_parser_destroy(self.raw);
            }
        }
    }
}

/// Reply for `OnnxMsg::Parse`: serialised engine plan ready for
/// `TrtRuntime::deserialize`.
pub type ParseReply = oneshot::Sender<Result<EnginePlan, TrtError>>;

/// Public messages for the ONNX parser actor surface.
pub enum OnnxMsg {
    /// Parse + build in one step. The actor implementation wires
    /// the parser into a fresh `INetworkDefinition`, runs the
    /// `IBuilder`, and returns the serialised plan.
    Parse {
        bytes: Arc<Vec<u8>>,
        config: Box<IBuilderConfig>,
        reply: ParseReply,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::Precision;

    #[test]
    fn parser_msg_constructs() {
        // The Parse variant must construct on hosts without
        // libnvonnxparser. Only the FFI calls are gated by
        // `tensorrt-link`.
        let (tx, _rx) = oneshot::channel();
        let _msg = OnnxMsg::Parse {
            bytes: Arc::new(b"\x08\x07onnx-bytes-here".to_vec()),
            config: Box::new(IBuilderConfig::new().with_precision(Precision::Fp16)),
            reply: tx,
        };

        // Newtype wrapper is Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OnnxParser>();

        let p = OnnxParser::for_test();
        assert!(p.raw().is_null());
    }
}
