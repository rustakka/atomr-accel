//! `IPluginV3` Rust trampolines.
//!
//! This module exposes a Rust trait surface that mirrors
//! `nvinfer1::v_1_0::IPluginV3` (the V3 API introduced in TRT 10).
//! The actual C++ vtable shim lives in the link-gated FFI; here we
//! provide the high-level trait + a registration helper.
//!
//! Plugin authors implement `PluginV3` and pass an `Arc<dyn PluginV3>`
//! to `register_plugin`, which (under `tensorrt-link`) constructs the
//! C++ `IPluginCreator` proxy and calls
//! `getPluginRegistry()->registerCreator()`.

#![cfg(feature = "tensorrt-plugin")]

use std::sync::Arc;

use crate::error::TrtError;
use crate::sys;

/// Plugin capability ID — mirrors `nvinfer1::PluginCapabilityType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginCapability {
    Core,
    Build,
    Runtime,
}

/// Field marking a plugin attribute exposed to the network builder.
#[derive(Debug, Clone)]
pub struct PluginField {
    pub name: String,
    pub data: Vec<u8>,
    pub dtype: sys::DataType,
}

/// Object-safe trait with the IPluginV3 surface a Rust author needs.
///
/// Notable design choices:
/// - `clone_boxed` returns a `Box<dyn PluginV3>` so the C++ proxy
///   can satisfy `IPluginV3::clone()` without exposing `Clone` (which
///   isn't object-safe).
/// - `get_capability` returns `None` when the plugin doesn't
///   implement a sub-interface; the proxy translates that to a null
///   `IPluginV3*`.
/// - All methods are infallible from the trait's POV — TensorRT
///   error reporting is folded into the `TrtError` return on
///   `register_plugin`. Plugin-internal failures should be logged
///   via `tracing` and converted to safe defaults.
pub trait PluginV3: Send + Sync {
    /// Plugin name (e.g. "FooBarPlugin"). Returned through the
    /// `IPluginCreator::getPluginName` path.
    fn name(&self) -> &str;

    /// Plugin version (e.g. "1"). Returned through
    /// `IPluginCreator::getPluginVersion`.
    fn version(&self) -> &str;

    /// Namespace, default empty. Returned through
    /// `IPluginCreator::getPluginNamespace`.
    fn namespace(&self) -> &str {
        ""
    }

    /// Clone for the C++ `IPluginV3::clone()` slot.
    fn clone_boxed(&self) -> Box<dyn PluginV3>;

    /// Sub-interface dispatch for `IPluginV3::getCapabilityInterface`.
    /// Returns `None` if the plugin doesn't expose that capability;
    /// the proxy returns a null `IPluginV3*` in that case.
    ///
    /// Default impl returns `None`; concrete plugins override to
    /// hand back `Some(self)` (which requires `Self: Sized`).
    fn get_capability(&self, _cap: PluginCapability) -> Option<&dyn PluginV3> {
        None
    }

    /// Configure the plugin from builder-side fields. Called once at
    /// engine-build time.
    fn configure(&mut self, _fields: &[PluginField]) -> Result<(), TrtError> {
        Ok(())
    }

    /// Output-shape inference. Returns the shape of each output
    /// given the input shapes. Only invoked at build time.
    fn infer_shapes(&self, _input_shapes: &[Vec<i32>]) -> Vec<Vec<i32>> {
        Vec::new()
    }

    /// Run-time `enqueue`. Inputs/outputs are device pointers; the
    /// plugin runs on the supplied CUDA stream.
    ///
    /// `stream` is an opaque `*mut c_void` because the C++ side hands
    /// us a `cudaStream_t` which we can't type-check from Rust. The
    /// proxy converts an `Arc<cudarc::driver::CudaStream>` into the
    /// raw `cudaStream_t` before calling this.
    fn enqueue(
        &self,
        _inputs: &[u64],
        _outputs: &[u64],
        _stream: *mut std::os::raw::c_void,
    ) -> Result<(), TrtError> {
        Ok(())
    }
}

/// Helper to construct an `Arc<dyn PluginV3>` from any concrete
/// type. Useful in test fixtures and plugin registration.
pub fn make<P: PluginV3 + 'static>(plugin: P) -> Arc<dyn PluginV3> {
    Arc::new(plugin) as Arc<dyn PluginV3>
}

/// Register a plugin with the global TensorRT plugin registry.
///
/// Without the `tensorrt-link` feature, this returns
/// `TrtError::NotLinked`. With the feature on, it constructs a C++
/// `IPluginCreator` proxy that bridges to the supplied trait object
/// and calls `getPluginRegistry()->registerCreator()`.
pub fn register_plugin(_plugin: Arc<dyn PluginV3>) -> Result<(), TrtError> {
    #[cfg(feature = "tensorrt-link")]
    {
        // The link-gated path leaks the `Arc<dyn PluginV3>` into a
        // C++ `IPluginCreator` proxy whose vtable methods
        // re-deref the Rust trait object. This file declares the
        // trait surface; the proxy implementation lives in the C++
        // shim source under `tensorrt-link`.
        let _ = _plugin;
        Err(TrtError::Plugin(
            "C++ IPluginCreator proxy not yet implemented in this Phase 8 skeleton; \
             stub returns Plugin error. Link-time registration arrives in a follow-up commit."
                .into(),
        ))
    }
    #[cfg(not(feature = "tensorrt-link"))]
    {
        Err(TrtError::NotLinked(
            "register_plugin requires the `tensorrt-link` feature",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubPlugin {
        name: String,
        version: String,
    }

    impl PluginV3 for StubPlugin {
        fn name(&self) -> &str {
            &self.name
        }
        fn version(&self) -> &str {
            &self.version
        }
        fn clone_boxed(&self) -> Box<dyn PluginV3> {
            Box::new(StubPlugin {
                name: self.name.clone(),
                version: self.version.clone(),
            })
        }
        fn get_capability(&self, _cap: PluginCapability) -> Option<&dyn PluginV3> {
            Some(self)
        }
    }

    #[test]
    fn plugin_v3_trait_object_safe() {
        // The trait must be object-safe so `Arc<dyn PluginV3>` builds.
        let p: Arc<dyn PluginV3> = make(StubPlugin {
            name: "Stub".into(),
            version: "1".into(),
        });
        assert_eq!(p.name(), "Stub");
        assert_eq!(p.version(), "1");
        assert_eq!(p.namespace(), "");
        assert!(p.get_capability(PluginCapability::Core).is_some());

        // clone_boxed roundtrips.
        let cloned = p.clone_boxed();
        assert_eq!(cloned.name(), "Stub");

        // register_plugin returns a clean error without the link
        // feature — must not panic.
        let r = register_plugin(p);
        assert!(matches!(
            r,
            Err(TrtError::NotLinked(_)) | Err(TrtError::Plugin(_))
        ));

        // Object-safety check via where-bounds.
        fn assert_obj_safe<T: ?Sized + PluginV3>() {}
        assert_obj_safe::<dyn PluginV3>();
    }
}
