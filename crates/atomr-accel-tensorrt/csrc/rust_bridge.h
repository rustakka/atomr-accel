// atomr-accel-tensorrt: C-ABI vtable shared between Rust and the
// `csrc/plugin_proxy.cpp` shim. Each function pointer dispatches a
// vtable-method call from the C++ proxy back to the corresponding
// Rust trait method on `dyn PluginV3`.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nvinfer_shim.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AtomrPluginVTable {
    // `user` is the leaked `Box<Arc<dyn PluginV3>>` raw pointer;
    // every method dereferences it back to a Rust trait object.
    const char* (*get_name)      (const void* user);
    const char* (*get_version)   (const void* user);
    const char* (*get_namespace) (const void* user);
    // Construct a per-instance plugin object. Returns an opaque
    // `void*` that the C++ proxy wraps in a `RustPluginV3Proxy`.
    // The returned pointer must be `destroy`able via the same
    // vtable's `destroy_instance`.
    void* (*create_plugin) (const void* user, const char* name);
    // Drop the leaked `Box<Arc<dyn PluginV3>>` carried as `user`.
    // Called from `~RustPluginCreatorProxy`.
    void (*destroy)        (void* user);
    // Drop a per-instance plugin pointer returned by `create_plugin`.
    void (*destroy_instance)(void* instance);
} AtomrPluginVTable;

// Construct a `nvinfer1::IPluginCreatorV3One` proxy (returned as the
// opaque `atomr_trt_IPluginCreator*` mirror) bound to the supplied
// vtable + user pointer. Caller retains ownership of the vtable copy
// by-value the proxy makes; `user` ownership transfers to the proxy
// (released via `vt->destroy(user)` at proxy destruction time).
atomr_trt_IPluginCreator* atomr_trt_make_plugin_creator(
    const AtomrPluginVTable* vt, void* user);

#ifdef __cplusplus
}
#endif
