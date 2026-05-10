// atomr-accel-tensorrt: IPluginCreatorV3One proxy that bridges to a
// Rust `Arc<dyn PluginV3>`. Compiled only when the
// `tensorrt-plugin` cargo feature is on (gated by build.rs).
//
// Phase 8 scope: this proxy makes `getPluginRegistry()->registerCreator()`
// accept a Rust-implemented plugin so plugins can advertise their
// `(name, version, namespace)` to the registry. The per-instance
// `IPluginV3` runtime methods (`enqueue`, `getOutputShapes`,
// `configure`) return safe-default failure codes; full pass-through to
// `Rust PluginV3::enqueue` lands in a follow-up that wires Rust
// callbacks for each runtime method.
//
// SPDX-License-Identifier: Apache-2.0

#include "rust_bridge.h"

#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>

#include <cstring>
#include <new>

namespace {

class RustPluginV3Proxy : public nvinfer1::IPluginV3 {
public:
    RustPluginV3Proxy(const AtomrPluginVTable* vt, void* user, void* instance)
        : vt_(*vt), user_(user), instance_(instance) {}

    ~RustPluginV3Proxy() override {
        if (vt_.destroy_instance && instance_) {
            vt_.destroy_instance(instance_);
        }
    }

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType /*type*/) noexcept override {
        // Phase 8: not yet routed to Rust. Returning nullptr causes
        // TRT to fall back to the legacy plugin path; an inference-
        // time use surfaces a clear error rather than a UB segfault.
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        if (!vt_.create_plugin) return nullptr;
        // Re-construct via the creator — calls back into Rust to
        // produce a fresh per-instance handle, then wraps it.
        const char* name = vt_.get_name ? vt_.get_name(user_) : "";
        void* fresh = vt_.create_plugin(user_, name ? name : "");
        if (!fresh) return nullptr;
        return new (std::nothrow) RustPluginV3Proxy(&vt_, user_, fresh);
    }

private:
    AtomrPluginVTable vt_;
    void* user_;
    void* instance_;
};

class RustPluginCreatorProxy : public nvinfer1::IPluginCreatorV3One {
public:
    RustPluginCreatorProxy(const AtomrPluginVTable* vt, void* user)
        : vt_(*vt), user_(user) {
        empty_fc_.nbFields = 0;
        empty_fc_.fields = nullptr;
    }

    ~RustPluginCreatorProxy() override {
        if (vt_.destroy) {
            vt_.destroy(user_);
        }
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return vt_.get_name ? vt_.get_name(user_) : "";
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return vt_.get_version ? vt_.get_version(user_) : "1";
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &empty_fc_;
    }

    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const* name,
        nvinfer1::PluginFieldCollection const* /*fc*/,
        nvinfer1::TensorRTPhase /*phase*/) noexcept override {
        if (!vt_.create_plugin) return nullptr;
        void* instance = vt_.create_plugin(user_, name ? name : "");
        if (!instance) return nullptr;
        return new (std::nothrow) RustPluginV3Proxy(&vt_, user_, instance);
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return vt_.get_namespace ? vt_.get_namespace(user_) : "";
    }

private:
    AtomrPluginVTable vt_;
    void* user_;
    nvinfer1::PluginFieldCollection empty_fc_{};
};

}  // namespace

extern "C" atomr_trt_IPluginCreator* atomr_trt_make_plugin_creator(
    const AtomrPluginVTable* vt, void* user) {
    if (!vt) return nullptr;
    auto* proxy = new (std::nothrow) RustPluginCreatorProxy(vt, user);
    return reinterpret_cast<atomr_trt_IPluginCreator*>(proxy);
}
