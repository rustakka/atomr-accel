// atomr-accel-tensorrt: C-ABI shim implementation for libnvinfer.
//
// Wraps the TensorRT 10.x C++ API as a set of `extern "C"` functions
// matching `src/sys.rs`. Every TRT call that may throw is wrapped in
// `try_or_null<T>(F&&)` so C++ exceptions never escape across the
// FFI boundary; null-pointer-returning failures are translated into
// either `nullptr` (for handle-returning shims) or `-1` (for int rc).
//
// SPDX-License-Identifier: Apache-2.0

#include "nvinfer_shim.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <utility>

namespace {

// Default log callback installed before atomr_trt_install_logger fires.
// Writes to stderr at WARNING+ so misuse during early init isn't silent.
void default_log(int sev, const char* msg, size_t len, void* /*user*/) {
    if (sev <= 2) {  // INTERNAL_ERROR, ERROR, WARNING per nvinfer1::ILogger::Severity
        std::fwrite("[atomr-trt] ", 1, 12, stderr);
        std::fwrite(msg, 1, len, stderr);
        std::fputc('\n', stderr);
    }
}

class RustBridgeLogger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        atomr_trt_log_cb cb;
        void* user;
        {
            std::lock_guard<std::mutex> g(mu_);
            cb = cb_;
            user = user_;
        }
        if (cb && msg) {
            cb(static_cast<int>(sev), msg, std::strlen(msg), user);
        }
    }

    void install(atomr_trt_log_cb cb, void* user) {
        std::lock_guard<std::mutex> g(mu_);
        cb_ = cb;
        user_ = user;
    }

private:
    std::mutex mu_;
    atomr_trt_log_cb cb_  = &default_log;
    void*            user_ = nullptr;
};

RustBridgeLogger g_logger;

template <typename F>
auto try_or_null(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception& e) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "TensorRT C++ exception: %s", e.what());
        g_logger.log(nvinfer1::ILogger::Severity::kERROR, buf);
        return nullptr;
    } catch (...) {
        g_logger.log(nvinfer1::ILogger::Severity::kERROR,
                     "TensorRT unknown C++ exception (...)");
        return nullptr;
    }
}

}  // namespace

extern "C" {

void atomr_trt_install_logger(atomr_trt_log_cb cb, void* user) {
    g_logger.install(cb, user);
}

// ─── Builder lifecycle ────────────────────────────────────────────────

atomr_trt_IBuilder* atomr_trt_builder_create(int /*logger_severity*/) {
    using namespace nvinfer1;
    return reinterpret_cast<atomr_trt_IBuilder*>(
        try_or_null([&]() -> IBuilder* { return createInferBuilder(g_logger); }));
}

void atomr_trt_builder_destroy(atomr_trt_IBuilder* builder) {
    delete reinterpret_cast<nvinfer1::IBuilder*>(builder);
}

atomr_trt_INetworkDefinition* atomr_trt_builder_create_network(
    atomr_trt_IBuilder* builder, uint32_t flags) {
    if (!builder) return nullptr;
    using namespace nvinfer1;
    auto* b = reinterpret_cast<IBuilder*>(builder);
    return reinterpret_cast<atomr_trt_INetworkDefinition*>(
        try_or_null([&]() -> INetworkDefinition* { return b->createNetworkV2(flags); }));
}

atomr_trt_IBuilderConfig* atomr_trt_builder_create_config(atomr_trt_IBuilder* builder) {
    if (!builder) return nullptr;
    using namespace nvinfer1;
    auto* b = reinterpret_cast<IBuilder*>(builder);
    return reinterpret_cast<atomr_trt_IBuilderConfig*>(
        try_or_null([&]() -> IBuilderConfig* { return b->createBuilderConfig(); }));
}

atomr_trt_IHostMemory* atomr_trt_builder_build_serialized(
    atomr_trt_IBuilder* builder,
    atomr_trt_INetworkDefinition* network,
    atomr_trt_IBuilderConfig* config) {
    if (!builder || !network || !config) return nullptr;
    using namespace nvinfer1;
    auto* b = reinterpret_cast<IBuilder*>(builder);
    auto* n = reinterpret_cast<INetworkDefinition*>(network);
    auto* c = reinterpret_cast<IBuilderConfig*>(config);
    return reinterpret_cast<atomr_trt_IHostMemory*>(
        try_or_null([&]() -> IHostMemory* { return b->buildSerializedNetwork(*n, *c); }));
}

// ─── BuilderConfig knobs ──────────────────────────────────────────────

void atomr_trt_config_destroy(atomr_trt_IBuilderConfig* config) {
    delete reinterpret_cast<nvinfer1::IBuilderConfig*>(config);
}

void atomr_trt_config_set_flag(atomr_trt_IBuilderConfig* config, uint32_t flag, int on) {
    if (!config) return;
    using namespace nvinfer1;
    auto* c = reinterpret_cast<IBuilderConfig*>(config);
    if (on) c->setFlag(static_cast<BuilderFlag>(flag));
    else    c->clearFlag(static_cast<BuilderFlag>(flag));
}

void atomr_trt_config_set_memory_pool_limit(
    atomr_trt_IBuilderConfig* config, int pool, size_t bytes) {
    if (!config) return;
    using namespace nvinfer1;
    reinterpret_cast<IBuilderConfig*>(config)
        ->setMemoryPoolLimit(static_cast<MemoryPoolType>(pool), bytes);
}

void atomr_trt_config_set_default_device_type(atomr_trt_IBuilderConfig* config, int dt) {
    if (!config) return;
    using namespace nvinfer1;
    reinterpret_cast<IBuilderConfig*>(config)
        ->setDefaultDeviceType(static_cast<DeviceType>(dt));
}

void atomr_trt_config_set_dla_core(atomr_trt_IBuilderConfig* config, int core) {
    if (!config) return;
    reinterpret_cast<nvinfer1::IBuilderConfig*>(config)->setDLACore(core);
}

void atomr_trt_config_set_tactic_sources(atomr_trt_IBuilderConfig* config, uint32_t mask) {
    if (!config) return;
    reinterpret_cast<nvinfer1::IBuilderConfig*>(config)->setTacticSources(mask);
}

void atomr_trt_config_set_int8_calibrator(
    atomr_trt_IBuilderConfig* config, atomr_trt_IInt8Calibrator* calibrator) {
    if (!config) return;
    reinterpret_cast<nvinfer1::IBuilderConfig*>(config)
        ->setInt8Calibrator(reinterpret_cast<nvinfer1::IInt8Calibrator*>(calibrator));
}

void atomr_trt_config_set_timing_cache(
    atomr_trt_IBuilderConfig* config, const uint8_t* blob, size_t len) {
    if (!config) return;
    using namespace nvinfer1;
    auto* c = reinterpret_cast<IBuilderConfig*>(config);
    auto* tc = c->createTimingCache(blob, len);
    if (tc) {
        c->setTimingCache(*tc, /*ignoreMismatch=*/false);
    }
}

// ─── Engine ───────────────────────────────────────────────────────────

void atomr_trt_engine_destroy(atomr_trt_ICudaEngine* engine) {
    delete reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
}

atomr_trt_IExecutionContext* atomr_trt_engine_create_execution_context(
    atomr_trt_ICudaEngine* engine) {
    if (!engine) return nullptr;
    using namespace nvinfer1;
    auto* e = reinterpret_cast<ICudaEngine*>(engine);
    return reinterpret_cast<atomr_trt_IExecutionContext*>(
        try_or_null([&]() -> IExecutionContext* { return e->createExecutionContext(); }));
}

atomr_trt_IHostMemory* atomr_trt_engine_serialize(atomr_trt_ICudaEngine* engine) {
    if (!engine) return nullptr;
    using namespace nvinfer1;
    auto* e = reinterpret_cast<ICudaEngine*>(engine);
    return reinterpret_cast<atomr_trt_IHostMemory*>(
        try_or_null([&]() -> IHostMemory* { return e->serialize(); }));
}

int atomr_trt_engine_num_io_tensors(atomr_trt_ICudaEngine* engine) {
    if (!engine) return -1;
    return reinterpret_cast<nvinfer1::ICudaEngine*>(engine)->getNbIOTensors();
}

const char* atomr_trt_engine_io_tensor_name(atomr_trt_ICudaEngine* engine, int idx) {
    if (!engine) return nullptr;
    auto* e = reinterpret_cast<nvinfer1::ICudaEngine*>(engine);
    if (idx < 0 || idx >= e->getNbIOTensors()) return nullptr;
    return e->getIOTensorName(idx);
}

atomr_trt_IRefitter* atomr_trt_engine_create_refitter(atomr_trt_ICudaEngine* engine) {
    if (!engine) return nullptr;
    using namespace nvinfer1;
    auto* e = reinterpret_cast<ICudaEngine*>(engine);
    return reinterpret_cast<atomr_trt_IRefitter*>(
        try_or_null([&]() -> IRefitter* { return createInferRefitter(*e, g_logger); }));
}

// ─── Refitter ─────────────────────────────────────────────────────────

void atomr_trt_refitter_destroy(atomr_trt_IRefitter* refitter) {
    delete reinterpret_cast<nvinfer1::IRefitter*>(refitter);
}

int atomr_trt_refitter_set_named_weights(
    atomr_trt_IRefitter* refitter,
    const char* name,
    const void* weights,
    size_t bytes,
    int dtype) {
    if (!refitter || !name || !weights) return -1;
    using namespace nvinfer1;
    auto* r = reinterpret_cast<IRefitter*>(refitter);
    Weights w{};
    w.type = static_cast<DataType>(dtype);
    w.values = weights;
    // count must be in element units, not bytes — caller passes bytes
    // and we let TRT compute element count via the dtype size.
    size_t elem_bytes = 4;  // sensible default for fp32
    switch (w.type) {
        case DataType::kHALF:  elem_bytes = 2; break;
        case DataType::kBF16:  elem_bytes = 2; break;
        case DataType::kFLOAT: elem_bytes = 4; break;
        case DataType::kINT32: elem_bytes = 4; break;
        case DataType::kINT64: elem_bytes = 8; break;
        case DataType::kINT8:  elem_bytes = 1; break;
        case DataType::kBOOL:  elem_bytes = 1; break;
        case DataType::kUINT8: elem_bytes = 1; break;
        case DataType::kFP8:   elem_bytes = 1; break;
        default: break;
    }
    w.count = static_cast<int64_t>(bytes / elem_bytes);
    return r->setNamedWeights(name, w) ? 0 : -1;
}

int atomr_trt_refitter_refit_engine(atomr_trt_IRefitter* refitter) {
    if (!refitter) return -1;
    return reinterpret_cast<nvinfer1::IRefitter*>(refitter)->refitCudaEngine() ? 0 : -1;
}

// ─── ExecutionContext ─────────────────────────────────────────────────

void atomr_trt_context_destroy(atomr_trt_IExecutionContext* ctx) {
    delete reinterpret_cast<nvinfer1::IExecutionContext*>(ctx);
}

int atomr_trt_context_set_input_shape(
    atomr_trt_IExecutionContext* ctx, const char* name, const atomr_trt_Dims* dims) {
    if (!ctx || !name || !dims) return -1;
    using namespace nvinfer1;
    Dims trt_dims{};
    trt_dims.nbDims = dims->nb_dims;
    for (int i = 0; i < dims->nb_dims && i < 8; ++i) {
        trt_dims.d[i] = dims->d[i];
    }
    return reinterpret_cast<IExecutionContext*>(ctx)
        ->setInputShape(name, trt_dims) ? 0 : -1;
}

int atomr_trt_context_set_tensor_address(
    atomr_trt_IExecutionContext* ctx, const char* name, void* addr) {
    if (!ctx || !name) return -1;
    return reinterpret_cast<nvinfer1::IExecutionContext*>(ctx)
        ->setTensorAddress(name, addr) ? 0 : -1;
}

int atomr_trt_context_enqueue_v3(atomr_trt_IExecutionContext* ctx, void* cuda_stream) {
    if (!ctx) return -1;
    return reinterpret_cast<nvinfer1::IExecutionContext*>(ctx)
        ->enqueueV3(static_cast<cudaStream_t>(cuda_stream)) ? 0 : -1;
}

// ─── Runtime ──────────────────────────────────────────────────────────

atomr_trt_IRuntime* atomr_trt_runtime_create(int /*logger_severity*/) {
    using namespace nvinfer1;
    return reinterpret_cast<atomr_trt_IRuntime*>(
        try_or_null([&]() -> IRuntime* { return createInferRuntime(g_logger); }));
}

void atomr_trt_runtime_destroy(atomr_trt_IRuntime* runtime) {
    delete reinterpret_cast<nvinfer1::IRuntime*>(runtime);
}

atomr_trt_ICudaEngine* atomr_trt_runtime_deserialize(
    atomr_trt_IRuntime* runtime, const uint8_t* blob, size_t len) {
    if (!runtime || !blob) return nullptr;
    using namespace nvinfer1;
    auto* r = reinterpret_cast<IRuntime*>(runtime);
    return reinterpret_cast<atomr_trt_ICudaEngine*>(
        try_or_null([&]() -> ICudaEngine* { return r->deserializeCudaEngine(blob, len); }));
}

// ─── HostMemory ───────────────────────────────────────────────────────

const uint8_t* atomr_trt_host_memory_data(atomr_trt_IHostMemory* mem) {
    if (!mem) return nullptr;
    return static_cast<const uint8_t*>(
        reinterpret_cast<nvinfer1::IHostMemory*>(mem)->data());
}

size_t atomr_trt_host_memory_size(atomr_trt_IHostMemory* mem) {
    if (!mem) return 0;
    return reinterpret_cast<nvinfer1::IHostMemory*>(mem)->size();
}

void atomr_trt_host_memory_destroy(atomr_trt_IHostMemory* mem) {
    delete reinterpret_cast<nvinfer1::IHostMemory*>(mem);
}

// ─── Plugin registry ──────────────────────────────────────────────────

int atomr_trt_register_plugin_creator(atomr_trt_IPluginCreator* creator) {
    if (!creator) return -1;
    using namespace nvinfer1;
    auto* reg = getPluginRegistry();
    if (!reg) return -1;
    auto* c = reinterpret_cast<IPluginCreator*>(creator);
    return reg->registerCreator(*c, "") ? 0 : -1;
}

}  // extern "C"
