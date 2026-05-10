// atomr-accel-tensorrt: C-ABI shim header for libnvinfer.
//
// Mirrors the `extern "C"` declarations in `src/sys.rs`. Included by
// every `.cpp` in `csrc/` so a missing prototype is a compile error
// rather than a link error.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Opaque object pointers (mirror the zero-sized structs in sys.rs).
typedef struct atomr_trt_IBuilder            atomr_trt_IBuilder;
typedef struct atomr_trt_IBuilderConfig      atomr_trt_IBuilderConfig;
typedef struct atomr_trt_INetworkDefinition  atomr_trt_INetworkDefinition;
typedef struct atomr_trt_ICudaEngine         atomr_trt_ICudaEngine;
typedef struct atomr_trt_IExecutionContext   atomr_trt_IExecutionContext;
typedef struct atomr_trt_IRuntime            atomr_trt_IRuntime;
typedef struct atomr_trt_IHostMemory         atomr_trt_IHostMemory;
typedef struct atomr_trt_IRefitter           atomr_trt_IRefitter;
typedef struct atomr_trt_IInt8Calibrator     atomr_trt_IInt8Calibrator;
typedef struct atomr_trt_IPluginCreator      atomr_trt_IPluginCreator;
typedef struct atomr_trt_IOnnxParser         atomr_trt_IOnnxParser;

typedef struct atomr_trt_Dims {
    int32_t nb_dims;
    int32_t d[8];
} atomr_trt_Dims;

// ─── Logger callback ─────────────────────────────────────────────────
typedef void (*atomr_trt_log_cb)(int severity, const char* msg, size_t len, void* user);
void atomr_trt_install_logger(atomr_trt_log_cb cb, void* user);

// ─── Builder lifecycle ────────────────────────────────────────────────
atomr_trt_IBuilder* atomr_trt_builder_create(int logger_severity);
void                atomr_trt_builder_destroy(atomr_trt_IBuilder* builder);
atomr_trt_INetworkDefinition* atomr_trt_builder_create_network(
    atomr_trt_IBuilder* builder, uint32_t flags);
atomr_trt_IBuilderConfig* atomr_trt_builder_create_config(atomr_trt_IBuilder* builder);
atomr_trt_IHostMemory* atomr_trt_builder_build_serialized(
    atomr_trt_IBuilder* builder,
    atomr_trt_INetworkDefinition* network,
    atomr_trt_IBuilderConfig* config);

// ─── BuilderConfig ────────────────────────────────────────────────────
void atomr_trt_config_destroy(atomr_trt_IBuilderConfig* config);
void atomr_trt_config_set_flag(atomr_trt_IBuilderConfig* config, uint32_t flag, int on);
void atomr_trt_config_set_memory_pool_limit(atomr_trt_IBuilderConfig* config, int pool, size_t bytes);
void atomr_trt_config_set_default_device_type(atomr_trt_IBuilderConfig* config, int dt);
void atomr_trt_config_set_dla_core(atomr_trt_IBuilderConfig* config, int core);
void atomr_trt_config_set_tactic_sources(atomr_trt_IBuilderConfig* config, uint32_t mask);
void atomr_trt_config_set_int8_calibrator(
    atomr_trt_IBuilderConfig* config, atomr_trt_IInt8Calibrator* calibrator);
void atomr_trt_config_set_timing_cache(
    atomr_trt_IBuilderConfig* config, const uint8_t* blob, size_t len);

// ─── Engine ───────────────────────────────────────────────────────────
void atomr_trt_engine_destroy(atomr_trt_ICudaEngine* engine);
atomr_trt_IExecutionContext* atomr_trt_engine_create_execution_context(
    atomr_trt_ICudaEngine* engine);
atomr_trt_IHostMemory* atomr_trt_engine_serialize(atomr_trt_ICudaEngine* engine);
int  atomr_trt_engine_num_io_tensors(atomr_trt_ICudaEngine* engine);
const char* atomr_trt_engine_io_tensor_name(atomr_trt_ICudaEngine* engine, int idx);
atomr_trt_IRefitter* atomr_trt_engine_create_refitter(atomr_trt_ICudaEngine* engine);

// ─── Refitter ─────────────────────────────────────────────────────────
void atomr_trt_refitter_destroy(atomr_trt_IRefitter* refitter);
int  atomr_trt_refitter_set_named_weights(
    atomr_trt_IRefitter* refitter,
    const char* name,
    const void* weights,
    size_t bytes,
    int dtype);
int  atomr_trt_refitter_refit_engine(atomr_trt_IRefitter* refitter);

// ─── ExecutionContext ─────────────────────────────────────────────────
void atomr_trt_context_destroy(atomr_trt_IExecutionContext* ctx);
int  atomr_trt_context_set_input_shape(
    atomr_trt_IExecutionContext* ctx, const char* name, const atomr_trt_Dims* dims);
int  atomr_trt_context_set_tensor_address(
    atomr_trt_IExecutionContext* ctx, const char* name, void* addr);
int  atomr_trt_context_enqueue_v3(atomr_trt_IExecutionContext* ctx, void* cuda_stream);

// ─── Runtime ──────────────────────────────────────────────────────────
atomr_trt_IRuntime* atomr_trt_runtime_create(int logger_severity);
void                atomr_trt_runtime_destroy(atomr_trt_IRuntime* runtime);
atomr_trt_ICudaEngine* atomr_trt_runtime_deserialize(
    atomr_trt_IRuntime* runtime, const uint8_t* blob, size_t len);

// ─── HostMemory ───────────────────────────────────────────────────────
const uint8_t* atomr_trt_host_memory_data(atomr_trt_IHostMemory* mem);
size_t         atomr_trt_host_memory_size(atomr_trt_IHostMemory* mem);
void           atomr_trt_host_memory_destroy(atomr_trt_IHostMemory* mem);

// ─── Plugin registry ──────────────────────────────────────────────────
int atomr_trt_register_plugin_creator(atomr_trt_IPluginCreator* creator);

// ─── ONNX parser (gated by ATOMR_TRT_ONNX) ────────────────────────────
#ifdef ATOMR_TRT_ONNX
atomr_trt_IOnnxParser* atomr_trt_onnx_parser_create(
    atomr_trt_INetworkDefinition* network, int logger_severity);
void atomr_trt_onnx_parser_destroy(atomr_trt_IOnnxParser* parser);
int  atomr_trt_onnx_parser_parse(
    atomr_trt_IOnnxParser* parser, const uint8_t* data, size_t len, const char* path);
int  atomr_trt_onnx_parser_num_errors(atomr_trt_IOnnxParser* parser);
#endif

#ifdef __cplusplus
}
#endif
