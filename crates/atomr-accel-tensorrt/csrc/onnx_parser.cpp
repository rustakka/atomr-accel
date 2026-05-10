// atomr-accel-tensorrt: ONNX parser C-ABI shim.
//
// Compiled only when the `tensorrt-onnx` cargo feature is on (gated
// in build.rs by `CARGO_FEATURE_TENSORRT_ONNX`). Mirrors the four
// `atomr_trt_onnx_parser_*` declarations in `src/sys.rs`.
//
// SPDX-License-Identifier: Apache-2.0

#include "nvinfer_shim.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <exception>

// Forward-declared accessor exported (with internal C++ linkage) by
// `nvinfer_shim.cpp`. Resolved at link time within the same static
// library, giving every translation unit access to the single
// process-wide `RustBridgeLogger` instance.
nvinfer1::ILogger& atomr_trt_logger();

extern "C" {

atomr_trt_IOnnxParser* atomr_trt_onnx_parser_create(
    atomr_trt_INetworkDefinition* network, int /*logger_severity*/) {
    if (!network) return nullptr;
    try {
        auto* n = reinterpret_cast<nvinfer1::INetworkDefinition*>(network);
        return reinterpret_cast<atomr_trt_IOnnxParser*>(
            nvonnxparser::createParser(*n, atomr_trt_logger()));
    } catch (const std::exception&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void atomr_trt_onnx_parser_destroy(atomr_trt_IOnnxParser* parser) {
    delete reinterpret_cast<nvonnxparser::IParser*>(parser);
}

int atomr_trt_onnx_parser_parse(
    atomr_trt_IOnnxParser* parser,
    const uint8_t* data,
    size_t len,
    const char* path) {
    if (!parser || !data) return 0;
    try {
        return reinterpret_cast<nvonnxparser::IParser*>(parser)
            ->parse(data, len, path) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int atomr_trt_onnx_parser_num_errors(atomr_trt_IOnnxParser* parser) {
    if (!parser) return 0;
    return reinterpret_cast<nvonnxparser::IParser*>(parser)->getNbErrors();
}

}  // extern "C"
