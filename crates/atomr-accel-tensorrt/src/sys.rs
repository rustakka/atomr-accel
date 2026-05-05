//! Hand-written FFI surface for libnvinfer (and libnvonnxparser when
//! the `tensorrt-onnx` feature is on).
//!
//! TensorRT is a C++ API. We expose just the C-ABI shim functions we
//! need from a thin C++ glue layer. The functions declared `extern "C"`
//! here are intentionally empty when the `tensorrt-link` feature is
//! off — the linker is never asked to resolve them, and the safe
//! wrappers in `builder.rs`/`engine.rs`/`runtime.rs`/`onnx.rs` only
//! call them through a `#[cfg(feature = "tensorrt-link")]` gate.
//!
//! The opaque pointer types (`IBuilder`, `IBuilderConfig`,
//! `INetworkDefinition`, `ICudaEngine`, `IExecutionContext`,
//! `IRuntime`, `IPluginCreator`) are zero-sized stand-ins for the
//! corresponding TensorRT C++ classes. They are only ever held as
//! `*mut` raw pointers; `Send`/`Sync` for the safe wrappers is granted
//! via newtypes (see `engine.rs`).
//!
//! The C-ABI shim itself (a hand-written `nvinfer_shim.cpp`) is not
//! shipped in this Phase 8 skeleton — it lives behind the
//! `tensorrt-link` feature in a follow-up commit. Until then the FFI
//! signatures here document the surface area and let downstream code
//! type-check against a stable shape.

#![allow(non_camel_case_types, dead_code, non_snake_case, unused_imports)]

use std::os::raw::{c_char, c_int, c_void};

// -------- Opaque object pointers --------

#[repr(C)]
pub struct IBuilder {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IBuilderConfig {
    _private: [u8; 0],
}

#[repr(C)]
pub struct INetworkDefinition {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ICudaEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IExecutionContext {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IRuntime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IHostMemory {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IRefitter {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IInt8Calibrator {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IPluginCreator {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IPluginV3 {
    _private: [u8; 0],
}

#[repr(C)]
pub struct IOnnxParser {
    _private: [u8; 0],
}

// -------- Enums (mirrored from NvInferRuntimeCommon.h / NvInferRuntime.h) --------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
    kINT4 = 9,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuilderFlag {
    kFP16 = 0,
    kINT8 = 1,
    kDEBUG = 2,
    kGPU_FALLBACK = 3,
    kREFIT = 4,
    kDISABLE_TIMING_CACHE = 5,
    kTF32 = 6,
    kSPARSE_WEIGHTS = 7,
    kSAFETY_SCOPE = 8,
    kOBEY_PRECISION_CONSTRAINTS = 9,
    kPREFER_PRECISION_CONSTRAINTS = 10,
    kDIRECT_IO = 11,
    kREJECT_EMPTY_ALGORITHMS = 12,
    kBF16 = 13,
    kFP8 = 14,
    kSTRIP_PLAN = 15,
    kVERSION_COMPATIBLE = 16,
    kEXCLUDE_LEAN_RUNTIME = 17,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    kGPU = 0,
    kDLA = 1,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TacticSource {
    kCUBLAS = 0,
    kCUBLAS_LT = 1,
    kCUDNN = 2,
    kEDGE_MASK_CONVOLUTIONS = 3,
    kJIT_CONVOLUTIONS = 4,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationAlgoType {
    kLEGACY_CALIBRATION = 0,
    kENTROPY_CALIBRATION = 1,
    kENTROPY_CALIBRATION_2 = 2,
    kMINMAX_CALIBRATION = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dims {
    pub nb_dims: c_int,
    pub d: [c_int; 8],
}

// -------- Function declarations (link probe is gated by `tensorrt-link`) --------
//
// The signatures below mirror the C-ABI shim that wraps the TensorRT
// C++ classes. With the `tensorrt-link` feature off these are present
// in the source as documentation; they're never referenced and so
// never produce link errors.

#[cfg(feature = "tensorrt-link")]
extern "C" {
    // Builder lifecycle
    pub fn atomr_trt_builder_create(logger_severity: c_int) -> *mut IBuilder;
    pub fn atomr_trt_builder_destroy(builder: *mut IBuilder);
    pub fn atomr_trt_builder_create_network(
        builder: *mut IBuilder,
        flags: u32,
    ) -> *mut INetworkDefinition;
    pub fn atomr_trt_builder_create_config(builder: *mut IBuilder) -> *mut IBuilderConfig;
    pub fn atomr_trt_builder_build_serialized(
        builder: *mut IBuilder,
        network: *mut INetworkDefinition,
        config: *mut IBuilderConfig,
    ) -> *mut IHostMemory;

    // BuilderConfig knobs
    pub fn atomr_trt_config_destroy(config: *mut IBuilderConfig);
    pub fn atomr_trt_config_set_flag(config: *mut IBuilderConfig, flag: u32, on: c_int);
    pub fn atomr_trt_config_set_memory_pool_limit(
        config: *mut IBuilderConfig,
        pool: c_int,
        bytes: usize,
    );
    pub fn atomr_trt_config_set_default_device_type(config: *mut IBuilderConfig, dt: c_int);
    pub fn atomr_trt_config_set_dla_core(config: *mut IBuilderConfig, core: c_int);
    pub fn atomr_trt_config_set_tactic_sources(config: *mut IBuilderConfig, mask: u32);
    pub fn atomr_trt_config_set_int8_calibrator(
        config: *mut IBuilderConfig,
        calibrator: *mut IInt8Calibrator,
    );
    pub fn atomr_trt_config_set_timing_cache(
        config: *mut IBuilderConfig,
        blob: *const u8,
        len: usize,
    );

    // Engine
    pub fn atomr_trt_engine_destroy(engine: *mut ICudaEngine);
    pub fn atomr_trt_engine_create_execution_context(
        engine: *mut ICudaEngine,
    ) -> *mut IExecutionContext;
    pub fn atomr_trt_engine_serialize(engine: *mut ICudaEngine) -> *mut IHostMemory;
    pub fn atomr_trt_engine_num_io_tensors(engine: *mut ICudaEngine) -> c_int;
    pub fn atomr_trt_engine_io_tensor_name(engine: *mut ICudaEngine, idx: c_int) -> *const c_char;
    pub fn atomr_trt_engine_create_refitter(engine: *mut ICudaEngine) -> *mut IRefitter;

    // Refitter
    pub fn atomr_trt_refitter_destroy(refitter: *mut IRefitter);
    pub fn atomr_trt_refitter_set_named_weights(
        refitter: *mut IRefitter,
        name: *const c_char,
        weights: *const c_void,
        bytes: usize,
        dtype: c_int,
    ) -> c_int;
    pub fn atomr_trt_refitter_refit_engine(refitter: *mut IRefitter) -> c_int;

    // ExecutionContext
    pub fn atomr_trt_context_destroy(ctx: *mut IExecutionContext);
    pub fn atomr_trt_context_set_input_shape(
        ctx: *mut IExecutionContext,
        name: *const c_char,
        dims: *const Dims,
    ) -> c_int;
    pub fn atomr_trt_context_set_tensor_address(
        ctx: *mut IExecutionContext,
        name: *const c_char,
        addr: *mut c_void,
    ) -> c_int;
    pub fn atomr_trt_context_enqueue_v3(
        ctx: *mut IExecutionContext,
        cuda_stream: *mut c_void,
    ) -> c_int;

    // Runtime + deserialise
    pub fn atomr_trt_runtime_create(logger_severity: c_int) -> *mut IRuntime;
    pub fn atomr_trt_runtime_destroy(runtime: *mut IRuntime);
    pub fn atomr_trt_runtime_deserialize(
        runtime: *mut IRuntime,
        blob: *const u8,
        len: usize,
    ) -> *mut ICudaEngine;

    // HostMemory
    pub fn atomr_trt_host_memory_data(mem: *mut IHostMemory) -> *const u8;
    pub fn atomr_trt_host_memory_size(mem: *mut IHostMemory) -> usize;
    pub fn atomr_trt_host_memory_destroy(mem: *mut IHostMemory);

    // Plugin registry (IPluginV3)
    pub fn atomr_trt_register_plugin_creator(creator: *mut IPluginCreator) -> c_int;
}

#[cfg(all(feature = "tensorrt-link", feature = "tensorrt-onnx"))]
extern "C" {
    pub fn atomr_trt_onnx_parser_create(
        network: *mut INetworkDefinition,
        logger_severity: c_int,
    ) -> *mut IOnnxParser;
    pub fn atomr_trt_onnx_parser_destroy(parser: *mut IOnnxParser);
    pub fn atomr_trt_onnx_parser_parse(
        parser: *mut IOnnxParser,
        data: *const u8,
        len: usize,
        path: *const c_char,
    ) -> c_int;
    pub fn atomr_trt_onnx_parser_num_errors(parser: *mut IOnnxParser) -> c_int;
}
