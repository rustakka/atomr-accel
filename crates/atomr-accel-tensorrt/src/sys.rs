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

/// Logger callback signature: invoked from the C++ shim's
/// `RustBridgeLogger::log()` once per TRT log line. Severity follows
/// `nvinfer1::ILogger::Severity` — 0 = INTERNAL_ERROR, 1 = ERROR,
/// 2 = WARNING, 3 = INFO, 4 = VERBOSE.
#[cfg(feature = "tensorrt-link")]
pub type AtomrTrtLogCb =
    unsafe extern "C" fn(severity: c_int, msg: *const c_char, len: usize, user: *mut c_void);

#[cfg(feature = "tensorrt-link")]
extern "C" {
    /// Install a Rust callback that the C++ shim's static `ILogger`
    /// forwards every TRT log line to. Idempotent — last call wins.
    pub fn atomr_trt_install_logger(cb: AtomrTrtLogCb, user: *mut c_void);

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

/// Vtable mirrored in `csrc/rust_bridge.h`. Each function pointer
/// dispatches a vtable-method call from the C++ proxy back to the
/// corresponding Rust trait method on `dyn PluginV3`.
#[cfg(all(feature = "tensorrt-link", feature = "tensorrt-plugin"))]
#[repr(C)]
pub struct AtomrPluginVTable {
    pub get_name: unsafe extern "C" fn(user: *const c_void) -> *const c_char,
    pub get_version: unsafe extern "C" fn(user: *const c_void) -> *const c_char,
    pub get_namespace: unsafe extern "C" fn(user: *const c_void) -> *const c_char,
    pub create_plugin:
        unsafe extern "C" fn(user: *const c_void, name: *const c_char) -> *mut c_void,
    pub destroy: unsafe extern "C" fn(user: *mut c_void),
    pub destroy_instance: unsafe extern "C" fn(instance: *mut c_void),
}

#[cfg(all(feature = "tensorrt-link", feature = "tensorrt-plugin"))]
extern "C" {
    pub fn atomr_trt_make_plugin_creator(
        vt: *const AtomrPluginVTable,
        user: *mut c_void,
    ) -> *mut IPluginCreator;
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
