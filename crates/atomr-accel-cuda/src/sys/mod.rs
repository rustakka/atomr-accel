//! Raw CUDA driver- and runtime-API wrappers used by the Phase 3
//! actors. cudarc's safe layer covers the common path (streams,
//! events, memory copy, graph capture/launch) but the IPC, memory
//! advisory, conditional-graph, and `cuModule*` paths are sys-only.
//!
//! Everything in here is `unsafe` at the FFI boundary; the modules
//! one level up build typed actor surfaces around these calls.

pub mod cuda_driver;
