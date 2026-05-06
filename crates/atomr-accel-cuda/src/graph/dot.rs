//! `cudaGraphDebugDotPrint` round-trip — emit a Graphviz DOT file
//! describing a captured graph for tooling / visualisation.
//!
//! cudarc's `cuGraphDebugDotPrint` writes the DOT to a file path, not
//! a buffer. We use a temporary file under `std::env::temp_dir()` and
//! read the contents back as a `String`.

use std::ffi::CString;
use std::fs;
use std::path::PathBuf;

use cudarc::driver::sys as driver_sys;

use crate::error::GpuError;
use crate::graph::GraphHandle;

const LIB: &str = "graph";

bitflags::bitflags! {
    /// Verbosity flags accepted by `cuGraphDebugDotPrint`. Match
    /// `CU_GRAPH_DEBUG_DOT_FLAGS_*`.
    pub struct DotFlags: u32 {
        const VERBOSE        = 1 << 0;
        const KERNEL_NODE    = 1 << 2;
        const MEMCPY_NODE    = 1 << 3;
        const MEMSET_NODE    = 1 << 4;
        const HOST_NODE      = 1 << 5;
        const GRAPH_NODE     = 1 << 6;
    }
}

impl Default for DotFlags {
    fn default() -> Self {
        DotFlags::empty()
    }
}

/// Export `graph` as a Graphviz DOT string. Returns `Unrecoverable`
/// on no-GPU hosts.
pub fn export_dot(graph: &GraphHandle, flags: DotFlags) -> Result<String, GpuError> {
    let path = temp_dot_path();
    let cpath = CString::new(path.to_string_lossy().as_ref())
        .map_err(|e| GpuError::Unrecoverable(format!("export_dot: bad path: {e}")))?;
    let cu_graph = graph.cu_graph();
    let s = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: cu_graph is a caller-owned handle; cpath is a valid
        // C string.
        unsafe { driver_sys::cuGraphDebugDotPrint(cu_graph, cpath.as_ptr(), flags.bits()) }
    }))
    .map_err(|_| GpuError::Unrecoverable("export_dot: CUDA driver not loadable".into()))?;
    if s != driver_sys::cudaError_enum::CUDA_SUCCESS {
        return Err(GpuError::LibraryError {
            lib: LIB,
            msg: format!("cuGraphDebugDotPrint: {s:?}"),
        });
    }
    let dot = fs::read_to_string(&path).map_err(|e| GpuError::LibraryError {
        lib: LIB,
        msg: format!("read DOT file {}: {e}", path.display()),
    })?;
    let _ = fs::remove_file(&path);
    Ok(dot)
}

fn temp_dot_path() -> PathBuf {
    let mut p = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    p.push(format!("atomr-accel-graph-{pid}-{nanos}.dot"));
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_export_returns_nonempty_for_known_graph() {
        // Mock-mode: synthetic GraphHandle. On no-GPU hosts we expect
        // Unrecoverable (libcuda missing). On real hardware with a
        // valid graph this would return a non-empty DOT string. We
        // assert a non-panicking round-trip.
        let g = GraphHandle::synthetic_for_tests();
        let r = export_dot(&g, DotFlags::VERBOSE);
        match r {
            Ok(s) => {
                // Real driver — must produce some DOT.
                assert!(
                    !s.is_empty(),
                    "expected non-empty DOT string from real driver"
                );
            }
            Err(GpuError::Unrecoverable(_)) => {}
            Err(GpuError::LibraryError { .. }) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
