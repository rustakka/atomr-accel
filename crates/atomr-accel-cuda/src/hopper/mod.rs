//! Hopper / Blackwell primitives (Phase 5).
//!
//! Hopper (sm_90 / sm_90a) introduced four kernel-side primitives that
//! materially change how high-throughput CUDA kernels are written:
//!
//! 1. **Tensor Memory Accelerator (TMA)** — bulk asynchronous tensor
//!    copies between global and shared memory described by an opaque
//!    `CUtensorMap` (built host-side via `cuTensorMapEncodeTiled`).
//!    See [`tma`].
//! 2. **WGMMA** — warp-group matrix multiply accumulate, the
//!    successor to MMA. Issued from a warpgroup of 128 threads via
//!    `wgmma.mma_async.sync`. See [`wgmma`].
//! 3. **`cp.async`** — already on Ampere, but Hopper adds the
//!    bulk-asynchronous TMA-driven `cp.async.bulk` that fences with
//!    barrier objects. See [`cp_async`].
//! 4. **Thread-block clusters** — a new launch dimension above grid /
//!    block that exposes Distributed Shared Memory (DSM) and the
//!    `cluster.sync` barrier. See [`cluster`].
//!
//! Blackwell (sm_100 / sm_120) adds the second-generation TMA, larger
//! cluster sizes, the new fp4 / fp6 / mxfp variants, and tensor memory
//! (TMEM) that backs `tcgen05.mma`. The `blackwell` cargo feature gates
//! the additional intrinsics; the host-side wrappers are shared with
//! Hopper through this module.
//!
//! ## Layout
//!
//! * [`tma`] — `TensorMapDescriptor` builder + the safe wrapper around
//!   `cuTensorMapEncodeTiled`.
//! * [`wgmma`] — public re-exports of the macro-defined `wgmma_*`
//!   intrinsics (definitions live in `include/atomr_hopper.cuh`).
//! * [`cp_async`] — `cp.async` pipeline macro shims.
//! * [`cluster`] — [`LaunchSpec`] and the safe wrapper around
//!   `cudaLaunchKernelExC` for cluster-dim launches; DSM helpers.

pub mod cluster;
pub mod cp_async;
pub mod tma;
pub mod wgmma;

pub use cluster::{ClusterDim, LaunchSpec};
pub use tma::{TensorMapDataType, TensorMapDescriptor, TensorMapInterleave, TensorMapSwizzle};

/// Path to the vendored hopper header (`atomr_hopper.cuh`) shipped
/// alongside the crate. NVRTC kernels can `--include-path` this and
/// `#include "atomr_hopper.cuh"` to pick up the wgmma / cp.async /
/// cluster macro shims.
pub const ATOMR_HOPPER_HEADER_REL_PATH: &str = "include/atomr_hopper.cuh";

/// Returns the absolute filesystem path to `atomr_hopper.cuh` if it
/// exists alongside the crate sources. Returns `None` for installations
/// that strip the `include/` directory (e.g. crates.io binary
/// publication of just the compiled lib).
pub fn atomr_hopper_header_path() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(ATOMR_HOPPER_HEADER_REL_PATH);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_path_resolves_in_workspace() {
        // The header ships in-tree under `include/`; in the workspace
        // build the file must exist.
        let p = atomr_hopper_header_path();
        assert!(
            p.is_some(),
            "atomr_hopper.cuh must ship alongside the crate"
        );
    }
}
