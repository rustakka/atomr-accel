//! Shared dispatch plumbing for the per-family CUB kernels.
//!
//! Every family follows the same flow:
//!
//! 1. Look up a cached [`KernelHandle`] in the actor-lifetime
//!    [`crate::KernelSourceCache`]. On hit, skip steps 2–3.
//! 2. Render the per-(op, dtype) `(source, kernel_name)` via the
//!    matching emitter in [`crate::kernels`].
//! 3. Send `NvrtcMsg::CompileAsync` with the right `NvrtcOpts` and
//!    `await` the reply (~2–3 s on first invocation, microseconds on
//!    a disk-cache hit).
//! 4. Push `KernelArg`s + a `LaunchConfig`, send `NvrtcMsg::Launch`,
//!    `await` reply.
//! 5. Forward the result through the request's `oneshot::Sender`.
//!
//! The compile / launch round-trip happens inside a `tokio::spawn`
//! task so the `CubActor`'s mailbox stays free during a 2-second NVRTC
//! template instantiation. Every helper here is `async fn` and
//! consumes its arguments by value.

use std::sync::Arc;

use atomr_core::actor::ActorRef;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::kernel::nvrtc::{CppStd, NvrtcOpts, SmArch};
use atomr_accel_cuda::kernel::{KernelArg, KernelHandle, NvrtcMsg};
use cudarc::driver::LaunchConfig;

use crate::KernelSourceCache;

/// Resolve the (`(op, dtype)` keyed) [`KernelHandle`] — either from
/// the in-actor cache or by sending `NvrtcMsg::CompileAsync` to the
/// supplied `NvrtcActor`.
///
/// Inserts the freshly compiled handle into the cache on success so
/// subsequent calls in the same actor lifetime skip the compile.
pub async fn compile_or_get_handle(
    nvrtc: Arc<ActorRef<NvrtcMsg>>,
    cache: Arc<Mutex<KernelSourceCache>>,
    op: String,
    dtype: String,
    src: String,
    kernel_name: String,
    arch: SmArch,
) -> Result<KernelHandle, GpuError> {
    if let Some(h) = cache.lock().get_handle(&op, &dtype) {
        return Ok(h);
    }
    let opts = nvrtc_opts_for_cub(arch);
    let (tx, rx) = oneshot::channel();
    nvrtc.tell(NvrtcMsg::CompileAsync {
        src,
        kernel_name,
        opts,
        reply: tx,
    });
    let handle = match rx.await {
        Ok(Ok(h)) => h,
        Ok(Err(e)) => return Err(e),
        Err(_) => {
            return Err(GpuError::Unrecoverable(
                "atomr-accel-cub: NvrtcActor dropped reply during CompileAsync".into(),
            ))
        }
    };
    cache.lock().insert_handle(&op, &dtype, handle.clone());
    Ok(handle)
}

/// Send `NvrtcMsg::Launch` and `await` the reply. Centralised so every
/// family-specific dispatcher converts a dropped-reply into the same
/// structured error.
pub async fn launch(
    nvrtc: &Arc<ActorRef<NvrtcMsg>>,
    kernel: KernelHandle,
    args: Vec<KernelArg>,
    cfg: LaunchConfig,
) -> Result<(), GpuError> {
    let (tx, rx) = oneshot::channel();
    nvrtc.tell(NvrtcMsg::Launch {
        kernel,
        args,
        cfg,
        reply: tx,
    });
    match rx.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(GpuError::Unrecoverable(
            "atomr-accel-cub: NvrtcActor dropped reply during Launch".into(),
        )),
    }
}

/// Build [`NvrtcOpts`] with the include-path, C++ standard, and arch
/// flags every CUB kernel needs. The include paths are baked into the
/// crate at build time via `build.rs` (`ATOMR_CUB_CUDA_INCLUDE`,
/// `ATOMR_CUB_CCCL_INCLUDE`, `ATOMR_CUB_INCLUDE`); missing-at-build
/// fall through to `/usr/local/cuda/include` so the runtime compile
/// still has a chance.
pub fn nvrtc_opts_for_cub(arch: SmArch) -> NvrtcOpts {
    let mut opts = NvrtcOpts::for_arch(arch).with_cpp_std(CppStd::Cpp17);

    // Crate-local re-export header (`atomr_cub_kernels.cuh`).
    if let Some(p) = option_env!("ATOMR_CUB_INCLUDE") {
        opts = opts.with_include_path(p);
    }
    // CUDA 12 layout: `<root>/include/cub/cub.cuh`.
    if let Some(p) = option_env!("ATOMR_CUB_CUDA_INCLUDE") {
        opts = opts.with_include_path(p);
    } else {
        opts = opts.with_include_path("/usr/local/cuda/include");
    }
    // CUDA 13 layout: `<root>/include/cccl/cub/cub.cuh`.
    if let Some(p) = option_env!("ATOMR_CUB_CCCL_INCLUDE") {
        opts = opts.with_include_path(p);
    }

    opts
}

/// Build a 1-D [`LaunchConfig`] sized to cover `n` elements at the
/// per-block tile of [`crate::kernels::TILE_ELEMENTS`].
pub fn launch_config_for(n: usize) -> LaunchConfig {
    let block = crate::kernels::BLOCK_THREADS;
    let tile = crate::kernels::TILE_ELEMENTS as usize;
    let grid = n.div_ceil(tile).max(1) as u32;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Single-block [`LaunchConfig`] used by the `_finalize` /
/// `_block_sums` cousins of the multi-launch reduce/scan flow, plus
/// every single-tile kernel (sort, select, partition).
pub fn launch_config_single_block() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (crate::kernels::BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Number of grid blocks the main kernel will use for an input of
/// `n` elements. The dispatcher passes this count to the finalize
/// kernel as `n_partials` so the second pass knows how many block
/// totals to consume.
pub fn grid_blocks_for(n: usize) -> u32 {
    let tile = crate::kernels::TILE_ELEMENTS as usize;
    n.div_ceil(tile).max(1) as u32
}
