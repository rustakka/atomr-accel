// atomr_cub_kernels.cuh — vendored re-export header for the
// per-(op, dtype) NVRTC kernels emitted by
// `crates/atomr-accel-cub/src/kernels/mod.rs`.
//
// The device-wide CUB API (`cub::DeviceReduce::*` etc.) is a host-side
// launcher and cannot be emitted from NVRTC. Phase 5.1 instead builds
// each kernel from CUB's *block-level* primitives
// (`cub::BlockReduce`, `cub::BlockScan`, `cub::BlockRadixSort`,
// `cub::BlockHistogram`, `cub::BlockDiscontinuity`) plus a grid-stride
// loop. Multi-block reductions/scans use a two-launch pattern (block
// partials → finalize). Sort / select / partition are single-tile in
// 5.1 (n ≤ BLOCK*ITEMS = 1024); larger inputs return a structured
// error from the dispatcher with a Phase 5.2 hint.
//
// CUB ships header-only with the CUDA toolkit (12.0+). NVRTC resolves
// `<cub/...>` and `<cuda_*.h>` via the include path that
// `crates/atomr-accel-cub/build.rs` discovers from
// CUDA_PATH / CUDA_HOME / /usr/local/cuda.
//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) NVIDIA Corporation; CUB headers are BSD-3-Clause.

#ifndef ATOMR_CUB_KERNELS_CUH
#define ATOMR_CUB_KERNELS_CUH

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/thread/thread_operators.cuh>

// Half / bfloat16 support — the emitter inserts the matching
// `#define ATOMR_CUB_USE_<…>` line above this include when the kernel
// uses these types. Gating keeps NVRTC happy on hosts with older CUDA
// toolkits that don't ship `<cuda_bf16.h>` standalone.
#if defined(ATOMR_CUB_USE_FP16)
#include <cuda_fp16.h>
#endif
#if defined(ATOMR_CUB_USE_BF16)
#include <cuda_bf16.h>
#endif

// Tiny multiply functor for the `Product` reduction. CUB does not ship
// a `cub::Multiplies` analogue, so we define one inline here.
namespace atomr_cub {
template <typename T>
struct Multiplies {
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return a * b;
    }
};
}  // namespace atomr_cub

#endif  // ATOMR_CUB_KERNELS_CUH
