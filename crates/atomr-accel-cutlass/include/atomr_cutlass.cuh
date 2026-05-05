// Local template adapter shim.
//
// The Rust-side `kernels::render_gemm` emits a `.cu` translation unit
// that #includes the vendored CUTLASS headers under
// `crates/atomr-accel-cutlass/cutlass/include/`. This file is the
// hand-written adapter shim that wraps CUTLASS host-side launch
// glue into a single extern-C symbol per template instantiation, so
// the NVRTC name-expression lookup in `atomr-accel-cuda` resolves
// cleanly.
//
// At present the shim is intentionally thin — Phase 6 ships the
// instantiation pipeline, the device-side launcher comes online in
// the follow-up that wires CUTLASS's host adapter through the
// `LaunchSpec` returned by `atomr-accel-cuda::hopper`.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>

namespace atomr_cutlass {

template <typename Gemm>
__device__ __forceinline__ void launch_gemm(typename Gemm::Arguments const&) {
  // CUTLASS's host-side launcher does the real work; this shim is the
  // device-side anchor point for NVRTC name expressions.
}

}  // namespace atomr_cutlass
