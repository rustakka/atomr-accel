// Vendored placeholder for cutlass/gemm/device/gemm_universal.h.
//
// Real CUTLASS exposes `cutlass::gemm::device::GemmUniversal<...>` here
// with a multi-arch template specialization tree. The placeholder
// exists so the host-side render path (used by unit tests) compiles
// the `#include` chain.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace cutlass {
namespace gemm {
namespace device {

template <typename... Ts>
struct GemmUniversal {
  struct Arguments {};
};

}  // namespace device
}  // namespace gemm
}  // namespace cutlass
