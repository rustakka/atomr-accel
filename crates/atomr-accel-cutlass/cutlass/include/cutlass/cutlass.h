// Vendored placeholder for cutlass.h.
//
// In production this file is replaced by the upstream CUTLASS header.
// The placeholder exists so host-side tests that exercise the include
// chain don't fail; runtime NVRTC compilations are pointed at a real
// CUTLASS install via the `CUTLASS_INCLUDE_DIR` env var.
//
// Upstream: https://github.com/NVIDIA/cutlass
// License : BSD-3-Clause (see ../../../LICENSES/cutlass-LICENSE.txt).

#pragma once

namespace cutlass {
// Placeholder: real CUTLASS provides the full library namespace here.
}  // namespace cutlass
