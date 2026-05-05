// SPDX-License-Identifier: Apache-2.0 OR BSD-3-Clause
//
// atomr-accel-flashattn / include / atomr_flash_adapter.h
//
// Adapter layer that maps the vendored FlashAttention csrc onto
// atomr-accel's `AccelDtype` / `GpuRef` abstractions.
//
// The vendored FA csrc references PyTorch's `<torch/extension.h>` for
// dtype tags (`at::Half`, `at::BFloat16`) and tensor accessors. We
// don't link against torch — instead the FA `.cu` files include this
// header in their place, and atomr ships plain pointer-based launches.
//
// This header is *not* compiled by Rust; it's #included by the
// vendored .cu files at NVRTC compile time. The `static_switch` style
// macros below mirror the upstream patterns so the kernel-name
// expressions resolved by `src/dispatch.rs` line up.

#pragma once

#ifdef __CUDACC__

// FlashAttention's tile shapes are templated on the Q/K/V element type.
// Map atomr's enum tags onto the upstream convention.
namespace atomr_flashattn {

enum class AccelDtype : int {
    F16    = 0,
    Bf16   = 1,
    F8E4m3 = 2,  // sm_90a only
    F8E5m2 = 3,  // sm_90a only
};

// Compile-time dtype dispatch. Mirrors `static_switch.h` from the
// vendored csrc but reduced to the dtypes atomr ships.
#define ATOMR_FA_DTYPE_SWITCH(DTYPE, NAME, ...)                          \
    [&] {                                                                \
        switch (DTYPE) {                                                 \
            case ::atomr_flashattn::AccelDtype::F16: {                   \
                using NAME = __half;                                     \
                return __VA_ARGS__();                                    \
            }                                                            \
            case ::atomr_flashattn::AccelDtype::Bf16: {                  \
                using NAME = __nv_bfloat16;                              \
                return __VA_ARGS__();                                    \
            }                                                            \
            case ::atomr_flashattn::AccelDtype::F8E4m3: {                \
                using NAME = __nv_fp8_e4m3;                              \
                return __VA_ARGS__();                                    \
            }                                                            \
            case ::atomr_flashattn::AccelDtype::F8E5m2: {                \
                using NAME = __nv_fp8_e5m2;                              \
                return __VA_ARGS__();                                    \
            }                                                            \
        }                                                                \
        return decltype(__VA_ARGS__()){};                                \
    }()

#define ATOMR_FA_HEAD_DIM_SWITCH(D, NAME, ...)                           \
    [&] {                                                                \
        switch (D) {                                                     \
            case 64:  { constexpr int NAME =  64; return __VA_ARGS__(); }\
            case 80:  { constexpr int NAME =  80; return __VA_ARGS__(); }\
            case 96:  { constexpr int NAME =  96; return __VA_ARGS__(); }\
            case 128: { constexpr int NAME = 128; return __VA_ARGS__(); }\
            case 192: { constexpr int NAME = 192; return __VA_ARGS__(); }\
            case 256: { constexpr int NAME = 256; return __VA_ARGS__(); }\
        }                                                                \
        return decltype(__VA_ARGS__()){};                                \
    }()

#define ATOMR_FA_BOOL_SWITCH(COND, NAME, ...)                            \
    [&] {                                                                \
        if (COND) { constexpr bool NAME = true;  return __VA_ARGS__(); } \
        else      { constexpr bool NAME = false; return __VA_ARGS__(); } \
    }()

}  // namespace atomr_flashattn

#endif  // __CUDACC__
