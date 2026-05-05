# Vendored CUTLASS subset

This directory holds the minimum CUTLASS header subset that
`atomr-accel-cutlass` needs to NVRTC-instantiate the basic
`cutlass::gemm::device::GemmUniversal` and `ImplicitGemmConvolution`
templates.

The full CUTLASS source is **not** vendored — only the headers
required for the basic GEMM template are included here, plus the
license. Full coverage (the rest of `cutlass/include/cutlass/...`,
plus `cute/...`) is a follow-up.

## Provenance

CUTLASS is licensed under the BSD 3-Clause license.  See
`LICENSES/cutlass-LICENSE.txt` for the full text. Upstream:
<https://github.com/NVIDIA/cutlass>.

## Layout

```
cutlass/
├── LICENSES/cutlass-LICENSE.txt   # BSD-3-Clause
├── README.md                      # this file
└── include/cutlass/               # vendored headers
    ├── cutlass.h
    ├── numeric_types.h
    └── gemm/device/gemm_universal.h
    └── conv/device/implicit_gemm_convolution.h
```

The placeholder headers shipped here are stubs that exist solely so
the `--include-path=` resolution chain succeeds in host-only test
runs. Real builds either bind-mount the upstream CUTLASS source over
this directory or pull in a full CUTLASS checkout via the
`CUTLASS_INCLUDE_DIR` environment variable that downstream build
scripts honor.
