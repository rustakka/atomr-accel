// atomr_cub_kernels.cuh — vendored CUB kernel sources, BSD-3-Clause.
//
// CUB ships header-only under <cub/...> in the CUDA toolkit. This file
// is a thin wrapper that includes the device-wide primitives we
// actually template-instantiate from `atomr-accel-cub`'s NVRTC
// compile path. Each `__global__` here is a minimal driver wrapper —
// the heavy lifting happens inside `cub::DeviceReduce::*`,
// `cub::DeviceScan::*`, etc.
//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) NVIDIA Corporation; redistributed under BSD-3-Clause.

#ifndef ATOMR_CUB_KERNELS_CUH
#define ATOMR_CUB_KERNELS_CUH

#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_segmented_reduce.cuh>

// Driver `__global__` wrappers are emitted per-(op, dtype) at NVRTC
// compile time by the actor; this header just forwards the includes.
// Sample shape (rendered into the per-(op,dtype) source string):
//
//   extern "C" __global__ void atomr_cub_reduce_sum_T(
//       const T* d_in, T* d_out, int n,
//       void* d_temp, size_t temp_bytes)
//   {
//       cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
//   }

#endif // ATOMR_CUB_KERNELS_CUH
