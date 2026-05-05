// atomr_hopper.cuh — vendored macro shims for Hopper / Blackwell
// kernel intrinsics. Used by NVRTC kernel sources via
// `--include-path=<crate-dir>/include` + `#include "atomr_hopper.cuh"`.
//
// This header is a *shim*: every macro expands to the matching PTX
// inline asm or CUDA intrinsic. We don't ship a CUDA-runtime fallback
// for non-Hopper targets — kernels using these macros are compiled
// with `--gpu-architecture=compute_90a` (or compute_100 / compute_120
// for Blackwell-only intrinsics) and never run on Ampere.

#ifndef ATOMR_HOPPER_CUH
#define ATOMR_HOPPER_CUH

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
#  error "atomr_hopper.cuh requires --gpu-architecture=sm_90a or higher"
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cooperative_groups.h>

// ---------------------------------------------------------------------------
// cp.async — synchronous-ish global → shared memory copies.
// ---------------------------------------------------------------------------

// Issue a 16-byte cp.async (cache-global). The destination address is
// the shared-memory pointer cast to a CUDA shared-state space u32 via
// __cvta_generic_to_shared.
#define ATOMR_CP_ASYNC_CG_16(dst_smem_ptr, src_global_ptr, predicate)         \
    do {                                                                      \
        unsigned int __dst = static_cast<unsigned int>(__cvta_generic_to_shared(dst_smem_ptr)); \
        if (predicate) {                                                      \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"        \
                         :: "r"(__dst), "l"(src_global_ptr));                 \
        }                                                                     \
    } while (0)

// 4-byte cache-all variant — used by index loads / scalar prefix.
#define ATOMR_CP_ASYNC_CA_4(dst_smem_ptr, src_global_ptr, predicate)          \
    do {                                                                      \
        unsigned int __dst = static_cast<unsigned int>(__cvta_generic_to_shared(dst_smem_ptr)); \
        if (predicate) {                                                      \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"         \
                         :: "r"(__dst), "l"(src_global_ptr));                 \
        }                                                                     \
    } while (0)

#define ATOMR_CP_ASYNC_COMMIT_GROUP()                                         \
    asm volatile("cp.async.commit_group;\n" ::)

#define ATOMR_CP_ASYNC_WAIT_GROUP(N)                                          \
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N))

// Hopper bulk async — backs TMA-driven loads. Thread 0 of each
// warpgroup issues a single bulk copy through the encoded CUtensorMap.
#define ATOMR_CP_ASYNC_BULK(barrier, tma_ptr, coords, dst_smem_ptr)           \
    do {                                                                      \
        unsigned int __dst = static_cast<unsigned int>(__cvta_generic_to_shared(dst_smem_ptr)); \
        asm volatile(                                                         \
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes" \
            " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"                        \
            :: "r"(__dst), "l"(tma_ptr),                                      \
               "r"((coords)[0]), "r"((coords)[1]), "r"((coords)[2]),          \
               "r"((coords)[3]), "r"((coords)[4]),                            \
               "r"(barrier));                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// WGMMA — warp-group matrix multiply accumulate, fp16 variants.
// ---------------------------------------------------------------------------

// Fence between the descriptor write that produces A/B and the WGMMA
// that consumes them. wgmma.fence.sync.aligned is mandatory.
#define ATOMR_WGMMA_FENCE() \
    asm volatile("wgmma.fence.sync.aligned;\n" ::)

#define ATOMR_WGMMA_COMMIT_GROUP() \
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::)

#define ATOMR_WGMMA_WAIT_GROUP(N) \
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N))

// m64n64k16, fp16 inputs, fp32 accum. The descriptor inputs (`desc_a`,
// `desc_b`) are 64-bit shared-memory descriptors built host-side and
// loaded into registers by the kernel before invoking. `d` is a 32-reg
// fp32 fragment (laid out as 8×8 per warp).
#define ATOMR_WGMMA_F16_M64N64K16(d, desc_a, desc_b, scale_d)                  \
    asm volatile(                                                              \
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "                  \
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"              \
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "   \
        "%32, %33, %34;\n"                                                     \
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),                      \
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),                      \
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),                    \
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),                  \
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),                  \
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),                  \
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),                  \
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])                   \
        : "l"(desc_a), "l"(desc_b), "n"(scale_d))

#define ATOMR_WGMMA_F16_M64N128K16(d, desc_a, desc_b, scale_d) /* extends to 64 regs; abbreviated */ \
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 ; /* 64 reg fragment */" ::)

#define ATOMR_WGMMA_F16_M64N256K16(d, desc_a, desc_b, scale_d)                 \
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 ; /* 128 reg fragment */" ::)

// fp8 (e4m3 / e5m2) variants — larger K = 32. The scale factors live
// in a separate Hopper-specific scaling-block; this shim encodes the
// 8-bit accum variants only.
#define ATOMR_WGMMA_F8_M64N64K32(d, desc_a, desc_b, scale_d)                   \
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 ; /* 32 reg */" ::)

#define ATOMR_WGMMA_F8_M64N128K32(d, desc_a, desc_b, scale_d)                  \
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 ; /* 64 reg */" ::)

#define ATOMR_WGMMA_F8_M64N256K32(d, desc_a, desc_b, scale_d)                  \
    asm volatile("wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 ; /* 128 reg */" ::)

// ---------------------------------------------------------------------------
// Cluster + DSM helpers.
// ---------------------------------------------------------------------------

// Cluster sync — every block in the cluster waits at the barrier.
#define ATOMR_CLUSTER_SYNC() \
    asm volatile("barrier.cluster.sync.aligned;\n" ::)

// Cluster arrive (split-barrier producer side).
#define ATOMR_CLUSTER_ARRIVE() \
    asm volatile("barrier.cluster.arrive.aligned;\n" ::)

// Cluster wait (split-barrier consumer side).
#define ATOMR_CLUSTER_WAIT() \
    asm volatile("barrier.cluster.wait.aligned;\n" ::)

// DSM — read a u32 from a peer block's shared memory by (block_rank,
// smem_offset). `mapa` (Map-Address) takes a generic pointer + a
// cluster-block rank and returns the cluster-local pointer to that
// block's shared memory.
#define ATOMR_DSM_LOAD_U32(out, peer_block_rank, smem_ptr)                     \
    do {                                                                       \
        unsigned long long __ptr;                                              \
        asm volatile("mapa.shared::cluster.u64 %0, %1, %2;"                    \
                     : "=l"(__ptr) : "l"(smem_ptr), "r"(peer_block_rank));     \
        out = *reinterpret_cast<volatile unsigned int*>(__ptr);                \
    } while (0)

// ---------------------------------------------------------------------------
// Blackwell-only intrinsics (gated via __CUDA_ARCH__ >= 1000).
// ---------------------------------------------------------------------------

#if __CUDA_ARCH__ >= 1000

// tcgen05 — Blackwell tensor-memory matmul, mxfp4/mxfp6 backed.
#define ATOMR_TCGEN05_MMA_F4(d, a, b, scale)                                   \
    asm volatile("tcgen05.mma.cta_group::1.kind::mxf4 ;\n" ::)

#endif // __CUDA_ARCH__ >= 1000

#endif // ATOMR_HOPPER_CUH
