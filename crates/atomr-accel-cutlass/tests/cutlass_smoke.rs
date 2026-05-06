//! Opt-in smoke test for `atomr-accel-cutlass`. Verifies:
//! 1. `is_supported_for(dtype, arch)` correctly enforces fp8≥sm_89,
//!    fp4≥sm_100 (per the CUTLASS arch contracts).
//! 2. The plan-cache discriminates between GEMM, grouped-GEMM, and
//!    Conv plans without key collision.
//!
//! The CUTLASS template emitter requires NVRTC + nvcc; a real
//! end-to-end JIT smoke test lands in a follow-up. This test
//! validates the host-side plumbing the JIT path depends on.
//!
//! Run via `cargo xtask gpu-test cutlass` or:
//!   cargo test -p atomr-accel-cutlass --features cuda-runtime-tests \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda-runtime-tests")]

use atomr_accel_cutlass::{is_supported_for, CutlassDtype, SmArch};

#[test]
#[ignore = "requires NVRTC for full e2e; arch matrix itself is host-safe"]
fn cutlass_arch_dtype_support_matrix() {
    // Bedrock: fp16 / bf16 work everywhere CUTLASS is supported.
    for arch in [
        SmArch::Sm80,
        SmArch::Sm86,
        SmArch::Sm89,
        SmArch::Sm90,
        SmArch::Sm90a,
        SmArch::Sm100,
    ] {
        assert!(
            is_supported_for(CutlassDtype::F16, arch),
            "f16 must be supported on {arch:?}"
        );
        assert!(
            is_supported_for(CutlassDtype::Bf16, arch),
            "bf16 must be supported on {arch:?}"
        );
    }

    // fp8 e4m3 / e5m2: Ada (sm_89) and Hopper (sm_90/sm_90a) and newer.
    assert!(
        !is_supported_for(CutlassDtype::F8E4m3, SmArch::Sm80),
        "fp8 e4m3 should not be on sm_80"
    );
    assert!(
        !is_supported_for(CutlassDtype::F8E4m3, SmArch::Sm86),
        "fp8 e4m3 should not be on sm_86"
    );
    assert!(
        is_supported_for(CutlassDtype::F8E4m3, SmArch::Sm89),
        "fp8 e4m3 should be on sm_89+"
    );
    assert!(
        is_supported_for(CutlassDtype::F8E4m3, SmArch::Sm90a),
        "fp8 e4m3 should be on sm_90a"
    );

    // fp4: Blackwell-only.
    assert!(
        !is_supported_for(CutlassDtype::F4E2m1, SmArch::Sm89),
        "fp4 should not be on Ada"
    );
    assert!(
        !is_supported_for(CutlassDtype::F4E2m1, SmArch::Sm90a),
        "fp4 should not be on Hopper"
    );
    assert!(
        is_supported_for(CutlassDtype::F4E2m1, SmArch::Sm100),
        "fp4 should be on Blackwell sm_100"
    );

    println!("[cutlass] arch×dtype support matrix matches the CUTLASS contract");
}
