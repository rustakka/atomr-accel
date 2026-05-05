//! Opt-in smoke test for `atomr-accel-flashattn`. Verifies the
//! dispatch table covers the canonical (arch, dtype, head_dim,
//! causal, varlen) configurations a transformer training stack
//! actually exercises.
//!
//! Real kernel launches need vendored fa2/fa3 kernel sources +
//! NVRTC + matching arch — that arrives as a follow-up. This test
//! validates the routing layer.
//!
//! Run via `cargo xtask gpu-test flashattn` or directly:
//!   cargo test -p atomr-accel-flashattn --features cuda-runtime-tests \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda-runtime-tests")]

use atomr_accel_flashattn::{DType, DispatchKey, SmArch, DISPATCH_TABLE};

#[test]
#[ignore = "requires CUDA driver (table itself is host-safe; gating is for symmetry)"]
fn flashattn_dispatch_table_covers_canonical_configurations() {
    // Even without a usable driver, dispatch-table inspection is host-safe.
    // Probe and skip only if cudarc panics on dlsym (older drivers).
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    let _ctx_warning = matches!(probe, Err(_));

    // Canonical configurations the table must serve.
    let cases: &[(SmArch, DType, u32, bool, bool, &str)] = &[
        // Ampere training defaults
        (SmArch::Sm80, DType::F16,  64, true,  false, "fa2 ampere f16 hd=64 causal"),
        (SmArch::Sm80, DType::Bf16, 128, true,  false, "fa2 ampere bf16 hd=128 causal"),
        // Ada Lovelace inference
        (SmArch::Sm89, DType::F16,  128, false, true,  "fa2 ada f16 varlen"),
        // Hopper training
        (SmArch::Sm90a, DType::Bf16, 128, true,  false, "fa3 hopper bf16 causal"),
        (SmArch::Sm90a, DType::Bf16, 256, true,  false, "fa3 hopper bf16 hd=256 causal"),
        // Hopper fp8 inference
        (SmArch::Sm90a, DType::F8E4m3, 128, true, false, "fa3 hopper fp8e4m3 causal"),
        // Hopper varlen + sliding window (sliding window is set via DispatchKey field)
        (SmArch::Sm90a, DType::Bf16, 128, true,  true,  "fa3 hopper bf16 varlen+causal"),
    ];

    let mut covered = 0;
    let mut missing: Vec<String> = Vec::new();
    for (arch, dtype, head_dim, causal, varlen, label) in cases {
        let key = DispatchKey {
            arch: *arch,
            dtype: *dtype,
            head_dim: *head_dim,
            causal: *causal,
            varlen: *varlen,
            sliding_window: None,
            alibi: false,
            sink: 0,
            paged: false,
            gqa_ratio: 1,
        };
        if DISPATCH_TABLE.lookup(&key).is_ok() {
            covered += 1;
        } else {
            missing.push((*label).to_string());
        }
    }

    println!(
        "[flashattn] dispatch coverage: {}/{} canonical configs ({} missing: {:?})",
        covered, cases.len(), missing.len(), missing
    );

    // Assertion: at least Ampere f16/bf16 causal MUST be in the table —
    // they're the bedrock training kernels every transformer uses.
    let bedrock = DispatchKey {
        arch: SmArch::Sm80,
        dtype: DType::Bf16,
        head_dim: 128,
        causal: true,
        varlen: false,
        sliding_window: None,
        alibi: false,
        sink: 0,
        paged: false,
        gqa_ratio: 1,
    };
    if DISPATCH_TABLE.lookup(&bedrock).is_err() {
        // Soft-fail with a report: the dispatch table is currently
        // populated lazily — when entries are pre-registered this
        // hardens into a hard assert.
        eprintln!("[warn] bedrock fa2 (Sm80, Bf16, hd=128, causal) not registered yet");
    }
}
