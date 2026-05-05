//! Opt-in smoke test for `atomr-accel-cub`. Validates the public
//! API surface — the kernel-source cache + dispatch-table key
//! generation — and exercises the host-only path against a real
//! CUDA driver when one is present.
//!
//! Run via `cargo xtask gpu-test cub` or directly:
//!   cargo test -p atomr-accel-cub --features cuda-runtime-tests \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda-runtime-tests")]

use std::sync::Arc;

use atomr_accel_cub::{kernel_key, KernelSourceCache, ReductionOp};

#[test]
#[ignore = "requires CUDA driver (the cache surface itself is host-safe; gating is for symmetry)"]
fn cub_kernel_source_cache_round_trip() {
    // Some hosts ship an older libcuda.so than cudarc 0.19's bindings
    // expect (missing newer symbols). cudarc panics on dlsym; catch
    // and skip so the test stays useful as a smoke probe.
    let probe = std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0));
    match probe {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            eprintln!("[skip] CUDA driver init failed: {e}");
            return;
        }
        Err(_) => {
            eprintln!("[skip] cudarc panicked on dlsym (driver likely older than its bindings)");
            return;
        }
    }
    let mut cache = KernelSourceCache::default();
    let ptx_blob: Arc<Vec<u8>> = Arc::new(b"// fake PTX".to_vec());
    cache.insert("reduce_sum", "f32", ptx_blob.clone());
    let got = cache.get("reduce_sum", "f32").expect("cache miss after insert");
    assert_eq!(&*got, &*ptx_blob, "round-trip mismatch");
    assert_eq!(cache.len(), 1);
    assert!(cache.get("reduce_sum", "f64").is_none(), "dtype namespace bleed");

    // Op-name distinctness: every reduction op produces a different cache key.
    let ops = [
        ReductionOp::Sum,
        ReductionOp::Max,
        ReductionOp::Min,
        ReductionOp::ArgMax,
        ReductionOp::ArgMin,
        ReductionOp::Product,
    ];
    let mut keys: Vec<String> = ops
        .iter()
        .map(|op| kernel_key(&format!("reduce_{:?}", op).to_lowercase(), "f32"))
        .collect();
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), ops.len(), "kernel keys collide across reduction ops");

    println!("[cub] kernel_source_cache round-trip + 6 distinct reduction-op keys verified");
}
