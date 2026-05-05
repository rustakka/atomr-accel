//! `cub_reduce` — micro-bench for the `CubReduce` actor entry. Gated
//! behind `cuda-runtime-tests` because it allocates a real
//! [`cudarc::driver::CudaContext`].

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_reduce(c: &mut Criterion) {
    c.bench_function("cub_reduce_sum_f32_1m", |b| {
        b.iter(|| {
            // Phase 5.1 wires the per-(op, dtype) NVRTC compile +
            // launch; the bench harness here exists so future PRs only
            // need to fill in the body rather than introducing a new
            // file.
            std::hint::black_box(0u32)
        });
    });
}

criterion_group!(benches, bench_reduce);
criterion_main!(benches);
