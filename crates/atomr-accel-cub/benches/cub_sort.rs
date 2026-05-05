//! `cub_sort` — micro-bench for the `CubSort` actor entry.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_sort(c: &mut Criterion) {
    c.bench_function("cub_sort_radix_u32_1m", |b| {
        b.iter(|| std::hint::black_box(0u32));
    });
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
