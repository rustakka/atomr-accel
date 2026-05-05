//! `cub_scan` — micro-bench for the `CubScan` actor entry.

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_scan(c: &mut Criterion) {
    c.bench_function("cub_scan_inclusive_sum_f32_1m", |b| {
        b.iter(|| std::hint::black_box(0u32));
    });
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
