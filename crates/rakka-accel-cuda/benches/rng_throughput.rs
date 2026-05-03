//! Benchmark `RngActor::FillUniformF32` throughput vs raw
//! `CudaRng::fill_with_uniform`. Target: actor path within 2% of
//! the raw path on N >= 1M.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use cudarc::curand::CudaRng;
use cudarc::driver::CudaContext;
use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use tokio::runtime::Runtime;

use rakka_accel_cuda::prelude::*;

const SIZES: &[usize] = &[1 << 16, 1 << 18, 1 << 20];

fn bench_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng_uniform_raw");
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &n| {
            let ctx = CudaContext::new(0).unwrap();
            let stream = ctx.new_stream().unwrap();
            let rng = CudaRng::new(0, stream.clone()).unwrap();
            let mut buf = stream.alloc_zeros::<f32>(n).unwrap();
            bencher.iter(|| {
                rng.fill_with_uniform(&mut buf).unwrap();
                stream.synchronize().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_actor(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng_uniform_actor");
    let rt = Runtime::new().unwrap();
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &n| {
            let (sys, device, rng) = rt.block_on(async {
                let sys = ActorSystem::create("bench-rng", Config::empty()).await.unwrap();
                let dev_cfg = DeviceConfig::new(0)
                    .with_libraries(EnabledLibraries::BLAS | EnabledLibraries::CURAND);
                let device = sys.actor_of(DeviceActor::props(dev_cfg), "dev").unwrap();
                tokio::time::sleep(Duration::from_millis(500)).await;
                let snap: Option<KernelChildren> = device
                    .ask_with(
                        move |tx| DeviceMsg::SnapshotChildren { reply: tx },
                        Duration::from_secs(5),
                    )
                    .await
                    .unwrap();
                let children = snap.unwrap();
                (sys, device, children.rng.unwrap())
            });
            bencher.iter(|| {
                rt.block_on(async {
                    let buf = device
                        .ask_with(
                            move |tx| DeviceMsg::AllocateF32 { len: n, reply: tx },
                            Duration::from_secs(2),
                        )
                        .await
                        .unwrap()
                        .unwrap();
                    let r: Result<(), _> = rng
                        .ask_with(
                            move |tx| RngMsg::FillUniformF32 { dst: buf, reply: tx },
                            Duration::from_secs(2),
                        )
                        .await
                        .unwrap();
                    r.unwrap();
                });
            });
            rt.block_on(async {
                sys.terminate().await;
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_raw, bench_actor);
criterion_main!(benches);
