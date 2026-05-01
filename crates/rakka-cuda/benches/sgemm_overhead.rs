//! `sgemm_overhead` — Criterion bench measuring the wall-clock overhead
//! the actor pipeline imposes on top of raw cudarc SGEMM.
//!
//! F1 exit criterion (§8): actor overhead < 5% for kernels above ~1 ms
//! (i.e. SGEMM at ≥ 2048²). Two measurements:
//!
//! - **raw**: `CudaBlas::sgemm` on a stream + `stream.synchronize()`.
//! - **actor**: `DeviceMsg::Sgemm` `ask_with` round-trip ending with the
//!   `HostFnCompletion` reply.
//!
//! The bench runs only with `--features cuda-runtime-tests`. Numbers are
//! produced on a GPU host; the dev box has no GPU and skips the bench.

use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::CudaContext;
use rakka_config::Config;
use rakka_core::actor::ActorSystem;
use rakka_cuda::prelude::*;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

const SIZES: &[i32] = &[1024, 2048, 4096];

fn bench_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgemm/raw");
    for &n in SIZES {
        group.throughput(Throughput::Elements((n as u64).pow(3)));
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = ctx.new_stream().expect("CUDA stream");
        let blas = CudaBlas::new(stream.clone()).expect("CudaBlas");
        let a = stream.alloc_zeros::<f32>((n * n) as usize).unwrap();
        let b = stream.alloc_zeros::<f32>((n * n) as usize).unwrap();
        let mut cm = stream.alloc_zeros::<f32>((n * n) as usize).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &n| {
            bencher.iter(|| {
                let cfg = gemm_cfg(n);
                unsafe { blas.gemm(cfg, &a, &b, &mut cm).unwrap() };
                stream.synchronize().unwrap();
            });
        });
    }
    group.finish();
}

fn bench_actor(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio rt");
    let mut group = c.benchmark_group("sgemm/actor");

    let system =
        rt.block_on(async { ActorSystem::create("bench", Config::empty()).await.unwrap() });
    let device = system
        .actor_of(DeviceActor::props(DeviceConfig::new(0)), "device-0")
        .unwrap();

    for &n in SIZES {
        group.throughput(Throughput::Elements((n as u64).pow(3)));

        // Allocate operands once per size; reuse across iterations.
        let (a, b, cdata) = rt.block_on(async {
            let a = alloc(&device, (n * n) as usize).await;
            let b = alloc(&device, (n * n) as usize).await;
            let c = alloc(&device, (n * n) as usize).await;
            (a, b, c)
        });

        let device_ref = device.clone();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &n| {
            bencher.iter(|| {
                rt.block_on(async {
                    let (tx, rx) = oneshot::channel();
                    device_ref.tell(DeviceMsg::Sgemm(Box::new(SgemmRequest {
                        a: a.clone(),
                        b: b.clone(),
                        // Note: `c` cannot be cloned for a write target —
                        // we re-allocate per iteration.
                        c: rt.block_on(alloc(&device_ref, (n * n) as usize)),
                        m: n,
                        n,
                        k: n,
                        alpha: 1.0,
                        beta: 0.0,
                        reply: tx,
                    })));
                    rx.await.unwrap().unwrap();
                });
            });
        });
        let _ = cdata; // silence unused
    }
    group.finish();

    rt.block_on(async {
        system.terminate().await;
    });
}

async fn alloc(device: &rakka_core::actor::ActorRef<DeviceMsg>, len: usize) -> GpuRef<f32> {
    device
        .ask_with::<Result<GpuRef<f32>, GpuError>, _>(
            move |tx| DeviceMsg::Allocate { len, reply: tx },
            Duration::from_secs(30),
        )
        .await
        .unwrap()
        .unwrap()
}

fn gemm_cfg(n: i32) -> GemmConfig<f32> {
    GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n,
        k: n,
        alpha: 1.0,
        lda: n,
        ldb: n,
        beta: 0.0,
        ldc: n,
    }
}

criterion_group!(benches, bench_raw, bench_actor);
criterion_main!(benches);

// Silence unused-import warnings when this file is built standalone.
#[allow(dead_code)]
fn _arc_unused() -> Arc<()> { Arc::new(()) }
