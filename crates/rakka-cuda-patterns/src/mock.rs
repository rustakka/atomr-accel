//! `GpuMockActor` — CPU-only stand-in for kernel actors.
//!
//! Implements a small subset of the kernel-actor message surface
//! using host-side `Vec<f32>` math, so pattern-level integration tests
//! can run on CI without a GPU.
//!
//! Currently supports:
//! - `MockSgemm`: f32 GEMM (no transpose, alpha=1, beta=0) computed
//!   on the host via naive triple-loop. Operands are owned `Vec<f32>`
//!   on the host — *not* `GpuRef<f32>` — because the mock has no
//!   device. Patterns that want to test against the mock pass host
//!   vectors directly.
//!
//! Real kernel-actor compatible mocks (those that accept `GpuRef<T>`
//! and route to a fake device) need `crate::host::PinnedBuf<T>`-style
//! plumbing; that's planned for F3 once the patterns crate has
//! integration tests demanding it.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;

pub struct MockSgemm {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
}

pub struct MockConv {
    /// NCHW input flattened.
    pub input: Vec<f32>,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    /// 3×3 filter (per-channel; broadcasts over input channels).
    pub kernel_3x3: Vec<f32>,
    pub reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
}

pub struct MockFftR2C {
    /// Real input.
    pub input: Vec<f32>,
    /// Reply: complex output as interleaved (re, im, re, im, ...).
    pub reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
}

pub struct MockRngFill {
    pub len: usize,
    pub seed: u64,
    pub reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
}

pub enum GpuMockMsg {
    Sgemm(Box<MockSgemm>),
    Conv(Box<MockConv>),
    FftR2C(Box<MockFftR2C>),
    RngFill(Box<MockRngFill>),
}

pub struct GpuMockActor;

impl GpuMockActor {
    pub fn props() -> Props<Self> {
        Props::create(|| GpuMockActor)
    }
}

#[async_trait]
impl Actor for GpuMockActor {
    type Msg = GpuMockMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: GpuMockMsg) {
        match msg {
            GpuMockMsg::Sgemm(req) => {
                let MockSgemm { a, b, m, n, k, reply } = *req;
                if a.len() != m * k {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "MockSgemm: A len {} != m*k {}",
                        a.len(),
                        m * k
                    ))));
                    return;
                }
                if b.len() != k * n {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "MockSgemm: B len {} != k*n {}",
                        b.len(),
                        k * n
                    ))));
                    return;
                }
                let mut c = vec![0.0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f32;
                        for kk in 0..k {
                            acc += a[i * k + kk] * b[kk * n + j];
                        }
                        c[i * n + j] = acc;
                    }
                }
                let _ = reply.send(Ok(c));
            }
            GpuMockMsg::Conv(req) => {
                let MockConv { input, n, c, h, w, kernel_3x3, reply } = *req;
                if kernel_3x3.len() != 9 {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "MockConv: kernel must be 3x3 (length 9)".into(),
                    )));
                    return;
                }
                if input.len() != n * c * h * w {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "MockConv: input len {} != n*c*h*w {}",
                        input.len(),
                        n * c * h * w
                    ))));
                    return;
                }
                let mut out = vec![0.0f32; n * c * h * w];
                for ni in 0..n {
                    for ci in 0..c {
                        for y in 0..h {
                            for x in 0..w {
                                let mut acc = 0.0f32;
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let sy = y as isize + ky as isize - 1;
                                        let sx = x as isize + kx as isize - 1;
                                        if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                                            continue;
                                        }
                                        let idx = ((ni * c + ci) * h + sy as usize) * w + sx as usize;
                                        acc += input[idx] * kernel_3x3[ky * 3 + kx];
                                    }
                                }
                                let oidx = ((ni * c + ci) * h + y) * w + x;
                                out[oidx] = acc;
                            }
                        }
                    }
                }
                let _ = reply.send(Ok(out));
            }
            GpuMockMsg::FftR2C(req) => {
                let MockFftR2C { input, reply } = *req;
                let n = input.len();
                if n == 0 {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "MockFftR2C: empty input".into(),
                    )));
                    return;
                }
                // Naive O(N^2) DFT — cuFFT-output shape is N/2+1 complex
                // bins for real input. Output is interleaved
                // (re, im, re, im, ...).
                let bins = n / 2 + 1;
                let mut out = Vec::with_capacity(bins * 2);
                let two_pi = std::f32::consts::TAU;
                for k in 0..bins {
                    let mut re = 0.0f32;
                    let mut im = 0.0f32;
                    for (j, x) in input.iter().enumerate() {
                        let theta = -two_pi * k as f32 * j as f32 / n as f32;
                        re += x * theta.cos();
                        im += x * theta.sin();
                    }
                    out.push(re);
                    out.push(im);
                }
                let _ = reply.send(Ok(out));
            }
            GpuMockMsg::RngFill(req) => {
                let MockRngFill { len, seed, reply } = *req;
                // Tiny deterministic LCG so tests with the same seed
                // get the same output.
                let mut state: u64 = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                let mut out = Vec::with_capacity(len);
                for _ in 0..len {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let bits = (state >> 33) as u32;
                    out.push((bits as f32) / (u32::MAX as f32));
                }
                let _ = reply.send(Ok(out));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mock_sgemm_2x2() {
        let sys = ActorSystem::create("mock-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(GpuMockActor::props(), "mock").unwrap();

        // 2x2 * 2x2 identity-ish: A = [[1,2],[3,4]], B = [[1,0],[0,1]] -> C = A.
        let (tx, rx) = oneshot::channel();
        actor.tell(GpuMockMsg::Sgemm(Box::new(MockSgemm {
            a: vec![1.0, 2.0, 3.0, 4.0],
            b: vec![1.0, 0.0, 0.0, 1.0],
            m: 2,
            n: 2,
            k: 2,
            reply: tx,
        })));
        let c = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);

        sys.terminate().await;
    }
}
