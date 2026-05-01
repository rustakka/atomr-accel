//! `ImageFilterPipeline` — applies a 3×3 convolution filter to a
//! frame.
//!
//! Two execution backends, selected at construction time:
//!
//! - **CPU reference** (default) — per-pixel 3×3 convolution +
//!   clamp on the host. Always available; lets realtime patterns
//!   build end-to-end pipelines without a GPU.
//! - **cuDNN** (feature `cudnn`) — routes to a [`CudnnActor`]
//!   instance via [`ImageFilterPipeline::with_cudnn`]. The user
//!   pre-allocates `GpuRef<f32>` input/output buffers and the
//!   filter sends a [`CudnnMsg::ConvForward`] to the actor.
//!
//! The `Process` message uses the host-side path because the
//! `Vec<u8>` payload makes the ergonomics straightforward; users who
//! want zero-copy GPU pipelines call [`ImageFilterPipeline::process_gpu`]
//! against the cuDNN backend.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
#[cfg(feature = "cudnn")]
use rakka_core::actor::ActorRef;
use tokio::sync::oneshot;

use rakka_cuda::error::GpuError;
#[cfg(feature = "cudnn")]
use rakka_cuda::gpu_ref::GpuRef;

#[cfg(feature = "cudnn")]
use rakka_cuda::kernel::{
    ActivationKind, ActivationRequest, ConvForwardRequest, ConvParams, CudnnMsg,
};

#[derive(Debug, Clone)]
pub struct ImageFilterConfig {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// 3×3 kernel weights, row-major (length 9).
    pub kernel_3x3: Vec<f32>,
}

pub enum ImageFilterMsg {
    /// Host-side per-pixel 3×3 convolve + clamp. Always works.
    Process {
        frame: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, GpuError>>,
    },
    /// GPU-side conv routed through the configured `CudnnActor`.
    /// Only available with feature `cudnn`. `input` and `output`
    /// must be NCHW f32 tensors with the dims described by the
    /// pipeline's config.
    #[cfg(feature = "cudnn")]
    ProcessGpu {
        input: GpuRef<f32>,
        weights: GpuRef<f32>,
        output: GpuRef<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    UpdateKernel {
        kernel_3x3: Vec<f32>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
}

pub struct ImageFilterPipeline {
    config: ImageFilterConfig,
    #[cfg(feature = "cudnn")]
    cudnn: Option<ActorRef<CudnnMsg>>,
}

impl ImageFilterPipeline {
    pub fn props(config: ImageFilterConfig) -> Props<Self> {
        Props::create(move || ImageFilterPipeline {
            config: config.clone(),
            #[cfg(feature = "cudnn")]
            cudnn: None,
        })
    }

    /// Construct a cuDNN-backed pipeline. The `ProcessGpu` message
    /// routes through `cudnn`; `Process` (host-side) still works.
    #[cfg(feature = "cudnn")]
    pub fn props_with_cudnn(
        config: ImageFilterConfig,
        cudnn: ActorRef<CudnnMsg>,
    ) -> Props<Self> {
        Props::create(move || ImageFilterPipeline {
            config: config.clone(),
            cudnn: Some(cudnn.clone()),
        })
    }

    fn convolve(&self, frame: &[u8]) -> Result<Vec<u8>, GpuError> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let c = self.config.channels as usize;
        let expected = w * h * c;
        if frame.len() != expected {
            return Err(GpuError::Unrecoverable(format!(
                "ImageFilter: frame len {} != expected {expected}",
                frame.len()
            )));
        }
        if self.config.kernel_3x3.len() != 9 {
            return Err(GpuError::Unrecoverable(format!(
                "ImageFilter: kernel len {} != 9",
                self.config.kernel_3x3.len()
            )));
        }
        let k = &self.config.kernel_3x3;
        let mut out = vec![0u8; expected];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    let mut acc = 0.0f32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let sy = y as isize + ky as isize - 1;
                            let sx = x as isize + kx as isize - 1;
                            if sy < 0 || sy >= h as isize || sx < 0 || sx >= w as isize {
                                continue;
                            }
                            let idx = ((sy as usize) * w + (sx as usize)) * c + ch;
                            acc += frame[idx] as f32 * k[ky * 3 + kx];
                        }
                    }
                    let clamped = acc.clamp(0.0, 255.0) as u8;
                    out[(y * w + x) * c + ch] = clamped;
                }
            }
        }
        Ok(out)
    }
}

#[async_trait]
impl Actor for ImageFilterPipeline {
    type Msg = ImageFilterMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ImageFilterMsg) {
        match msg {
            ImageFilterMsg::Process { frame, reply } => {
                let _ = reply.send(self.convolve(&frame));
            }
            #[cfg(feature = "cudnn")]
            ImageFilterMsg::ProcessGpu { input, weights, output, reply } => {
                let Some(cudnn) = self.cudnn.clone() else {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "ProcessGpu: pipeline constructed without a CudnnActor; \
                         call ImageFilterPipeline::props_with_cudnn".into(),
                    )));
                    return;
                };
                // NCHW dims: 1 × C × H × W. Weights: C_out=C × C_in=C × 3 × 3.
                let n = 1i32;
                let c = self.config.channels as i32;
                let h = self.config.height as i32;
                let w = self.config.width as i32;
                cudnn.tell(CudnnMsg::ConvForward(Box::new(ConvForwardRequest {
                    x: input,
                    x_dims: [n, c, h, w],
                    w: weights,
                    w_dims: [c, c, 3, 3],
                    y: output,
                    y_dims: [n, c, h, w],
                    conv: ConvParams {
                        pad: [1, 1],
                        stride: [1, 1],
                        dilation: [1, 1],
                    },
                    alpha: 1.0,
                    beta: 0.0,
                    reply,
                })));
            }
            ImageFilterMsg::UpdateKernel { kernel_3x3, reply } => {
                if kernel_3x3.len() != 9 {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "UpdateKernel: kernel must be length 9".into(),
                    )));
                } else {
                    self.config.kernel_3x3 = kernel_3x3;
                    let _ = reply.send(Ok(()));
                }
            }
        }
    }
}

// `ActivationRequest` and `ActivationKind` are re-exported from the
// kernel module — keep them imported so the use line above isn't
// flagged. Future work: chain conv → activation in a single
// `ProcessGpu` to mirror the cuDNN epilogue.
#[cfg(feature = "cudnn")]
const _UNUSED_ACTIVATION_REEXPORTS: Option<(ActivationKind, ActivationRequest)> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use rakka_config::Config;
    use rakka_core::actor::ActorSystem;
    use std::time::Duration;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn identity_kernel_passes_frame_through() {
        // Identity 3x3 = center 1, others 0.
        let mut k = vec![0.0f32; 9];
        k[4] = 1.0;
        let cfg = ImageFilterConfig {
            width: 4,
            height: 4,
            channels: 1,
            kernel_3x3: k,
        };
        let frame: Vec<u8> = (0..16).map(|i| i as u8 * 8).collect();

        let sys = ActorSystem::create("filter-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(ImageFilterPipeline::props(cfg), "filter").unwrap();

        let (tx, rx) = oneshot::channel();
        actor.tell(ImageFilterMsg::Process { frame: frame.clone(), reply: tx });
        let out = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
        assert_eq!(out, frame);

        sys.terminate().await;
    }
}
