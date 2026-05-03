//! `ReductionAnalysisActor` — host-side reductions over a frame
//! (sum, mean, min, max, argmax). F8+ swaps to cuDNN
//! `ReduceTensor` for GPU-resident input.

use async_trait::async_trait;
use rakka_core::actor::{Actor, Context, Props};
use tokio::sync::oneshot;

use rakka_accel_cuda::error::GpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    Sum,
    Mean,
    Min,
    Max,
    ArgMax,
}

pub enum ReductionMsg {
    Reduce {
        kind: ReductionKind,
        data: Vec<f32>,
        reply: oneshot::Sender<Result<f32, GpuError>>,
    },
    /// Multi-channel reduction: returns one value per channel.
    /// `data.len() == channels * width * height`.
    ReduceChannels {
        kind: ReductionKind,
        data: Vec<f32>,
        channels: usize,
        reply: oneshot::Sender<Result<Vec<f32>, GpuError>>,
    },
}

pub struct ReductionAnalysisActor;

impl ReductionAnalysisActor {
    pub fn props() -> Props<Self> {
        Props::create(|| ReductionAnalysisActor)
    }
}

fn reduce_one(kind: ReductionKind, data: &[f32]) -> Result<f32, GpuError> {
    if data.is_empty() {
        return Err(GpuError::Unrecoverable("Reduction: empty input".into()));
    }
    Ok(match kind {
        ReductionKind::Sum => data.iter().sum(),
        ReductionKind::Mean => data.iter().sum::<f32>() / data.len() as f32,
        ReductionKind::Min => data.iter().copied().fold(f32::INFINITY, f32::min),
        ReductionKind::Max => data.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        ReductionKind::ArgMax => {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in data.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            best_idx as f32
        }
    })
}

#[async_trait]
impl Actor for ReductionAnalysisActor {
    type Msg = ReductionMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: ReductionMsg) {
        match msg {
            ReductionMsg::Reduce { kind, data, reply } => {
                let _ = reply.send(reduce_one(kind, &data));
            }
            ReductionMsg::ReduceChannels { kind, data, channels, reply } => {
                if channels == 0 {
                    let _ = reply.send(Err(GpuError::Unrecoverable(
                        "ReduceChannels: channels == 0".into(),
                    )));
                    return;
                }
                if data.len() % channels != 0 {
                    let _ = reply.send(Err(GpuError::Unrecoverable(format!(
                        "ReduceChannels: data len {} not divisible by {channels}",
                        data.len()
                    ))));
                    return;
                }
                let per_chan = data.len() / channels;
                let mut out = Vec::with_capacity(channels);
                for c in 0..channels {
                    let chunk: Vec<f32> = (0..per_chan).map(|i| data[i * channels + c]).collect();
                    match reduce_one(kind, &chunk) {
                        Ok(v) => out.push(v),
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            return;
                        }
                    }
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
    async fn sum_mean_max() {
        let sys = ActorSystem::create("reduce-test", Config::empty()).await.unwrap();
        let actor = sys.actor_of(ReductionAnalysisActor::props(), "r").unwrap();

        for (kind, expected) in [
            (ReductionKind::Sum, 10.0),
            (ReductionKind::Mean, 2.5),
            (ReductionKind::Max, 4.0),
            (ReductionKind::Min, 1.0),
            (ReductionKind::ArgMax, 3.0),
        ] {
            let (tx, rx) = oneshot::channel();
            actor.tell(ReductionMsg::Reduce {
                kind,
                data: vec![1.0, 2.0, 3.0, 4.0],
                reply: tx,
            });
            let v = tokio::time::timeout(Duration::from_secs(2), rx).await.unwrap().unwrap().unwrap();
            assert!((v - expected).abs() < 1e-5, "{kind:?} expected {expected}, got {v}");
        }
        sys.terminate().await;
    }
}
