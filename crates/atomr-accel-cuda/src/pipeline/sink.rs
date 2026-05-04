//! Bounded-mpsc Source/Sink adapters around `PipelineExecutorN`.
//!
//! Produces an async ergonomics layer on top of the synchronous
//! executor: a producer (`PipelineSink<I>`) holds the head of a
//! bounded `tokio::mpsc::Sender<I>`; a driver task pulls items off
//! the channel one at a time, runs them through the executor, and
//! pushes results onto a tail `tokio::mpsc::Sender<Result<O>>`.
//! The consumer (`PipelineSource<O>`) wraps the tail receiver as
//! a `Stream<Result<O, GpuError>>`.
//!
//! Backpressure: the head channel's bound caps how many items can
//! be queued while the executor is busy. Real atomr-streams
//! integration (with `OverflowStrategy::Backpressure`) is a
//! drop-in once that crate is added as a workspace dependency.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::completion::CompletionStrategy;
use crate::error::GpuError;
use crate::pipeline::executor::PipelineExecutorN;

/// Producer end. `submit` blocks (awaits) when the channel is full
/// — that's the backpressure signal.
#[derive(Clone)]
pub struct PipelineSink<I: Send + 'static> {
    tx: mpsc::Sender<I>,
}

impl<I: Send + 'static> PipelineSink<I> {
    pub async fn submit(&self, item: I) -> Result<(), GpuError> {
        self.tx
            .send(item)
            .await
            .map_err(|_| GpuError::Unrecoverable("PipelineSink: driver dropped".into()))
    }

    pub fn try_submit(&self, item: I) -> Result<(), GpuError> {
        self.tx
            .try_send(item)
            .map_err(|e| GpuError::Unrecoverable(format!("PipelineSink try_submit: {e}")))
    }
}

/// Consumer end. Returns a `ReceiverStream<Result<O, GpuError>>`
/// that yields one item per processed input.
pub struct PipelineSource<O: Send + 'static> {
    rx: mpsc::Receiver<Result<O, GpuError>>,
}

impl<O: Send + 'static> PipelineSource<O> {
    pub fn into_stream(self) -> ReceiverStream<Result<O, GpuError>> {
        ReceiverStream::new(self.rx)
    }
}

/// Spawn a backpressured async pipeline driver around an executor.
/// Returns `(PipelineSink<I>, PipelineSource<O>)`. The driver runs
/// on the ambient tokio runtime.
pub fn spawn_pipeline<I, O>(
    mut executor: PipelineExecutorN,
    streams: Vec<Arc<CudaStream>>,
    completion: Arc<dyn CompletionStrategy>,
    head_capacity: usize,
    tail_capacity: usize,
) -> (PipelineSink<I>, PipelineSource<O>)
where
    I: Send + 'static,
    O: Send + 'static,
{
    let (in_tx, mut in_rx) = mpsc::channel::<I>(head_capacity.max(1));
    let (out_tx, out_rx) = mpsc::channel::<Result<O, GpuError>>(tail_capacity.max(1));
    tokio::spawn(async move {
        while let Some(item) = in_rx.recv().await {
            let result = executor.run::<I, O>(&streams, &completion, item).await;
            if out_tx.send(result).await.is_err() {
                break;
            }
        }
    });
    (PipelineSink { tx: in_tx }, PipelineSource { rx: out_rx })
}
