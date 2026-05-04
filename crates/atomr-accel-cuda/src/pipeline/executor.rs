//! Minimal pipeline executor: takes N boxed homogeneous stages and
//! runs them in sequence with event-based handoff.
//!
//! Intentionally simple — no Source / Sink integration, no
//! backpressure adapters. Those land in F3 once concrete patterns
//! exist that benefit from the full DSL.

use std::sync::Arc;

use cudarc::driver::CudaStream;

use crate::completion::CompletionStrategy;
use crate::error::GpuError;
use crate::pipeline::stage::PipelineStage;

/// Run a homogeneous sequence of stages on `streams[i]` for stage i.
///
/// Caller supplies one stream per stage (use [`crate::stream::PerActorAllocator`]
/// to mint them). The executor enqueues all stages, hooking each stage's
/// returned event into the next stage's `wait_for`, then awaits one
/// `HostFnCompletion` on the last stream.
pub async fn run_pipeline<S: PipelineStage>(
    stages: &mut [S],
    streams: &[Arc<CudaStream>],
    completion: &Arc<dyn CompletionStrategy>,
    input: S::In,
) -> Result<S::Out, GpuError>
where
    S::Out: From<S::In>, // Stage chain identity helper for trivial through-pipelines.
{
    if stages.is_empty() {
        return Err(GpuError::Unrecoverable("pipeline has zero stages".into()));
    }
    if stages.len() != streams.len() {
        return Err(GpuError::Unrecoverable(format!(
            "stage count {} != stream count {}",
            stages.len(),
            streams.len()
        )));
    }
    let stages_len = stages.len();
    let mut input = Some(input);
    let mut wait_event = None;
    let mut last_out: Option<S::Out> = None;
    for (i, stage) in stages.iter_mut().enumerate() {
        let stream = &streams[i];
        let in_v = input.take().expect("pipeline input consumed prematurely");
        let (ev, out) = stage.enqueue(stream, wait_event.as_ref(), in_v)?;
        wait_event = Some(ev);
        last_out = Some(out);
        if i + 1 < stages_len {
            return Err(GpuError::Unrecoverable(
                "run_pipeline currently supports only single-stage chains; \
                 use PipelineExecutor for multi-stage heterogeneous pipelines"
                    .into(),
            ));
        }
    }
    // Tail completion.
    let tail_stream = streams.last().unwrap();
    completion.await_completion(tail_stream).await?;
    last_out.ok_or_else(|| GpuError::Unrecoverable("pipeline produced no output".into()))
}

/// Two-stage type-state executor — the simplest non-trivial chain.
pub struct PipelineExecutor<S0, S1>
where
    S0: PipelineStage,
    S1: PipelineStage<In = S0::Out>,
{
    pub s0: S0,
    pub s1: S1,
}

impl<S0, S1> PipelineExecutor<S0, S1>
where
    S0: PipelineStage,
    S1: PipelineStage<In = S0::Out>,
{
    pub async fn run(
        &mut self,
        s0_stream: &Arc<CudaStream>,
        s1_stream: &Arc<CudaStream>,
        completion: &Arc<dyn CompletionStrategy>,
        input: S0::In,
    ) -> Result<S1::Out, GpuError> {
        let (ev0, out0) = self.s0.enqueue(s0_stream, None, input)?;
        let (_ev1, out1) = self.s1.enqueue(s1_stream, Some(&ev0), out0)?;
        completion.await_completion(s1_stream).await?;
        Ok(out1)
    }
}

/// Heterogeneous N-stage executor.
///
/// Each stage's `In` and `Out` types are erased into `Box<dyn Any +
/// Send>`. Stage adapters wrap their typed `PipelineStage` impl into
/// a `BoxedStage` and the executor drives the chain. Dynamic typing
/// gives up some compile-time safety in exchange for arbitrarily-long
/// chains; type mismatches at stage boundaries surface as
/// `GpuError::Unrecoverable("…downcast failed…")` at runtime.
pub trait BoxedStage: Send + 'static {
    fn enqueue_boxed(
        &mut self,
        stream: &Arc<CudaStream>,
        wait_for: Option<&cudarc::driver::CudaEvent>,
        input: Box<dyn std::any::Any + Send>,
    ) -> Result<(cudarc::driver::CudaEvent, Box<dyn std::any::Any + Send>), GpuError>;
}

/// Adapter wrapping any typed `PipelineStage` into a `BoxedStage`.
pub struct StageBox<S: PipelineStage> {
    inner: S,
}

impl<S: PipelineStage> StageBox<S> {
    pub fn new(s: S) -> Self {
        Self { inner: s }
    }
}

impl<S: PipelineStage> BoxedStage for StageBox<S> {
    fn enqueue_boxed(
        &mut self,
        stream: &Arc<CudaStream>,
        wait_for: Option<&cudarc::driver::CudaEvent>,
        input: Box<dyn std::any::Any + Send>,
    ) -> Result<(cudarc::driver::CudaEvent, Box<dyn std::any::Any + Send>), GpuError> {
        let typed = input.downcast::<S::In>().map_err(|_| {
            GpuError::Unrecoverable(format!(
                "PipelineExecutorN: stage input downcast to `{}` failed",
                std::any::type_name::<S::In>()
            ))
        })?;
        let (ev, out) = self.inner.enqueue(stream, wait_for, *typed)?;
        Ok((ev, Box::new(out) as Box<dyn std::any::Any + Send>))
    }
}

/// N-stage heterogeneous executor.
pub struct PipelineExecutorN {
    stages: Vec<Box<dyn BoxedStage>>,
}

impl PipelineExecutorN {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn push<S: PipelineStage>(mut self, stage: S) -> Self {
        self.stages.push(Box::new(StageBox::new(stage)));
        self
    }

    /// Run the chain across `streams` (one per stage). On success
    /// returns the tail stage's output. On any stage failure, the
    /// error short-circuits the chain.
    pub async fn run<I, O>(
        &mut self,
        streams: &[Arc<CudaStream>],
        completion: &Arc<dyn CompletionStrategy>,
        input: I,
    ) -> Result<O, GpuError>
    where
        I: Send + 'static,
        O: Send + 'static,
    {
        if self.stages.is_empty() {
            return Err(GpuError::Unrecoverable(
                "PipelineExecutorN: no stages".into(),
            ));
        }
        if streams.len() != self.stages.len() {
            return Err(GpuError::Unrecoverable(format!(
                "PipelineExecutorN: stage count {} != stream count {}",
                self.stages.len(),
                streams.len()
            )));
        }
        let mut payload: Box<dyn std::any::Any + Send> = Box::new(input);
        let mut wait_event: Option<cudarc::driver::CudaEvent> = None;
        for (stage, stream) in self.stages.iter_mut().zip(streams.iter()) {
            let (ev, next) = stage.enqueue_boxed(stream, wait_event.as_ref(), payload)?;
            wait_event = Some(ev);
            payload = next;
        }
        completion.await_completion(streams.last().unwrap()).await?;
        let out = payload.downcast::<O>().map_err(|_| {
            GpuError::Unrecoverable(format!(
                "PipelineExecutorN: tail downcast to `{}` failed",
                std::any::type_name::<O>()
            ))
        })?;
        Ok(*out)
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

impl Default for PipelineExecutorN {
    fn default() -> Self {
        Self::new()
    }
}
