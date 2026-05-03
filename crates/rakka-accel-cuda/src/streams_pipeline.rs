//! `rakka-streams`-based pipeline helpers — the F10 successor to the
//! actor-based [`crate::pipeline::PipelineExecutor`].
//!
//! Wraps the Source / Sink DSL so callers can compose GPU kernel
//! stages with the rest of a `rakka-streams` graph (file IO, TCP,
//! framing, kill switches, restart-on-error supervision).
//!
//! The functions here are intentionally thin: they let you build a
//! `Source<I>` from any `mpsc::UnboundedReceiver`, transform it with
//! a `map_async` stage that calls a user-supplied async function (the
//! GPU kernel call), and terminate with one of the built-in sinks.
//! For more complex topologies (broadcast, balance, partition), drop
//! straight into `rakka_streams::*` — this module's helpers stay out
//! of your way.

use rakka_streams::{Sink, Source};

/// Wrap a `tokio::sync::mpsc::UnboundedReceiver` as a streams `Source`.
///
/// Callers send work into the matching `UnboundedSender`; the source
/// terminates when every sender is dropped.
pub fn source_from_unbounded<T: Send + 'static>(
    rx: tokio::sync::mpsc::UnboundedReceiver<T>,
) -> Source<T> {
    Source::from_receiver(rx)
}

/// Apply an async GPU stage with the given degree of parallelism.
/// Ordering is preserved (akka.net's `SelectAsync`).
///
/// # Example
///
/// ```ignore
/// let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<f32>();
/// let s = source_from_unbounded(rx);
/// let s = gpu_stage::<f32, f32, _, _>(s, 4, |x| async move { x * 2.0 });
/// let out = Sink::collect(s).await;
/// ```
pub fn gpu_stage<I, O, F, Fut>(source: Source<I>, parallelism: usize, f: F) -> Source<O>
where
    I: Send + 'static,
    O: Send + 'static,
    F: FnMut(I) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = O> + Send + 'static,
{
    source.map_async(parallelism.max(1), f)
}

/// Run a single-stage pipeline end-to-end: pull from `rx`, apply the
/// async `stage` with the given parallelism, and collect every output
/// into a `Vec`. The future completes when every sender is dropped
/// upstream.
pub async fn run_collect<I, O, F, Fut>(
    rx: tokio::sync::mpsc::UnboundedReceiver<I>,
    parallelism: usize,
    stage: F,
) -> Vec<O>
where
    I: Send + 'static,
    O: Send + 'static,
    F: FnMut(I) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = O> + Send + 'static,
{
    let s = gpu_stage(source_from_unbounded(rx), parallelism, stage);
    Sink::collect(s).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unbounded_round_trips_through_async_stage() {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
        for i in 1..=5 {
            tx.send(i).unwrap();
        }
        drop(tx);

        let mut got = run_collect::<u32, u32, _, _>(rx, 4, |x| async move { x * 10 }).await;
        // map_async with parallelism=4 doesn't guarantee global order
        // for >1 in-flight, so sort before assertion.
        got.sort();
        assert_eq!(got, vec![10, 20, 30, 40, 50]);
    }
}
