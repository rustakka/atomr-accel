//! `GpuDispatcher` (§5.1) — pinned single-thread runtime that ensures the
//! actor's CUDA context stays current on the same OS thread for the
//! actor's whole lifetime.
//!
//! Tokio's default work-stealing scheduler moves tasks between worker
//! threads, which would break the "context is current on this thread"
//! invariant. This dispatcher owns its own dedicated OS thread, builds a
//! Tokio runtime on it (multi-threaded with `worker_threads = 1` so
//! background tasks make progress without anyone calling `block_on`),
//! and forwards `Dispatcher::spawn_task` to that runtime via a
//! [`DefaultDispatcher`] composed at construction time.
//!
//! Library actors that share a context with their `DeviceActor` should
//! use the same `GpuDispatcher`. F1 wires the dispatcher
//! programmatically; rakka-config integration is deferred to F2.

use std::sync::Arc;
use std::thread;

use futures_util::future::BoxFuture;
use rakka_core::dispatch::{DefaultDispatcher, Dispatcher, DispatcherHandle};
use tokio::sync::oneshot;

pub struct GpuDispatcher {
    inner: Arc<GpuDispatcherInner>,
}

struct GpuDispatcherInner {
    /// Wraps the runtime handle. We compose rather than re-implement so
    /// we don't have to construct `DispatcherHandle` from outside
    /// rakka-core (its inner field is `pub(crate)`).
    delegate: DefaultDispatcher,
    /// Held to keep the runtime thread alive until drop.
    _join: Option<thread::JoinHandle<()>>,
    shutdown_tx: parking_lot::Mutex<Option<oneshot::Sender<()>>>,
}

impl GpuDispatcher {
    /// Spawn the dedicated thread and its runtime, returning a ready-to-use
    /// dispatcher.
    pub fn new() -> std::io::Result<Self> {
        let (handle_tx, handle_rx) = std::sync::mpsc::sync_channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();

        let join = thread::Builder::new()
            .name("rakka-accel-cuda-gpu".into())
            .spawn(move || {
                // worker_threads(1) — exactly one tokio worker, on this
                // OS thread (the one we just spawned). All tasks
                // submitted via the runtime handle land here.
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_name("rakka-accel-cuda-gpu-worker")
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = handle_tx.send(Err(e));
                        return;
                    }
                };
                let _ = handle_tx.send(Ok(rt.handle().clone()));
                rt.block_on(async move {
                    let _ = shutdown_rx.await;
                });
            })?;

        let rt_handle = match handle_rx.recv() {
            Ok(Ok(h)) => h,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(std::io::Error::other(
                    "GpuDispatcher thread died before yielding its runtime handle",
                ));
            }
        };

        Ok(Self {
            inner: Arc::new(GpuDispatcherInner {
                delegate: DefaultDispatcher::new(rt_handle, 16),
                _join: Some(join),
                shutdown_tx: parking_lot::Mutex::new(Some(shutdown_tx)),
            }),
        })
    }
}

impl Dispatcher for GpuDispatcher {
    fn spawn_task(&self, task: BoxFuture<'static, ()>) -> DispatcherHandle {
        self.inner.delegate.spawn_task(task)
    }

    fn throughput(&self) -> u32 {
        self.inner.delegate.throughput()
    }
}

impl Drop for GpuDispatcherInner {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.lock().take() {
            let _ = tx.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn pinned_dispatcher_runs_on_dedicated_thread() {
        let d = GpuDispatcher::new().expect("spawn dispatcher");
        let (tx, rx) = std::sync::mpsc::channel::<thread::ThreadId>();

        for _ in 0..3 {
            let tx = tx.clone();
            d.spawn_task(Box::pin(async move {
                let _ = tx.send(thread::current().id());
            }));
        }

        let mut ids = Vec::new();
        for _ in 0..3 {
            ids.push(rx.recv_timeout(Duration::from_secs(2)).unwrap());
        }
        // All tasks ran on the same dispatcher thread...
        assert!(ids.windows(2).all(|w| w[0] == w[1]), "tasks ran on different threads: {:?}", ids);
        // ...and not the calling test thread.
        assert_ne!(ids[0], thread::current().id());
    }
}
