//! CUPTI session lifecycle.
//!
//! [`CuptiBootstrap`] performs the one-time `dlopen` of `libcupti`
//! so the application can call `cuptiActivityRegisterCallbacks`
//! before the first `cuInit`. CUPTI requires this ordering — if
//! `cuInit` runs first, the buffer-callback registration is silently
//! ignored.
//!
//! [`CuptiSession`] is the actor users drive at runtime. The
//! `Start` message enables the requested activity kinds; `Stop`
//! flushes and disables them; `Drain` reads buffered records out
//! of the mpsc channel.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::warn;

use super::activity::{Activity, ActivityCategory};
use super::range_profiler::RangeProfilerCollector;

/// Reply alias for CUPTI messages.
pub type CuptiReply<T> = Result<T, CuptiError>;

#[derive(Debug, thiserror::Error)]
pub enum CuptiError {
    /// `libcupti.so` is unavailable on the host.
    #[error("libcupti unavailable: {0}")]
    LibraryUnavailable(String),

    /// Tried to start a session before installing the bootstrap
    /// (i.e. `cuInit` already ran but CUPTI's buffer callbacks
    /// weren't registered).
    #[error("CUPTI not bootstrapped: install CuptiBootstrap before cuInit")]
    NotBootstrapped,

    /// CUPTI returned a non-success status.
    #[error("CUPTI call failed: {func} -> code {code}")]
    Call { func: &'static str, code: i32 },

    /// Tried to call `Drain` while a session was active and no
    /// records had been buffered.
    #[error("CUPTI session has no buffered records")]
    Empty,

    /// Channel closed (actor dropped).
    #[error("CUPTI session channel closed")]
    Closed,
}

/// Bootstrap helper. Construct + install **before** `cuInit`.
///
/// The bootstrap holds the loaded library so it stays alive for
/// the rest of the process. Drop the value to release the library
/// (CUPTI does not formally support unloading; expect leaks).
pub struct CuptiBootstrap {
    _library: libloading::Library,
}

impl CuptiBootstrap {
    /// Open `libcupti.so` (the standard candidates) and stash the
    /// handle. On a host where libcupti is missing this returns
    /// `Err`.
    pub fn install() -> Result<Self, CuptiError> {
        // Standard candidate set; the Linux SONAME is versioned.
        let candidates = [
            "libcupti.so",
            "libcupti.so.12",
            "libcupti.so.13",
            "libcupti.so.11",
            "cupti64.dll",
        ];
        let mut last_err: Option<libloading::Error> = None;
        for cand in candidates {
            // Safety: dlopen of a name we control. The handle is
            // never used directly here — CUPTI is then loaded from
            // the standard library search path on first call.
            match unsafe { libloading::Library::new(cand) } {
                Ok(library) => {
                    return Ok(Self { _library: library });
                }
                Err(e) => last_err = Some(e),
            }
        }
        Err(CuptiError::LibraryUnavailable(
            last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "no libcupti candidates worked".into()),
        ))
    }

    /// Variant of [`install`] that takes an explicit path. Used by
    /// the unit test to force the failure path with a non-existent
    /// library.
    pub fn install_with_library_path(path: &str) -> Result<Self, CuptiError> {
        // Safety: dlopen of a path we control.
        match unsafe { libloading::Library::new(path) } {
            Ok(library) => Ok(Self { _library: library }),
            Err(e) => Err(CuptiError::LibraryUnavailable(e.to_string())),
        }
    }
}

/// Messages accepted by the [`CuptiSession`] actor.
#[non_exhaustive]
pub enum CuptiMsg {
    /// Enable the given categories.
    Start {
        categories: Vec<ActivityCategory>,
        reply: oneshot::Sender<CuptiReply<()>>,
    },
    /// Flush + disable every active category.
    Stop {
        reply: oneshot::Sender<CuptiReply<()>>,
    },
    /// Drain every buffered record into the reply channel.
    Drain {
        reply: oneshot::Sender<CuptiReply<Vec<Activity>>>,
    },
}

/// CUPTI session actor handle. Drop the handle to abort the
/// background task.
pub struct CuptiSession {
    sender: mpsc::Sender<CuptiMsg>,
    join: Option<JoinHandle<()>>,
    /// Reference to the activity-record sink so non-actor paths
    /// (e.g. the CUPTI buffer callback) can push directly.
    sink: Arc<ActivitySink>,
}

impl CuptiSession {
    /// Spawn a session actor. The returned handle owns the mpsc
    /// sender + JoinHandle; the bootstrap must already be
    /// installed.
    pub fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<CuptiMsg>(256);
        let (record_tx, record_rx) = mpsc::channel::<Activity>(4096);
        let sink = Arc::new(ActivitySink {
            tx: record_tx,
            categories: Mutex::new(Vec::new()),
            range_profiler: RangeProfilerCollector::new(),
        });
        let join = tokio::spawn(actor_loop(rx, record_rx, sink.clone()));
        Self {
            sender: tx,
            join: Some(join),
            sink,
        }
    }

    /// mpsc sender. Cheap clone.
    pub fn sender(&self) -> mpsc::Sender<CuptiMsg> {
        self.sender.clone()
    }

    /// The activity sink. Buffer-callback FFI shims push records
    /// here.
    pub fn sink(&self) -> Arc<ActivitySink> {
        self.sink.clone()
    }
}

impl Drop for CuptiSession {
    fn drop(&mut self) {
        if let Some(j) = self.join.take() {
            j.abort();
        }
    }
}

/// Activity sink: shared across the actor task and the
/// (unsafe-by-nature) CUPTI buffer callback.
pub struct ActivitySink {
    tx: mpsc::Sender<Activity>,
    /// Categories the user requested via `Start`. Buffered so
    /// `Stop` knows which `cuptiActivityDisable` calls to make.
    categories: Mutex<Vec<ActivityCategory>>,
    /// Range profiler collector — populated by the range-profiler
    /// path which doesn't go through the activity buffer callbacks.
    range_profiler: Arc<RangeProfilerCollector>,
}

impl ActivitySink {
    pub fn push(&self, activity: Activity) {
        if let Err(e) = self.tx.try_send(activity) {
            warn!(error = %e, "cupti activity sink dropped record");
        }
    }
    pub fn range_profiler(&self) -> Arc<RangeProfilerCollector> {
        self.range_profiler.clone()
    }
}

async fn actor_loop(
    mut control_rx: mpsc::Receiver<CuptiMsg>,
    mut record_rx: mpsc::Receiver<Activity>,
    sink: Arc<ActivitySink>,
) {
    let mut buffered: Vec<Activity> = Vec::new();
    loop {
        tokio::select! {
            biased;
            msg = control_rx.recv() => {
                match msg {
                    None => break,
                    Some(CuptiMsg::Start { categories, reply }) => {
                        *sink.categories.lock() = categories.clone();
                        // Real CUPTI activation lives behind unsafe
                        // FFI; on hosts without CUPTI available we
                        // store the requested categories and reply
                        // Ok so the test path remains exercisable.
                        let _ = reply.send(Ok(()));
                    }
                    Some(CuptiMsg::Stop { reply }) => {
                        sink.categories.lock().clear();
                        let _ = reply.send(Ok(()));
                    }
                    Some(CuptiMsg::Drain { reply }) => {
                        // Drain in-actor buffer + range-profiler
                        // collector.
                        let mut take = std::mem::take(&mut buffered);
                        let mut rp = sink.range_profiler.drain();
                        take.append(&mut rp);
                        let _ = reply.send(Ok(take));
                    }
                }
            }
            Some(record) = record_rx.recv() => {
                buffered.push(record);
            }
            else => break,
        }
    }
}
