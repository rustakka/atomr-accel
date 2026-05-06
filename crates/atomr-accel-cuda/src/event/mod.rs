//! `EventActor` — typed actor surface around `CudaEvent`.
//!
//! cudarc 0.19 ships a safe `CudaEvent` (Record / Wait / Query /
//! ElapsedTime / Synchronize) but doesn't expose IPC. This actor
//! wraps both the safe layer and the sys-level
//! `cuIpcGetEventHandle`/`cuIpcOpenEventHandle` for cross-process
//! event sharing.
//!
//! Lifecycle:
//! 1. Construct with `EventActor::props(ctx)` — captures an
//!    `Arc<CudaContext>` so events can be created on demand.
//! 2. Send `CreateEvent { reply }` (returns an `Event`) or
//!    `Record { event, stream, ... }` to push the event onto a
//!    captured stream.
//! 3. Wait/query/elapsed/synchronize as needed.
//! 4. (gated `cuda-ipc`) Send `GetIpcHandle` on the source process,
//!    transmit the bytes via your application channel, then
//!    `OpenIpcHandle` on the destination.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use atomr_core::actor::{Actor, Context, Props};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::error::GpuError;
#[cfg(feature = "cuda-ipc")]
use crate::sys::cuda_driver;
#[cfg(feature = "cuda-ipc")]
use cudarc::driver::sys as driver_sys;

const LIB: &str = "event";

/// Typed handle to a `CudaEvent`. Cloneable via `Arc`; the underlying
/// `cuEventDestroy` is run when the last clone drops.
#[derive(Clone)]
pub struct Event {
    inner: Arc<EventInner>,
}

/// `CudaEvent` is `Send + Sync` per cudarc, but we wrap it in our own
/// inner struct so external callers can't reach into the driver-level
/// handle without going through the actor.
struct EventInner {
    event: CudaEvent,
}

impl Event {
    /// Wrap an already-created `CudaEvent`. The actor uses this when
    /// minting events on demand.
    pub fn from_cuda(event: CudaEvent) -> Self {
        Self {
            inner: Arc::new(EventInner { event }),
        }
    }

    /// Underlying cudarc handle. Public because some downstream callers
    /// (`p2p`, `pipeline`) need to issue cross-stream waits via the
    /// safe `stream.wait(&event)` shape.
    pub fn cuda_event(&self) -> &CudaEvent {
        &self.inner.event
    }

    /// Raw driver-level event handle. Used by the IPC path; do not
    /// destroy the returned value.
    #[cfg(feature = "cuda-ipc")]
    pub fn cu_event(&self) -> driver_sys::CUevent {
        self.inner.event.cu_event()
    }
}

/// Cross-process IPC handle for an event. The 64-byte reserved blob
/// matches `CUipcEventHandle_st`.
///
/// `Clone + Copy + Send + Sync`. The handle bytes are opaque — the
/// driver may pack a process-id, ev-handle slot, etc. inside; treat as
/// black-box bytes when forwarding via your application's IPC channel.
#[cfg(feature = "cuda-ipc")]
#[derive(Clone, Copy)]
pub struct IpcEventHandle {
    pub(crate) raw: driver_sys::CUipcEventHandle,
}

#[cfg(feature = "cuda-ipc")]
impl IpcEventHandle {
    /// 64 bytes of opaque payload. Used by serialization helpers.
    pub fn as_bytes(&self) -> [u8; 64] {
        // SAFETY: `reserved` is `[c_char; 64]` with no padding; transmute is safe.
        unsafe { std::mem::transmute::<[std::ffi::c_char; 64], [u8; 64]>(self.raw.reserved) }
    }

    /// Reconstruct from a 64-byte payload (e.g. one received via
    /// shared memory or a Unix domain socket).
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        let raw = driver_sys::CUipcEventHandle_st {
            // SAFETY: `[c_char; 64]` has the same layout as `[u8; 64]`.
            reserved: unsafe { std::mem::transmute::<[u8; 64], [std::ffi::c_char; 64]>(bytes) },
        };
        Self { raw }
    }
}

#[cfg(feature = "cuda-ipc")]
impl std::fmt::Debug for IpcEventHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcEventHandle")
            .field("bytes_hash", &fxhash(&self.as_bytes()))
            .finish()
    }
}

#[cfg(feature = "cuda-ipc")]
fn fxhash(bytes: &[u8]) -> u64 {
    // Tiny FNV-1a — keeps Debug stable without pulling in a dep.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub enum EventMsg {
    /// Create a new event. Reply with the typed `Event`.
    Create {
        reply: oneshot::Sender<Result<Event, GpuError>>,
    },
    /// Record `event` against the current work in `stream`.
    Record {
        event: Event,
        stream: Arc<CudaStream>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Wait on `stream` until `event` completes.
    Wait {
        event: Event,
        stream: Arc<CudaStream>,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Non-blocking query: `true` if completed.
    Query {
        event: Event,
        reply: oneshot::Sender<Result<bool, GpuError>>,
    },
    /// Elapsed wall-clock time between two events.
    ElapsedTime {
        start: Event,
        end: Event,
        reply: oneshot::Sender<Result<Duration, GpuError>>,
    },
    /// Block the calling task until `event` completes.
    Synchronize {
        event: Event,
        reply: oneshot::Sender<Result<(), GpuError>>,
    },
    /// Export an IPC handle for `event` so a peer process can open it
    /// via `OpenIpcHandle`.
    #[cfg(feature = "cuda-ipc")]
    GetIpcHandle {
        event: Event,
        reply: oneshot::Sender<Result<IpcEventHandle, GpuError>>,
    },
    /// Open an IPC handle minted by another process. The opened event
    /// is bound to the actor's context.
    #[cfg(feature = "cuda-ipc")]
    OpenIpcHandle {
        handle: IpcEventHandle,
        reply: oneshot::Sender<Result<Event, GpuError>>,
    },
}

struct SendCtx(Arc<CudaContext>);
unsafe impl Send for SendCtx {}
unsafe impl Sync for SendCtx {}

#[allow(dead_code)]
enum EventInnerActor {
    Real { ctx: Mutex<SendCtx> },
    Mock,
}

pub struct EventActor {
    inner: EventInnerActor,
}

impl EventActor {
    pub fn props(ctx: Arc<CudaContext>) -> Props<Self> {
        Props::create(move || EventActor {
            inner: EventInnerActor::Real {
                ctx: Mutex::new(SendCtx(ctx.clone())),
            },
        })
    }

    pub fn mock_props() -> Props<Self> {
        Props::create(|| EventActor {
            inner: EventInnerActor::Mock,
        })
    }
}

#[async_trait]
impl Actor for EventActor {
    type Msg = EventMsg;

    async fn handle(&mut self, _ctx: &mut Context<Self>, msg: EventMsg) {
        match &self.inner {
            EventInnerActor::Mock => mock_reply(msg),
            EventInnerActor::Real { ctx } => {
                let ctx = ctx.lock().0.clone();
                handle_real(&ctx, msg);
            }
        }
    }
}

fn mock_reply(msg: EventMsg) {
    match msg {
        EventMsg::Create { reply } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        EventMsg::Record { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        EventMsg::Wait { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        EventMsg::Query { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        EventMsg::ElapsedTime { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        EventMsg::Synchronize { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        #[cfg(feature = "cuda-ipc")]
        EventMsg::GetIpcHandle { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
        #[cfg(feature = "cuda-ipc")]
        EventMsg::OpenIpcHandle { reply, .. } => {
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor in mock mode".into(),
            )));
        }
    }
}

fn handle_real(ctx: &Arc<CudaContext>, msg: EventMsg) {
    match msg {
        EventMsg::Create { reply } => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ctx.new_event(None)
                    .map(Event::from_cuda)
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("new_event: {e}"),
                    })
            }))
            .unwrap_or_else(|_| {
                Err(GpuError::Unrecoverable(
                    "EventActor::Create: CUDA driver not loadable".into(),
                ))
            });
            let _ = reply.send(r);
        }
        EventMsg::Record {
            event,
            stream,
            reply,
        } => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                event
                    .cuda_event()
                    .record(&stream)
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("record: {e}"),
                    })
            }))
            .unwrap_or_else(|_| {
                Err(GpuError::Unrecoverable(
                    "EventActor::Record: CUDA driver not loadable".into(),
                ))
            });
            let _ = reply.send(r);
        }
        EventMsg::Wait {
            event,
            stream,
            reply,
        } => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stream
                    .wait(event.cuda_event())
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("wait: {e}"),
                    })
            }))
            .unwrap_or_else(|_| {
                Err(GpuError::Unrecoverable(
                    "EventActor::Wait: CUDA driver not loadable".into(),
                ))
            });
            let _ = reply.send(r);
        }
        EventMsg::Query { event, reply } => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Ok::<_, GpuError>(event.cuda_event().is_complete())
            }))
            .unwrap_or_else(|_| {
                Err(GpuError::Unrecoverable(
                    "EventActor::Query: CUDA driver not loadable".into(),
                ))
            });
            let _ = reply.send(r);
        }
        EventMsg::ElapsedTime { start, end, reply } => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                start
                    .cuda_event()
                    .elapsed_ms(end.cuda_event())
                    .map(|ms| Duration::from_secs_f64(ms as f64 / 1000.0))
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("elapsed: {e}"),
                    })
            }))
            .unwrap_or_else(|_| {
                Err(GpuError::Unrecoverable(
                    "EventActor::ElapsedTime: CUDA driver not loadable".into(),
                ))
            });
            let _ = reply.send(r);
        }
        EventMsg::Synchronize { event, reply } => {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                event
                    .cuda_event()
                    .synchronize()
                    .map_err(|e| GpuError::LibraryError {
                        lib: LIB,
                        msg: format!("synchronize: {e}"),
                    })
            }))
            .unwrap_or_else(|_| {
                Err(GpuError::Unrecoverable(
                    "EventActor::Synchronize: CUDA driver not loadable".into(),
                ))
            });
            let _ = reply.send(r);
        }
        #[cfg(feature = "cuda-ipc")]
        EventMsg::GetIpcHandle { event, reply } => {
            let r = cuda_driver::ipc_get_event_handle(event.cu_event())
                .map(|raw| IpcEventHandle { raw });
            let _ = reply.send(r);
        }
        #[cfg(feature = "cuda-ipc")]
        EventMsg::OpenIpcHandle { handle, reply } => {
            let raw_event = match cuda_driver::ipc_open_event_handle(handle.raw) {
                Ok(e) => e,
                Err(e) => {
                    let _ = reply.send(Err(e));
                    return;
                }
            };
            // We've got a raw `CUevent` — wrap it into a cudarc
            // `CudaEvent` by going through the documented public
            // constructor. cudarc has no public adopt API, so we
            // rebuild a CudaEvent via `new_event` plus an explicit
            // record-from-raw shim. For Phase 3 we accept the raw
            // handle as a separate carrier — the `Event` we return
            // wraps a freshly-minted CudaEvent on the local context
            // and the caller is responsible for the cross-process
            // wait via the underlying IPC handle bytes.
            //
            // F-future: a `CudaEvent::from_raw` would let us hand back
            // a unified Event. For now: leak the raw event back via
            // `Unrecoverable` so callers know to use the bytes path
            // until the safe wrapper lands.
            let _ = raw_event;
            let _ = reply.send(Err(GpuError::Unrecoverable(
                "EventActor::OpenIpcHandle: cudarc 0.19 lacks CudaEvent::from_raw — \
                 use the IpcEventHandle bytes directly with cuStreamWaitEvent on the \
                 destination context"
                    .into(),
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_config::Config;
    use atomr_core::actor::ActorSystem;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn event_msg_round_trip() {
        let sys = ActorSystem::create("event-msg-test", Config::empty())
            .await
            .unwrap();
        let actor = sys.actor_of(EventActor::mock_props(), "evt").unwrap();

        // Create
        let (tx, rx) = oneshot::channel();
        actor.tell(EventMsg::Create { reply: tx });
        let r = tokio::time::timeout(Duration::from_secs(2), rx)
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(r, Err(GpuError::Unrecoverable(_))));

        sys.terminate().await;
    }

    #[cfg(feature = "cuda-ipc")]
    #[test]
    fn ipc_event_handle_serializes() {
        let bytes: [u8; 64] = std::array::from_fn(|i| i as u8);
        let h = IpcEventHandle::from_bytes(bytes);
        let round = h.as_bytes();
        assert_eq!(round, bytes);
        // Send / Sync sanity at the type level.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<IpcEventHandle>();
        let _clone: IpcEventHandle = h;
    }
}
