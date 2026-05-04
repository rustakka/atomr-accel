//! Error taxonomy and the supervisor decider for context-poisoning recovery
//! (§5.3, §5.11 of the architecture document).
//!
//! Rakka's supervisor (`atomr_core::supervision::Decider`) inspects a panic
//! message string rather than a typed error value. To trigger the doc's
//! `OneForOne::default().on::<ContextPoisoned>(Restart)` behaviour, an
//! actor that detects context poisoning panics with a message containing
//! the string `"ContextPoisoned"`. The decider exposed here parses these
//! markers back into supervisor directives.

use atomr_core::supervision::{Directive, OneForOneStrategy, SupervisorOf, SupervisorStrategy};
use std::time::Duration;
use thiserror::Error;

/// Marker prefix used in panic messages to signal a poisoned-context error.
/// Matched by [`decider`].
pub const CONTEXT_POISONED_TAG: &str = "ContextPoisoned";
/// Marker prefix for OOM errors that the supervisor should `Resume` past.
pub const OUT_OF_MEMORY_TAG: &str = "OutOfMemory";
/// Marker prefix for fatal errors that should stop the device entirely.
pub const UNRECOVERABLE_TAG: &str = "Unrecoverable";

#[derive(Debug, Error)]
pub enum GpuError {
    /// CUDA context is in a sticky-error state (§5.3). Triggers
    /// `ContextActor` restart and a generation bump on `DeviceState`.
    #[error("ContextPoisoned: {0}")]
    ContextPoisoned(String),

    /// Allocation failed but the context is still usable. Supervisor
    /// `Resume`s the actor.
    #[error("OutOfMemory: {0}")]
    OutOfMemory(String),

    /// Hardware fault or repeated poisoning past the retry budget.
    #[error("Unrecoverable: {0}")]
    Unrecoverable(String),

    /// `GpuRef::access()` was called on a buffer whose context was
    /// rebuilt or whose `DeviceActor` is shutting down (§5.8).
    #[error("GpuRef stale: {0}")]
    GpuRefStale(&'static str),

    #[error("cudarc driver error: {0}")]
    Driver(String),

    /// cuBLAS-specific error. Retained for back-compat — new library
    /// actors should emit [`GpuError::LibraryError`] with `lib = "cublas"`
    /// instead. Will be removed in a future release.
    #[deprecated(note = "use GpuError::LibraryError { lib: \"cublas\", msg } instead")]
    #[error("cudarc cuBLAS error: {0}")]
    Cublas(String),

    /// Generic library error tagged with the originating CUDA library
    /// name (e.g. `"cudnn"`, `"cufft"`, `"curand"`, `"cusolver"`,
    /// `"cublaslt"`, `"nvrtc"`, `"nccl"`). Callers that need to
    /// discriminate library failures match on `lib`.
    #[error("cudarc {lib} error: {msg}")]
    LibraryError { lib: &'static str, msg: String },

    #[error("ask timed out before GPU completion")]
    Timeout,
}

impl GpuError {
    /// Construct a tagged library error.
    pub fn lib(lib: &'static str, msg: impl Into<String>) -> Self {
        Self::LibraryError {
            lib,
            msg: msg.into(),
        }
    }
}

impl GpuError {
    /// Format suitable for panicking out of an actor handler so that the
    /// atomr supervisor's decider can route it.
    pub fn panic_message(&self) -> String {
        self.to_string()
    }
}

/// The supervisor decider used by `DeviceActor` to route `ContextActor`
/// failures (§5.11).
///
/// Rakka 0.2.0 ships a typed `SupervisorOf<C>` trait (see the
/// [`device_supervisor`] impl below) that lets `DeviceActor` pattern-
/// match on `&GpuError` directly. The closure-based `decider()` here
/// is retained as the runtime fallback used by
/// [`device_supervisor_strategy`] — actors without an explicit
/// `SupervisorOf<C>` impl fall through to this string-matching path,
/// and panicking remains the failure transport regardless (since
/// `Actor::handle` returns `()`). The typed trait simply replaces the
/// receive-side parsing.
pub fn decider() -> impl Fn(&str) -> Directive + Send + Sync + 'static {
    |panic_msg: &str| {
        if panic_msg.contains(CONTEXT_POISONED_TAG) {
            Directive::Restart
        } else if panic_msg.contains(OUT_OF_MEMORY_TAG) {
            Directive::Resume
        } else if panic_msg.contains(UNRECOVERABLE_TAG) {
            Directive::Stop
        } else {
            // Default: surface the failure rather than masking it.
            Directive::Escalate
        }
    }
}

/// Build the `SupervisorStrategy` `DeviceActor` applies to its
/// `ContextActor` child (§5.11). Three retries inside a one-minute window;
/// past that, the circuit opens and the device stops.
pub fn device_supervisor_strategy() -> SupervisorStrategy {
    OneForOneStrategy::new()
        .with_max_retries(3)
        .with_within(Duration::from_secs(60))
        .with_decider(decider())
        .into()
}

/// Typed `SupervisorOf<ContextActor>` adapter for `DeviceActor`.
///
/// atomr 0.2.0 added the [`SupervisorOf`] trait so a parent can decide
/// child failures by pattern-matching a typed error rather than parsing
/// the panic-string. The implementation here lives behind the
/// [`device_supervisor`] zero-sized type so it can be used either
/// independently (`DeviceSupervisor.decide(&err)`) or attached to
/// future call sites that take a `SupervisorOf<C>` constraint.
///
/// We attach the impl to a marker rather than directly to `DeviceActor`
/// so that the `error` module stays free of a circular dependency on
/// `device::DeviceActor` / `device::ContextActor`. The decision logic
/// is identical to the closure in [`decider`] — and indeed
/// [`DeviceSupervisor::decide_str`] is what the closure-based code path
/// internally calls.
pub struct DeviceSupervisor;

impl DeviceSupervisor {
    /// Typed decider over `&GpuError`. Mirrors the panic-string match
    /// in [`decider`].
    pub fn decide(err: &GpuError) -> Directive {
        match err {
            GpuError::ContextPoisoned(_) => Directive::Restart,
            GpuError::OutOfMemory(_) => Directive::Resume,
            GpuError::Unrecoverable(_) => Directive::Stop,
            GpuError::Timeout
            | GpuError::GpuRefStale(_)
            | GpuError::Driver(_)
            | GpuError::LibraryError { .. } => Directive::Escalate,
            #[allow(deprecated)]
            GpuError::Cublas(_) => Directive::Escalate,
        }
    }

    /// Convenience: decide directly from the panic-string transport.
    /// Equivalent to invoking [`decider`] but available as a free
    /// function for callers who already have the panic message in
    /// hand.
    pub fn decide_str(panic_msg: &str) -> Directive {
        if panic_msg.contains(CONTEXT_POISONED_TAG) {
            Directive::Restart
        } else if panic_msg.contains(OUT_OF_MEMORY_TAG) {
            Directive::Resume
        } else if panic_msg.contains(UNRECOVERABLE_TAG) {
            Directive::Stop
        } else {
            Directive::Escalate
        }
    }
}

/// Blanket `SupervisorOf<C>` impl: any atomr actor `C` whose failures
/// the application classifies as [`GpuError`] can be supervised by
/// this marker. The trait's `decide` method dispatches to
/// [`DeviceSupervisor::decide`].
impl<C> SupervisorOf<C> for DeviceSupervisor
where
    C: atomr_core::actor::Actor,
{
    type ChildError = GpuError;

    fn decide(&self, err: &GpuError) -> Directive {
        DeviceSupervisor::decide(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decider_routes_context_poisoned_to_restart() {
        let d = decider();
        assert_eq!(d("ContextPoisoned: cuInit failed"), Directive::Restart);
    }

    #[test]
    fn decider_routes_out_of_memory_to_resume() {
        let d = decider();
        assert_eq!(d("OutOfMemory: alloc 1GB"), Directive::Resume);
    }

    #[test]
    fn decider_routes_unrecoverable_to_stop() {
        let d = decider();
        assert_eq!(d("Unrecoverable: hardware fault"), Directive::Stop);
    }

    #[test]
    fn decider_escalates_unknown_panics() {
        let d = decider();
        assert_eq!(d("some other panic"), Directive::Escalate);
    }

    #[test]
    fn typed_supervisor_routes_context_poisoned_to_restart() {
        let err = GpuError::ContextPoisoned("simulated".into());
        assert_eq!(DeviceSupervisor::decide(&err), Directive::Restart);
    }

    #[test]
    fn typed_supervisor_routes_oom_to_resume() {
        let err = GpuError::OutOfMemory("alloc 1GB".into());
        assert_eq!(DeviceSupervisor::decide(&err), Directive::Resume);
    }

    #[test]
    fn typed_supervisor_routes_unrecoverable_to_stop() {
        let err = GpuError::Unrecoverable("hw fault".into());
        assert_eq!(DeviceSupervisor::decide(&err), Directive::Stop);
    }

    #[test]
    fn typed_supervisor_escalates_other() {
        let err = GpuError::Timeout;
        assert_eq!(DeviceSupervisor::decide(&err), Directive::Escalate);
        let err = GpuError::GpuRefStale("stale");
        assert_eq!(DeviceSupervisor::decide(&err), Directive::Escalate);
        let err = GpuError::lib("cublas", "x");
        assert_eq!(DeviceSupervisor::decide(&err), Directive::Escalate);
    }
}
