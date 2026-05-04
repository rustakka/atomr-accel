//! Process-wide tokio runtime for the Python bridge.
//!
//! pyo3-async-runtimes bridges Python `asyncio` futures to a Tokio
//! runtime; the first call to `runtime()` initializes the
//! multi-threaded scheduler with sensible defaults for GPU work
//! (long-lived blocking allowed on dedicated threads, full I/O
//! drivers enabled).

use std::sync::Once;

use tokio::runtime::Builder;

static INIT: Once = Once::new();

pub fn ensure_initialized() {
    INIT.call_once(|| {
        let mut b = Builder::new_multi_thread();
        b.enable_all().thread_name("atomr-accel-py");
        pyo3_async_runtimes::tokio::init(b);
    });
}

pub fn runtime() -> &'static tokio::runtime::Runtime {
    ensure_initialized();
    pyo3_async_runtimes::tokio::get_runtime()
}
