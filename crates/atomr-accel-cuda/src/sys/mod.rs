//! Local sys-level safe wrappers around cudarc's `*::sys` modules for
//! entry points cudarc 0.19 doesn't expose through its safe layer.
//!
//! Every wrapper:
//! - takes typed Rust pointers / cudarc handles,
//! - returns `Result<(), GpuError>` with a `LibraryError { lib, msg }`
//!   on failure,
//! - confines its `unsafe` to a single `extern "C"` call.
//!
//! Putting the unsafe in one place keeps cudarc-upgrade diffs small —
//! when cudarc adds a safe wrapper, the corresponding free function
//! here can be retired.

pub mod cublas;
