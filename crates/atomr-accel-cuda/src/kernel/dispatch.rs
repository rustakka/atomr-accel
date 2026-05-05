//! Per-actor dispatch traits for typed, dtype-generic kernel
//! requests.
//!
//! Each library actor exposes a flat enum (`SolverMsg`, `BlasMsg`,
//! …) over its concrete request structs *and* a single `Op(Box<dyn
//! _Dispatch>)` arm that routes any future op without forcing a new
//! enum variant. The dispatch trait is implemented per request so
//! the actor can hand it the runtime cells (`handle`, `stream`,
//! `workspace`, …) it needs to execute.
//!
//! This module is small on purpose: it intentionally only exposes
//! the trait the public API consumes (`SolverDispatch`). Per-actor
//! cell types remain crate-private to keep the surface area narrow
//! while we ramp up coverage in subsequent phases.

#[cfg(feature = "cusolver")]
pub use crate::kernel::solver::SolverDispatch;
