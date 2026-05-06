//! Typed scope guard around `ncclGroupStart` / `ncclGroupEnd`.
//!
//! Usage at the world level:
//!
//! ```ignore
//! NcclWorld::group(|w| {
//!     w.send(buf, peer)?;
//!     w.recv(buf2, peer)?;
//!     Ok(())
//! })?;
//! ```
//!
//! At the actor level (one rank), `GroupGuard` issues `group_start`
//! on construction and `group_end` on `Drop` (or on `commit()`),
//! while emitting `BeginGroup` / `EndGroup` markers via a tracker so
//! tests can assert the begin/end pair fires exactly once.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cudarc::nccl::{group_end, group_start};

use super::LIB;
use crate::error::GpuError;

/// Counter of begin/end events. Tests construct one, hand it to a
/// `GroupGuard`, and assert the begin/end pair is balanced.
#[derive(Debug, Default)]
pub struct GroupTracker {
    pub begins: AtomicUsize,
    pub ends: AtomicUsize,
}

/// RAII scope guard for a group call on a single rank. Issues
/// `ncclGroupStart` on construction; `ncclGroupEnd` on `Drop` if
/// `commit()` was not called (errors logged but swallowed in Drop).
pub struct GroupGuard {
    tracker: Option<Arc<GroupTracker>>,
    committed: bool,
    /// Set to true on construction error so Drop is a no-op.
    inert: bool,
}

impl GroupGuard {
    /// Begin a group. If `tracker` is `Some`, increments
    /// `tracker.begins` on construction and `tracker.ends` on the
    /// matching end.
    pub fn begin(tracker: Option<Arc<GroupTracker>>) -> Result<Self, GpuError> {
        match group_start() {
            Ok(_) => {
                if let Some(t) = &tracker {
                    t.begins.fetch_add(1, Ordering::SeqCst);
                }
                Ok(Self {
                    tracker,
                    committed: false,
                    inert: false,
                })
            }
            Err(e) => Err(GpuError::LibraryError {
                lib: LIB,
                msg: format!("group_start: {e:?}"),
            }),
        }
    }

    /// Begin a group without invoking NCCL — for tests on hosts
    /// without a working NCCL install. Bumps the tracker but issues
    /// no FFI call.
    pub fn begin_inert(tracker: Option<Arc<GroupTracker>>) -> Self {
        if let Some(t) = &tracker {
            t.begins.fetch_add(1, Ordering::SeqCst);
        }
        Self {
            tracker,
            committed: false,
            inert: true,
        }
    }

    /// End the group. Returns the FFI result. Subsequent `Drop` is
    /// a no-op.
    pub fn commit(mut self) -> Result<(), GpuError> {
        self.committed = true;
        if let Some(t) = &self.tracker {
            t.ends.fetch_add(1, Ordering::SeqCst);
        }
        if self.inert {
            return Ok(());
        }
        group_end().map(|_| ()).map_err(|e| GpuError::LibraryError {
            lib: LIB,
            msg: format!("group_end: {e:?}"),
        })
    }
}

impl Drop for GroupGuard {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        if let Some(t) = &self.tracker {
            t.ends.fetch_add(1, Ordering::SeqCst);
        }
        if self.inert {
            return;
        }
        if let Err(e) = group_end() {
            tracing::warn!(error = ?e, "GroupGuard::drop: group_end failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructing a guard via `begin_inert` and dropping it must
    /// produce exactly one begin and one end event.
    #[test]
    fn group_scope_guard_emits_begin_end_pair() {
        let tracker = Arc::new(GroupTracker::default());
        {
            let _g = GroupGuard::begin_inert(Some(tracker.clone()));
            // dropped at end of scope
        }
        assert_eq!(tracker.begins.load(Ordering::SeqCst), 1);
        assert_eq!(tracker.ends.load(Ordering::SeqCst), 1);
    }

    /// Calling `commit()` must not double-fire the end counter.
    #[test]
    fn commit_then_drop_does_not_double_count() {
        let tracker = Arc::new(GroupTracker::default());
        let g = GroupGuard::begin_inert(Some(tracker.clone()));
        g.commit().unwrap();
        assert_eq!(tracker.begins.load(Ordering::SeqCst), 1);
        assert_eq!(tracker.ends.load(Ordering::SeqCst), 1);
    }
}
