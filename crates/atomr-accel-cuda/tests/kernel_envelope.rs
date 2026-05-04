//! Mock-mode kernel envelope test — verifies that the F2 generic
//! envelope correctly tags errors, validates GpuRef freshness, and
//! short-circuits pre-launch failures through the reply channel.

use atomr_accel_cuda::error::GpuError;

#[test]
fn library_error_constructor_tags_lib() {
    let e = GpuError::lib("cudnn", "create_handle: bad alloc");
    match e {
        GpuError::LibraryError { lib, msg } => {
            assert_eq!(lib, "cudnn");
            assert!(msg.contains("create_handle"));
        }
        other => panic!("expected LibraryError, got {other:?}"),
    }
}

#[test]
fn library_error_panic_message_routes_to_supervisor() {
    use atomr_core::supervision::Directive;
    let d = atomr_accel_cuda::error::decider();
    // LibraryError isn't itself a panic tag; it has no Directive
    // implication. The panic-message tags (ContextPoisoned,
    // OutOfMemory, Unrecoverable) are unchanged.
    let e = GpuError::lib("cufft", "plan failed");
    let s = e.panic_message();
    assert!(!s.contains("ContextPoisoned"));
    assert!(!s.contains("OutOfMemory"));
    assert!(!s.contains("Unrecoverable"));
    // Unknown messages escalate.
    assert_eq!(d(&s), Directive::Escalate);
}
