"""``Memory`` handle — Phase 1.5 CUDA memory ops surface checks.

``atomr_accel.memory.Memory`` is unconditional once the merger wires
``mod memory`` into ``lib.rs``; until then the facade returns ``None``
and these tests skip. We verify:

- The class is exposed via the facade and has the expected method
  surface.
- Spawning under a ``System`` returns a handle and ``__repr__`` works.
- Mock-mode error paths (allocate / prefetch / advise / stats) surface
  the right typed exceptions without panicking.
- ``cuda-ipc``-gated symbols are present only when the wheel was built
  with that feature.

End-to-end allocation / prefetch / advise need a CUDA driver, so we
keep the assertions focused on the surface and the error envelope —
mirroring ``test_cudnn.py`` and the ``patterns`` test family.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import memory as memory_mod


pytestmark = pytest.mark.skipif(
    memory_mod.Memory is None,
    reason="Memory native class not wired (lib.rs registration pending merge)",
)


# ─── module surface ────────────────────────────────────────────────


PHASE_1_5_METHODS = (
    "allocate_managed_f32",
    "prefetch_f32",
    "advise_f32",
    "stats",
)


def test_memory_class_exposed_via_facade():
    """The facade re-exports the native class."""
    assert memory_mod.Memory.__name__ == "Memory"


def test_managed_buffer_class_exposed():
    """``ManagedBufferF32`` is exposed alongside ``Memory``."""
    assert memory_mod.ManagedBufferF32 is not None
    assert memory_mod.ManagedBufferF32.__name__ == "ManagedBufferF32"


@pytest.mark.parametrize("name", PHASE_1_5_METHODS)
def test_memory_method_exists(name: str) -> None:
    """Every Phase 1.5 method is a callable attribute on ``Memory``."""
    Memory = memory_mod.Memory
    assert hasattr(Memory, name), f"missing method: {name}"
    assert callable(getattr(Memory, name)), f"{name} not callable"


def test_memory_method_count_floor():
    """Pin a floor of 5 callable methods (4 listed + ``spawn``)."""
    Memory = memory_mod.Memory
    method_names = [
        n
        for n in dir(Memory)
        if not n.startswith("_") and callable(getattr(Memory, n, None))
    ]
    assert len(method_names) >= 5, (
        f"expected >= 5 Memory methods, got {len(method_names)}: "
        f"{sorted(method_names)}"
    )


# ─── spawn + handle envelope ───────────────────────────────────────


def test_memory_spawn_returns_handle():
    """``Memory.spawn`` succeeds against a fresh ``System`` — the
    underlying ``ManagedAllocatorActor`` does not need a CUDA driver
    just to come up."""
    with atomr_accel.System.open("memory-spawn") as sys:
        m = memory_mod.Memory.spawn(sys)
        assert m is not None
        assert "Memory" in repr(m)


def test_memory_stats_returns_pair_in_mock_mode():
    """``stats`` returns a ``(allocations, bytes_allocated)`` tuple.
    Fresh actor → both zero."""
    with atomr_accel.System.open("memory-stats") as sys:
        m = memory_mod.Memory.spawn(sys)
        allocs, nbytes = m.stats(timeout_secs=2.0)
        assert allocs == 0
        assert nbytes == 0


def test_allocate_managed_f32_typed_error_on_no_driver():
    """``allocate_managed_f32`` returns either a ``ManagedBufferF32``
    (real GPU host) or ``OutOfMemory`` (no driver). Both are
    acceptable — what matters is that we never panic and we always
    surface a typed exception subclass."""
    with atomr_accel.System.open("memory-alloc") as sys:
        m = memory_mod.Memory.spawn(sys)
        try:
            buf = m.allocate_managed_f32(64, timeout_secs=5.0)
            assert isinstance(buf, memory_mod.ManagedBufferF32)
            assert len(buf) == 64
            assert buf.dtype == "f32"
            assert "ManagedBufferF32" in repr(buf)
        except atomr_accel.OutOfMemory:
            pass
        except atomr_accel.GpuRuntimeError:
            # Other typed errors (Unrecoverable / LibraryError) are fine
            # too — the contract is "no naked panic".
            pass


def test_allocate_managed_f32_rejects_bad_flags():
    """Unknown `flags` strings are caught before the actor call and
    surface as ``GpuRuntimeError``."""
    with atomr_accel.System.open("memory-bad-flags") as sys:
        m = memory_mod.Memory.spawn(sys)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            m.allocate_managed_f32(16, flags="not-a-real-flag")


def test_advise_f32_rejects_unknown_advice():
    """Unknown `advice` strings surface as ``GpuRuntimeError`` before
    the buffer is even consulted — but we still need a real
    ``ManagedBufferF32`` to typecheck the call. Skip if the host can't
    allocate."""
    with atomr_accel.System.open("memory-advise-bad") as sys:
        m = memory_mod.Memory.spawn(sys)
        try:
            buf = m.allocate_managed_f32(16, timeout_secs=5.0)
        except atomr_accel.GpuRuntimeError:
            pytest.skip("no CUDA driver — can't construct a ManagedBufferF32")
        with pytest.raises(atomr_accel.GpuRuntimeError):
            m.advise_f32(buf, advice="bogus-advice")


def test_advise_f32_requires_target_for_set_preferred_location():
    """``set_preferred_location`` requires a ``target`` argument; the
    helper should reject the call before dispatching to the actor."""
    with atomr_accel.System.open("memory-advise-needs-target") as sys:
        m = memory_mod.Memory.spawn(sys)
        try:
            buf = m.allocate_managed_f32(16, timeout_secs=5.0)
        except atomr_accel.GpuRuntimeError:
            pytest.skip("no CUDA driver — can't construct a ManagedBufferF32")
        with pytest.raises(atomr_accel.GpuRuntimeError):
            m.advise_f32(buf, advice="set_preferred_location")


def test_prefetch_f32_rejects_unknown_target():
    """Bad `target` strings surface as ``GpuRuntimeError``."""
    with atomr_accel.System.open("memory-prefetch-bad") as sys:
        m = memory_mod.Memory.spawn(sys)
        try:
            buf = m.allocate_managed_f32(16, timeout_secs=5.0)
        except atomr_accel.GpuRuntimeError:
            pytest.skip("no CUDA driver — can't construct a ManagedBufferF32")
        with pytest.raises(atomr_accel.GpuRuntimeError):
            m.prefetch_f32(buf, target="not-a-target")


# ─── string-keyed enum vocabularies (documented surface) ────────────


ACCEPTED_FLAGS = ("global", "host", "attach_global", "attach_host")
ACCEPTED_PREFETCH_TARGETS = ("cpu", "device", "gpu", "host")
ACCEPTED_ADVICE = (
    "set_read_mostly",
    "unset_read_mostly",
    "set_preferred_location",
    "unset_preferred_location",
    "set_accessed_by",
    "unset_accessed_by",
)


def test_memory_string_enum_vocabularies_documented():
    """Smoke-check: documented string vocabularies are non-empty so
    future authors know what the helpers accept."""
    for vocab in (ACCEPTED_FLAGS, ACCEPTED_PREFETCH_TARGETS, ACCEPTED_ADVICE):
        assert len(vocab) > 0


# ─── IPC (cuda-ipc-feature-gated) ──────────────────────────────────


@pytest.mark.skipif(
    memory_mod.IpcMemHandle is None,
    reason="cuda-ipc feature not compiled in",
)
def test_ipc_handle_round_trip():
    """``IpcMemHandle`` round-trips through 64 bytes. PyO3 marshals a
    ``[u8; 64]`` as either ``bytes`` (when wrapped in PyBytes) or a
    ``list[int]`` of length 64; we accept both."""
    h = memory_mod.IpcMemHandle.from_bytes(bytes(range(64)))
    payload = h.bytes()
    assert hasattr(payload, "__len__")
    assert len(payload) == 64
    assert list(payload) == list(range(64))


@pytest.mark.skipif(
    memory_mod.IpcMemHandle is None,
    reason="cuda-ipc feature not compiled in",
)
def test_ipc_open_returns_typed_error_on_no_driver():
    """``ipc_open_mem_handle`` against a zero handle on a no-driver
    host surfaces ``GpuRuntimeError`` (Unrecoverable / LibraryError)."""
    h = memory_mod.IpcMemHandle.from_bytes(bytes(64))
    with pytest.raises(atomr_accel.GpuRuntimeError):
        memory_mod.ipc_open_mem_handle(h, 0)


def test_ipc_symbols_consistent():
    """Either all four IPC symbols are present (cuda-ipc on) or all
    four are ``None`` (cuda-ipc off)."""
    syms = (
        memory_mod.IpcMemHandle,
        memory_mod.IpcOpenedMem,
        memory_mod.ipc_get_mem_handle,
        memory_mod.ipc_open_mem_handle,
    )
    presence = {s is None for s in syms}
    assert len(presence) == 1, f"mixed cuda-ipc symbol availability: {syms}"
