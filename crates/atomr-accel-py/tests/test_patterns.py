"""``atomr_accel.patterns`` surface tests.

Phase 2 shipped the seven canonical pattern actors as structural
anchors. Phase 2.5 adds representative methods on every one of them
(spawned with internal echo / mock helpers; opaque ``Vec<u8>``
payloads cross PyO3 cleanly until typed contracts land in 2.6).

These tests verify both the original symbol-surface contract and the
new spawn + method dispatch path.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import patterns


PATTERN_NAMES = [
    "DynamicBatchingServer",
    "InferenceCascade",
    "ModelReplicaPool",
    "FairShareScheduler",
    "HotSwapServer",
    "SpeculativeDecoder",
    "MoeRouter",
]


# ─── Original symbol-surface contract ─────────────────────────────


def test_patterns_module_exposes_handles():
    """Each handle is importable from the facade."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        assert cls is not None, name


def test_patterns_classes_are_types():
    """They're classes, not modules / placeholders."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        assert isinstance(cls, type), name


def test_pattern_handles_have_repr():
    """Every PyClass has a ``__repr__`` method (defined via PyO3)."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        assert hasattr(cls, "__repr__"), name


def test_native_module_has_pattern_classes():
    """The ``_native`` module hosts the same handle names."""
    native = atomr_accel._native  # type: ignore[attr-defined]
    for name in PATTERN_NAMES:
        assert hasattr(native, name), name


def test_pattern_handles_not_constructable_from_python():
    """Phase 2.5 still ships them without ``__new__`` — direct
    construction raises ``TypeError``. Spawning goes through
    ``cls.spawn(system, ...)``."""
    for name in PATTERN_NAMES:
        cls = getattr(patterns, name)
        with pytest.raises(TypeError):
            cls()


# ─── Phase 2.5: spawn + method dispatch ────────────────────────────


def test_dynamic_batching_server_submit():
    """`submit(payload)` echoes the input through the internal
    echo BatchFn. Bytes round-trip as ``list[int]`` from PyO3."""
    with atomr_accel.System.open("batching-py") as sys:
        bs = patterns.DynamicBatchingServer.spawn(
            sys, max_batch=2, max_wait_ms=20
        )
        assert "DynamicBatchingServer" in repr(bs)
        out = bs.submit(b"hello", timeout_secs=5.0)
        assert out == list(b"hello")


def test_inference_cascade_infer():
    """`infer(input)` returns `(response, stage_index, confidence)`."""
    with atomr_accel.System.open("cascade-py") as sys:
        c = patterns.InferenceCascade.spawn(sys)
        resp, idx, conf = c.infer(b"abc", timeout_secs=5.0)
        assert resp == list(b"abc")
        assert idx == 0
        assert conf == 1.0


def test_model_replica_pool_submit():
    """`submit(payload)` round-robins across the internal replicas."""
    with atomr_accel.System.open("replica-pool-py") as sys:
        pool = patterns.ModelReplicaPool.spawn(sys, n_replicas=3)
        for i in range(6):
            out = pool.submit(bytes([i]), timeout_secs=5.0)
            assert out == [i]


def test_fair_share_scheduler_submit():
    """`submit(tenant, payload)` echoes via the internal dispatcher."""
    with atomr_accel.System.open("fair-py") as sys:
        sched = patterns.FairShareScheduler.spawn(
            sys, tenants=[(1, 1), (2, 3)], max_in_flight=2
        )
        # Tenant 1 and 2 each get their requests serviced.
        for t in (1, 2):
            out = sched.submit(t, b"payload", timeout_secs=5.0)
            assert out == list(b"payload")


def test_hot_swap_server_serve_and_swap():
    """`serve` runs through current backend; `swap()` rotates it."""
    with atomr_accel.System.open("hot-swap-py") as sys:
        hs = patterns.HotSwapServer.spawn(sys)
        # Initial backend tags with version=0.
        out0 = hs.serve(b"x", timeout_secs=5.0)
        assert out0 == [ord("x"), 0]
        gen = hs.swap(timeout_secs=5.0)
        assert gen == 1
        out1 = hs.serve(b"x", timeout_secs=5.0)
        # The swapped-in backend uses a different version tag.
        assert out1 != out0
        assert out1[0] == ord("x")


def test_speculative_decoder_decode():
    """`decode(prompt, budget)` returns `(tokens, iterations,
    accepted_tokens)`."""
    with atomr_accel.System.open("spec-py") as sys:
        dec = patterns.SpeculativeDecoder.spawn(
            sys, k=4, max_total_tokens=16
        )
        tokens, iters, accepted = dec.decode(
            prompt=[0], budget=8, timeout_secs=5.0
        )
        assert len(tokens) <= 8
        assert iters >= 1
        assert accepted >= 1


def test_moe_router_route():
    """`route(input)` dispatches to the top-k expert(s) (last expert
    wins under the linear gate scoring)."""
    with atomr_accel.System.open("moe-py") as sys:
        r = patterns.MoeRouter.spawn(sys, n_experts=3, top_k=1)
        out = r.route([1.0, 2.0], timeout_secs=5.0)
        # Top expert (idx=2) adds bias 2.0 → [3.0, 4.0].
        assert len(out) == 2
        assert abs(out[0] - 3.0) < 1e-3
        assert abs(out[1] - 4.0) < 1e-3
