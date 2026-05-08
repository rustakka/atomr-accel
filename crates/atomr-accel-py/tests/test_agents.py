"""``atomr_accel.agents`` surface tests.

Phase 2 ships ``SharedGpuStateCoordinator``, ``EmbeddingCache``, and
``CpuVectorIndex`` with working ``spawn`` constructors plus
representative methods, and ``RagPipeline`` / ``LangGraphGpuActor``
as structural anchors.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import agents


STRUCTURAL_ANCHOR_NAMES = ["RagPipeline", "LangGraphGpuActor"]


def test_agents_module_exposes_handles():
    """All five handles are importable."""
    for name in [
        "SharedGpuStateCoordinator",
        "EmbeddingCache",
        "CpuVectorIndex",
        "RagPipeline",
        "LangGraphGpuActor",
    ]:
        assert getattr(agents, name) is not None, name


def test_shared_state_coordinator_spawn_and_acquire():
    """Acquire a write token, then release it. Pure CPU; the
    coordinator is dtype-free."""
    with atomr_accel.System.open("shared-state-py") as sys:
        coord = agents.SharedGpuStateCoordinator.spawn(sys)
        assert "SharedGpuStateCoordinator" in repr(coord)
        token = coord.acquire_write(agent_id=1, timeout_secs=5.0)
        assert isinstance(token, int)
        assert token > 0
        coord.release_write(token)


def test_embedding_cache_spawn_get_insert():
    """End-to-end cache miss → insert → hit cycle."""
    with atomr_accel.System.open("embed-py") as sys:
        cache = agents.EmbeddingCache.spawn(
            sys, capacity_entries=4, embedding_dim=3
        )
        assert "EmbeddingCache" in repr(cache)
        miss = cache.get(b"hello", timeout_secs=5.0)
        assert miss is None
        cache.insert(b"hello", [1.0, 2.0, 3.0], timeout_secs=5.0)
        hit = cache.get(b"hello", timeout_secs=5.0)
        assert hit == [1.0, 2.0, 3.0]


def test_cpu_vector_index_spawn_insert_search():
    """Linear-scan top-k cosine search on a tiny corpus."""
    with atomr_accel.System.open("vec-py") as sys:
        idx = agents.CpuVectorIndex.spawn(sys, dim=3)
        assert "CpuVectorIndex" in repr(idx)
        idx.insert(id=1, embedding=[1.0, 0.0, 0.0])
        idx.insert(id=2, embedding=[0.0, 1.0, 0.0])
        results = idx.search(query=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0] == 1


def test_cpu_vector_index_dim_mismatch_errors():
    """Inserting an embedding of the wrong dim raises a typed
    ``Unrecoverable`` error from the underlying actor."""
    with atomr_accel.System.open("vec-py-bad") as sys:
        idx = agents.CpuVectorIndex.spawn(sys, dim=3)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            idx.insert(id=1, embedding=[1.0, 0.0])


def test_structural_anchors_have_repr():
    """``RagPipeline`` and ``LangGraphGpuActor`` ship as anchors."""
    for name in STRUCTURAL_ANCHOR_NAMES:
        cls = getattr(agents, name)
        assert hasattr(cls, "__repr__"), name


def test_structural_anchors_not_constructable():
    """Anchors don't expose ``__new__``."""
    for name in STRUCTURAL_ANCHOR_NAMES:
        cls = getattr(agents, name)
        with pytest.raises(TypeError):
            cls()
