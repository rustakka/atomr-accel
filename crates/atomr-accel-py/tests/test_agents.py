"""``atomr_accel.agents`` surface tests.

Phase 2 shipped ``SharedGpuStateCoordinator``, ``EmbeddingCache``, and
``CpuVectorIndex`` with working ``spawn`` constructors plus
representative methods, and ``RagPipeline`` / ``LangGraphGpuActor``
as structural anchors. Phase 2.5 adds ``spawn(...)`` + a
representative method on the two anchors as well — the RAG pipeline
wires through an internal deterministic embedding fn + stub LLM, and
the LangGraph actor monomorphizes over ``Vec<f32>`` state with a
linear chain of scale-by-(i+1) nodes.
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
    """``RagPipeline`` and ``LangGraphGpuActor`` expose ``__repr__``."""
    for name in STRUCTURAL_ANCHOR_NAMES:
        cls = getattr(agents, name)
        assert hasattr(cls, "__repr__"), name


def test_structural_anchors_not_constructable():
    """Anchors don't expose ``__new__`` — spawn via ``cls.spawn(...)``."""
    for name in STRUCTURAL_ANCHOR_NAMES:
        cls = getattr(agents, name)
        with pytest.raises(TypeError):
            cls()


# ─── Phase 2.5: spawn + method dispatch ────────────────────────────


def test_rag_pipeline_query():
    """`query(text, k)` runs embed + vector search + LLM stub."""
    with atomr_accel.System.open("rag-py") as sys:
        cache = agents.EmbeddingCache.spawn(
            sys, capacity_entries=8, embedding_dim=3
        )
        idx = agents.CpuVectorIndex.spawn(sys, dim=3)
        # Seed the index so vector search has something to return.
        idx.insert(id=1, embedding=[1.0, 0.0, 0.0])
        idx.insert(id=2, embedding=[0.0, 1.0, 0.0])

        rag = agents.RagPipeline.spawn(
            sys, cache, idx, embedding_dim=3, timeout_secs=5.0
        )
        assert "RagPipeline" in repr(rag)
        answer, sources, was_cached = rag.query(
            "alpha", k=2, timeout_secs=10.0
        )
        assert "alpha" in answer
        assert len(sources) >= 1
        # First call: cache miss.
        assert was_cached is False

        # Second call same text: cache hit.
        _, _, was_cached2 = rag.query("alpha", k=2, timeout_secs=10.0)
        assert was_cached2 is True


def test_langgraph_gpu_actor_step():
    """`step(state)` runs the linear node chain. With 3 nodes
    (factors 1, 2, 3) the state scales by 1*2*3 = 6."""
    with atomr_accel.System.open("langgraph-py") as sys:
        graph = agents.LangGraphGpuActor.spawn(sys, n_nodes=3)
        assert "LangGraphGpuActor" in repr(graph)
        out = graph.step(state=[1.0, 2.0], timeout_secs=5.0)
        assert len(out) == 2
        assert abs(out[0] - 6.0) < 1e-3
        assert abs(out[1] - 12.0) < 1e-3
