"""``atomr_accel.train`` surface tests.

Phase 2 shipped ``AsyncParameterServer`` with a working ``spawn`` +
``push_gradient`` / ``pull_weights`` surface, and structural anchors
for the three generic trainers (``DataParallelTrainer``,
``PipelineParallelTrainer``, ``TensorParallelTrainer``). Phase 2.5
adds ``spawn(...)`` + a representative method on each generic
trainer (internal echo replicas / stages / shards stand in for real
GPU compute until typed protocols land in 2.6).
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import train


GENERIC_TRAINER_NAMES = [
    "DataParallelTrainer",
    "PipelineParallelTrainer",
    "TensorParallelTrainer",
]


def test_train_module_exposes_handles():
    """All four handles are importable."""
    assert train.AsyncParameterServer is not None
    for name in GENERIC_TRAINER_NAMES:
        assert getattr(train, name) is not None, name


def test_async_parameter_server_spawn_and_push():
    """End-to-end: spawn an `AsyncParameterServer`, push a gradient,
    pull the weights back. Pure CPU; no GPU required."""
    with atomr_accel.System.open("ps-train") as sys:
        ps = train.AsyncParameterServer.spawn(
            sys,
            initial_weights=[10.0, 20.0],
            optimizer="sgd",
            lr=0.1,
            max_staleness=4,
        )
        assert "AsyncParameterServer" in repr(ps)
        v = ps.push_gradient(
            worker_id=1,
            worker_version=0,
            gradient=[1.0, 2.0],
            timeout_secs=5.0,
        )
        assert v == 1
        weights, version = ps.pull_weights(worker_id=1, timeout_secs=5.0)
        assert version == 1
        assert len(weights) == 2
        # w[0] = 10 - 0.1 * 1 = 9.9
        assert abs(weights[0] - 9.9) < 1e-5


def test_async_parameter_server_unknown_optimizer_errors():
    """Unrecognized optimizer tag surfaces a `GpuRuntimeError`."""
    with atomr_accel.System.open("ps-bad-opt") as sys:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            train.AsyncParameterServer.spawn(
                sys, initial_weights=[1.0], optimizer="not-a-real-optimizer"
            )


def test_generic_trainers_have_repr():
    """The three generic trainers expose ``__repr__``."""
    for name in GENERIC_TRAINER_NAMES:
        cls = getattr(train, name)
        assert hasattr(cls, "__repr__"), name


def test_generic_trainers_not_constructable():
    """Generic trainers expose no ``__new__`` — direct construction
    raises ``TypeError``. Spawning goes through ``cls.spawn(...)``."""
    for name in GENERIC_TRAINER_NAMES:
        cls = getattr(train, name)
        with pytest.raises(TypeError):
            cls()


# ─── Phase 2.5: spawn + step dispatch ──────────────────────────────


def test_data_parallel_trainer_step():
    """`step(batch)` aggregates loss across internal echo replicas."""
    with atomr_accel.System.open("dp-train") as sys:
        t = train.DataParallelTrainer.spawn(sys, n_replicas=2)
        assert "DataParallelTrainer" in repr(t)
        # Replica 0 sees [[1,2]] → loss 3; replica 1 sees [[3,4]] → loss 7.
        # Weighted avg = (3+7)/2 = 5.
        loss, grad_norm, step_micros = t.step(
            batch=[[1.0, 2.0], [3.0, 4.0]], timeout_secs=5.0
        )
        assert abs(loss - 5.0) < 1e-3
        assert grad_norm == 1.0
        assert step_micros >= 0


def test_pipeline_parallel_trainer_step():
    """`step(microbatch)` walks identity stages and reads the final
    stage's `(loss, grad_norm)`."""
    with atomr_accel.System.open("pp-train") as sys:
        t = train.PipelineParallelTrainer.spawn(sys, n_stages=3)
        assert "PipelineParallelTrainer" in repr(t)
        # Final stage reports loss = mean(input).
        loss, grad_norm, _step_us = t.step(
            microbatch=[1.0, 2.0, 3.0, 4.0], timeout_secs=5.0
        )
        assert abs(loss - 2.5) < 1e-3
        assert grad_norm == 1.0


def test_tensor_parallel_trainer_forward_and_backward():
    """`forward(x)` reconstructs the input via shard sum;
    `backward(grad)` returns the grad-norm reported by shards."""
    with atomr_accel.System.open("tp-train") as sys:
        t = train.TensorParallelTrainer.spawn(sys, n_shards=2)
        assert "TensorParallelTrainer" in repr(t)
        # With 2 shards splitting [1,2,3,4] by rows, partial outputs
        # are [1,2] and [3,4]; sum is [4,6] (length 2 from
        # padding-to-longest in the sum step).
        out, loss = t.forward(x=[1.0, 2.0, 3.0, 4.0], timeout_secs=5.0)
        assert len(out) >= 2
        assert loss >= 0.0
        gn = t.backward(grad=[0.1, 0.2, 0.3, 0.4], timeout_secs=5.0)
        assert gn == 1.0
