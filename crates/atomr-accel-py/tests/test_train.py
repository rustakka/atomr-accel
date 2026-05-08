"""``atomr_accel.train`` surface tests.

Phase 2 ships ``AsyncParameterServer`` with a working ``spawn`` +
``push_gradient`` / ``pull_weights`` surface, and structural anchors
for the three generic trainers (``DataParallelTrainer``,
``PipelineParallelTrainer``, ``TensorParallelTrainer``).
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
    """The three generic trainers are structural anchors with
    ``__repr__`` set."""
    for name in GENERIC_TRAINER_NAMES:
        cls = getattr(train, name)
        assert hasattr(cls, "__repr__"), name


def test_generic_trainers_not_constructable():
    """Generic trainers expose no ``__new__`` — direct construction
    raises ``TypeError``."""
    for name in GENERIC_TRAINER_NAMES:
        cls = getattr(train, name)
        with pytest.raises(TypeError):
            cls()
