"""``atomr_accel.train`` — distributed training blueprints.

Phase 2 ships:

- ``AsyncParameterServer`` — non-generic; full ``spawn`` + ``push_gradient``
  / ``pull_weights`` / ``stats`` surface. Construct with
  ``AsyncParameterServer.spawn(system, initial_weights, optimizer="sgd",
  lr=0.01)``.
- ``DataParallelTrainer``, ``PipelineParallelTrainer``,
  ``TensorParallelTrainer`` — generic over a user-supplied protocol;
  shipped as structural anchors. Phase 2.5 will widen the surface.
"""

from ._native import (
    AsyncParameterServer,
    DataParallelTrainer,
    PipelineParallelTrainer,
    TensorParallelTrainer,
)

__all__ = [
    "AsyncParameterServer",
    "DataParallelTrainer",
    "PipelineParallelTrainer",
    "TensorParallelTrainer",
]
