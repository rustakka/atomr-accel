"""``atomr_accel.patterns`` — universal GPU actor blueprints.

Phase 2 ships handle classes for the seven canonical patterns:
``DynamicBatchingServer``, ``InferenceCascade``, ``ModelReplicaPool``,
``FairShareScheduler``, ``HotSwapServer``, ``SpeculativeDecoder``,
``MoeRouter``. Each is generic over a user-supplied request /
response or expert-protocol type, so Phase 2 ships them as
**structural anchors** (``__repr__`` only) — Python-side typed
marshaling lands in the Phase 2.5 tracking issue.
"""

from ._native import (
    DynamicBatchingServer,
    InferenceCascade,
    ModelReplicaPool,
    FairShareScheduler,
    HotSwapServer,
    SpeculativeDecoder,
    MoeRouter,
)

__all__ = [
    "DynamicBatchingServer",
    "InferenceCascade",
    "ModelReplicaPool",
    "FairShareScheduler",
    "HotSwapServer",
    "SpeculativeDecoder",
    "MoeRouter",
]
