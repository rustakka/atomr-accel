"""``atomr_accel.memory`` — CUDA memory ops (Phase 1.5).

Surfaces the upstream ``atomr-accel-cuda::memory`` primitives:

- ``Memory`` — handle wrapping ``ManagedAllocatorActor``. Spawn under
  a ``System`` and use ``allocate_managed_f32`` / ``prefetch_f32`` /
  ``advise_f32`` / ``stats``.
- ``ManagedBufferF32`` — opaque token returned by
  ``Memory.allocate_managed_f32``. Pass to ``prefetch_f32`` and
  ``advise_f32``.
- ``IpcMemHandle`` / ``IpcOpenedMem`` and the module-level
  ``ipc_get_mem_handle`` / ``ipc_open_mem_handle`` functions —
  available only when the wheel was built with
  ``--features cuda-ipc``.

Names that aren't compiled in resolve to ``None`` so callers can probe
with ``if memory.Memory is None: skip``.
"""

from __future__ import annotations


def _try_import(name):
    try:
        from . import _native
        return getattr(_native, name)
    except (ImportError, AttributeError):
        return None


# ─── Always-present surface ────────────────────────────────────────
Memory = _try_import("Memory")
ManagedBufferF32 = _try_import("ManagedBufferF32")

# ─── cuda-ipc-feature-gated surface ────────────────────────────────
IpcMemHandle = _try_import("IpcMemHandle")
IpcOpenedMem = _try_import("IpcOpenedMem")
ipc_get_mem_handle = _try_import("ipc_get_mem_handle")
ipc_open_mem_handle = _try_import("ipc_open_mem_handle")


__all__ = [
    "Memory",
    "ManagedBufferF32",
    "IpcMemHandle",
    "IpcOpenedMem",
    "ipc_get_mem_handle",
    "ipc_open_mem_handle",
]
