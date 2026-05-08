"""``atomr_accel.graph`` — CUDA-graph capture / replay handles.

Phase 1.5 surface (issue #1):

* :class:`GraphCapture` — wraps an ``ActorRef<GraphMsg>``. Spawned in
  *mock mode* via :py:meth:`GraphCapture.spawn`; ``record(script)``
  and ``launch(handle)`` round-trip through the actor. Mock-mode
  replies surface as :class:`atomr_accel.Unrecoverable`. Real-mode
  capture (against a live ``CudaStream``) lands once
  ``Device.graph()`` is wired in Phase 5.
* :class:`GraphScript` — host-side builder accumulating
  ``Box<dyn GraphOp>`` ops (``add_memcpy`` / ``add_sgemm``). The
  script is consumed by :py:meth:`GraphCapture.record`.
* :class:`GraphHandle` — opaque captured + instantiated graph.
  Returned by ``record``; pass back to ``launch``. Supports
  :py:meth:`GraphHandle.export_dot` for Graphviz DOT round-trips
  (works on synthetic handles too — useful for tooling tests).

Example
-------

>>> import atomr_accel
>>> from atomr_accel import graph as graph_mod
>>>
>>> with atomr_accel.System.open("graph-demo") as sys:
...     cap = graph_mod.GraphCapture.spawn(sys, name="demo")
...     script = graph_mod.GraphScript()
...     # In mock mode the actor rejects Record:
...     try:
...         cap.record(script, timeout_secs=2.0)
...     except atomr_accel.Unrecoverable:
...         pass
...
"""

try:
    from ._native import GraphCapture, GraphHandle, GraphScript
except ImportError:  # pragma: no cover - graph module always-on at build
    GraphCapture = None  # type: ignore[assignment]
    GraphHandle = None  # type: ignore[assignment]
    GraphScript = None  # type: ignore[assignment]

__all__ = ["GraphCapture", "GraphHandle", "GraphScript"]
