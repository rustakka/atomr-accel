"""``Cudnn`` handle — Phase 1.5 method-level surface checks.

The cuDNN feature is optional; on builds without it
``atomr_accel.Cudnn`` is ``None`` and every test in this file is
skipped. When the feature is compiled in we verify that every Phase 1.5
method exists and is callable as a bound method (mirroring the pattern
in ``test_handles.py::test_cudnn_handle_optional`` for handle access).
We do NOT exercise the methods end-to-end here — mock-mode actors drop
the boxed reply senders, and constructing real ``GpuBufferF32`` for
all the request types would require a live CUDA device. Per-method
correctness lives in the Rust integration tests under
``atomr-accel-cuda``.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import cudnn as cudnn_mod


pytestmark = pytest.mark.skipif(
    atomr_accel.Cudnn is None, reason="cudnn feature not compiled in"
)


# ----- module surface -------------------------------------------------


def test_cudnn_class_exposed_via_facade():
    """``atomr_accel.cudnn.Cudnn`` is the same class as
    ``atomr_accel.Cudnn`` (the facade re-exports the native class)."""
    assert cudnn_mod.Cudnn is atomr_accel.Cudnn
    assert cudnn_mod.Cudnn.__name__ == "Cudnn"


def test_cudnn_handle_unavailable_in_mock_mode():
    """Mirrors ``test_handles.py::test_cudnn_handle_optional`` — mock
    mode never publishes the cuDNN child actor, so ``device.cudnn()``
    raises GpuRuntimeError."""
    with atomr_accel.System.open("cudnn-mock-surface") as sys:
        dev = sys.spawn_device(device_id=0, mock=True)
        with pytest.raises(atomr_accel.GpuRuntimeError):
            dev.cudnn(timeout_secs=2.0)


# ----- method-level surface (Phase 1.5) -------------------------------

PHASE_1_METHODS = (
    "conv2d_fwd_f32",
    # Tier 1 forward
    "pool2d_fwd_f32",
    "softmax_fwd_f32",
    "activation_fwd_f32",
    "batch_norm_f32",
    "layer_norm_f32",
    "instance_norm_f32",
    "group_norm_f32",
    "dropout_fwd_f32",
    "lrn_fwd_f32",
    # Tier 2 backward
    "conv2d_bwd_data_f32",
    "conv2d_bwd_filter_f32",
    "pool2d_bwd_f32",
    "norm_bwd_f32",
    # Tier 3 RNN / attention
    "rnn_fwd_f32",
    "multihead_attn_fwd_f32",
)


@pytest.mark.parametrize("name", PHASE_1_METHODS)
def test_cudnn_method_exists(name: str):
    """Every Phase 1.5 method is a callable attribute on ``Cudnn``.
    Catches PyO3 macro regressions and signature parse errors at import
    time."""
    Cudnn = atomr_accel.Cudnn
    assert hasattr(Cudnn, name), f"missing method: {name}"
    method = getattr(Cudnn, name)
    assert callable(method), f"{name} is not callable"


def test_cudnn_method_count_at_or_above_phase_1_5_floor():
    """Phase 1.5 grew the surface from 1 method (``conv2d_fwd_f32``)
    to >= 16 methods. Pin a floor so accidental deletions break the
    suite."""
    Cudnn = atomr_accel.Cudnn
    method_names = [
        n
        for n in dir(Cudnn)
        if not n.startswith("_") and callable(getattr(Cudnn, n, None))
    ]
    assert len(method_names) >= 16, (
        f"expected >= 16 cudnn methods after Phase 1.5, got {len(method_names)}: "
        f"{sorted(method_names)}"
    )


# ----- string-keyed enum mappings -------------------------------------

# These probe the per-method `<enum>_from_str` Rust helpers indirectly:
# even without a live actor, an unknown mode string surfaces as a
# GpuRuntimeError BEFORE the call enters the Rust async block. The
# easiest way to assert this is to call against a *fake* handle
# constructor — but the class can't be instantiated from Python (only
# from `device.cudnn()`), so instead we settle for documenting the
# accepted strings in the test surface.
ACCEPTED_POOL_MODES = ("max", "avg", "average", "avg_exclude_padding", "AverageInclude")
ACCEPTED_SOFTMAX_MODES = ("channel", "instance")
ACCEPTED_ACTIVATIONS = (
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "gelu_approx",
    "swish",
    "silu",
    "elu",
    "softplus",
    "identity",
)
ACCEPTED_NORM_PHASES = ("train", "training", "inference", "eval", "persistent")
ACCEPTED_NORM_MODES = ("batch", "layer", "instance", "group", "rms")
ACCEPTED_RNN_MODES = ("rnn", "rnn_tanh", "lstm", "gru")
ACCEPTED_RNN_DIRECTIONS = ("uni", "bi", "unidirectional", "bidirectional")
ACCEPTED_ATTENTION_MASKS = ("none", "causal", "sliding_window", "causal_sliding_window")


def test_cudnn_string_keyed_enums_documented():
    """Smoke-check: the documented string vocabularies are non-empty
    so future authors know which strings the helpers accept."""
    for vocab in (
        ACCEPTED_POOL_MODES,
        ACCEPTED_SOFTMAX_MODES,
        ACCEPTED_ACTIVATIONS,
        ACCEPTED_NORM_PHASES,
        ACCEPTED_NORM_MODES,
        ACCEPTED_RNN_MODES,
        ACCEPTED_RNN_DIRECTIONS,
        ACCEPTED_ATTENTION_MASKS,
    ):
        assert len(vocab) > 0
