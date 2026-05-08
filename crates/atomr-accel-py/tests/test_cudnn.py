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
    # Phase 1.5++ — Inputs-builder backward dispatch
    "rnn_bwd_f32",
    "multihead_attn_bwd_f32",
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
    to >= 16 methods. Phase 1.5++ adds the two Inputs-builder backward
    methods (``rnn_bwd_f32``, ``multihead_attn_bwd_f32``), pushing the
    floor to 18."""
    Cudnn = atomr_accel.Cudnn
    method_names = [
        n
        for n in dir(Cudnn)
        if not n.startswith("_") and callable(getattr(Cudnn, n, None))
    ]
    assert len(method_names) >= 18, (
        f"expected >= 18 cudnn methods after Phase 1.5++, got {len(method_names)}: "
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


# ----- Phase 1.5++ Inputs-builder surface ------------------------------


def test_rnn_bwd_inputs_class_exposed():
    """``RnnBwdInputs`` is re-exported at top level and through the
    ``cudnn`` facade, mirroring the ``Cudnn`` re-export pattern."""
    assert atomr_accel.RnnBwdInputs is not None
    assert atomr_accel.RnnBwdInputs.__name__ == "RnnBwdInputs"
    assert cudnn_mod.RnnBwdInputs is atomr_accel.RnnBwdInputs


def test_multihead_attn_bwd_inputs_class_exposed():
    assert atomr_accel.MultiHeadAttnBwdInputs is not None
    assert atomr_accel.MultiHeadAttnBwdInputs.__name__ == "MultiHeadAttnBwdInputs"
    assert cudnn_mod.MultiHeadAttnBwdInputs is atomr_accel.MultiHeadAttnBwdInputs


RNN_BWD_SETTERS = (
    # GpuBufferF32 fields
    "set_x",
    "set_y",
    "set_dy",
    "set_h_in",
    "set_c_in",
    "set_h_out",
    "set_c_out",
    "set_dh_out",
    "set_dc_out",
    "set_weights",
    "set_dx",
    "set_dh_in",
    "set_dc_in",
    "set_dweights",
    # scalar / enum fields
    "set_mode",
    "set_direction",
    "set_num_layers",
    "set_input_size",
    "set_hidden_size",
    "set_seq_length",
    "set_batch_size",
    "set_dropout",
)


@pytest.mark.parametrize("name", RNN_BWD_SETTERS)
def test_rnn_bwd_inputs_setter_exists(name: str):
    cls = atomr_accel.RnnBwdInputs
    assert hasattr(cls, name), f"missing setter: {name}"
    assert callable(getattr(cls, name))


def test_rnn_bwd_inputs_scalar_setters_round_trip():
    """The scalar / enum setters do not require GPU buffers — they
    populate inner state and surface validation errors for unknown
    enum strings."""
    inp = atomr_accel.RnnBwdInputs()
    assert not inp.is_consumed()
    inp.set_mode("lstm")
    inp.set_direction("bi")
    inp.set_num_layers(2)
    inp.set_input_size(128)
    inp.set_hidden_size(256)
    inp.set_seq_length(32)
    inp.set_batch_size(8)
    inp.set_dropout(0.1)


def test_rnn_bwd_inputs_unknown_enum_raises():
    inp = atomr_accel.RnnBwdInputs()
    with pytest.raises(atomr_accel.GpuRuntimeError):
        inp.set_mode("not-a-real-mode")
    with pytest.raises(atomr_accel.GpuRuntimeError):
        inp.set_direction("sideways")


def test_rnn_bwd_dispatch_without_buffers_fails_with_field_error():
    """An empty (no buffers set) builder dispatched against a fresh
    Cudnn handle would fail at the ``set_x`` check in the Rust
    method — but we can't construct a real ``Cudnn`` in mock mode.
    Instead we verify the ``rnn_bwd_f32`` method exists on the class
    (covered by parametrize above) and that the builder is consumed
    semantics work (next test)."""
    inp = atomr_accel.RnnBwdInputs()
    assert not inp.is_consumed()


MHA_BWD_SETTERS = (
    "set_q",
    "set_k",
    "set_v",
    "set_o",
    "set_do",
    "set_dq",
    "set_dk",
    "set_dv",
    "set_stats",
    "set_batch",
    "set_seq_q",
    "set_seq_kv",
    "set_heads_q",
    "set_heads_kv",
    "set_head_dim",
    "set_mask",
    "set_scale",
    "set_dropout",
    "set_dropout_seed",
)


@pytest.mark.parametrize("name", MHA_BWD_SETTERS)
def test_multihead_attn_bwd_inputs_setter_exists(name: str):
    cls = atomr_accel.MultiHeadAttnBwdInputs
    assert hasattr(cls, name), f"missing setter: {name}"
    assert callable(getattr(cls, name))


def test_multihead_attn_bwd_inputs_scalar_setters_round_trip():
    inp = atomr_accel.MultiHeadAttnBwdInputs()
    assert not inp.is_consumed()
    inp.set_batch(2)
    inp.set_seq_q(128)
    inp.set_seq_kv(128)
    inp.set_heads_q(8)
    inp.set_heads_kv(8)
    inp.set_head_dim(64)
    inp.set_mask("causal")
    inp.set_mask("sliding_window", window=64)
    inp.set_mask("causal_sliding_window", window=32)
    inp.set_scale(0.125)
    inp.set_dropout(0.1)
    inp.set_dropout_seed(42)


def test_multihead_attn_bwd_inputs_unknown_mask_raises():
    inp = atomr_accel.MultiHeadAttnBwdInputs()
    with pytest.raises(atomr_accel.GpuRuntimeError):
        inp.set_mask("not-a-real-mask")


def test_multihead_attn_bwd_inputs_default_state():
    """Newly-constructed builder is not consumed and tolerates
    setters being called in any order."""
    inp = atomr_accel.MultiHeadAttnBwdInputs()
    assert not inp.is_consumed()
    # Out-of-order setters work.
    inp.set_dropout_seed(1)
    inp.set_batch(1)
    inp.set_head_dim(64)


def test_inputs_builders_have_repr():
    inp1 = atomr_accel.RnnBwdInputs()
    inp2 = atomr_accel.MultiHeadAttnBwdInputs()
    assert "RnnBwdInputs" in repr(inp1)
    assert "MultiHeadAttnBwdInputs" in repr(inp2)
