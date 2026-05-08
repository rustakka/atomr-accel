"""``atomr_accel.realtime`` surface tests.

Phase 2 ships all six realtime / simulation actors with ``spawn``
constructors plus one representative method each. The CPU reference
implementations work end-to-end without a GPU.
"""
from __future__ import annotations

import pytest

import atomr_accel
from atomr_accel import realtime


REALTIME_NAMES = [
    "ClothSimulationActor",
    "FluidSimulationActor",
    "ParticleSystemActor",
    "SpatialIndexActor",
    "GpuHashMapActor",
    "ImageFilterPipeline",
]


def test_realtime_module_exposes_handles():
    """All six handles are importable."""
    for name in REALTIME_NAMES:
        assert getattr(realtime, name) is not None, name


def test_cloth_spawn_and_step():
    with atomr_accel.System.open("cloth-py") as sys:
        cloth = realtime.ClothSimulationActor.spawn(
            sys, width=4, height=4, spacing=0.1, stiffness=0.5, iterations=2
        )
        assert "ClothSimulationActor" in repr(cloth)
        cloth.step(dt=0.016, timeout_secs=5.0)


def test_fluid_spawn_and_step():
    with atomr_accel.System.open("fluid-py") as sys:
        fluid = realtime.FluidSimulationActor.spawn(
            sys, width=8, height=8, viscosity=0.1
        )
        assert "FluidSimulationActor" in repr(fluid)
        fluid.step(dt=0.016, timeout_secs=5.0)


def test_particle_spawn_and_step():
    with atomr_accel.System.open("particle-py") as sys:
        ps = realtime.ParticleSystemActor.spawn(sys, gravity_y=-9.8)
        assert "ParticleSystemActor" in repr(ps)
        # Empty system steps cleanly.
        n = ps.step(dt=0.016, timeout_secs=5.0)
        assert n == 0


def test_spatial_index_spawn_and_query():
    with atomr_accel.System.open("spatial-py") as sys:
        idx = realtime.SpatialIndexActor.spawn(sys, cell_size=1.0)
        assert "SpatialIndexActor" in repr(idx)
        # Empty index returns an empty neighbor list.
        neighbors = idx.query_neighbors(0.0, 0.0, 0.0, timeout_secs=2.0)
        assert neighbors == []


def test_gpu_hashmap_spawn_and_insert():
    with atomr_accel.System.open("hashmap-py") as sys:
        m = realtime.GpuHashMapActor.spawn(
            sys, capacity=16, key_size_bytes=4, value_size_bytes=4
        )
        assert "GpuHashMapActor" in repr(m)
        # Two 4-byte keys + two 4-byte values.
        n = m.insert(
            keys=[1, 0, 0, 0, 2, 0, 0, 0],
            values=[10, 0, 0, 0, 20, 0, 0, 0],
        )
        assert n == 2


def test_gpu_hashmap_size_mismatch_errors():
    """Mismatched key/value sizes surface ``Unrecoverable``."""
    with atomr_accel.System.open("hashmap-py-bad") as sys:
        m = realtime.GpuHashMapActor.spawn(
            sys, capacity=4, key_size_bytes=4, value_size_bytes=4
        )
        with pytest.raises(atomr_accel.GpuRuntimeError):
            m.insert(keys=[1, 0, 0], values=[10, 0, 0, 0])


def test_image_filter_spawn_and_process():
    """Identity 3×3 kernel returns the input frame unchanged for
    interior pixels."""
    with atomr_accel.System.open("image-py") as sys:
        identity = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        f = realtime.ImageFilterPipeline.spawn(
            sys, width=4, height=4, channels=1, kernel_3x3=identity
        )
        assert "ImageFilterPipeline" in repr(f)
        frame = bytes(range(16))  # 4×4×1 = 16 bytes
        out = f.process(frame=list(frame), timeout_secs=5.0)
        # Identity convolution preserves interior pixel values.
        assert out[5] == frame[5]


def test_image_filter_bad_kernel_size_errors():
    """A length-mismatched kernel raises before the actor spawns."""
    with atomr_accel.System.open("image-py-bad") as sys:
        with pytest.raises(atomr_accel.GpuRuntimeError):
            realtime.ImageFilterPipeline.spawn(
                sys,
                width=4,
                height=4,
                channels=1,
                kernel_3x3=[0.0, 0.0, 0.0],
            )
