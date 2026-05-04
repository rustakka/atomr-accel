"""pytest fixtures shared across the atomr-accel Python tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_per_test():
    """Each test opens + closes its own `System`. The shared tokio
    runtime persists across tests (it's process-wide), but actor
    systems are scoped to the test. No-op fixture for documentation."""
    yield
