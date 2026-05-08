"""``atomr_accel.system`` — `System` lifecycle.

Re-export of the native ``System`` class for users who prefer per-domain
imports (mirrors upstream atomr's ``atomr.system``).
"""

from ._native import System

__all__ = ["System"]
