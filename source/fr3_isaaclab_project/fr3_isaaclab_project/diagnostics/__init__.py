"""FR3 experimental diagnostics.

Pure-Python helpers in this package encode load-bearing invariants about the
FR3 USD asset and the Isaac Lab v2.1.0 manager-based MDP. They intentionally
import **no** Isaac Sim / Isaac Lab code so they can be unit-tested locally
(e.g. on macOS) without a GPU. The headless scripts under ``scripts/`` wrap
these helpers with a live ``SimulationApp``.
"""

from .sim_close import close_with_exit_code
from .invariants import (
    EXPECTED_ARM_JOINTS,
    EXPECTED_EE_BODY,
    MIN_RIGID_BODIES,
    AssetReport,
    classify_asset,
    check_arm_joints_present,
    check_ee_body_present,
    check_nonzero_rigid_bodies,
)

__all__ = [
    "EXPECTED_ARM_JOINTS",
    "EXPECTED_EE_BODY",
    "MIN_RIGID_BODIES",
    "AssetReport",
    "classify_asset",
    "check_arm_joints_present",
    "check_ee_body_present",
    "check_nonzero_rigid_bodies",
    "close_with_exit_code",
]
