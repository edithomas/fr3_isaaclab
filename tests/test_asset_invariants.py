"""Static tests for FR3 asset invariants (no Isaac Sim needed).

These tests are the regression net for the swallowed-error behaviour of the old
``check_fr3_asset.py``: the old script returned exit code 0 even when the asset
was missing, invariants failed, or an exception was raised. The new script must
exit nonzero whenever :func:`classify_asset` reports a violation, and these
tests pin that contract against sampled USD data.
"""
from __future__ import annotations

import pytest

from fr3_isaaclab_project.diagnostics import (
    EXPECTED_ARM_JOINTS,
    EXPECTED_EE_BODY,
    MIN_RIGID_BODIES,
    AssetReport,
    check_arm_joints_present,
    check_ee_body_present,
    check_nonzero_rigid_bodies,
    classify_asset,
)


# A realistic, healthy FR3 USD snapshot (joints + the fr3_hand body).
HEALTHY_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
    "fr3_finger_joint1", "fr3_finger_joint2",
]
HEALTHY_BODIES = [
    "fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3", "fr3_link4",
    "fr3_link5", "fr3_link6", "fr3_link7", "fr3_hand", "fr3_leftfinger",
    "fr3_rightfinger", "fr3_hand_tcp",
]


# ---------------------------------------------------------------------------
# Individual invariant checks
# ---------------------------------------------------------------------------
class TestCheckArmJointsPresent:
    def test_all_present(self):
        missing, extra = check_arm_joints_present(list(EXPECTED_ARM_JOINTS))
        assert missing == [] and extra == []

    def test_missing_joint_detected(self):
        joints = list(EXPECTED_ARM_JOINTS)
        joints.remove("fr3_joint4")
        missing, extra = check_arm_joints_present(joints)
        assert missing == ["fr3_joint4"]
        assert extra == []

    def test_extra_revolute_joint_detected(self):
        # A USD variant that promoted fr3_joint8 to a revolute DOF is unexpected.
        joints = list(EXPECTED_ARM_JOINTS) + ["fr3_joint8"]
        missing, extra = check_arm_joints_present(joints)
        assert missing == []
        assert extra == ["fr3_joint8"]

    def test_finger_joints_not_arm(self):
        missing, extra = check_arm_joints_present(
            list(EXPECTED_ARM_JOINTS) + ["fr3_finger_joint1", "fr3_finger_joint2"]
        )
        assert missing == [] and extra == []

    def test_empty_input_reports_all_missing(self):
        missing, extra = check_arm_joints_present([])
        assert missing == list(EXPECTED_ARM_JOINTS)


class TestCheckEeBody:
    def test_present(self):
        assert check_ee_body_present(HEALTHY_BODIES) is True

    def test_missing(self):
        assert check_ee_body_present([b for b in HEALTHY_BODIES if b != "fr3_hand"]) is False

    def test_case_sensitive(self):
        # USD prim names are case-sensitive; a capitalised variant must NOT match.
        assert check_ee_body_present(["FR3_Hand"]) is False


class TestCheckNonzeroRigidBodies:
    def test_nonzero(self):
        assert check_nonzero_rigid_bodies(HEALTHY_BODIES) is True

    def test_zero_is_failure(self):
        assert check_nonzero_rigid_bodies([]) is False


# ---------------------------------------------------------------------------
# classify_asset — the report the script exits on
# ---------------------------------------------------------------------------
class TestClassifyAsset:
    def test_healthy_report_is_ok(self):
        report = classify_asset(HEALTHY_JOINTS, HEALTHY_BODIES, ["/fr3"])
        assert report.ok is True
        assert report.violations == [] or all(
            "not an invariant failure" in v for v in report.violations
        )
        assert report.ee_body_found is True
        assert report.num_rigid_bodies == len(HEALTHY_BODIES)
        assert report.missing_arm_joints == []
        assert report.extra_arm_joints == []
        assert "fr3_hand" in report.ee_candidates

    def test_missing_ee_body_fails(self):
        bodies = [b for b in HEALTHY_BODIES if b != "fr3_hand"]
        report = classify_asset(HEALTHY_JOINTS, bodies, ["/fr3"])
        assert report.ok is False
        assert any("fr3_hand" in v for v in report.violations)
        assert report.ee_body_found is False

    def test_missing_arm_joint_fails(self):
        joints = [j for j in HEALTHY_JOINTS if j != "fr3_joint3"]
        report = classify_asset(joints, HEALTHY_BODIES, ["/fr3"])
        assert report.ok is False
        assert "fr3_joint3" in report.missing_arm_joints
        assert any("fr3_joint3" in v for v in report.violations)

    def test_zero_rigid_bodies_fails(self):
        report = classify_asset(HEALTHY_JOINTS, [], [])
        assert report.ok is False
        assert report.num_rigid_bodies == 0
        assert any("rigid" in v.lower() for v in report.violations)

    def test_extra_revolute_joint_fails(self):
        joints = list(EXPECTED_ARM_JOINTS) + ["fr3_joint8"]
        report = classify_asset(joints, HEALTHY_BODIES, ["/fr3"])
        assert report.ok is False
        assert "fr3_joint8" in report.extra_arm_joints

    def test_empty_articulation_roots_is_not_a_failure(self):
        # A missing ArticulationRootAPI traversal is diagnostic, not invariant.
        report = classify_asset(HEALTHY_JOINTS, HEALTHY_BODIES, [])
        assert report.ok is True

    def test_report_remains_mutable_and_ordered(self):
        report = AssetReport()
        assert report.ok is True
        report.add_violation("x")
        report.add_violation("y")
        assert report.ok is False
        assert report.violations == ["x", "y"]


# ---------------------------------------------------------------------------
# Regression: the OLD check_fr3_asset.py swallowed these and returned 0.
# Pin the contract that the new script MUST exit nonzero on each.
# ---------------------------------------------------------------------------
class TestOldSwallowedErrorBehaviour:
    """Each of these failure modes must produce ``report.ok is False``.

    The pre-fix script printed ``[ERROR] ...`` and ``return``-ed, leaving the
    process exit code at 0 (the ``finally: simulation_app.close()`` swallowed
    the real outcome). These tests assert the invariant layer now flags each.
    """

    @pytest.mark.parametrize(
        "joints,bodies,root,label",
        [
            # Asset-resolution failure path: nothing resolved -> empty everything.
            ([], [], [], "asset_unresolved"),
            # fr3_hand body missing.
            (HEALTHY_JOINTS, [b for b in HEALTHY_BODIES if b != "fr3_hand"], ["/fr3"], "ee_missing"),
            # arm joint missing.
            (
                [j for j in HEALTHY_JOINTS if j != "fr3_joint6"],
                HEALTHY_BODIES,
                ["/fr3"],
                "arm_joint_missing",
            ),
        ],
    )
    def test_failure_mode_is_nonzero(self, joints, bodies, root, label):
        report = classify_asset(joints, bodies, root)
        assert report.ok is False, f"{label}: expected nonzero (ok=False) report"
