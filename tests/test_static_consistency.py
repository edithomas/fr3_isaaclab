"""T-10 static consistency: the asset-invariant constants must match the live
config in ``assets/fr3.py`` (no Isaac Sim needed; AST/string parse only).

This is the zero-dependency guard against "script output vs config drift" the
test-review report recommended (§5.4): the EE body, arm-joint regex, and init
joint-pos keys declared in code must all agree.
"""
from __future__ import annotations

import ast
import os

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FR3_CFG_PATH = os.path.join(
    PROJECT_ROOT, "source", "fr3_isaaclab_project", "fr3_isaaclab_project", "assets", "fr3.py"
)

from fr3_isaaclab_project.diagnostics import EXPECTED_ARM_JOINTS, EXPECTED_EE_BODY


def _module_constants(path: str) -> dict:
    """Parse top-level NAME = literal assignments from a module (no imports)."""
    tree = ast.parse(open(path).read(), filename=path)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    try:
                        consts[tgt.id] = ast.literal_eval(node.value)
                    except Exception:
                        consts[tgt.id] = None
    return consts


class TestStaticConsistency:
    def test_ee_body_constant_matches_invariants(self):
        consts = _module_constants(FR3_CFG_PATH)
        assert consts.get("FR3_EE_BODY") == EXPECTED_EE_BODY

    def test_arm_joint_regex_matches_expected_prefix(self):
        consts = _module_constants(FR3_CFG_PATH)
        regex = consts.get("FR3_ARM_JOINTS_REGEX", "")
        # regex must match every expected arm joint name
        import re
        pat = re.compile(regex)
        for j in EXPECTED_ARM_JOINTS:
            assert pat.fullmatch(j) is not None, f"{regex} does not match {j}"

    def test_init_joint_pos_keys_are_expected_arm_joints(self):
        consts = _module_constants(FR3_CFG_PATH)
        keys = set((consts.get("FR3_INIT_JOINT_POS") or {}).keys())
        assert set(EXPECTED_ARM_JOINTS) == keys, (
            f"FR3_INIT_JOINT_POS keys {sorted(keys)} != expected {sorted(EXPECTED_ARM_JOINTS)}"
        )

    def test_shoulder_and_forearm_patterns_partition_arm_joints(self):
        import re
        consts = _module_constants(FR3_CFG_PATH)
        shoulder = re.compile(consts["FR3_SHOULDER_PATTERN"])
        forearm = re.compile(consts["FR3_FOREARM_PATTERN"])
        for j in EXPECTED_ARM_JOINTS[:4]:
            assert shoulder.fullmatch(j), f"{j} should match shoulder pattern"
        for j in EXPECTED_ARM_JOINTS[4:]:
            assert forearm.fullmatch(j), f"{j} should match forearm pattern"
        # no overlap
        for j in EXPECTED_ARM_JOINTS:
            assert not (shoulder.fullmatch(j) and forearm.fullmatch(j))
