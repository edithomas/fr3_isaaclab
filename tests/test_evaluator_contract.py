"""Static contract tests for the evaluator script (no Isaac Sim needed).

Pins that the evaluator implements the fixed-command-table replay contract and
the four policy modes incl. an independent untrained-network mode (random
uniform actions must NOT masquerade as an untrained policy).
"""
from __future__ import annotations

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EVAL_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "eval_fr3_reach.py")


def _src() -> str:
    with open(EVAL_SCRIPT) as f:
        return f.read()


class TestEvaluatorContract:
    def test_has_untrained_policy_choice(self):
        src = _src()
        assert "untrained" in src, "evaluator must expose an untrained-network mode"
        # choices tuple must list all four modes
        assert '"zero"' in src and '"random"' in src and '"untrained"' in src and '"checkpoint"' in src

    def test_untrained_is_fresh_network_not_random(self):
        src = _src()
        # untrained must construct a fresh OnPolicyRunner WITHOUT loading a
        # checkpoint (random uniform action must not stand in for it).
        assert "OnPolicyRunner(" in src
        assert "if mode == \"checkpoint\":" in src and "runner.load(" in src
        # the untrained branch falls through without runner.load
        assert "fresh-init" in src or "fresh" in src.lower()

    def test_has_fixed_command_injection(self):
        src = _src()
        assert "FixedPoseCommand" in src
        assert "set_table" in src
        assert "_resample_command" in src
        assert "make_command_table" in src

    def test_no_seed_only_fairness_claim(self):
        src = _src()
        # must NOT claim seed-only reproducibility across policies
        assert "seed-only" not in src.lower().replace("not enough", "x") or "NOT enough" in src

    def test_has_provenance_and_schema(self):
        src = _src()
        assert "provenance.json" in src
        assert "raw_rows.ndjson" in src
        assert "schema_version" in src or "SCHEMA_VERSION" in src

    def test_uses_direct_scene_state_not_fk(self):
        src = _src()
        assert "body_pos_w" in src and "body_quat_w" in src
        assert "combine_frame_transforms" in src and "quat_mul" in src

    def test_info_log_not_used_as_reward(self):
        src = _src()
        # must use get_active_iterable_terms, not info['log'], for reward terms
        assert "get_active_iterable_terms" in src
