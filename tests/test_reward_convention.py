"""Static tests for the reward integration convention (no Isaac Sim needed).

Pins the Isaac Lab v2.1.0 ``RewardManager`` convention that the headless reward
diagnostic proves:

* ``net_reward == sum(get_active_iterable_terms rates) * step_dt``
* ``info["log"]`` is episode-level logging, NOT per-step reward terms.
"""
from __future__ import annotations

import math

import pytest

from fr3_isaaclab_project.diagnostics.reward_terms import (
    INFO_LOG_IS_PER_STEP,
    REWARD_TERM_RATE_DOC,
    reconstruct_net_reward,
    verify_reward_convention,
)


class TestReconstructNetReward:
    def test_single_term(self):
        # rate form: func*weight; net = rate*dt
        assert reconstruct_net_reward([2.0], step_dt=0.033) == pytest.approx(0.066)

    def test_multiple_terms(self):
        rates = [-0.2 * 1.0, 0.1 * 1.0, -0.1 * 0.5]
        assert reconstruct_net_reward(rates, step_dt=0.1) == pytest.approx(sum(rates) * 0.1)

    def test_empty_terms(self):
        assert reconstruct_net_reward([], step_dt=0.1) == 0.0

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError):
            reconstruct_net_reward([1.0], step_dt=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError):
            reconstruct_net_reward([1.0], step_dt=-0.1)


class TestVerifyRewardConvention:
    def test_exact_match(self):
        rates = [-0.2, 0.1, -0.0001]
        dt = 0.033
        net = sum(rates) * dt
        r = verify_reward_convention(net, rates, dt)
        assert r.ok is True
        assert r.abs_err < 1e-12

    def test_mislabelled_info_log_detected(self):
        """If someone fed episode-level ``info['log']`` values as 'per-step
        terms', the rates would already be dt-integrated and the convention
        check must FAIL (catching the regression)."""
        rates = [-0.2, 0.1]  # correct rate form
        dt = 0.033
        net = sum(rates) * dt
        # wrong: pretend info['log'] values (already episode-averaged, ~ net)
        wrong_rates = [net, net]
        r = verify_reward_convention(net, wrong_rates, dt)
        assert r.ok is False

    def test_float_tolerance(self):
        rates = [0.3333333333, -0.1111111111]
        dt = 0.0333333333
        net = sum(rates) * dt
        r = verify_reward_convention(net, rates, dt, abs_tol=1e-6, rel_tol=1e-5)
        assert r.ok is True

    def test_result_fields_populated(self):
        r = verify_reward_convention(0.1, [0.1], 0.1)
        assert r.net_reward == 0.1
        assert r.reconstructed == pytest.approx(0.01 * 1.0)  # 0.1*0.1
        assert r.abs_err >= 0.0
        assert r.rel_err >= 0.0


class TestInfoLogSemantics:
    def test_info_log_is_not_per_step(self):
        assert INFO_LOG_IS_PER_STEP is False

    def test_rate_doc_mentions_get_active_iterable_terms_and_dt(self):
        assert "get_active_iterable_terms" in REWARD_TERM_RATE_DOC
        assert "step_dt" in REWARD_TERM_RATE_DOC
