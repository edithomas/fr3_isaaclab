"""Pure-logic tests for the evaluator contract (no Isaac Sim needed).

Pins the fixed-command-table replay contract:
* command table shape/distribution/determinism (decoupled from torch RNG),
* init-joint table shape/distribution,
* pre-registered success logic (segment + episode),
* aggregator schema (bootstrap CI, success rate).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from fr3_isaaclab_project.diagnostics import evaluator as E


class TestCommandTable:
    def test_shape(self):
        t = E.make_command_table(7, 1234)
        assert t.shape == (7, E.SEGMENTS_PER_EPISODE, 7)

    def test_position_within_ranges(self):
        t = E.make_command_table(500, 1)
        pos = t[..., :3]
        assert pos[..., 0].min() >= E.POS_X[0] and pos[..., 0].max() <= E.POS_X[1]
        assert pos[..., 1].min() >= E.POS_Y[0] and pos[..., 1].max() <= E.POS_Y[1]
        assert pos[..., 2].min() >= E.POS_Z[0] and pos[..., 2].max() <= E.POS_Z[1]

    def test_quaternions_normalized(self):
        t = E.make_command_table(50, 1)
        q = t[..., 3:7]
        n = np.linalg.norm(q, axis=-1)
        assert np.allclose(n, 1.0, atol=1e-6)

    def test_deterministic_with_seed(self):
        a = E.make_command_table(10, 42)
        b = E.make_command_table(10, 42)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        a = E.make_command_table(10, 1)
        b = E.make_command_table(10, 2)
        assert not np.array_equal(a, b)


class TestInitTable:
    def test_shape(self):
        t = E.make_init_table(4, 1)
        assert t.shape == (4, 9)

    def test_within_scale_range(self):
        d = E.FR3_DEFAULT_JOINT_POS_9
        t = E.make_init_table(1000, 1)
        for j in range(9):
            dj = d[j]
            if dj == 0.0:
                # zero-default joints stay zero (default*s with s any)
                assert (t[:, j] == 0.0).all()
            else:
                r = t[:, j] / dj
                assert r.min() >= 0.5 - 1e-9 and r.max() <= 1.5 + 1e-9

    def test_deterministic(self):
        a = E.make_init_table(10, 7)
        b = E.make_init_table(10, 7)
        assert np.array_equal(a, b)


class TestSuccess:
    def _thr(self):
        return E.SuccessThresholds(pos=0.05, ori=0.20, sustain=10, segments_required=2)

    def test_segment_success_all_below(self):
        thr = self._thr()
        pos = np.full(E.STEPS_PER_SEGMENT, 0.01)
        ori = np.full(E.STEPS_PER_SEGMENT, 0.01)
        assert E.segment_success(pos, ori, thr) is True

    def test_segment_fail_all_above(self):
        thr = self._thr()
        pos = np.full(E.STEPS_PER_SEGMENT, 1.0)
        ori = np.full(E.STEPS_PER_SEGMENT, 1.0)
        assert E.segment_success(pos, ori, thr) is False

    def test_segment_requires_sustain_consecutive(self):
        thr = self._thr()  # sustain=10
        # last 30 steps; only 9 consecutive below -> fail
        pos = np.ones(E.STEPS_PER_SEGMENT) * 1.0
        ori = np.ones(E.STEPS_PER_SEGMENT) * 1.0
        pos[-9:] = 0.01
        ori[-9:] = 0.01
        assert E.segment_success(pos, ori, thr) is False
        # 10 consecutive -> success
        pos[-10:] = 0.01
        ori[-10:] = 0.01
        assert E.segment_success(pos, ori, thr) is True

    def test_segment_only_tail_counts(self):
        thr = self._thr()
        # early sustained success but tail above -> fail
        pos = np.ones(E.STEPS_PER_SEGMENT) * 1.0
        ori = np.ones(E.STEPS_PER_SEGMENT) * 1.0
        pos[:30] = 0.01
        ori[:30] = 0.01
        assert E.segment_success(pos, ori, thr) is False

    def test_episode_success_requires_two_segments(self):
        thr = self._thr()  # need 2/3
        good = np.full(E.STEPS_PER_SEGMENT, 0.01)
        bad = np.full(E.STEPS_PER_SEGMENT, 1.0)
        # 1 good, 2 bad -> fail
        ok, segs = E.episode_success(np.concatenate([good, bad, bad]),
                                     np.concatenate([good, bad, bad]), thr)
        assert ok is False and segs == [True, False, False]
        # 2 good -> success
        ok, segs = E.episode_success(np.concatenate([good, good, bad]),
                                     np.concatenate([good, good, bad]), thr)
        assert ok is True and segs == [True, True, False]


class TestAggregate:
    def test_returns_schema_keys(self):
        n = 40
        agg = E.aggregate_episodes(
            np.random.rand(n), np.full(n, 360), np.random.rand(n),
            np.random.rand(n), np.random.rand(n), np.random.rand(n),
            np.random.randint(0, 2, n), np.random.randint(0, 2, (n, 3)).astype(bool),
            bootstrap=500, seed=0,
        )
        for k in ("n_episodes", "mean_return", "mean_episode_length", "final_pos_err_m",
                  "success_rate_episode", "success_rate_segment_s0",
                  "success_rate_segment_s1", "success_rate_segment_s2"):
            assert k in agg
        assert agg["n_episodes"] == n
        assert 0.0 <= agg["success_rate_episode"]["mean"] <= 1.0
        # CI ordering
        assert agg["mean_return"]["ci_lo"] <= agg["mean_return"]["mean"] <= agg["mean_return"]["ci_hi"]

    def test_episode_length_constant_360(self):
        n = 5
        agg = E.aggregate_episodes(
            np.zeros(n), np.full(n, 360), np.zeros(n), np.zeros(n),
            np.zeros(n), np.zeros(n), np.zeros(n, dtype=bool),
            np.zeros((n, 3), dtype=bool), bootstrap=100, seed=0,
        )
        assert agg["mean_episode_length"]["mean"] == 360.0
