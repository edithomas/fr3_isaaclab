"""Static tests for the observation reconstruction convention (no Isaac Sim).

Pins the fact that the FR3 reach policy ``joint_pos`` slice is *relative* to the
default joint positions and is the documented reconciliation target — never an
absolute baseline. This is the regression net for the prior F3/H7/P0-1
"runtime default mismatch" claim, which was disproved because recorded
observations were relative/noisy.
"""
from __future__ import annotations

import pytest

from fr3_isaaclab_project.diagnostics.observation import (
    OBS_NOISE_HALF_WIDTH,
    reconstruct_relative_joint_pos,
    verify_observation_reconstruction,
)


DEFAULTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
ABSOLUTE = [0.05, -0.5, 0.01, -2.7, 0.0, 3.0, 0.74]


class TestReconstructRelativeJointPos:
    def test_reconstruction(self):
        rel = reconstruct_relative_joint_pos(ABSOLUTE, DEFAULTS)
        assert rel == [a - d for a, d in zip(ABSOLUTE, DEFAULTS)]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            reconstruct_relative_joint_pos([1, 2, 3], [1, 2])

    def test_zero_drift_when_at_defaults(self):
        rel = reconstruct_relative_joint_pos(DEFAULTS, DEFAULTS)
        assert rel == [0.0] * len(DEFAULTS)


class TestVerifyObservationReconstruction:
    def test_clean_reconstruction_ok(self):
        rel = reconstruct_relative_joint_pos(ABSOLUTE, DEFAULTS)
        r = verify_observation_reconstruction(rel, ABSOLUTE, DEFAULTS)
        assert r.ok is True
        assert r.n_within_noise == r.n_total
        assert r.max_abs_err < 1e-12

    def test_within_noise_band_ok(self):
        rel = reconstruct_relative_joint_pos(ABSOLUTE, DEFAULTS)
        # add noise within +/-0.01 (the configured Unoise half-width)
        noisy = [v + (0.005 if i % 2 == 0 else -0.005) for i, v in enumerate(rel)]
        r = verify_observation_reconstruction(noisy, ABSOLUTE, DEFAULTS)
        assert r.ok is True
        assert r.max_abs_err <= OBS_NOISE_HALF_WIDTH

    def test_outside_noise_band_fails(self):
        rel = reconstruct_relative_joint_pos(ABSOLUTE, DEFAULTS)
        bad = [v + 0.5 for v in rel]  # 0.5 rad >> 0.01 noise band
        r = verify_observation_reconstruction(bad, ABSOLUTE, DEFAULTS)
        assert r.ok is False
        assert r.n_within_noise < r.n_total
        assert r.max_abs_err > OBS_NOISE_HALF_WIDTH

    def test_must_not_be_reported_as_absolute_mismatch(self):
        """If the obs slice were mis-compared to absolute joint_pos, the error
        would be huge (~defaults). The reconstruction must use (abs - default)."""
        rel = reconstruct_relative_joint_pos(ABSOLUTE, DEFAULTS)
        # Wrong target: compare obs directly to absolute (the disproved claim).
        wrong_errs = [abs(o - a) for o, a in zip(rel, ABSOLUTE)]
        assert max(wrong_errs) > OBS_NOISE_HALF_WIDTH  # would falsely "fail"
        # Correct target: relative reconstruction.
        r = verify_observation_reconstruction(rel, ABSOLUTE, DEFAULTS)
        assert r.ok is True

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            verify_observation_reconstruction([1, 2], [1, 2, 3], [1, 2, 3])
