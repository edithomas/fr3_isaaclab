"""Observation reconstruction helpers — pure Python.

Encodes the relationship between the *policy observation* and the *absolute
robot state* for the FR3 reach task under Isaac Lab v2.1.0, so the headless
observation-triangulation diagnostic can verify the reconstruction locally.

From ``isaaclab_tasks/.../reach/reach_env_cfg.py`` (v2.1.0) the policy group is::

    joint_pos      = mdp.joint_pos_rel      # joint_pos - default_joint_pos  (noisy)
    joint_vel      = mdp.joint_vel_rel      # joint_vel - default_joint_vel  (noisy)
    pose_command   = mdp.generated_commands("ee_pose")
    actions        = mdp.last_action

``mdp.joint_pos_rel`` returns ``asset.data.joint_pos - asset.data.default_joint_pos``
(see ``isaaclab/envs/mdp/observations.py``). So a recorded policy observation of
the joint-position slice **must** reconstruct as ``absolute - default`` within
the additive uniform noise band (``n_min=-0.01, n_max=0.01``).

This is the load-bearing fact for the diagnostic: prior "runtime default
mismatch" claims (F3/H7/P0-1) were disproved because the recorded observations
were *relative* and *noisy* — the triangulation must compare against the
relative reconstruction, never against an absolute baseline.
"""

from __future__ import annotations

from dataclasses import dataclass


# Additive uniform noise configured on joint_pos/joint_vel observations in the
# reach env cfg (Unoise(n_min=-0.01, n_max=0.01)).
OBS_NOISE_HALF_WIDTH: float = 0.01


@dataclass
class ObsReconstructionResult:
    """Result of verifying ``obs_joint_rel == joint_pos - default_joint_pos``."""

    max_abs_err: float
    n_within_noise: int
    n_total: int
    ok: bool
    sample_errors: list[float]


def reconstruct_relative_joint_pos(
    absolute_joint_pos: list[float], default_joint_pos: list[float]
) -> list[float]:
    """Reconstruct the policy ``joint_pos`` slice from absolute state.

    ``mdp.joint_pos_rel`` returns ``asset.data.joint_pos - asset.data.default_joint_pos``.
    """
    if len(absolute_joint_pos) != len(default_joint_pos):
        raise ValueError(
            f"length mismatch: absolute={len(absolute_joint_pos)} "
            f"default={len(default_joint_pos)}"
        )
    return [a - d for a, d in zip(absolute_joint_pos, default_joint_pos)]


def verify_observation_reconstruction(
    obs_joint_pos_rel: list[float],
    absolute_joint_pos: list[float],
    default_joint_pos: list[float],
    *,
    noise_half_width: float = OBS_NOISE_HALF_WIDTH,
    max_sample_errors: int = 10,
) -> ObsReconstructionResult:
    """Verify the policy ``joint_pos`` slice reconstructs as relative state.

    Args:
        obs_joint_pos_rel: The ``joint_pos`` slice of ``obs["policy"]`` for one env.
        absolute_joint_pos: ``robot.data.joint_pos`` for the same env/step.
        default_joint_pos: ``robot.data.default_joint_pos`` for the same env.
        noise_half_width: Max |obs - (abs - default)| allowed before it counts as
            a genuine mismatch. Defaults to the 0.01 Unoise band.

    A failure means the observation pipeline diverged from the documented
    relative convention; it must **not** be reported as a "runtime default
    mismatch" because the observation is *defined* relative to defaults.
    """
    expected = reconstruct_relative_joint_pos(absolute_joint_pos, default_joint_pos)
    if len(obs_joint_pos_rel) != len(expected):
        raise ValueError(
            f"obs slice length {len(obs_joint_pos_rel)} != expected {len(expected)}"
        )
    errs = [abs(o - e) for o, e in zip(obs_joint_pos_rel, expected)]
    within = sum(1 for e in errs if e <= noise_half_width)
    ok = within == len(errs)
    return ObsReconstructionResult(
        max_abs_err=max(errs) if errs else 0.0,
        n_within_noise=within,
        n_total=len(errs),
        ok=ok,
        sample_errors=errs[:max_sample_errors],
    )
