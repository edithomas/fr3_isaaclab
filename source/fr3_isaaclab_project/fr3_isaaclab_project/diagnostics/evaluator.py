"""Pure-Python evaluator helpers (no Isaac Sim imports).

Implements the offline-testable core of the finite-step evaluator contract
(``data/fr3-evaluator-api-design/report.md``):

* **fixed command table** generation — a ``(E, 3, 7)`` array of root-frame
  pose commands (pos3 + quat_wxyz4) sampled from the same distribution as the
  reach task's ``UniformPoseCommandCfg`` ranges. Decoupling the command stream
  from the policy/RNG is what makes zero/random/untrained/checkpoint comparable
  (seed-only fairness is NOT enough: torch RNG consumption points depend on the
  policy, so the resampled command sets would diverge — contract §3.2/L4).
* **fixed init-joint table** — ``(E, 9)`` from ``default*s, s~U(0.5,1.5)`` per
  joint, mirroring ``reset_joints_by_scale``.
* **success logic** — segment success (last 25% of each 120-step segment, ≥10
  consecutive steps with pos≤thr and ori≤thr) and episode success (≥2/3
  segments). Pre-registered, not an official benchmark.
* **aggregator** — mean return + bootstrap 95% CI, per-term mean rate, episode
  length, final/avg errors, success rates.

All functions are pure NumPy/stdlib so they unit-test locally without a GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Reach task command distribution (reach_env_cfg.py, UniformPoseCommandCfg.Ranges)
# ---------------------------------------------------------------------------
POS_X = (0.35, 0.65)
POS_Y = (-0.2, 0.2)
POS_Z = (0.15, 0.5)
ROLL = (0.0, 0.0)
PITCH = (math.pi, math.pi)   # FR3 override
YAW = (-3.14, 3.14)

SEGMENTS_PER_EPISODE = 3
STEPS_PER_SEGMENT = 120
STEPS_PER_EPISODE = SEGMENTS_PER_EPISODE * STEPS_PER_SEGMENT  # 360
STEP_DT = 1.0 / 30.0

# FR3 articulation default joint positions (7 arm + 2 finger = 9 DOF).
# Mirror of assets/fr3.py FR3_INIT_JOINT_POS + zero finger joints.
FR3_DEFAULT_JOINT_POS_9 = np.array(
    [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.0, 0.0], dtype=np.float64
)

# ---------------------------------------------------------------------------
# Raw-row schema (column names + units) — contract §3.3
# ---------------------------------------------------------------------------
RAW_COLUMNS = (
    "run_id", "policy", "seed", "num_envs", "device",
    "global_step", "env_idx", "episode_idx", "step_in_episode",
    "sim_time_s", "terminated", "truncated", "done",
    "cmd_pos_xyz_0", "cmd_pos_xyz_1", "cmd_pos_xyz_2",
    "cmd_quat_wxyz_0", "cmd_quat_wxyz_1", "cmd_quat_wxyz_2", "cmd_quat_wxyz_3",
    "ee_pos_w_0", "ee_pos_w_1", "ee_pos_w_2",
    "ee_quat_w_0", "ee_quat_w_1", "ee_quat_w_2", "ee_quat_w_3",
    "joint_pos_abs_0", "joint_pos_abs_1", "joint_pos_abs_2", "joint_pos_abs_3",
    "joint_pos_abs_4", "joint_pos_abs_5", "joint_pos_abs_6",
    "joint_pos_abs_7", "joint_pos_abs_8",
    "pos_err_m", "ori_err_rad",
    "raw_action_0", "raw_action_1", "raw_action_2", "raw_action_3",
    "raw_action_4", "raw_action_5", "raw_action_6",
    "reward_net", "return_sofar",
    "reward_term_rate_end_effector_position_tracking",
    "reward_term_rate_end_effector_position_tracking_fine_grained",
    "reward_term_rate_end_effector_orientation_tracking",
    "reward_term_rate_action_rate",
    "reward_term_rate_joint_vel",
    "success_step",
)
SCHEMA_VERSION = 1

# Pre-registered success thresholds (contract §3.7). Defaults; overridable via
# CLI but every change must be a NEW registry entry (date/reason), never a
# silent edit of these defaults.
DEFAULT_SUCCESS_POS = 0.05       # m
DEFAULT_SUCCESS_ORI = 0.20      # rad
DEFAULT_SUCCESS_TAIL = 0.25     # last fraction of each 120-step segment
DEFAULT_SUCCESS_SUSTAIN = 10   # consecutive steps within the tail


@dataclass
class SuccessThresholds:
    pos: float = DEFAULT_SUCCESS_POS
    ori: float = DEFAULT_SUCCESS_ORI
    tail: float = DEFAULT_SUCCESS_TAIL
    sustain: int = DEFAULT_SUCCESS_SUSTAIN
    segments_required: int = 2  # of 3


# ---------------------------------------------------------------------------
# Command / init-joint table generation (numpy PCG64, decoupled from torch RNG)
# ---------------------------------------------------------------------------
def _quat_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Euler (rpy) -> quaternion (w,x,y,z), per Isaac Sim convention."""
    cr, cp, cy = np.cos(roll / 2), np.cos(pitch / 2), np.cos(yaw / 2)
    sr, sp, sy = np.sin(roll / 2), np.sin(pitch / 2), np.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.stack([w, x, y, z], axis=-1)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n == 0, 1.0, n)


def make_command_table(num_episodes: int, seed: int) -> np.ndarray:
    """Fixed command table, shape (num_episodes, 3, 7): [pos3, quat_wxyz4].

    Sampled from the reach task's UniformPoseCommand distribution, root frame.
    3 segments per episode (resampling at 4.0s intervals => steps 119/239).
    """
    rng = np.random.default_rng(seed)
    n = num_episodes * SEGMENTS_PER_EPISODE
    px = rng.uniform(POS_X[0], POS_X[1], n)
    py = rng.uniform(POS_Y[0], POS_Y[1], n)
    pz = rng.uniform(POS_Z[0], POS_Z[1], n)
    r = rng.uniform(ROLL[0], ROLL[1], n)
    p = rng.uniform(PITCH[0], PITCH[1], n)
    y = rng.uniform(YAW[0], YAW[1], n)
    q = _quat_from_rpy(r, p, y)
    pos = np.stack([px, py, pz], axis=-1)
    cmd = np.concatenate([pos, q], axis=-1)  # (n, 7)
    return cmd.reshape(num_episodes, SEGMENTS_PER_EPISODE, 7)


def make_init_table(num_episodes: int, seed: int, defaults: np.ndarray | None = None) -> np.ndarray:
    """Fixed init-joint table, shape (num_episodes, 9), default*s, s~U(0.5,1.5).

    Mirrors ``reset_joints_by_scale`` (position_range=(0.5,1.5)) so the same
    initial conditions are replayed across policies.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(defaults if defaults is not None else FR3_DEFAULT_JOINT_POS_9, dtype=np.float64)
    s = rng.uniform(0.5, 1.5, size=(num_episodes, d.shape[0]))
    return d * s


# ---------------------------------------------------------------------------
# Success logic
# ---------------------------------------------------------------------------
def segment_success(
    pos_err: np.ndarray, ori_err: np.ndarray, thr: SuccessThresholds
) -> bool:
    """True if, within the last ``tail`` fraction of the segment, there exist
    ``sustain`` consecutive steps with pos_err<=thr.pos AND ori_err<=thr.ori."""
    n = len(pos_err)
    if n == 0:
        return False
    k = max(1, int(n * thr.tail))
    tail_pos = pos_err[-k:]
    tail_ori = ori_err[-k:]
    ok = (tail_pos <= thr.pos) & (tail_ori <= thr.ori)
    run = 0
    for v in ok:
        run = run + 1 if v else 0
        if run >= thr.sustain:
            return True
    return False


def episode_success(
    pos_err: np.ndarray, ori_err: np.ndarray, thr: SuccessThresholds
) -> tuple[bool, list[bool]]:
    """Episode success = >= segments_required of 3 segments succeed."""
    segs = []
    for s in range(SEGMENTS_PER_EPISODE):
        a = s * STEPS_PER_SEGMENT
        b = a + STEPS_PER_SEGMENT
        segs.append(segment_success(pos_err[a:b], ori_err[a:b], thr))
    return sum(segs) >= thr.segments_required, segs


# ---------------------------------------------------------------------------
# Aggregator (bootstrap CI)
# ---------------------------------------------------------------------------
def _bootstrap_mean(x: np.ndarray, b: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_episodes(
    returns: np.ndarray,
    lengths: np.ndarray,
    final_pos_err: np.ndarray,
    final_ori_err: np.ndarray,
    avg_pos_err: np.ndarray,
    avg_ori_err: np.ndarray,
    success: np.ndarray,
    seg_success: np.ndarray,
    *,
    bootstrap: int = 10000,
    seed: int = 0,
) -> dict:
    """Build the aggregate schema (contract §3.4). ``seg_success`` is
    (n_episodes, 3)."""
    rng = np.random.default_rng(seed)
    n = len(returns)

    def ci(x):
        lo, hi = _bootstrap_mean(np.asarray(x, dtype=np.float64), bootstrap, rng)
        return {"mean": float(np.mean(x)) if n else float("nan"),
                "ci_lo": lo, "ci_hi": hi}

    seg_rates = {f"success_rate_segment_s{s}": float(seg_success[:, s].mean()) if n else float("nan")
                 for s in range(SEGMENTS_PER_EPISODE)}
    agg = {
        "n_episodes": int(n),
        "mean_return": ci(returns),
        "mean_episode_length": ci(lengths),
        "final_pos_err_m": ci(final_pos_err),
        "final_ori_err_rad": ci(final_ori_err),
        "avg_pos_err_m": ci(avg_pos_err),
        "avg_ori_err_rad": ci(avg_ori_err),
        "success_rate_episode": ci(success.astype(np.float64)),
        **seg_rates,
        "bootstrap_samples": bootstrap,
    }
    return agg


def aggregate_from_rows(rows: np.ndarray, thr: SuccessThresholds, **kw) -> dict:
    """Aggregate raw rows (structured array with named fields per RAW_COLUMNS)
    into the aggregate schema, recomputing success from pos_err/ori_err."""
    # group rows by (env_idx, episode_idx)
    eps = {}
    for r in rows:
        key = (int(r["env_idx"]), int(r["episode_idx"]))
        eps.setdefault(key, []).append(r)
    returns, lengths, fpos, fori, apos, aori, succ, seg_succ = [], [], [], [], [], [], [], []
    for _, rs in eps.items():
        rs = np.array(rs)
        order = np.argsort(rs["step_in_episode"])
        rs = rs[order]
        pos = rs["pos_err_m"].astype(np.float64)
        ori = rs["ori_err_rad"].astype(np.float64)
        ok, segs = episode_success(pos, ori, thr)
        returns.append(float(rs["return_sofar"][-1]))
        lengths.append(int(rs["step_in_episode"][-1]) + 1)
        fpos.append(float(pos[-1]))
        fori.append(float(ori[-1]))
        apos.append(float(pos.mean()))
        aori.append(float(ori.mean()))
        succ.append(bool(ok))
        seg_succ.append(segs)
    return aggregate_episodes(
        np.array(returns), np.array(lengths), np.array(fpos), np.array(fori),
        np.array(apos), np.array(aori), np.array(succ), np.array(seg_succ, dtype=bool),
        **kw,
    )
