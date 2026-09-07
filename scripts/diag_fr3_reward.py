#!/usr/bin/env python
"""D2 — Headless reward integration diagnostic for FR3-Reach-v0 (Isaac Lab v2.1.0).

Records genuine per-step active reward terms and proves the net-reward
integration convention of :class:`~isaaclab.managers.RewardManager`::

    compute(dt):
        value = func(env, params) * weight * dt
        reward_buf += value
        _step_reward[:, i] = value / dt          # = func * weight  (RATE form)

    get_active_iterable_terms(env_idx) -> [(name, [_step_reward_i])]
        # per-step term value in RATE form (func*weight, BEFORE dt)

Convention to prove::

    net_reward == sum(active_term_rates) * step_dt      (tolerance 1e-5)

Per-term manual recompute::

    rate_i == func(env, **params).mean() * get_term_cfg(name).weight

The weight is read LIVE via ``get_term_cfg(name).weight`` because the reach
curriculum modifies ``action_rate``/``joint_vel`` weights over training
(``reach_env_cfg.py`` CurriculumCfg); never cache constants.

NEVER label ``info["log"]`` as per-step reward decomposition: ``extras["log"]``
is the per-episode logging dict populated only inside ``_reset_idx`` on reset
steps (``Episode_Reward/<name>`` = episodic sum / ``max_episode_length_s``).
This diagnostic records whether ``info["log"]`` was present each step to make
that boundary explicit.

Exit: 0 = convention verified; 6 = unexpected; 7 = inconclusive/violated.

Usage::

    ~/IsaacLab/isaaclab.sh -p scripts/diag_fr3_reward.py --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="D2 FR3 reward integration diagnostic.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch

import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.diagnostics.reward_terms import verify_reward_convention
from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code

EXIT_OK = 0
EXIT_UNEXPECTED = 6
EXIT_INCONCLUSIVE = 7

NUM_STEPS = 10
OUT_PATH = os.path.join("outputs", "diagnostics", "reward.json")


def __new_cfg():
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import (
        FR3ReachEnvCfg,
    )

    cfg = FR3ReachEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    return cfg


def main() -> int:
    env = gym.make("FR3-Reach-v0", cfg=__new_cfg())
    try:
        device = env.unwrapped.device
        rm = env.unwrapped.reward_manager
        step_dt = env.unwrapped.step_dt
        active_terms = list(rm.active_terms)

        obs, _ = env.reset()
        records = []
        convention_ok = True
        per_term_ok = True
        for step in range(NUM_STEPS):
            action = torch.zeros(
                env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device
            )
            obs, reward, terminated, truncated, info = env.step(action)

            # GENUINE per-step active reward terms (rate form: func*weight).
            iterable = rm.get_active_iterable_terms(env_idx=0)
            term_names = [t for t, _ in iterable]
            term_rates = [float(v[0]) for _, v in iterable]
            net_reward = float(reward[0].detach().cpu().item())

            conv = verify_reward_convention(net_reward, term_rates, step_dt)

            # Per-term manual recompute with LIVE weights (curriculum-aware).
            term_recompute = []
            for name, rate in zip(term_names, term_rates):
                tc = rm.get_term_cfg(name)
                # func returns (num_envs, ...) per-env; mean over envs
                raw = float(tc.func(env.unwrapped, **tc.params)[0].mean().item())
                weight = float(tc.weight)
                expected_rate = raw * weight
                ok = abs(rate - expected_rate) <= 1e-4 * max(1.0, abs(expected_rate))
                per_term_ok = per_term_ok and ok
                term_recompute.append({
                    "name": name,
                    "rate_recorded": rate,            # func*weight (from manager)
                    "raw_func_mean": raw,             # tc.func(...).mean()
                    "weight_live": weight,            # get_term_cfg(name).weight
                    "expected_rate": expected_rate,
                    "ok": ok,
                })

            has_log = isinstance(info, dict) and "log" in info
            records.append({
                "step": step,
                "active_terms": active_terms,
                "term_names": term_names,
                "term_rates": term_rates,
                "net_reward": net_reward,
                "reconstructed": conv.reconstructed,
                "abs_err": conv.abs_err,
                "rel_err": conv.rel_err,
                "convention_ok": conv.ok,
                "per_term_recompute": term_recompute,
                "info_log_present": has_log,
                "info_log_note": (
                    "extras['log'] is per-episode logging populated only on "
                    "reset (Episode_Reward/<name> = episodic_sum / "
                    "max_episode_length_s); NOT per-step reward terms."
                ),
            })
            convention_ok = convention_ok and conv.ok

        summary = {
            "task": "D2 FR3-Reach-v0 reward integration",
            "api": "isaaclab.managers.RewardManager (Isaac Lab v2.1.0)",
            "convention": "net_reward == sum(get_active_iterable_terms rates) * step_dt",
            "step_dt": step_dt,
            "num_steps": NUM_STEPS,
            "active_terms": active_terms,
            "convention_ok_all_steps": convention_ok,
            "per_term_recompute_ok": per_term_ok,
            "verdict": "VERIFIED" if (convention_ok and per_term_ok) else "VIOLATED",
            "records": records,
        }
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({k: summary[k] for k in
                          ("convention_ok_all_steps", "per_term_recompute_ok",
                           "verdict", "step_dt", "active_terms")}, indent=2))
        print(f"Saved: {OUT_PATH}")
        return EXIT_OK if (convention_ok and per_term_ok) else EXIT_INCONCLUSIVE
    finally:
        env.close()


if __name__ == "__main__":
    code = EXIT_UNEXPECTED
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        close_with_exit_code(simulation_app, exit_code=code, timeout_sec=15)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(code)
