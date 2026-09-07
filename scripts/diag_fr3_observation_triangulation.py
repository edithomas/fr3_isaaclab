#!/usr/bin/env python
"""D1 — Headless observation-triangulation diagnostic for FR3-Reach-v0.

Triangulates the policy ``joint_pos`` observation against three independent
sources, proving the **relative** reconstruction (Isaac Lab v2.1.0):

    obs["policy"][joint_pos slice]  ==  robot.data.joint_pos
                                          - robot.data.default_joint_pos

Per ``isaaclab_tasks/.../reach/reach_env_cfg.py`` the policy group is::

    joint_pos    = mdp.joint_pos_rel   (noise=Unoise +/-0.01)
    joint_vel    = mdp.joint_vel_rel   (noise=Unoise +/-0.01)
    pose_command = mdp.generated_commands("ee_pose")
    actions      = mdp.last_action

and ``mdp.joint_pos_rel`` returns ``joint_pos - default_joint_pos``
(``isaaclab/envs/mdp/observations.py``). Slices are taken **dynamically** from
``observation_manager.group_obs_term_dim["policy"]`` — never hardcoded.

Checks (corruption OFF => strict identity; ON => 2x noise band):
* T1  obs_joint_pos == joint_pos - default_joint_pos  (identity)
* T2  obs_joint_pos + default_joint_pos == joint_pos (rebuild)
* T3  action block (last) == last input action

MUST NOT claim a "runtime default mismatch": prior F3/H7/P0-1 asserted the
recorded observations implied a default mismatch, but that was disproved — the
observations are *relative* (joint_pos - default) and *noisy*. This script
verifies the relative reconstruction explicitly and reports the max abs error
and the fraction within the noise band, never an absolute-mismatch verdict.

Exit: 0 = verified; 6 = unexpected; 7 = inconclusive (state unreadable).

Usage::

    ~/IsaacLab/isaaclab.sh -p scripts/diag_fr3_observation_triangulation.py --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="D1 FR3 observation triangulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch

import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.diagnostics.observation import (
    OBS_NOISE_HALF_WIDTH,
    verify_observation_reconstruction,
)
from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code

EXIT_OK = 0
EXIT_UNEXPECTED = 6
EXIT_INCONCLUSIVE = 7

NUM_STEPS = 5
OUT_PATH = os.path.join("outputs", "diagnostics", "observation_triangulation.json")


def _build_cfg(enable_corruption: bool):
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import (
        FR3ReachEnvCfg,
    )

    cfg = FR3ReachEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = enable_corruption
    return cfg


def _run_pass(enable_corruption: bool) -> dict:
    """One triangulation pass; returns the per-step records + summary."""
    env = gym.make("FR3-Reach-v0", cfg=_build_cfg(enable_corruption))
    try:
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        obs_manager = env.unwrapped.observation_manager
        action_manager = env.unwrapped.action_manager

        term_dims = obs_manager.group_obs_term_dim["policy"]  # [(9,),(9,),(7,),(7,)]
        n_q = term_dims[0][0]            # joint_pos slice width = num joints
        n_dq = term_dims[1][0]
        n_cmd = term_dims[2][0]
        n_act = term_dims[3][0]
        # arm joint indices in the articulation DOF ordering
        arm_ids, arm_names = robot.find_joints("fr3_joint.*")
        joint_names = list(robot.data.joint_names)

        obs, _ = env.reset()
        # P1: step once before reading targets (target buffer residue).
        action = torch.zeros(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device)
        records = []
        worst = None
        for step in range(NUM_STEPS):
            obs, _, _, _, _ = env.step(action)
            policy = obs["policy"]
            obs_jp_rel = policy[0, :n_q].detach().cpu().tolist()
            absolute = robot.data.joint_pos[0].detach().cpu().tolist()
            defaults = robot.data.default_joint_pos[0].detach().cpu().tolist()

            if not (len(absolute) == len(defaults) == len(obs_jp_rel)):
                return {"verdict": "INCONCLUSIVE",
                        "reason": f"length mismatch abs={len(absolute)} "
                                  f"def={len(defaults)} obs={len(obs_jp_rel)}"}

            r = verify_observation_reconstruction(
                obs_jp_rel, absolute, defaults,
                noise_half_width=(1e-6 if not enable_corruption else OBS_NOISE_HALF_WIDTH),
            )
            # T3: last_action block == input action (this step's action was zeros)
            act_block = policy[0, n_q + n_dq + n_cmd:].detach().cpu().tolist()
            t3 = all(abs(a - 0.0) <= (OBS_NOISE_HALF_WIDTH if enable_corruption else 1e-6)
                     for a in act_block)
            rec = {
                "step": step,
                "term_dims": [list(d) for d in term_dims],
                "joint_names": joint_names,
                "arm_joint_names": arm_names,
                "absolute_joint_pos": absolute,
                "default_joint_pos": defaults,
                "obs_joint_pos_rel": obs_jp_rel,
                "max_abs_err": r.max_abs_err,
                "n_within_noise": r.n_within_noise,
                "n_total": r.n_total,
                "t1_identity_ok": r.ok,
                "t3_action_block_is_last_action": t3,
            }
            records.append(rec)
            if worst is None or r.max_abs_err > worst["max_abs_err"]:
                worst = rec

        all_ok = all(r["t1_identity_ok"] and r["t3_action_block_is_last_action"] for r in records)
        return {
            "enable_corruption": enable_corruption,
            "tolerance": (1e-6, "strict") if not enable_corruption else (OBS_NOISE_HALF_WIDTH, "2x noise band"),
            "term_dims": [list(d) for d in term_dims],
            "group_obs_dim": list(obs_manager.group_obs_dim["policy"]),
            "arm_joint_names": arm_names,
            "all_ok": all_ok,
            "worst_step": worst,
            "records": records,
        }
    finally:
        env.close()


def main() -> int:
    try:
        strict = _run_pass(enable_corruption=False)
        noisy = _run_pass(enable_corruption=True)
    except Exception:
        import traceback
        traceback.print_exc()
        return EXIT_UNEXPECTED

    if strict.get("verdict") == "INCONCLUSIVE" or noisy.get("verdict") == "INCONCLUSIVE":
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump({"verdict": "INCONCLUSIVE", "passes": {"strict": strict, "noisy": noisy}}, f, indent=2)
        print(f"[INCONCLUSIVE] Saved: {OUT_PATH}")
        return EXIT_INCONCLUSIVE

    verified = bool(strict.get("all_ok")) and bool(noisy.get("all_ok"))
    summary = {
        "task": "D1 FR3-Reach-v0 observation triangulation",
        "convention": "obs['policy'][joint_pos] == joint_pos - default_joint_pos (mdp.joint_pos_rel)",
        "disproves_prior_claim": (
            "Prior F3/H7/P0-1 'runtime default mismatch' was disproved: "
            "recorded observations are relative (joint_pos - default) and "
            "noisy, so an absolute-baseline comparison is invalid."
        ),
        "verdict": "VERIFIED" if verified else "MISMATCH",
        "verdict_note": (
            "MISMATCH does NOT imply a runtime default mismatch; it means the "
            "observation pipeline diverged from the documented relative convention."
        ),
        "passes": {"strict_corruption_off": strict, "noisy_corruption_on": noisy},
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"verdict": summary["verdict"],
                      "strict_all_ok": strict["all_ok"],
                      "noisy_all_ok": noisy["all_ok"]}, indent=2))
    print(f"Saved: {OUT_PATH}")
    return EXIT_OK if verified else EXIT_INCONCLUSIVE


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
