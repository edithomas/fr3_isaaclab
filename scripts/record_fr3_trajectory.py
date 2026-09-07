#!/usr/bin/env python
"""Record FR3 trajectory data (headless, state-space only).

Runs a short FR3-Reach-v0 episode in headless mode and records, per step:

* **absolute joint state** — ``robot.data.joint_pos`` / ``joint_vel``
  (absolute, in rad / rad/s)
* **relative policy observation** — the ``joint_pos``/``joint_vel`` slices of
  ``obs["policy"]`` (these are *relative* to defaults, per ``mdp.joint_pos_rel``)
* **actual EE pose** — ``robot.data.body_state_w`` for the ``fr3_hand`` body
  (world-frame pos+quat+vel, NOT reconstructed via approximate FK)
* **per-step reward terms (rate form)** — via
  ``reward_manager.get_active_iterable_terms`` (func*weight, before dt), plus
  the net reward and the reconstructed net reward (sum(rates)*dt)

Important corrections vs. the prior recorder:
* The device is derived from the environment (``env.unwrapped.device``), never
  hardcoded ``cuda:0``.
* ``info["log"]`` is NOT recorded as per-step reward decomposition — it is the
  per-episode logging dict populated only on reset steps. Per-step terms come
  from ``get_active_iterable_terms``.
* A **zero joint-position action** is NOT "zero torque / free fall": the
  ``JointPositionAction(use_default_offset=True, scale=0.5)`` sets the position
  target to the default pose and the implicit PD (stiffness=80, damping=4) holds
  it. The recorder records this fact, it does not claim free fall.

Output schema is versioned so downstream consumers can detect breakage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record FR3 trajectory data.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.assets.fr3 import FR3_EE_BODY
from fr3_isaaclab_project.diagnostics.reward_terms import reconstruct_net_reward
from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code

OUTPUT_ROOT = "outputs/record"
NUM_STEPS = 50
SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_UNEXPECTED = 6


def _new_cfg():
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import (
        FR3ReachEnvCfg,
    )

    cfg = FR3ReachEnvCfg()
    cfg.scene.num_envs = 1
    cfg.commands.ee_pose.debug_vis = False
    return cfg


def main() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"record_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    env = gym.make("FR3-Reach-v0", cfg=_new_cfg())
    try:
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        reward_manager = env.unwrapped.reward_manager
        obs_manager = env.unwrapped.observation_manager
        step_dt = env.unwrapped.step_dt

        # Dynamic obs layout (never hardcode 9/7/32).
        term_dims = obs_manager.group_obs_term_dim["policy"]
        n_q, n_dq, n_cmd, n_act = (d[0] for d in term_dims)
        joint_names = list(robot.data.joint_names)
        arm_ids, arm_names = robot.find_joints("fr3_joint.*")
        ee_idx = robot.data.body_names.index(FR3_EE_BODY)

        obs, _ = env.reset()
        print(f"Env reset. policy obs shape: {obs['policy'].shape}; "
              f"joints: {joint_names}; device: {device}")

        records = []
        t0 = datetime.now().timestamp()
        for step in range(NUM_STEPS):
            action = torch.zeros(
                env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device
            )
            obs, reward, terminated, truncated, info = env.step(action)

            policy = obs["policy"]
            # Relative policy observation slices (joint_pos_rel, joint_vel_rel).
            obs_jp_rel = policy[0, :n_q].detach().cpu().tolist()
            obs_jv_rel = policy[0, n_q : n_q + n_dq].detach().cpu().tolist()
            pose_cmd = policy[0, n_q + n_dq : n_q + n_dq + n_cmd].detach().cpu().tolist()
            last_action = policy[0, n_q + n_dq + n_cmd :].detach().cpu().tolist()

            # Absolute joint state (direct from the articulation).
            abs_jp = robot.data.joint_pos[0].detach().cpu().tolist()
            abs_jv = robot.data.joint_vel[0].detach().cpu().tolist()

            # Actual EE pose (world frame), NOT reconstructed via FK.
            ee_state = robot.data.body_state_w[0, ee_idx].detach().cpu().tolist()

            # Genuine per-step active reward terms (rate form: func*weight).
            iterable = reward_manager.get_active_iterable_terms(env_idx=0)
            term_names = [t for t, _ in iterable]
            term_rates = [float(v[0]) for _, v in iterable]
            net_reward = float(reward[0].detach().cpu().item())
            reconstructed = reconstruct_net_reward(term_rates, step_dt)

            # info["log"] is episode-level logging (only present on reset steps).
            has_log = isinstance(info, dict) and "log" in info

            records.append({
                "step": step,
                "joint_names": joint_names,
                "arm_joint_names": arm_names,
                "absolute_joint_pos": abs_jp,
                "absolute_joint_vel": abs_jv,
                "obs_joint_pos_rel": obs_jp_rel,
                "obs_joint_vel_rel": obs_jv_rel,
                "pose_command": pose_cmd,
                "last_action": last_action,
                "ee_body": FR3_EE_BODY,
                "ee_state_w": ee_state,           # [pos3, quat_wxyz4, linvel3, angvel3]
                "reward_term_names": term_names,
                "reward_term_rates": term_rates,  # func*weight (rate form, no dt)
                "reward_net": net_reward,          # sum(rates)*dt
                "reward_reconstructed": reconstructed,
                "reward_convention_abs_err": abs(net_reward - reconstructed),
                "info_log_present": has_log,
                "info_log_note": (
                    "extras['log'] is per-episode logging (only on reset steps), "
                    "NOT per-step reward terms."
                ),
            })
            if step % 10 == 0:
                print(f"  step {step}/{NUM_STEPS}")
        elapsed = datetime.now().timestamp() - t0
        print(f"\nCompleted {NUM_STEPS} steps in {elapsed:.1f}s "
              f"({NUM_STEPS/elapsed:.0f} steps/s)")
    finally:
        env.close()

    meta = {
        "schema_version": SCHEMA_VERSION,
        "task": "FR3-Reach-v0",
        "num_steps": NUM_STEPS,
        "step_dt": step_dt,
        "device": str(device),
        "ee_body": FR3_EE_BODY,
        "obs_term_dims_policy": [list(d) for d in term_dims],
        "group_obs_dim_policy": list(obs_manager.group_obs_dim["policy"]),
        "joint_names": joint_names,
        "arm_joint_names": arm_names,
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / NUM_STEPS,
        "action_note": (
            "Zero joint-position action => target = default_joint_pos "
            "(JointPositionAction use_default_offset=True, scale=0.5) held by "
            "implicit PD (stiffness=80, damping=4); NOT zero torque / free fall."
        ),
        "records": records,
    }
    json_path = os.path.join(out_dir, "trajectory.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {json_path}")

    for p in _generate_plots(records, out_dir, arm_names):
        print(f"Saved: {p}")
    _write_report(out_dir, [json_path], NUM_STEPS, elapsed)
    return out_dir


def _generate_plots(records, out_dir, arm_names):
    steps = [r["step"] for r in records]
    paths = []

    fig, ax = plt.subplots(figsize=(12, 6))
    abs_jp = np.array([r["absolute_joint_pos"] for r in records])
    for j in range(min(abs_jp.shape[1], 7)):
        ax.plot(steps, abs_jp[:, j], label=f"{arm_names[j] if j < len(arm_names) else j}", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Joint position (rad, absolute)")
    ax.set_title("FR3 Absolute Joint Position Trajectory")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "joint_positions.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(12, 6))
    ee = np.array([r["ee_state_w"][:3] for r in records])
    for i, lbl in enumerate(["X", "Y", "Z"]):
        ax.plot(steps, ee[:, i], label=f"ee_{lbl}", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("EE position (m, world)")
    ax.set_title("Actual EE Pose (body_state_w, not FK-reconstructed)")
    ax.legend(); ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "ee_pose.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(12, 4))
    nets = [r["reward_net"] for r in records]
    recon = [r["reward_reconstructed"] for r in records]
    ax.plot(steps, nets, "b-", label="net reward", alpha=0.8)
    ax.plot(steps, recon, "g--", label="reconstructed sum(rates)*dt", alpha=0.6)
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.set_title("Net reward vs reconstructed (convention check)")
    ax.legend(); ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "rewards.png")
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def _write_report(out_dir, files, num_steps, elapsed):
    lines = [
        "# FR3 Trajectory Record Report",
        "",
        f"- Steps recorded: {num_steps}",
        f"- Total time: {elapsed:.1f}s",
        f"- Avg step rate: {num_steps/elapsed:.0f} steps/s",
        "",
        "## Output Files",
        "",
    ]
    for f in files:
        lines.append(f"- `{f}`")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Headless, **compute-only** (CUDA/PhysX); no GPU *rendering* required.",
        "- Data is state-space: absolute joint state, relative policy obs, "
        "actual EE pose (world frame), and per-step reward terms (rate form).",
        "- The EE pose is read directly from `robot.data.body_state_w` — it is "
        "NOT reconstructed via the approximate FK used by the offline animator.",
        "- A zero joint-position action sets the position target to the default "
        "pose (held by implicit PD); it is NOT zero torque / free fall.",
        "- `reward_term_rates` are func*weight (rate form, before dt); "
        "`reward_net == sum(rates) * step_dt`.",
    ])
    with open(os.path.join(out_dir, "record_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    code = EXIT_UNEXPECTED
    out = None
    try:
        out = main()
        code = EXIT_OK
    except Exception:
        import traceback
        print(f"\n[FATAL] recorder failed")
        traceback.print_exc()
    finally:
        close_with_exit_code(simulation_app, exit_code=code, timeout_sec=10)
        sys.stdout.flush()
        sys.stderr.flush()
        if out:
            print(f"\nDone. Output: {out}")
        sys.exit(code)
