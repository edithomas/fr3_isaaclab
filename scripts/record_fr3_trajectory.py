#!/usr/bin/env python
"""
Record FR3 trajectory data and generate state-space visualizations.

Runs a short FR3-Reach-v0 episode in headless mode, records:
  - Joint positions over time
  - End-effector pose trajectory
  - Command vs actual pose tracking
  - Reward components

Saves trajectory data as JSON + matplotlib PNG plots.
Works WITHOUT Vulkan/GPU rendering — uses state data only.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Trigger registration
import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import FR3ReachEnvCfg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_ROOT = "outputs/record"
NUM_STEPS = 50  # ~1.67s of simulation at 0.033s step


def _close_with_timeout(sim_app, timeout_sec: int = 10) -> None:
    def _force_exit():
        print(f"\n[WARN] simulation_app.close() hung for {timeout_sec}s, force-exiting.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    timer = threading.Timer(timeout_sec, _force_exit)
    timer.daemon = True
    timer.start()
    try:
        sim_app.close()
    finally:
        timer.cancel()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, f"record_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Create environment with explicit config
    env_cfg = FR3ReachEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("FR3-Reach-v0", cfg=env_cfg)
    print(f"Env created: FR3-Reach-v0 (num_envs=1)")

    obs, _ = env.reset()
    print(f"Env reset. Observation shape: {obs['policy'].shape}")

    # Storage
    records: list[dict] = []

    print(f"\nRunning {NUM_STEPS} steps...")
    t0 = time.time()
    for step in range(NUM_STEPS):
        # Use zero action (stay still) — records initial pose + gravity settling
        action = torch.zeros(1, 7, device="cuda:0")
        obs, reward, terminated, truncated, info = env.step(action)

        # Extract trajectory data from obs
        joint_pos = obs["policy"][0, :9].cpu().numpy().tolist()
        joint_vel = obs["policy"][0, 9:18].cpu().numpy().tolist()
        pose_cmd = obs["policy"][0, 18:25].cpu().numpy().tolist()
        prev_action = obs["policy"][0, 25:32].cpu().numpy().tolist()

        record = {
            "step": step,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "pose_command": pose_cmd,
            "prev_action": prev_action,
            "reward": float(reward.mean().item()),
        }

        # Extract reward components if available
        if "log" in info:
            record["reward_components"] = {
                k: float(v.mean().item()) if isinstance(v, torch.Tensor) else float(v)
                for k, v in info["log"].items()
            }

        records.append(record)

        if step % 10 == 0:
            print(f"  step {step}/{NUM_STEPS}")

    elapsed = time.time() - t0
    print(f"\nCompleted {NUM_STEPS} steps in {elapsed:.1f}s ({NUM_STEPS/elapsed:.0f} steps/s)")

    env.close()
    print("Env closed.")

    # ---- Save JSON trajectory ----
    json_path = os.path.join(out_dir, "trajectory.json")
    with open(json_path, "w") as f:
        json.dump({"num_steps": NUM_STEPS, "step_time_sec": elapsed / NUM_STEPS, "records": records}, f, indent=2)
    print(f"Saved: {json_path}")

    # ---- Generate plots ----
    plot_paths = _generate_plots(records, out_dir)
    for p in plot_paths:
        print(f"Saved: {p}")

    # ---- Generate summary report ----
    report_path = os.path.join(out_dir, "record_report.md")
    _write_report(out_dir, plot_paths + [json_path], NUM_STEPS, elapsed)
    print(f"Saved: {report_path}")

    return out_dir


def _generate_plots(records: list[dict], out_dir: str) -> list[str]:
    steps = [r["step"] for r in records]
    paths = []

    # --- Plot 1: Joint positions ---
    fig, ax = plt.subplots(figsize=(12, 6))
    joint_data = np.array([r["joint_pos"] for r in records])
    for j in range(min(joint_data.shape[1], 7)):  # first 7 joints (arm)
        ax.plot(steps, joint_data[:, j], label=f"joint_{j+1}", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title("FR3 Joint Position Trajectory")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "joint_positions.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- Plot 2: EE pose command vs action ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmd_data = np.array([r["pose_command"] for r in records])
    act_data = np.array([r["prev_action"] for r in records])
    # Position part (first 3 of command)
    for i, label in enumerate(["X", "Y", "Z"]):
        ax1.plot(steps, cmd_data[:, i], label=f"cmd_{label}", alpha=0.7)
    ax1.set_title("EE Pose Command (Position)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Position")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    # Action
    for j in range(min(act_data.shape[1], 7)):
        ax2.plot(steps, act_data[:, j], alpha=0.5, linewidth=0.5)
    ax2.set_title("Previous Action (7-DOF)")
    ax2.set_xlabel("Step"); ax2.set_ylabel("Action")
    ax2.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "ee_pose_and_actions.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- Plot 3: Reward ---
    fig, ax = plt.subplots(figsize=(12, 4))
    rewards = [r["reward"] for r in records]
    ax.plot(steps, rewards, "b-", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Step Reward")
    ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "rewards.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- Plot 4: Joint velocities ---
    fig, ax = plt.subplots(figsize=(12, 6))
    vel_data = np.array([r["joint_vel"] for r in records])
    for j in range(min(vel_data.shape[1], 7)):
        ax.plot(steps, vel_data[:, j], label=f"joint_{j+1}", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint velocity (rad/s)")
    ax.set_title("FR3 Joint Velocity")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    p = os.path.join(out_dir, "joint_velocities.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def _write_report(out_dir: str, files: list[str], num_steps: int, elapsed: float) -> None:
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
        "- This recording was generated in **headless mode** (no Vulkan/GPU rendering).",
        "- Data is state-space only: joint positions, velocities, pose commands, actions, rewards.",
        "- No camera images — Vulkan `ERROR_INCOMPATIBLE_DRIVER` prevents GPU rendering in this container.",
        "- For demo-quality videos with camera views, run on a system with full Vulkan/GPU support.",
    ])
    with open(os.path.join(out_dir, "record_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    try:
        out = main()
        print(f"\nDone. Output: {out}")
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _close_with_timeout(simulation_app, timeout_sec=10)
