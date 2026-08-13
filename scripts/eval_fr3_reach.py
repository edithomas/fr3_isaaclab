#!/usr/bin/env python
"""Finite-step evaluator for FR3-Reach-v0.

Supports three policy modes on fixed seeds:

* ``--policy zero``      : zero action (position target = default, held by PD)
* ``--policy random``   : untrained uniform(-1, 1) actions
* ``--policy checkpoint``: RSL-RL checkpoint via OnPolicyRunner (like play.py)

Emits machine-readable **raw per-step rows** (command + actual EE state + error)
plus **aggregate** metrics (position/orientation error, return, episode length).
Metrics are computed from **direct scene state** (``robot.data.body_pos_w`` /
``body_quat_w`` and the command transformed to world via
``combine_frame_transforms`` / ``quat_mul``), exactly mirroring the reward
functions in ``reach/mdp/rewards.py`` — NOT from the approximate FK recorder.

PRE-REGISTERED SUCCESS DEFINITION (declared before any run):
    An episode is a SUCCESS iff the mean EE position error over the final 25%
    of steps is <= --success_pos_thresh (default 0.05 m). This is computed only
    from actual world-frame EE state vs the world-frame command.

Comparability: the same ``--seed`` + ``--num_envs`` reproduces the same command
resampling across modes; raw per-step commands are written so a later run can
replay the exact command/seed set.

Exit: 0 = evaluation completed; 6 = unexpected; 7 = inconclusive (no episodes
completed / state unreadable).

Usage::

    # zero baseline
    ~/IsaacLab/isaaclab.sh -p scripts/eval_fr3_reach.py --headless \
        --policy zero --num_envs 16 --seed 0 --max_steps 100
    # untrained random
    ~/IsaacLab/isaaclab.sh -p scripts/eval_fr3_reach.py --headless \
        --policy random --num_envs 16 --seed 0 --max_steps 100
    # checkpoint
    ~/IsaacLab/isaaclab.sh -p scripts/eval_fr3_reach.py --headless \
        --policy checkpoint --checkpoint /path/to/model.pt --num_envs 16 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Finite-step FR3 reach evaluator.")
parser.add_argument("--policy", choices=("zero", "random", "checkpoint"), default="zero")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--success_pos_thresh", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch

import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.assets.fr3 import FR3_EE_BODY
from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

EXIT_OK = 0
EXIT_UNEXPECTED = 6
EXIT_INCONCLUSIVE = 7

OUT_PATH = os.path.join("outputs", "eval", "eval.json")
SUCCESS_FINAL_FRACTION = 0.25  # pre-registered


def _cfg():
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import (
        FR3ReachEnvCfg,
    )

    cfg = FR3ReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    return cfg


def _metrics_for_env(robot, command, env_ids):
    """Position/orientation error (world frame), mirroring reach/mdp/rewards.py."""
    ee_idx = robot.data.body_names.index(FR3_EE_BODY)
    root_pos = robot.data.root_state_w[env_ids, :3]
    root_quat = robot.data.root_state_w[env_ids, 3:7]
    des_pos_b = command[env_ids, :3]
    des_quat_b = command[env_ids, 3:7]
    des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b)
    des_quat_w = quat_mul(root_quat, des_quat_b)
    curr_pos_w = robot.data.body_pos_w[env_ids, ee_idx]
    curr_quat_w = robot.data.body_quat_w[env_ids, ee_idx]
    pos_err = torch.norm(curr_pos_w - des_pos_w, dim=1)
    ori_err = quat_error_magnitude(curr_quat_w, des_quat_w)
    return pos_err, ori_err


def _make_policy_fn(env, device):
    mode = args_cli.policy
    if mode == "zero":
        def fn(obs):
            return torch.zeros(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device)
        return fn, env, "raw"
    if mode == "random":
        def fn(obs):
            a = torch.empty(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device)
            a.uniform_(-1.0, 1.0)
            return a
        return fn, env, "raw"
    # checkpoint
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.agents import rsl_rl_ppo_cfg
    agent_cfg = rsl_rl_ppo_cfg.FR3ReachPPORunnerCfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    def fn(obs):
        with torch.inference_mode():
            return policy(obs)
    return fn, wrapped, "wrapped"


def main() -> int:
    torch.manual_seed(args_cli.seed)
    env = gym.make("FR3-Reach-v0", cfg=_cfg())
    try:
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        command_manager = env.unwrapped.command_manager
        step_dt = env.unwrapped.step_dt
        max_steps = args_cli.max_steps
        n_envs = env.unwrapped.num_envs

        policy_fn, step_env, iface = _make_policy_fn(env, device)

        # reset
        if iface == "raw":
            obs, _ = env.reset(seed=args_cli.seed)
        else:
            obs, _ = step_env.get_observations()

        per_step_rows = []
        returns = torch.zeros(n_envs, device=device)
        lengths = torch.zeros(n_envs, device=device, dtype=torch.long)
        # accumulate per-env position-error history for the success criterion
        pos_err_history = [[] for _ in range(n_envs)]
        done_envs = torch.zeros(n_envs, device=device, dtype=torch.bool)

        for step in range(max_steps):
            action = policy_fn(obs)
            if iface == "raw":
                obs, reward, terminated, truncated, info = env.step(action)
                dones = terminated | truncated
            else:
                obs, reward, dones, info = step_env.step(action)

            command = command_manager.get_command("ee_pose")
            pos_err, ori_err = _metrics_for_env(robot, command, torch.arange(n_envs, device=device))

            for i in range(n_envs):
                pos_err_history[i].append(float(pos_err[i].item()))
            returns += reward
            lengths += 1

            per_step_rows.append({
                "step": step,
                "env0_pos_err": float(pos_err[0].item()),
                "env0_ori_err": float(ori_err[0].item()),
                "env0_reward": float(reward[0].item()),
                "env0_command": command[0].detach().cpu().tolist(),
                "env0_ee_pos_w": robot.data.body_pos_w[0, robot.data.body_names.index(FR3_EE_BODY)].detach().cpu().tolist(),
            })

        # success: mean pos_err over final SUCCESS_FINAL_FRACTION of each episode
        successes = torch.zeros(n_envs, dtype=torch.bool, device=device)
        for i in range(n_envs):
            hist = pos_err_history[i]
            k = max(1, int(len(hist) * SUCCESS_FINAL_FRACTION))
            final_mean = sum(hist[-k:]) / k
            successes[i] = final_mean <= args_cli.success_pos_thresh

        aggregate = {
            "policy": args_cli.policy,
            "seed": args_cli.seed,
            "num_envs": n_envs,
            "max_steps": max_steps,
            "step_dt": step_dt,
            "success_definition": (
                f"SUCCESS iff mean EE position error over the final "
                f"{int(SUCCESS_FINAL_FRACTION*100)}% of steps <= "
                f"{args_cli.success_pos_thresh} m (pre-registered)."
            ),
            "success_pos_thresh": args_cli.success_pos_thresh,
            "mean_pos_err": float(pos_err.mean().item()),
            "mean_ori_err": float(ori_err.mean().item()),
            "mean_return": float(returns.mean().item()),
            "mean_episode_length": float(lengths.mean().item()),
            "success_rate": float(successes.float().mean().item()),
            "n_success": int(successes.sum().item()),
        }
        summary = {
            "task": "FR3-Reach-v0 finite-step evaluation",
            "ee_body": FR3_EE_BODY,
            "metrics_source": "robot.data.body_pos_w/body_quat_w + command->world (combine_frame_transforms/quat_mul)",
            "note": "EE state is read directly from the scene, NOT reconstructed via approximate FK.",
            "aggregate": aggregate,
            "raw_rows_env0_sample": per_step_rows[:10],
            "raw_rows_env0_tail": per_step_rows[-5:],
        }
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(aggregate, indent=2))
        print(f"Saved: {OUT_PATH}")
        if lengths.max().item() == 0:
            print("[INCONCLUSIVE] no steps completed")
            return EXIT_INCONCLUSIVE
        return EXIT_OK
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
        close_with_exit_code(simulation_app, exit_code=code, timeout_sec=30)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(code)
