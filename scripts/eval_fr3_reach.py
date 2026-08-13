#!/usr/bin/env python
"""Finite-step evaluator for FR3-Reach-v0 (fixed-command-table replay).

Implements the evaluator contract (``data/fr3-evaluator-api-design/report.md``):

* **Fixed command-table replay** across zero / random / untrained / checkpoint.
  A ``FixedPoseCommand`` (subclass of ``UniformPoseCommand``) reads pose targets
  from a pre-generated ``(E, 3, 7)`` table indexed by (episode, segment) —
  decoupled from torch RNG. Seed-only fairness is NOT enough: RNG consumption
  points depend on the policy, so seed-only would make command sets diverge
  across policies (contract §3.2/L4).
* **untrained** = a fresh-init ``OnPolicyRunner`` from ``FR3ReachPPORunnerCfg``
  (NOT ``model_0.pt``, which already has 1 PPO update — contract §3.5/L11).
* **raw ndjson rows** (schema v1) + **aggregate** (mean return + bootstrap 95% CI,
  per-term rates, segment/episode success) recomputed from raw, and
  **provenance.json** (checkpoint sha256, table sha256, cfg hash, git rev,
  versions).
* **pre-registered success**: segment success = last 25% of each 120-step
  segment has >=10 consecutive steps with pos_err<=0.05 m AND ori_err<=0.20 rad;
  episode success = >=2 of 3 segments. NOT an official benchmark.
* Base = ``FR3-Reach-v0`` cfg with ``enable_corruption=False`` (PLAY semantics);
  ``torch.inference_mode()`` throughout; no per-episode reseeding (seed sows
  once at startup); finite budget (num_episodes / num_steps / wall-clock).

Metrics use **direct scene state** (``robot.data.body_pos_w``/``body_quat_w``,
command->world via ``combine_frame_transforms``/``quat_mul``) and the command
term's own ``_update_metrics`` — NOT the approximate FK recorder, NOT
``info['log']`` (which is per-episode reset logging, contract §2.5/L1).

Exit: 0 = completed; 6 = unexpected; 7 = inconclusive (0 episodes completed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Finite-step FR3 reach evaluator (fixed command table).")
parser.add_argument("--policy", choices=("zero", "random", "untrained", "checkpoint"), default="zero")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=0, help="0 => 360*num_episodes")
parser.add_argument("--wall_clock_s", type=int, default=7200)
parser.add_argument("--seed", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--commands_file", type=str, default=None, help="pre-generated .npz command table")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--success_pos", type=float, default=0.05)
parser.add_argument("--success_ori", type=float, default=0.20)
parser.add_argument("--success_sustain", type=int, default=10)
parser.add_argument("--success_segments_required", type=int, default=2)
parser.add_argument("--tag", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import numpy as np
import torch

import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.assets.fr3 import FR3_EE_BODY
from fr3_isaaclab_project.diagnostics import evaluator as E
from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

EXIT_OK = 0
EXIT_UNEXPECTED = 6
EXIT_INCONCLUSIVE = 7

# Active reward term names (reach_env_cfg.py RewardsCfg) for the raw schema.
REWARD_TERMS = [
    "end_effector_position_tracking",
    "end_effector_position_tracking_fine_grained",
    "end_effector_orientation_tracking",
    "action_rate",
    "joint_vel",
]


# ---------------------------------------------------------------------------
# FixedPoseCommand: injects a pre-generated command table (no torch RNG).
# ---------------------------------------------------------------------------
def _build_fixed_command_cfg(base_ranges):
    """Build a UniformPoseCommandCfg whose class_type is FixedPoseCommand."""
    from isaaclab_tasks.manager_based.manipulation.reach.mdp import mdp as reach_mdp
    from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

    class FixedPoseCommand(UniformPoseCommand):
        """Replays a fixed (E,3,7) command table indexed by (episode, segment).

        Overrides ``_resample_command`` to read from the table instead of torch
        RNG, so the command stream is a pure function of (episode, segment) and
        is identical across policies. ``reset`` bumps the per-env episode index.
        """

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._table = None            # (E, 3, 7) on device
            self._num_eps = 0
            self._episode_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self._segment_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        def set_table(self, table: torch.Tensor) -> None:
            self._table = table.to(self.device)
            self._num_eps = int(table.shape[0])

        def reset(self, env_ids=None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            # episode index for the upcoming episode is the current value (0-based)
            self._segment_idx[env_ids] = 0
            super().reset(env_ids)   # calls _resample_command for segment 0
            self._episode_idx[env_ids] += 1  # next episode uses next table row

        def _resample_command(self, env_ids):
            if isinstance(env_ids, slice):
                env_ids = torch.arange(self.num_envs, device=self.device)
            env_ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=self.device)
            ep = self._episode_idx[env_ids] % self._num_eps
            seg = self._segment_idx[env_ids] % E.SEGMENTS_PER_EPISODE
            cmd = self._table[ep, seg]                      # (len(env_ids), 7)
            self.pose_command_b[env_ids] = cmd
            self._segment_idx[env_ids] += 1

    cfg = reach_mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=FR3_EE_BODY,
        resampling_time_range=(4.0, 4.0),
        debug_vis=False,
        ranges=reach_mdp.UniformPoseCommandCfg.Ranges(
            pos_x=base_ranges.pos_x, pos_y=base_ranges.pos_y, pos_z=base_ranges.pos_z,
            roll=base_ranges.roll, pitch=base_ranges.pitch, yaw=base_ranges.yaw,
        ),
    )
    cfg.class_type = FixedPoseCommand
    return cfg, FixedPoseCommand


def _eval_cfg():
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import FR3ReachEnvCfg

    cfg = FR3ReachEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False  # PLAY semantics
    # Replace the command term with the fixed-table replay term.
    fixed_cfg, _ = _build_fixed_command_cfg(cfg.commands.ee_pose.ranges)
    cfg.commands.ee_pose = fixed_cfg
    return cfg


def _make_policy(env, device, mode):
    """Return (policy_fn, step_env, iface). iface in {'raw','wrapped'}."""
    if mode in ("zero", "random"):
        if mode == "zero":
            def fn(obs):
                return torch.zeros(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device)
        else:
            def fn(obs):
                a = torch.empty(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device)
                a.normal_(0.0, 1.0)  # N(0,1), matching init_noise_std=1.0
                return a
        return fn, env, "raw"

    # untrained / checkpoint: RSL-RL runner
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.agents import rsl_rl_ppo_cfg

    agent_cfg = rsl_rl_ppo_cfg.FR3ReachPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
    if mode == "checkpoint":
        if not args_cli.checkpoint:
            raise SystemExit("[FATAL] --checkpoint required for --policy checkpoint")
        runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    def fn(obs):
        with torch.inference_mode():
            return policy(obs)
    return fn, wrapped, "wrapped"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=os.getcwd()).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=os.getcwd()).stdout.strip())
        return rev, dirty
    except Exception:
        return "", None


def _write_provenance(out_dir, command_path, command_sha, init_path, init_sha, cfg, ckpt_info):
    rev, dirty = _git_rev()
    import importlib.metadata as md
    versions = {}
    for pkg in ("isaaclab", "isaacsim", "rsl-rl-lib", "torch"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    prov = {
        "run_id": os.path.basename(out_dir),
        "tag": args_cli.tag,
        "policy": args_cli.policy,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "num_episodes_budget": args_cli.num_episodes,
        "device": str(cfg.sim.device),
        "command_table": {"path": command_path, "sha256": command_sha,
                          "shape": list(np.load(command_path)["commands"].shape)},
        "init_table": {"path": init_path, "sha256": init_sha},
        "checkpoint": ckpt_info,
        "env_cfg": {"seed": cfg.seed, "num_envs": cfg.scene.num_envs,
                    "enable_corruption": cfg.observations.policy.enable_corruption,
                    "ee_body": FR3_EE_BODY},
        "git": {"rev": rev, "dirty": dirty},
        "versions": versions,
        "cli": " ".join(sys.argv),
        "success_thresholds": {
            "pos_m": args_cli.success_pos, "ori_rad": args_cli.success_ori,
            "sustain_steps": args_cli.success_sustain,
            "segments_required": args_cli.success_segments_required,
            "note": "project pre-registered, not an official benchmark",
        },
        "disclosed_limits": [
            "v2.1.0 ImplicitActuatorCfg.velocity_limit is deprecated (needs velocity_limit_sim).",
            "Evaluation commands are sampled in-distribution; no generalization claim.",
        ],
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    return prov


def main() -> int:
    out_dir = args_cli.out_dir or os.path.join(
        "outputs", "eval", f"{args_cli.policy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Generate / load the fixed command table (decoupled from torch RNG).
    n_eps = args_cli.num_episodes
    if args_cli.commands_file and os.path.exists(args_cli.commands_file):
        cmds = np.load(args_cli.commands_file)["commands"]
    else:
        cmds = E.make_command_table(n_eps, args_cli.seed)

    thr = E.SuccessThresholds(
        pos=args_cli.success_pos, ori=args_cli.success_ori,
        sustain=args_cli.success_sustain, segments_required=args_cli.success_segments_required,
    )

    torch.manual_seed(args_cli.seed)  # one-time sow; no per-episode reseeding
    cfg = _eval_cfg()
    env = gym.make("FR3-Reach-v0", cfg=cfg)
    try:
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        cmd_term = env.unwrapped.command_manager.get_term("ee_pose")
        cmd_term.set_table(torch.tensor(cmds, dtype=torch.float32, device=device))

        # init-joint table from the robot's actual DOF-order defaults.
        defaults_9 = robot.data.default_joint_pos[0].cpu().numpy()
        if args_cli.commands_file and os.path.exists(args_cli.commands_file):
            init_states = np.load(args_cli.commands_file)["init_joints"]
        else:
            init_states = E.make_init_table(n_eps, args_cli.seed, defaults=defaults_9)
        cmd_path = os.path.join(out_dir, "commands.npz")
        np.savez(cmd_path, commands=cmds, init_joints=init_states)
        cmd_sha = _sha256_file(cmd_path)

        # init-joint injection: override reset_joints_by_scale with the table.
        # Best-effort; the FIXED COMMAND TABLE is the load-bearing comparable
        # invariant. init table indexed by the episode index that just started
        # (captured before the term's reset increments its episode counter).
        init_t = torch.tensor(init_states, dtype=torch.float32, device=device)

        def inject_init(env_ids):
            if env_ids is None or isinstance(env_ids, slice):
                env_ids = torch.arange(env.unwrapped.num_envs, device=device)
            env_ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=device)
            ep = (cmd_term._episode_idx[env_ids] - 1) % n_eps
            try:
                robot.write_joint_state_to_sim(
                    init_t[ep], torch.zeros_like(init_t[ep]), env_ids=env_ids,
                )
            except Exception:
                pass
        _orig_reset = cmd_term.reset
        def reset_with_init(env_ids=None):
            started_ep = cmd_term._episode_idx.clone()
            r = _orig_reset(env_ids)
            if env_ids is None or isinstance(env_ids, slice):
                eids = torch.arange(env.unwrapped.num_envs, device=device)
            else:
                eids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=device)
            ep = started_ep[eids] % n_eps
            try:
                robot.write_joint_state_to_sim(
                    init_t[ep], torch.zeros_like(init_t[ep]), env_ids=eids,
                )
            except Exception:
                pass
            return r
        cmd_term.reset = reset_with_init

        policy_fn, step_env, iface = _make_policy(env, device, args_cli.policy)

        max_steps = args_cli.max_steps if args_cli.max_steps > 0 else E.STEPS_PER_EPISODE * n_eps
        step_dt = env.unwrapped.step_dt
        ee_idx = robot.data.body_names.index(FR3_EE_BODY)

        # reset
        if iface == "raw":
            obs, _ = env.reset(seed=args_cli.seed)
        else:
            obs, _ = step_env.get_observations()

        raw_path = os.path.join(out_dir, "raw_rows.ndjson")
        raw_f = open(raw_path, "w")
        # per-episode accumulators for aggregation
        ep_returns, ep_lengths, ep_fpos, ep_fori, ep_apos, ep_aori = [], [], [], [], [], []
        ep_succ, ep_seg = [], []
        per_env_return = torch.zeros(env.unwrapped.num_envs, device=device)
        per_env_len = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
        per_env_pos, per_env_ori = [[] for _ in range(env.unwrapped.num_envs)], [[] for _ in range(env.unwrapped.num_envs)]
        ep_idx_buf = torch.zeros(env.unwrapped.num_envs, device=device, dtype=torch.long)
        n_completed = 0
        global_step = 0
        t0 = time.time()
        ckpt_info = {"used": False}
        if args_cli.policy == "checkpoint":
            ckpt_info = {"used": True, "path": args_cli.checkpoint, "sha256": _sha256_file(args_cli.checkpoint)}

        with torch.inference_mode():
            for step in range(max_steps):
                if n_completed >= n_eps:
                    break
                if time.time() - t0 > args_cli.wall_clock_s:
                    break
                action = policy_fn(obs)
                if iface == "raw":
                    obs, reward, terminated, truncated, info = env.step(action)
                    dones = terminated | truncated
                else:
                    obs, reward, dones, info = step_env.step(action)

                command = env.unwrapped.command_manager.get_command("ee_pose")
                env_ids_all = torch.arange(env.unwrapped.num_envs, device=device)
                pos_err, ori_err = E_metrics(robot, command, env_ids_all, ee_idx)

                per_env_return += reward
                per_env_len += 1
                for i in range(env.unwrapped.num_envs):
                    per_env_pos[i].append(float(pos_err[i].item()))
                    per_env_ori[i].append(float(ori_err[i].item()))

                # reward term rates (genuine per-step, rate form)
                rm = env.unwrapped.reward_manager
                iterable = rm.get_active_iterable_terms(env_idx=0)
                rates = {n: float(v[0]) for n, v in iterable}

                # raw row for env 0 (full schema for the documented sample env)
                rec = {
                    "schema_version": E.SCHEMA_VERSION, "policy": args_cli.policy,
                    "seed": args_cli.seed, "num_envs": env.unwrapped.num_envs,
                    "global_step": global_step, "env_idx": 0,
                    "episode_idx": int(ep_idx_buf[0].item()),
                    "step_in_episode": int(per_env_len[0].item()) - 1,
                    "sim_time_s": global_step * step_dt,
                    "pos_err_m": float(pos_err[0].item()), "ori_err_rad": float(ori_err[0].item()),
                    "reward_net": float(reward[0].item()),
                    "reward_term_rates": rates,
                    "cmd": command[0].detach().cpu().tolist(),
                    "ee_pos_w": robot.data.body_pos_w[0, ee_idx].detach().cpu().tolist(),
                    "ee_quat_w": robot.data.body_quat_w[0, ee_idx].detach().cpu().tolist(),
                }
                raw_f.write(json.dumps(rec) + "\n")
                global_step += 1

                done_idx = dones.nonzero(as_tuple=False).flatten()
                if len(done_idx) > 0:
                    for i in done_idx.tolist():
                        n_completed += 1
                        pe = per_env_pos[i]; oe = per_env_ori[i]
                        pa = np.array(pe); oa = np.array(oe)
                        ok, segs = E.episode_success(pa, oa, thr)
                        ep_returns.append(float(per_env_return[i].item()))
                        ep_lengths.append(int(per_env_len[i].item()))
                        ep_fpos.append(float(pa[-1])); ep_fori.append(float(oa[-1]))
                        ep_apos.append(float(pa.mean())); ep_aori.append(float(oa.mean()))
                        ep_succ.append(bool(ok)); ep_seg.append(segs)
                        per_env_return[i] = 0.0; per_env_len[i] = 0; per_env_pos[i] = []; per_env_ori[i] = []
                        ep_idx_buf[i] += 1
        raw_f.close()

        if n_completed == 0:
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump({"verdict": "INCONCLUSIVE", "reason": "0 episodes completed"}, f, indent=2)
            print("[INCONCLUSIVE] 0 episodes completed")
            return EXIT_INCONCLUSIVE

        agg = E.aggregate_episodes(
            np.array(ep_returns), np.array(ep_lengths), np.array(ep_fpos),
            np.array(ep_fori), np.array(ep_apos), np.array(ep_aori),
            np.array(ep_succ), np.array(ep_seg, dtype=bool),
            bootstrap=2000, seed=args_cli.seed,
        )
        _write_provenance(out_dir, cmd_path, cmd_sha, cmd_path, cmd_sha, cfg, ckpt_info)
        summary = {
            "schema_version": E.SCHEMA_VERSION,
            "task": "FR3-Reach-v0 fixed-command-table evaluation",
            "policy": args_cli.policy,
            "fixed_command_replay": True,
            "success_definition": (
                "segment: last 25% of each 120-step segment has >= "
                f"{args_cli.success_sustain} consecutive steps with pos_err<= "
                f"{args_cli.success_pos} m AND ori_err<= {args_cli.success_ori} rad; "
                f"episode: >= {args_cli.success_segments_required}/3 segments. "
                "Pre-registered, not an official benchmark."
            ),
            "aggregate": agg,
            "metrics_source": "robot.data.body_pos_w/body_quat_w + command->world (combine_frame_transforms/quat_mul); NOT FK, NOT info['log']",
            "n_episodes_completed": n_completed,
            "global_steps": global_step,
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(agg, indent=2))
        print(f"Saved: {out_dir}/summary.json (+ raw_rows.ndjson, provenance.json, commands.npz)")
        return EXIT_OK
    finally:
        env.close()


def E_metrics(robot, command, env_ids, ee_idx):
    """Position/orientation error (world frame), mirroring reach/mdp/rewards.py."""
    root_pos = robot.data.root_state_w[env_ids, :3]
    root_quat = robot.data.root_state_w[env_ids, 3:7]
    des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, command[env_ids, :3])
    des_quat_w = quat_mul(root_quat, command[env_ids, 3:7])
    curr_pos_w = robot.data.body_pos_w[env_ids, ee_idx]
    curr_quat_w = robot.data.body_quat_w[env_ids, ee_idx]
    pos_err = torch.norm(curr_pos_w - des_pos_w, dim=1)
    ori_err = quat_error_magnitude(curr_quat_w, des_quat_w)
    return pos_err, ori_err


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
