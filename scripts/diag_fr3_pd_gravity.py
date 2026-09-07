#!/usr/bin/env python
"""D3 — Headless PD/gravity A-B-C diagnostic for FR3-Reach-v0.

Probes whether a **zero joint-position action** means "zero torque / free fall"
(it does NOT) by running three bounded rollouts and reporting evidence from
**direct absolute robot state** and the **actual EE pose** — never from the
approximate FK recorder.

Physics (Isaac Lab v2.1.0, ``isaaclab/envs/mdp/actions/joint_actions.py``):
``JointPositionAction(use_default_offset=True, scale=0.5)`` computes
``processed = raw*0.5 + default_joint_pos`` and calls
``set_joint_position_target(processed)``; implicit PD actuators
(stiffness=80, damping=4) hold the target. A zero action sets the target to the
**default** pose — NOT zero torque, NOT free fall. ``robot.data.computed_torque``
exposes the PD-computed torque (``actuator_pd.py``).

Three legs (each a short bounded rollout, 4 envs, fixed seed, zero action):

* **A** gravity ON  (0,0,-9.81), PD ON  (80,4): PD pulls toward default against
  gravity; bounded steady-state error; computed_torque small but nonzero.
* **B** gravity OFF (0,0,0),     PD ON  (80,4): PD converges with no gravity
  bias; transient cleaner; computed_torque -> ~0 at settle.
* **C** gravity ON  (0,0,-9.81), PD OFF (0,0): true "gravity only"; arm
  collapses; max|q - q_default| >> A/B; computed_torque == ~0 (no PD).

Evidence reported per leg: max|q - q_default|, EE z displacement, max
computed_torque, final joint config. The expected ordering ``C > A >= B`` is
reported as a hypothesis assessment, NOT a brittle universal threshold; the
experiment reports the actual numbers and a nonzero exit is returned only on
invalid/inconclusive output (e.g. computed_torque unreadable, or all three legs
degenerate). This refutes the "zero action == free fall" claim by showing A/B
hold (PD active) while C collapses (PD off).

Exit: 0 = evidence collected; 6 = unexpected; 7 = inconclusive.

Usage::

    ~/IsaacLab/isaaclab.sh -p scripts/diag_fr3_pd_gravity.py --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="D3 FR3 PD/gravity A-B-C diagnostic.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch

import fr3_isaaclab_project.tasks  # noqa: F401
from fr3_isaaclab_project.assets.fr3 import FR3_CFG, FR3_EE_BODY
from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code

EXIT_OK = 0
EXIT_UNEXPECTED = 6
EXIT_INCONCLUSIVE = 7

NUM_STEPS = 180  # 6s @ step_dt=1/30
NUM_ENVS = 4
SEED = 42
OUT_PATH = os.path.join("outputs", "diagnostics", "pd_gravity_abc.json")


def _leg_cfg(gravity, stiffness_damping):
    from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import (
        FR3ReachEnvCfg,
    )

    cfg = FR3ReachEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = SEED
    cfg.sim.gravity = gravity
    cfg.commands.ee_pose.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    # rebuild robot cfg so the actuator overrides apply cleanly
    cfg.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if stiffness_damping is not None:
        k, d = stiffness_damping
        for act in cfg.scene.robot.actuators.values():
            act.stiffness = k
            act.damping = d
    return cfg


def _run_leg(gravity, stiffness_damping, label) -> dict:
    env = gym.make("FR3-Reach-v0", cfg=_leg_cfg(gravity, stiffness_damping))
    try:
        device = env.unwrapped.device
        robot = env.unwrapped.scene["robot"]
        env.reset()
        arm_ids, _ = robot.find_joints("fr3_joint.*")
        if not arm_ids:
            return {"label": label, "verdict": "INCONCLUSIVE", "reason": "no fr3_joint.* DOFs"}
        ee_idx = robot.data.body_names.index(FR3_EE_BODY)
        q_default = robot.data.default_joint_pos[:, arm_ids].clone()
        z0 = robot.data.body_state_w[:, ee_idx, 2].detach().clone()

        action = torch.zeros(env.unwrapped.num_envs, env.unwrapped.single_action_space.shape[0], device=device)
        max_q_dev = 0.0
        max_tau = 0.0
        z_end = z0.clone()
        per_step = []
        for step in range(NUM_STEPS):
            env.step(action)
            q = robot.data.joint_pos[:, arm_ids]
            dev = (q - q_default).abs().max().item()
            max_q_dev = max(max_q_dev, dev)
            tau = robot.data.computed_torque[:, arm_ids].abs().max().item() \
                if hasattr(robot.data, "computed_torque") else float("nan")
            max_tau = max(max_tau, tau)
            z_end = robot.data.body_state_w[:, ee_idx, 2].detach().clone()
            if step % 20 == 0:
                per_step.append({
                    "step": step,
                    "max_q_dev_rad": dev,
                    "max_computed_torque": tau,
                    "ee_z_mean": float(z_end.float().mean().item()),
                })

        ee_dz_mean = float((z_end - z0).float().mean().item())
        return {
            "label": label,
            "gravity": list(gravity),
            "stiffness_damping": list(stiffness_damping) if stiffness_damping else None,
            "max_q_deviation_from_default_rad": float(max_q_dev),
            "ee_dz_mean_m": ee_dz_mean,
            "max_computed_torque": float(max_tau),
            "final_arm_pos_mean": [float(x) for x in robot.data.joint_pos[:, arm_ids].float().mean(dim=0).cpu().tolist()],
            "default_arm_pos": [float(x) for x in q_default[0].cpu().tolist()],
            "per_step_sample": per_step,
        }
    finally:
        env.close()


def main() -> int:
    try:
        A = _run_leg((0.0, 0.0, -9.81), (80.0, 4.0), "A_grav_on_pd_on")
        B = _run_leg((0.0, 0.0, 0.0), (80.0, 4.0), "B_grav_off_pd_on")
        C = _run_leg((0.0, 0.0, -9.81), (0.0, 0.0), "C_grav_on_pd_off")
    except Exception:
        import traceback
        traceback.print_exc()
        return EXIT_UNEXPECTED

    inconclusive = any(
        leg.get("verdict") == "INCONCLUSIVE" for leg in (A, B, C)
    ) or any(
        leg.get("max_computed_torque", 0.0) != leg.get("max_computed_torque", 0.0)  # NaN check
        for leg in (A, B, C)
    )

    # Expected ordering (evidence-based, NOT a universal physics threshold):
    #   C.max_q_dev > A.max_q_dev >= 0.95 * B.max_q_dev
    #   C collapses (ee_dz more negative than A); C.max_tau ~ 0 (no PD).
    ordering = {
        "C_max_q_dev_gt_A": C["max_q_deviation_from_default_rad"] > A["max_q_deviation_from_default_rad"],
        "A_max_q_dev_ge_B_slack": A["max_q_deviation_from_default_rad"] >= 0.95 * B["max_q_deviation_from_default_rad"],
        "C_ee_dz_more_negative_than_A": C["ee_dz_mean_m"] < A["ee_dz_mean_m"],
        "C_computed_torque_near_zero": C["max_computed_torque"] < 1.0,
    }

    summary = {
        "task": "D3 FR3-Reach-v0 PD/gravity A-B-C",
        "api": "JointPositionAction(use_default_offset=True, scale=0.5) + ImplicitActuator PD",
        "num_steps_per_leg": NUM_STEPS,
        "num_envs": NUM_ENVS,
        "seed": SEED,
        "hypotheses": {
            "A": "Zero action => target=default; PD (80,4) holds the arm against gravity; bounded steady-state error; computed_torque small but nonzero.",
            "B": "Gravity off => PD converges with no gravity bias; computed_torque -> ~0 at settle.",
            "C": "PD off (stiffness=damping=0) + gravity on => true free fall; max|q-q_default| >> A/B; computed_torque == ~0 (no PD).",
            "refutes": "A/B hold (PD active) while C collapses => 'zero action == free fall' is false; zero action is a position target with PD.",
        },
        "evidence_not_thresholds": (
            "Reported quantities are evidence; the ordering is a hypothesis "
            "assessment, not a brittle universal physics threshold."
        ),
        "legs": {"A": A, "B": B, "C": C},
        "ordering_assessment": ordering,
        "verdict": "INCONCLUSIVE" if inconclusive else "EVIDENCE_COLLECTED",
        "verdict_note": (
            "Nonzero exit is reserved for invalid/inconclusive output, not for "
            "'the arm moved'. Read A vs C to refute 'zero action == free fall'."
        ),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "A_max_q_dev": A["max_q_deviation_from_default_rad"],
        "B_max_q_dev": B["max_q_deviation_from_default_rad"],
        "C_max_q_dev": C["max_q_deviation_from_default_rad"],
        "C_max_tau": C["max_computed_torque"],
        "ordering": ordering,
        "verdict": summary["verdict"],
    }, indent=2))
    print(f"Saved: {OUT_PATH}")
    return EXIT_INCONCLUSIVE if inconclusive else EXIT_OK


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
