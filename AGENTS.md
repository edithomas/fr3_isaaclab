# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab v2.1.0 external project for the **Franka Research 3 (FR3)** robot arm. Currently implements a joint-position-controlled reach task (FR3-Reach-v0), modeled after the official Franka Panda reach task.

Dependencies: Isaac Sim 4.5.0, Isaac Lab v2.1.0, Python 3.10, PyTorch 2.5.1 (CUDA 12.1), RSL-RL 2.3.1.

## Common Commands

All scripts run through Isaac Lab's Python interpreter (`~/IsaacLab/isaaclab.sh -p`):

```bash
# Install the project (editable)
~/IsaacLab/isaaclab.sh -p -m pip install -e source/fr3_isaaclab_project

# List registered FR3 environments
~/IsaacLab/isaaclab.sh -p scripts/list_envs.py --headless

# Inspect FR3 USD asset structure (joint/body names)
~/IsaacLab/isaaclab.sh -p scripts/check_fr3_asset.py --headless

# Generate a local FR3 USD from URDF (when Nucleus is unreachable)
~/IsaacLab/isaaclab.sh -p scripts/generate_fr3_usd.py --headless

# Train FR3 Reach (quick test)
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task FR3-Reach-v0 --headless --num_envs 64 --max_iterations 10

# Play a trained checkpoint
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task FR3-Reach-Play-v0 --headless --num_envs 8 --checkpoint /path/to/model.pt
```

## Architecture

```
source/fr3_isaaclab_project/fr3_isaaclab_project/
├── assets/fr3.py              # FR3_CFG articulation config (joints, actuators, USD path)
└── tasks/manager_based/fr3_reach/
    ├── __init__.py             # Gym env registration (FR3-Reach-v0, FR3-Reach-Play-v0)
    ├── fr3_reach_env_cfg.py    # FR3ReachEnvCfg — extends ReachEnvCfg, overrides robot/actions/rewards
    └── agents/rsl_rl_ppo_cfg.py # PPO agent hyperparams (64-dim hidden, same as official Panda reach)
```

**Registration flow**: `tasks/__init__.py` imports `fr3_reach/__init__.py`, which calls `gym.register()` mapping task names to `ManagerBasedRLEnv` with `FR3ReachEnvCfg` and `FR3ReachPPORunnerCfg`. Training scripts import `fr3_isaaclab_project.tasks` to trigger registration before calling `gym.make("FR3-Reach-v0")`.

**Config hierarchy**: `FR3ReachEnvCfg` extends Isaac Lab's `ReachEnvCfg`, overriding the robot articulation to `FR3_CFG`, the end-effector body to `fr3_hand`, arm actions to 7 FR3 joints (`fr3_joint.*`), and pose command ranges.

**FR3 vs Panda differences**: joint names use `fr3_joint*` prefix (not `panda_joint*`), USD path uses `ISAAC_NUCLEUS_DIR` (not `ISAACLAB_NUCLEUS_DIR`), velocity limits are 2.62 rad/s for all joints, and `fr3_hand` is the end-effector body.

## Key Conventions

- The project follows Isaac Lab's external project pattern: a `source/<name>/<name>/` layout installable via `pip install -e`
- Task configs use `@configclass` decorator (Hydra/Isaac Lab pattern), with `__post_init__` for initialization
- Gym registration happens at import time in `__init__.py` files
- Actuator groups: `fr3_shoulder` (joints 1-4, effort_limit=87.0) and `fr3_forearm` (joints 5-7, effort_limit=12.0)
- Logs go to `logs/rsl_rl/<experiment_name>/<timestamp>/`
- Hydra configs are dumped to `outputs/<date>/<time>/.hydra/`

## Adding New Tasks

1. Create `tasks/manager_based/fr3_<task>/` with `__init__.py`, `*_env_cfg.py`, and `agents/rsl_rl_ppo_cfg.py`
2. Register the Gym environment in `__init__.py`
3. Import the new task module in `tasks/__init__.py`
4. Follow the existing pattern: extend the base task config from `isaaclab_tasks`, override the robot to `FR3_CFG`, and set end-effector/action joint names

## Offline Analysis: `ensemble_gate` (Isaac-Sim-free)

`ensemble_gate/` is a **pure standard-library Python** package + CLI
(`python -m ensemble_gate`, or `scripts/ensemble_gate_metrics.py`) for offline
ensemble-disagreement safety-gate metrics. It has **no** Isaac Sim / Lab /
NumPy / SciPy / sklearn dependency, so it runs with any Python ≥ 3.9 and is
tested without the simulator. Data contract, formulas, and the
**research-instrument disclaimer** (not a safety guarantee / not sim-to-real)
live in `docs/ensemble_gate.md`; a worked input is in
`examples/ensemble_gate_sample.jsonl`. Tests: `python -m pytest tests/ -q`
(conftest adds the repo root to `sys.path`, no install needed). Add new
analysis tooling under this pattern rather than dragging in simulator deps.

## Headless Server Considerations

- Always use `--headless` on servers without a display
- Vulkan `ERROR_INCOMPATIBLE_DRIVER` warnings are non-fatal; physics (PhysX on CUDA) still works
- May need system libraries: `libsm6 libxrender1 libfontconfig1 libxt6 libxfixes3 libxrandr2 libxcursor1 libxi6`

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
