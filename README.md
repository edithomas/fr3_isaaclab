# FR3 Isaac Lab Project

Isaac Lab external project for the **Franka Research 3 (FR3)** robot arm.

Currently supports:
- **FR3-Reach-v0** — joint-position-controlled reach task (7-DOF arm, no gripper)
- **FR3-Reach-Play-v0** — play variant (50 envs, no observation corruption)

---

## Dependencies

| Component | Version |
|---|---|
| Isaac Sim | 4.5.0 |
| Isaac Lab | v2.1.0 |
| Python | 3.10 |
| PyTorch | 2.5.1 (CUDA 12.1) |
| RSL-RL | 2.3.1 |

---

## Installation

```bash
# 1) Clone & install Isaac Lab (if not already done)
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd ~/IsaacLab
git checkout v2.1.0

# 2) Install dependencies
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
./isaaclab.sh --install

# 3) Install FR3 external project
cd ~/fr3_isaaclab_project
~/IsaacLab/isaaclab.sh -p -m pip install -e source/fr3_isaaclab_project
```

**System libraries** required for headless servers:
```bash
sudo apt-get install -y libsm6 libxrender1 libfontconfig1 libxt6 libxfixes3 libxrandr2 libxcursor1 libxi6
```

---

## Verify Installation

### List registered FR3 environments
```bash
~/IsaacLab/isaaclab.sh -p scripts/list_envs.py --headless
```

Expected output:
```
+----------------------------------------------------------------------------------------------------------------------------+
|                                            FR3 Registered Environments                                                      |
+--------+-------------------+---------------------------------+--------------------------------------------------------------+
| S. No. | Task Name         | Entry Point                     | Config                                                       |
+--------+-------------------+---------------------------------+--------------------------------------------------------------+
|   1    | FR3-Reach-v0      | isaaclab.envs:ManagerBasedRLEnv | ...FR3ReachEnvCfg                                             |
|   2    | FR3-Reach-Play-v0 | isaaclab.envs:ManagerBasedRLEnv | ...FR3ReachEnvCfg_PLAY                                        |
+--------+-------------------+---------------------------------+--------------------------------------------------------------+
```

### Check FR3 USD asset (joint/body names)
```bash
~/IsaacLab/isaaclab.sh -p scripts/check_fr3_asset.py --headless
```

Prints all joint names, rigid body names, and suggests end-effector candidates
from the FR3 USD on Nucleus.

### Train FR3 Reach (quick test)
```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
    --task FR3-Reach-v0 \
    --headless \
    --num_envs 64 \
    --max_iterations 10
```

### Play with a trained checkpoint
```bash
~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
    --task FR3-Reach-Play-v0 \
    --headless \
    --num_envs 8 \
    --checkpoint /path/to/model.pt
```

---

## Project Structure

```
~/fr3_isaaclab_project/
├── pyproject.toml
├── README.md
├── source/fr3_isaaclab_project/
│   └── fr3_isaaclab_project/
│       ├── __init__.py
│       ├── assets/
│       │   ├── __init__.py
│       │   └── fr3.py                     # FR3_CFG articulation config
│       └── tasks/
│           ├── __init__.py
│           └── manager_based/
│               ├── __init__.py
│               └── fr3_reach/
│                   ├── __init__.py         # Gym registration
│                   ├── fr3_reach_env_cfg.py # FR3ReachEnvCfg
│                   └── agents/
│                       ├── __init__.py
│                       └── rsl_rl_ppo_cfg.py  # PPO agent config
└── scripts/
    ├── list_envs.py
    ├── check_fr3_asset.py
    └── rsl_rl/
        ├── train.py
        ├── play.py
        └── cli_args.py
```

---

## Key Design Decisions

| Aspect | Choice | Reason |
|---|---|---|
| USD path | `ISAAC_NUCLEUS_DIR/Robots/Franka/FR3/fr3.usd` | Verified to exist on Nucleus (Isaac Sim 4.5.0) |
| Arm joints | `fr3_joint.*` (matches fr3_joint1–7) | Confirmed by check_fr3_asset.py and FR3 URDF |
| End-effector body | `fr3_hand` | Confirmed in USD (analogous to `panda_hand`) |
| Actuator groups | `fr3_shoulder` (j1-4), `fr3_forearm` (j5-7) | Same grouping as Panda |
| Action space | Joint-position, scale=0.5, use_default_offset | Inherited from Panda reach |
| Agent | PPO via RSL-RL, 64-dim hidden layers | Same hyperparams as official Franka reach |

---

## FR3 vs Panda Differences

- **Joint names**: `fr3_joint*` instead of `panda_joint*`
- **USD path**: Uses `ISAAC_NUCLEUS_DIR` (not `ISAACLAB_NUCLEUS_DIR`)
- **Velocity limits**: 2.62 rad/s for all joints (FR3 URDF spec)
- **Fixed joints**: `fr3_joint8` and `fr3_hand_joint` are FixedJoints in the USD (not DOFs)
- **TCP**: `fr3_hand_tcp` is a child frame of `fr3_hand`

---

## Troubleshooting

### "Unable to find the file: ...FR3/fr3.usd"
The FR3 USD asset is unavailable on Nucleus. Check:
- Internet connectivity from the server
- Isaac Sim version (requires 4.5.0+)
- DNS resolution (campus network may block S3)
Workaround: Download the USD manually and update `FR3_USD_PATH` in `assets/fr3.py`.

### "joint_names regex didn't match any joints"
Run `scripts/check_fr3_asset.py --headless` to see the actual joint names
in the FR3 USD. Update `FR3_SHOULDER_PATTERN` / `FR3_FOREARM_PATTERN` in
`assets/fr3.py` accordingly.

### "end-effector body name wrong"
The USD body for the end-effector might differ from `fr3_hand`.
Run `check_fr3_asset.py` and update `EE_BODY` in `fr3_reach_env_cfg.py`.

### Vulkan / GPU errors on headless server
```
VkResult: ERROR_INCOMPATIBLE_DRIVER
```
Non-fatal. Isaac Sim's rendering backend requires a display/Vulkan, but
the physics (PhysX on CUDA) still works. Use `--headless` flag.

### "torch cuda unavailable"
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, check:
- NVIDIA driver: `nvidia-smi`
- PyTorch CUDA version matches driver: `torch.version.cuda`
- Reinstall: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

---

## Adding New Tasks

To add a new FR3 task (e.g., lift, stack):
1. Create a new directory under `tasks/manager_based/fr3_lift/`
2. Copy the pattern from `fr3_reach/`: `__init__.py`, `*_env_cfg.py`, `agents/`
3. Register the Gym environment in the `__init__.py`
4. Add the agent config in `agents/rsl_rl_ppo_cfg.py`
5. Import the new task in `tasks/__init__.py`

---

## License

This project follows the same license terms as Isaac Lab.
