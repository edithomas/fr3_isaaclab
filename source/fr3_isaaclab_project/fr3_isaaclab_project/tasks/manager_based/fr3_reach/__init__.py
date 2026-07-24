"""Register FR3 Reach environments with gymnasium."""

import gymnasium as gym

from fr3_isaaclab_project.tasks.manager_based.fr3_reach.agents import rsl_rl_ppo_cfg
from fr3_isaaclab_project.tasks.manager_based.fr3_reach.fr3_reach_env_cfg import (
    FR3ReachEnvCfg,
    FR3ReachEnvCfg_PLAY,
)

# ---------------------------------------------------------------------------
# Joint-position-controlled reach tasks
# ---------------------------------------------------------------------------

gym.register(
    id="FR3-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FR3ReachEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.FR3ReachPPORunnerCfg,
    },
)

gym.register(
    id="FR3-Reach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FR3ReachEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.FR3ReachPPORunnerCfg,
    },
)
