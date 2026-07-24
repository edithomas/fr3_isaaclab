"""FR3 Reach environment configuration.

Defines FR3ReachEnvCfg – a joint-position-controlled reach task
for the Franka Research 3 arm, registered as FR3-Reach-v0.

Built by adapting the official Franka Panda reach config
(isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg).
"""

import math

from isaaclab.utils import configclass

# Base reach environment config (same parent as official Franka reach)
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# Reach-task MDP building blocks (JointPositionActionCfg, …)
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# Project assets
from fr3_isaaclab_project.assets import FR3_CFG


@configclass
class FR3ReachEnvCfg(ReachEnvCfg):
    """Configuration for the FR3 Reach task (joint-position control).

    This class overrides the robot, action, command, and reward
    configuration from the base ReachEnvCfg to work with the FR3 arm.

    The gripper is NOT controlled in the reach task – only the 7 arm joints.
    """

    def __post_init__(self):
        # ------------------------------------------------------------------
        # Parent initialisation (sets up scene, commands, observations, …)
        # ------------------------------------------------------------------
        super().__post_init__()

        # ------------------------------------------------------------------
        # Robot: replace the placeholder with the FR3 articulation config
        # ------------------------------------------------------------------
        self.scene.robot = FR3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------------------
        # End-effector body name – verified against the FR3 USD
        # ------------------------------------------------------------------
        # fr3_hand is the gripper base body (analogous to panda_hand).
        # fr3_hand_tcp (child of fr3_hand) is an alternative for IK tasks.
        EE_BODY = "fr3_hand"

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [EE_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [EE_BODY]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [EE_BODY]

        # ------------------------------------------------------------------
        # Arm action: joint-position targets for the 7 FR3 arm joints
        # ------------------------------------------------------------------
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["fr3_joint.*"],          # matches fr3_joint1 … fr3_joint7
            scale=0.5,
            use_default_offset=True,
        )

        # ------------------------------------------------------------------
        # Pose commands
        # ------------------------------------------------------------------
        self.commands.ee_pose.body_name = EE_BODY
        # FR3 end-effector points downward (same convention as Panda reach)
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FR3ReachEnvCfg_PLAY(FR3ReachEnvCfg):
    """Play variant: fewer envs, no observation corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.observations.policy.enable_corruption = False
