# Copyright (c) 2024, Uncertainty Robotics Research
# Configuration for Franka Lift Cube with degraded observations

from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import GaussianNoiseCfg

# Import the original Franka configuration
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
    FrankaCubeLiftEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.manipulation.lift import mdp


@configclass
class NoisyObservationsCfg:
    """Observation specifications with configurable noise for uncertainty experiments."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with noise."""

        # Joint positions with noise (proprioceptive noise)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),  # ~0.5 degree noise
        )

        # Joint velocities with noise
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        )

        # Object position with HIGH noise (simulates poor vision/occlusion)
        # This is where uncertainty matters most!
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),  # 5cm noise - significant!
        )

        # Target position (command) - no noise (known goal)
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"}
        )

        # Previous actions - no noise
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # Enable noise
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaCubeLiftEnvCfg_NOISY(FrankaCubeLiftEnvCfg):
    """Franka Lift Cube with noisy observations for uncertainty experiments."""

    def __post_init__(self):
        # Initialize parent
        super().__post_init__()

        # Replace observations with noisy version
        self.observations = NoisyObservationsCfg()

        # Reduce number of environments for experiments
        self.scene.num_envs = 64


@configclass
class FrankaCubeLiftEnvCfg_NOISY_PLAY(FrankaCubeLiftEnvCfg_NOISY):
    """Play configuration with noisy observations."""

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for visualization
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5


# Different noise levels for ablation studies
NOISE_LEVELS = {
    "low": {
        "joint_pos_std": 0.005,
        "joint_vel_std": 0.02,
        "object_pos_std": 0.02,  # 2cm
    },
    "medium": {
        "joint_pos_std": 0.01,
        "joint_vel_std": 0.05,
        "object_pos_std": 0.05,  # 5cm
    },
    "high": {
        "joint_pos_std": 0.02,
        "joint_vel_std": 0.1,
        "object_pos_std": 0.10,  # 10cm - very challenging
    },
    "extreme": {
        "joint_pos_std": 0.05,
        "joint_vel_std": 0.2,
        "object_pos_std": 0.15,  # 15cm - nearly impossible
    },
}


def create_noisy_env_cfg(noise_level: str = "medium"):
    """Factory function to create environment config with specific noise level."""

    if noise_level not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level: {noise_level}. Choose from {list(NOISE_LEVELS.keys())}")

    noise_params = NOISE_LEVELS[noise_level]

    @configclass
    class DynamicNoisyObsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                noise=GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"]),
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                noise=GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"]),
            )
            object_position = ObsTerm(
                func=mdp.object_position_in_robot_root_frame,
                noise=GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"]),
            )
            target_object_position = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "object_pose"}
            )
            actions = ObsTerm(func=mdp.last_action)

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class DynamicEnvCfg(FrankaCubeLiftEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.observations = DynamicNoisyObsCfg()
            self.scene.num_envs = 64

    return DynamicEnvCfg()
