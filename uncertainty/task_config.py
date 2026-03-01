"""
Task-Specific Configuration for Multi-Task Uncertainty Evaluation

Handles differences between Lift, Reach, Stack, and Cabinet tasks:
- Observation structure (dimensions, what's included)
- Ground truth observation extraction
- Success criteria
- Noise injection indices
- OOD perturbation targets

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TaskObsConfig:
    """Observation structure for a specific task."""
    joint_pos_slice: Tuple[int, int]    # (start, end) for joint positions
    joint_vel_slice: Tuple[int, int]    # (start, end) for joint velocities
    noisy_slice: Optional[Tuple[int, int]]  # (start, end) for the noisy non-joint obs (object_pos or pose_command)
    command_name: str                   # Name in command_manager ("object_pose" or "ee_pose")
    has_object: bool                    # Whether scene has rigid_objects["object"]
    obs_dim: int                        # Total observation dimension
    action_dim: int                     # Action dimension (for prev_actions)
    success_type: str                   # "height" or "tracking"
    success_threshold: float            # Threshold for success criterion


# Task configurations
TASK_CONFIGS = {
    "Isaac-Lift-Cube-Franka-v0": TaskObsConfig(
        joint_pos_slice=(0, 9),
        joint_vel_slice=(9, 18),
        noisy_slice=(18, 21),           # object_pos (3D)
        command_name="object_pose",
        has_object=True,
        obs_dim=36,
        action_dim=8,
        success_type="height",
        success_threshold=0.2,
    ),
    "Isaac-Reach-Franka-v0": TaskObsConfig(
        joint_pos_slice=(0, 9),
        joint_vel_slice=(9, 18),
        noisy_slice=None,               # No object_pos to add noise to; pose_command is clean
        command_name="ee_pose",
        has_object=False,
        obs_dim=32,
        action_dim=7,
        success_type="tracking",
        success_threshold=0.05,         # End-effector within 5cm of target
    ),
}


def get_task_config(task_name: str) -> TaskObsConfig:
    """Get configuration for a task. Falls back to Lift config for unknown tasks."""
    clean_name = task_name.split(":")[-1]
    if clean_name in TASK_CONFIGS:
        return TASK_CONFIGS[clean_name]
    # Default to Lift behavior for backward compatibility
    print(f"  Warning: Unknown task '{clean_name}', using Lift config as default")
    return TASK_CONFIGS["Isaac-Lift-Cube-Franka-v0"]


def get_ground_truth_obs(env, task_cfg: TaskObsConfig) -> Optional[torch.Tensor]:
    """
    Get ground truth observation from environment state, adapted per task.

    Lift (36D): [joint_pos(9), joint_vel(9), object_pos(3), target(7), actions(8)]
    Reach (32D): [joint_pos(9), joint_vel(9), ee_pose_command(7), actions(7)]
    """
    try:
        unwrapped = env.unwrapped
        device = unwrapped.device
        robot = unwrapped.scene.articulations["robot"]
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        batch_size = joint_pos.shape[0]

        parts = [joint_pos[:, :9], joint_vel[:, :9]]

        if task_cfg.has_object:
            # Lift/Stack: include object position
            obj = unwrapped.scene.rigid_objects["object"]
            object_pos = obj.data.root_pos_w - unwrapped.scene.env_origins
            parts.append(object_pos[:, :3])

        # Get command (target pose)
        if hasattr(unwrapped, 'command_manager'):
            try:
                target_pose = unwrapped.command_manager.get_command(task_cfg.command_name)
            except Exception:
                target_pose = torch.zeros(batch_size, 7, device=device)
                target_pose[:, 3] = 1.0
        else:
            target_pose = torch.zeros(batch_size, 7, device=device)
            target_pose[:, 3] = 1.0
        parts.append(target_pose[:, :7])

        # Get previous actions
        if hasattr(unwrapped, 'action_manager'):
            prev_actions = unwrapped.action_manager.action
        else:
            prev_actions = torch.zeros(batch_size, task_cfg.action_dim, device=device)
        parts.append(prev_actions[:, :task_cfg.action_dim])

        return torch.cat(parts, dim=-1)
    except Exception:
        return None


def add_noise_to_samples(samples: torch.Tensor, noise_params: Dict,
                         task_cfg: TaskObsConfig, device: torch.device) -> torch.Tensor:
    """
    Add noise to observation samples, respecting task-specific structure.

    For Lift: noise on joint_pos[0:9], joint_vel[9:18], object_pos[18:21]
    For Reach: noise on joint_pos[0:9], joint_vel[9:18] (no object_pos)
    """
    batch_size = samples.shape[0]
    jp_start, jp_end = task_cfg.joint_pos_slice
    jv_start, jv_end = task_cfg.joint_vel_slice

    if noise_params.get("joint_pos_std", 0) > 0:
        n = jp_end - jp_start
        samples[:, jp_start:jp_end] += torch.randn(batch_size, n, device=device) * noise_params["joint_pos_std"]

    if noise_params.get("joint_vel_std", 0) > 0:
        n = jv_end - jv_start
        samples[:, jv_start:jv_end] += torch.randn(batch_size, n, device=device) * noise_params["joint_vel_std"]

    if task_cfg.noisy_slice is not None and noise_params.get("object_pos_std", 0) > 0:
        ns_start, ns_end = task_cfg.noisy_slice
        n = ns_end - ns_start
        samples[:, ns_start:ns_end] += torch.randn(batch_size, n, device=device) * noise_params["object_pos_std"]

    return samples


def check_success(env, env_max_metric: torch.Tensor, task_cfg: TaskObsConfig) -> torch.Tensor:
    """
    Update and return the task-specific success metric per environment.

    For Lift: tracks max object height
    For Reach: tracks min end-effector distance to target
    """
    unwrapped = env.unwrapped

    if task_cfg.success_type == "height":
        # Object height tracking (Lift, Stack)
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            try:
                heights = unwrapped.scene.rigid_objects["object"].data.root_pos_w[:, 2]
                env_max_metric = torch.maximum(env_max_metric, heights)
            except Exception:
                pass
    elif task_cfg.success_type == "tracking":
        # End-effector tracking error (Reach)
        try:
            robot = unwrapped.scene.articulations["robot"]
            ee_pos = robot.data.body_pos_w[:, -3, :3]  # panda_hand body
            if hasattr(unwrapped, 'command_manager'):
                cmd = unwrapped.command_manager.get_command(task_cfg.command_name)
                target_pos = cmd[:, :3]
                # For Reach commands, target is in robot root frame; ee is in world frame
                # Need to compute in same frame — command is already in base frame
                ee_pos_local = ee_pos - unwrapped.scene.env_origins
                dist = torch.norm(ee_pos_local - target_pos, dim=-1)
                # Track MINIMUM distance (lower = better for reach)
                # We store negative distance so we can use max() and threshold >
                env_max_metric = torch.maximum(env_max_metric, -dist)
        except Exception:
            pass

    return env_max_metric


def is_episode_success(metric_value: float, task_cfg: TaskObsConfig) -> bool:
    """Check if an episode was successful based on the tracked metric."""
    if task_cfg.success_type == "height":
        return metric_value > task_cfg.success_threshold
    elif task_cfg.success_type == "tracking":
        # metric stored as negative distance; success if -dist > -threshold
        # i.e., dist < threshold
        return metric_value > -task_cfg.success_threshold
    return False


def get_ood_scenarios(task_cfg: TaskObsConfig) -> Dict:
    """Get OOD scenarios appropriate for the task."""
    if task_cfg.has_object:
        # Lift/Stack: object mass, friction, gravity
        return {
            "E3_mass_2x":      {"ood_type": "mass",     "ood_params": {"scale": 2.0},  "description": "Object 2x heavier"},
            "E3_mass_5x":      {"ood_type": "mass",     "ood_params": {"scale": 5.0},  "description": "Object 5x heavier"},
            "E3_mass_10x":     {"ood_type": "mass",     "ood_params": {"scale": 10.0}, "description": "Object 10x heavier"},
            "E4_friction_0.5":  {"ood_type": "friction", "ood_params": {"scale": 0.5},  "description": "Friction 0.5x (slippery)"},
            "E4_friction_0.2":  {"ood_type": "friction", "ood_params": {"scale": 0.2},  "description": "Friction 0.2x (very slippery)"},
            "E6_gravity_1.5":   {"ood_type": "gravity",  "ood_params": {"scale": 1.5},  "description": "Gravity 1.5x (heavier world)"},
        }
    else:
        # Reach: robot joint damping, gravity, joint limits (no object to modify)
        return {
            "E6_gravity_1.5":   {"ood_type": "gravity",  "ood_params": {"scale": 1.5},  "description": "Gravity 1.5x (heavier world)"},
            "E6_gravity_2.0":   {"ood_type": "gravity",  "ood_params": {"scale": 2.0},  "description": "Gravity 2.0x (much heavier)"},
            "E7_joint_damping_3x": {"ood_type": "joint_damping", "ood_params": {"scale": 3.0}, "description": "Joint damping 3x"},
            "E7_joint_damping_5x": {"ood_type": "joint_damping", "ood_params": {"scale": 5.0}, "description": "Joint damping 5x"},
        }
