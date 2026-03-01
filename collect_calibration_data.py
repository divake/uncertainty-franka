#!/usr/bin/env python3
"""
Calibration Data Collection Script

Runs the pretrained policy in a CLEAN (no noise) environment and collects
all 36-dimensional observations from successful trajectories.

This calibration dataset X_cal is used to fit:
  - Mahalanobis aleatoric uncertainty estimator (mean μ, covariance Σ)
  - Spectral/Repulsive epistemic uncertainty estimators (manifold structure)
  - Conformal prediction thresholds

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect calibration data from clean environment")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to collect")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for calibration data")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after Isaac Sim
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime
from typing import List, Dict

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


def get_ground_truth_obs(env) -> torch.Tensor:
    """
    Get ground truth observation from environment state.
    Reconstructs observation from actual robot and object state.
    """
    unwrapped = env.unwrapped
    device = unwrapped.device

    # Get robot state
    robot = unwrapped.scene.articulations["robot"]
    joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel = robot.data.joint_vel - robot.data.default_joint_vel

    # Get object state
    obj = unwrapped.scene.rigid_objects["object"]
    object_pos = obj.data.root_pos_w - unwrapped.scene.env_origins

    # Get target pose
    if hasattr(unwrapped, 'command_manager'):
        target_pose = unwrapped.command_manager.get_command("object_pose")
    else:
        target_pose = torch.zeros(joint_pos.shape[0], 7, device=device)
        target_pose[:, 3] = 1.0

    # Get previous actions
    if hasattr(unwrapped, 'action_manager'):
        prev_actions = unwrapped.action_manager.action
    else:
        prev_actions = torch.zeros(joint_pos.shape[0], 8, device=device)

    # Concatenate: [joint_pos(9), joint_vel(9), obj_pos(3), target(7), actions(8)] = 36
    ground_truth = torch.cat([
        joint_pos[:, :9],
        joint_vel[:, :9],
        object_pos[:, :3],
        target_pose[:, :7],
        prev_actions[:, :8]
    ], dim=-1)

    return ground_truth


def obs_to_tensor(obs) -> torch.Tensor:
    """Convert observation (TensorDict or Tensor) to tensor."""
    if hasattr(obs, 'get'):
        return obs.get('policy', obs)
    elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
        try:
            return obs['policy']
        except (KeyError, TypeError):
            pass
    return obs


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect calibration data from clean environment."""
    print("\n" + "=" * 70)
    print("CALIBRATION DATA COLLECTION")
    print("Clean environment — no noise — successful trajectories only")
    print("=" * 70)

    # Setup clean environment (NO noise)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Ensure no noise corruption
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        env_cfg.observations.policy.enable_corruption = False

    # Get checkpoint
    task_name = args_cli.task.split(":")[-1]
    checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
    print(f"Checkpoint: {checkpoint_path}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    print(f"\nConfig:")
    print(f"  Task: {args_cli.task}")
    print(f"  Num envs: {num_envs}")
    print(f"  Target episodes: {args_cli.num_episodes}")
    print(f"  Seed: {args_cli.seed}")

    # Storage for calibration data
    all_observations = []  # Will store ground truth observations
    all_actions = []
    episode_observations = {i: [] for i in range(num_envs)}  # Per-env temp storage
    episode_actions = {i: [] for i in range(num_envs)}
    episode_rewards = {i: 0.0 for i in range(num_envs)}
    episode_max_heights = {i: 0.0 for i in range(num_envs)}

    completed_episodes = 0
    successful_episodes = 0
    total_steps = 0

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)

    print(f"\nCollecting calibration data...")

    while completed_episodes < args_cli.num_episodes:
        # Get ground truth observation
        gt_obs = get_ground_truth_obs(env)

        # Store observation for each env
        for i in range(num_envs):
            episode_observations[i].append(gt_obs[i].cpu().numpy())

        # Get action from policy
        with torch.inference_mode():
            actions = policy(obs)

        # Store actions
        for i in range(num_envs):
            episode_actions[i].append(actions[i].cpu().numpy())

        # Step environment
        obs, rewards, dones, _ = env.step(actions)

        env_steps += 1
        total_steps += 1

        # Track rewards and heights
        for i in range(num_envs):
            episode_rewards[i] += rewards[i].item()

        # Track object height
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            heights = unwrapped.scene.rigid_objects["object"].data.root_pos_w[:, 2]
            for i in range(num_envs):
                episode_max_heights[i] = max(episode_max_heights[i], heights[i].item())

        # Process done episodes
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()
            success = episode_max_heights[idx] > 0.2

            if success:
                # Only store observations from SUCCESSFUL episodes
                all_observations.extend(episode_observations[idx])
                all_actions.extend(episode_actions[idx])
                successful_episodes += 1

            completed_episodes += 1

            # Reset per-env storage
            episode_observations[idx] = []
            episode_actions[idx] = []
            episode_rewards[idx] = 0.0
            episode_max_heights[idx] = 0.0
            env_steps[idx] = 0

            if completed_episodes % 50 == 0:
                print(f"  Episodes: {completed_episodes}/{args_cli.num_episodes} | "
                      f"Successful: {successful_episodes} ({successful_episodes/completed_episodes:.1%}) | "
                      f"Observations collected: {len(all_observations)}")

    # Convert to numpy arrays
    X_cal = np.array(all_observations)
    A_cal = np.array(all_actions)

    print(f"\n{'=' * 70}")
    print(f"CALIBRATION DATA COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total episodes: {completed_episodes}")
    print(f"Successful episodes: {successful_episodes} ({successful_episodes/completed_episodes:.1%})")
    print(f"Total observations: {X_cal.shape[0]}")
    print(f"Observation dim: {X_cal.shape[1]}")
    print(f"Action dim: {A_cal.shape[1]}")

    # Print observation statistics
    print(f"\nObservation Statistics:")
    print(f"  Joint positions (0:9):  mean={X_cal[:, :9].mean():.4f}, std={X_cal[:, :9].std():.4f}")
    print(f"  Joint velocities (9:18): mean={X_cal[:, 9:18].mean():.4f}, std={X_cal[:, 9:18].std():.4f}")
    print(f"  Object position (18:21): mean={X_cal[:, 18:21].mean():.4f}, std={X_cal[:, 18:21].std():.4f}")
    print(f"  Target pose (21:28):     mean={X_cal[:, 21:28].mean():.4f}, std={X_cal[:, 21:28].std():.4f}")
    print(f"  Prev actions (28:36):    mean={X_cal[:, 28:36].mean():.4f}, std={X_cal[:, 28:36].std():.4f}")

    # Save calibration data
    if args_cli.output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "calibration_data",
            f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        output_dir = args_cli.output_dir

    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "X_cal.npy"), X_cal)
    np.save(os.path.join(output_dir, "A_cal.npy"), A_cal)

    # Also save metadata
    metadata = {
        "task": args_cli.task,
        "num_envs": num_envs,
        "total_episodes": completed_episodes,
        "successful_episodes": successful_episodes,
        "total_observations": X_cal.shape[0],
        "observation_dim": X_cal.shape[1],
        "action_dim": A_cal.shape[1],
        "seed": args_cli.seed,
        "obs_structure": {
            "joint_pos": "0:9",
            "joint_vel": "9:18",
            "object_pos": "18:21",
            "target_pose": "21:28",
            "prev_actions": "28:36"
        },
        "obs_stats": {
            "mean": X_cal.mean(axis=0).tolist(),
            "std": X_cal.std(axis=0).tolist(),
            "min": X_cal.min(axis=0).tolist(),
            "max": X_cal.max(axis=0).tolist(),
        }
    }

    import json
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nCalibration data saved to: {output_dir}")
    print(f"  X_cal.npy: {X_cal.shape}")
    print(f"  A_cal.npy: {A_cal.shape}")
    print(f"  metadata.json")

    env.close()
    return X_cal, A_cal, output_dir


if __name__ == "__main__":
    result = main()
    simulation_app.close()
