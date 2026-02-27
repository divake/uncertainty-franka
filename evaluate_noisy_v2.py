#!/usr/bin/env python3
"""
Evaluation script for Franka Lift Cube with noisy observations.
Based on Isaac Lab's play.py structure for proper checkpoint loading.

For IROS 2026 paper: Uncertainty Decomposition for Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate Franka Lift Cube with noisy observations")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
parser.add_argument("--noise_level", type=str, default="medium",
                    choices=["none", "low", "medium", "high", "extreme"],
                    help="Noise level for observations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0", help="Task name")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", default=True)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after Isaac Sim
import torch
import numpy as np
import gymnasium as gym
import json
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg

import isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Noise configurations
NOISE_LEVELS = {
    "none": {"joint_pos_std": 0.0, "joint_vel_std": 0.0, "object_pos_std": 0.0},
    "low": {"joint_pos_std": 0.005, "joint_vel_std": 0.02, "object_pos_std": 0.02},
    "medium": {"joint_pos_std": 0.01, "joint_vel_std": 0.05, "object_pos_std": 0.05},
    "high": {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10},
    "extreme": {"joint_pos_std": 0.05, "joint_vel_std": 0.2, "object_pos_std": 0.15},
}


def add_noise_to_observations(env_cfg, noise_params: Dict[str, float]):
    """Add noise to observation terms in the environment config."""
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        policy_cfg = env_cfg.observations.policy

        # Add noise to joint_pos
        if hasattr(policy_cfg, 'joint_pos') and noise_params["joint_pos_std"] > 0:
            policy_cfg.joint_pos.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"])

        # Add noise to joint_vel
        if hasattr(policy_cfg, 'joint_vel') and noise_params["joint_vel_std"] > 0:
            policy_cfg.joint_vel.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"])

        # Add noise to object_position (most important!)
        if hasattr(policy_cfg, 'object_position') and noise_params["object_pos_std"] > 0:
            policy_cfg.object_position.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"])

        # Enable noise corruption
        policy_cfg.enable_corruption = True

    return env_cfg


@dataclass
class EpisodeStats:
    success: bool
    steps: int
    max_height: float
    reward: float


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate with noisy observations."""
    print("\n" + "="*70, flush=True)
    print("FRANKA LIFT CUBE - NOISY OBSERVATION EVALUATION", flush=True)
    print("For IROS 2026: Uncertainty Decomposition for Robot Manipulation", flush=True)
    print("="*70, flush=True)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]

    print(f"\n[INFO] Noise level: {noise_level}", flush=True)
    print(f"[INFO] Noise params: {noise_params}", flush=True)

    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Add noise to observations
    if noise_level != "none":
        env_cfg = add_noise_to_observations(env_cfg, noise_params)
        print(f"[INFO] Added noise to observations", flush=True)

    # Get checkpoint
    task_name = args_cli.task.split(":")[-1]
    checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
    if not checkpoint_path:
        print("[ERROR] Pretrained checkpoint not found!")
        return
    print(f"[INFO] Checkpoint: {checkpoint_path}", flush=True)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create runner and load checkpoint
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    # Get policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    print(f"[INFO] Policy loaded, starting evaluation...", flush=True)

    # Evaluation
    num_envs = args_cli.num_envs
    num_episodes = args_cli.num_episodes

    obs = env.get_observations()

    env_steps = torch.zeros(num_envs, device=env.unwrapped.device)
    env_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
    env_max_height = torch.zeros(num_envs, device=env.unwrapped.device)

    episode_stats: List[EpisodeStats] = []
    completed = 0
    step_count = 0

    print(f"\n[INFO] Running {num_episodes} episodes with {num_envs} parallel envs...", flush=True)

    while completed < num_episodes and step_count < num_episodes * 500:
        step_count += 1

        with torch.inference_mode():
            actions = policy(obs)

        obs, rewards, dones, _ = env.step(actions)
        policy_nn.reset(dones)

        env_steps += 1
        env_rewards += rewards

        # Get object height
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            heights = unwrapped.scene.rigid_objects["object"].data.root_pos_w[:, 2]
            env_max_height = torch.maximum(env_max_height, heights)

        # Process done episodes
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()

            # Success: object lifted above 0.2m (table is ~0.05m, target is ~0.4m)
            max_h = env_max_height[idx].item()
            success = max_h > 0.2

            episode_stats.append(EpisodeStats(
                success=success,
                steps=int(env_steps[idx].item()),
                max_height=max_h,
                reward=env_rewards[idx].item(),
            ))
            completed += 1

            # Reset trackers
            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_max_height[idx] = 0

            if completed % 10 == 0:
                sr = sum(1 for s in episode_stats if s.success) / len(episode_stats)
                print(f"  Episodes: {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    # Compute results
    successes = [s.success for s in episode_stats]
    rewards = [s.reward for s in episode_stats]
    heights = [s.max_height for s in episode_stats]

    results = {
        "noise_level": noise_level,
        "noise_params": noise_params,
        "num_episodes": len(episode_stats),
        "success_rate": float(np.mean(successes)),
        "success_rate_std": float(np.std(successes)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_max_height": float(np.mean(heights)),
        "std_max_height": float(np.std(heights)),
    }

    # Print results
    print("\n" + "="*60, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("="*60, flush=True)
    print(f"Noise Level: {results['noise_level']}", flush=True)
    print(f"Object Noise: {results['noise_params']['object_pos_std']*100:.1f} cm", flush=True)
    print(f"Episodes: {results['num_episodes']}", flush=True)
    print(f"Success Rate: {results['success_rate']:.1%}", flush=True)
    print(f"Avg Reward: {results['avg_reward']:.2f}", flush=True)
    print(f"Avg Max Height: {results['avg_max_height']:.3f} m", flush=True)
    print("="*60, flush=True)

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, f"results_{noise_level}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}", flush=True)

    env.close()
    return results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
