#!/usr/bin/env python3
"""
Run ablation study across all noise levels for Franka Lift Cube.

This script evaluates the pretrained policy under all noise conditions
to generate data for the IROS 2026 paper.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Run ablation study across noise levels")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Episodes per noise level")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after Isaac Sim launch
import torch
import numpy as np
import gymnasium as gym
import json
from datetime import datetime
from collections import defaultdict

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


NOISE_LEVELS = {
    "none": {"joint_pos_std": 0.0, "joint_vel_std": 0.0, "object_pos_std": 0.0},
    "low": {"joint_pos_std": 0.005, "joint_vel_std": 0.02, "object_pos_std": 0.02},
    "medium": {"joint_pos_std": 0.01, "joint_vel_std": 0.05, "object_pos_std": 0.05},
    "high": {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10},
    "extreme": {"joint_pos_std": 0.05, "joint_vel_std": 0.2, "object_pos_std": 0.15},
}


def create_noisy_obs_cfg(noise_params):
    """Create observation config with specified noise."""

    @configclass
    class NoisyPolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"])
                  if noise_params["joint_pos_std"] > 0 else None,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"])
                  if noise_params["joint_vel_std"] > 0 else None,
        )
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"])
                  if noise_params["object_pos_std"] > 0 else None,
        )
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoisyObservationsCfg:
        policy: NoisyPolicyCfg = NoisyPolicyCfg()

    return NoisyObservationsCfg


def evaluate_noise_level(noise_level: str, num_envs: int, num_episodes: int, seed: int):
    """Evaluate a single noise level."""
    print(f"\n{'='*60}")
    print(f"Evaluating noise level: {noise_level}")
    print(f"{'='*60}")

    noise_params = NOISE_LEVELS[noise_level]

    # Create environment
    env_cfg = FrankaCubeLiftEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    if noise_level != "none":
        env_cfg.observations = create_noisy_obs_cfg(noise_params)()

    env = gym.make("Isaac-Lift-Cube-Franka-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=True)

    # Load policy
    checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", "Isaac-Lift-Cube-Franka-v0")
    agent_cfg = {
        "seed": seed,
        "device": str(env.unwrapped.device),
        "num_steps_per_env": 24,
        "max_iterations": 1,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 0.001,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=str(env.unwrapped.device))
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Evaluation loop
    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=env.unwrapped.device)
    env_rewards = torch.zeros(num_envs, device=env.unwrapped.device)
    env_max_height = torch.zeros(num_envs, device=env.unwrapped.device)

    episode_stats = []
    completed = 0
    step_count = 0
    max_steps = num_episodes * 500

    while completed < num_episodes and step_count < max_steps:
        step_count += 1

        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)

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
            success = env_max_height[idx].item() > 0.15

            episode_stats.append({
                "success": success,
                "steps": int(env_steps[idx].item()),
                "max_height": env_max_height[idx].item(),
                "reward": env_rewards[idx].item(),
            })
            completed += 1

            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_max_height[idx] = 0

            if completed % 20 == 0:
                sr = sum(1 for s in episode_stats if s["success"]) / len(episode_stats)
                print(f"  Progress: {completed}/{num_episodes} | Success: {sr:.1%}")

    env.close()

    # Compute stats
    successes = [s["success"] for s in episode_stats]
    rewards = [s["reward"] for s in episode_stats]
    heights = [s["max_height"] for s in episode_stats]

    return {
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


def main():
    print("\n" + "="*70)
    print("FRANKA LIFT CUBE - NOISE ABLATION STUDY")
    print("IROS 2026: Uncertainty Decomposition for Robot Manipulation")
    print("="*70 + "\n")

    # Output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for noise_level in ["none", "low", "medium", "high", "extreme"]:
        results = evaluate_noise_level(
            noise_level=noise_level,
            num_envs=args_cli.num_envs,
            num_episodes=args_cli.num_episodes,
            seed=args_cli.seed,
        )
        all_results.append(results)

        # Save intermediate results
        with open(os.path.join(output_dir, f"results_{noise_level}.json"), 'w') as f:
            json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(f"{'Noise Level':<12} {'Obj Noise (cm)':<15} {'Success Rate':<15} {'Avg Reward':<12}")
    print("-"*70)
    for r in all_results:
        obj_noise = r["noise_params"]["object_pos_std"] * 100
        print(f"{r['noise_level']:<12} {obj_noise:<15.1f} "
              f"{r['success_rate']:.1%} ± {r['success_rate_std']:.1%}     "
              f"{r['avg_reward']:.2f}")
    print("="*70)

    # Save combined results
    combined_file = os.path.join(output_dir, "ablation_results.json")
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
