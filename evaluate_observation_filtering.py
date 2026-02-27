#!/usr/bin/env python3
"""
Observation Filtering Evaluation Script

This script compares:
1. Baseline policy (raw noisy observations)
2. Filtered policy (temporal averaging to reduce noise)

The idea: If we can average multiple noisy observations, we reduce
aleatoric uncertainty and improve action quality.

For IROS 2026: Uncertainty Decomposition for Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate observation filtering")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--filter_window", type=int, default=3, help="Moving average window")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
from collections import deque

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg

import isaaclab_tasks
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


class ObservationFilter:
    """
    Temporal observation filter using exponential moving average.

    This reduces aleatoric (sensor) noise by averaging recent observations.
    """

    def __init__(self, num_envs: int, obs_dim: int, alpha: float = 0.3, device: str = "cuda:0"):
        """
        Args:
            num_envs: Number of parallel environments
            obs_dim: Observation dimension
            alpha: EMA smoothing factor (higher = more weight on new obs)
            device: Device for tensors
        """
        self.alpha = alpha
        self.device = device
        self.num_envs = num_envs
        self.obs_dim = obs_dim

        # Initialize filtered observation
        self.filtered_obs = None
        self.initialized = False

    def filter(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply exponential moving average filter."""
        if not self.initialized or self.filtered_obs is None:
            self.filtered_obs = obs.clone()
            self.initialized = True
            return obs

        # EMA update: filtered = alpha * new + (1-alpha) * old
        self.filtered_obs = self.alpha * obs + (1 - self.alpha) * self.filtered_obs

        return self.filtered_obs.clone()

    def reset(self, env_ids: torch.Tensor):
        """Reset filter for specific environments."""
        if self.filtered_obs is not None and len(env_ids) > 0:
            # Will be re-initialized on next observation
            pass


class FilteredPolicyWrapper:
    """Wraps a policy with observation filtering."""

    def __init__(self, policy, policy_nn, filter: ObservationFilter):
        self.policy = policy
        self.policy_nn = policy_nn
        self.filter = filter

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from filtered observation."""
        filtered_obs = self.filter.filter(obs)
        with torch.inference_mode():
            return self.policy(filtered_obs)

    def reset(self, dones: torch.Tensor):
        """Reset policy and filter for done environments."""
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        if hasattr(self.policy_nn, 'reset'):
            self.policy_nn.reset(dones)
        self.filter.reset(done_idx)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    max_height: float
    reward: float


def add_noise_to_observations(env_cfg, noise_params: Dict[str, float]):
    """Add noise to observation terms."""
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        policy_cfg = env_cfg.observations.policy
        if hasattr(policy_cfg, 'joint_pos') and noise_params["joint_pos_std"] > 0:
            policy_cfg.joint_pos.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"])
        if hasattr(policy_cfg, 'joint_vel') and noise_params["joint_vel_std"] > 0:
            policy_cfg.joint_vel.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"])
        if hasattr(policy_cfg, 'object_position') and noise_params["object_pos_std"] > 0:
            policy_cfg.object_position.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"])
        policy_cfg.enable_corruption = True
    return env_cfg


def run_evaluation(env, policy, num_episodes: int) -> List[EpisodeResult]:
    """Run evaluation and collect results."""
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)
    env_rewards = torch.zeros(num_envs, device=device)
    env_max_height = torch.zeros(num_envs, device=device)

    results: List[EpisodeResult] = []
    completed = 0

    while completed < num_episodes:
        with torch.inference_mode():
            actions = policy(obs)

        obs, rewards, dones, _ = env.step(actions)

        if hasattr(policy, 'reset'):
            policy.reset(dones)

        env_steps += 1
        env_rewards += rewards

        # Track height
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
            heights = unwrapped.scene.rigid_objects["object"].data.root_pos_w[:, 2]
            env_max_height = torch.maximum(env_max_height, heights)

        # Process done episodes
        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()
            success = env_max_height[idx].item() > 0.2

            results.append(EpisodeResult(
                success=success,
                steps=int(env_steps[idx].item()),
                max_height=env_max_height[idx].item(),
                reward=env_rewards[idx].item(),
            ))
            completed += 1

            # Reset trackers
            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_max_height[idx] = 0

            if completed % 20 == 0:
                sr = sum(1 for r in results if r.success) / len(results)
                print(f"  Episodes: {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    return results


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Compare baseline vs filtered observations."""
    print("\n" + "="*70, flush=True)
    print("OBSERVATION FILTERING FOR NOISE REDUCTION", flush=True)
    print("IROS 2026: Reducing Aleatoric Uncertainty through Filtering", flush=True)
    print("="*70, flush=True)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]

    print(f"\n[CONFIG]", flush=True)
    print(f"  Noise level: {noise_level}", flush=True)
    print(f"  Object noise: {noise_params['object_pos_std']*100:.1f} cm", flush=True)
    print(f"  Filter window/alpha: {args_cli.filter_window}", flush=True)

    # Setup environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if noise_level != "none":
        env_cfg = add_noise_to_observations(env_cfg, noise_params)

    # Get checkpoint
    task_name = args_cli.task.split(":")[-1]
    checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", task_name)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    base_policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Get observation dimension
    obs = env.get_observations()
    obs_dim = obs.shape[-1]

    # =========================================================================
    # Evaluate BASELINE (no filtering)
    # =========================================================================
    print(f"\n{'='*60}", flush=True)
    print("BASELINE EVALUATION (raw noisy observations)", flush=True)
    print(f"{'='*60}", flush=True)

    class BaselinePolicy:
        def __init__(self, policy, policy_nn):
            self.policy = policy
            self.policy_nn = policy_nn

        def __call__(self, obs):
            with torch.inference_mode():
                return self.policy(obs)

        def reset(self, dones):
            if hasattr(self.policy_nn, 'reset'):
                self.policy_nn.reset(dones)

    baseline_policy = BaselinePolicy(base_policy, policy_nn)
    baseline_results = run_evaluation(env, baseline_policy, num_episodes=args_cli.num_episodes)

    baseline_success = sum(1 for r in baseline_results if r.success) / len(baseline_results)
    baseline_reward = np.mean([r.reward for r in baseline_results])
    baseline_height = np.mean([r.max_height for r in baseline_results])

    print(f"\nBaseline Results:", flush=True)
    print(f"  Success Rate: {baseline_success:.1%}", flush=True)
    print(f"  Avg Reward: {baseline_reward:.2f}", flush=True)
    print(f"  Avg Max Height: {baseline_height:.3f} m", flush=True)

    # =========================================================================
    # Evaluate with OBSERVATION FILTERING
    # =========================================================================
    print(f"\n{'='*60}", flush=True)
    print("FILTERED EVALUATION (EMA smoothing)", flush=True)
    print(f"{'='*60}", flush=True)

    # Reset environment
    env.reset()

    # Create filter with alpha based on window size
    # alpha = 2/(N+1) for EMA approximation of N-period SMA
    alpha = 2.0 / (args_cli.filter_window + 1)

    obs_filter = ObservationFilter(
        num_envs=args_cli.num_envs,
        obs_dim=obs_dim,
        alpha=alpha,
        device=str(env.unwrapped.device)
    )

    filtered_policy = FilteredPolicyWrapper(base_policy, policy_nn, obs_filter)

    filtered_results = run_evaluation(env, filtered_policy, num_episodes=args_cli.num_episodes)

    filtered_success = sum(1 for r in filtered_results if r.success) / len(filtered_results)
    filtered_reward = np.mean([r.reward for r in filtered_results])
    filtered_height = np.mean([r.max_height for r in filtered_results])

    print(f"\nFiltered Results:", flush=True)
    print(f"  Success Rate: {filtered_success:.1%}", flush=True)
    print(f"  Avg Reward: {filtered_reward:.2f}", flush=True)
    print(f"  Avg Max Height: {filtered_height:.3f} m", flush=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Metric':<25} {'Baseline':<15} {'Filtered':<15} {'Improvement':<15}", flush=True)
    print("-"*70, flush=True)

    sr_improve = (filtered_success - baseline_success) * 100
    print(f"{'Success Rate':<25} {baseline_success*100:>12.1f}% {filtered_success*100:>12.1f}% {sr_improve:>+12.1f}%", flush=True)

    rw_improve = filtered_reward - baseline_reward
    print(f"{'Avg Reward':<25} {baseline_reward:>13.2f} {filtered_reward:>13.2f} {rw_improve:>+13.2f}", flush=True)

    ht_improve = (filtered_height - baseline_height) * 100
    print(f"{'Avg Max Height (cm)':<25} {baseline_height*100:>13.1f} {filtered_height*100:>13.1f} {ht_improve:>+13.1f}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"filtering_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "noise_level": noise_level,
        "noise_params": noise_params,
        "filter_alpha": alpha,
        "num_episodes": args_cli.num_episodes,
        "baseline": {
            "success_rate": baseline_success,
            "avg_reward": baseline_reward,
            "avg_max_height": baseline_height,
        },
        "filtered": {
            "success_rate": filtered_success,
            "avg_reward": filtered_reward,
            "avg_max_height": filtered_height,
        },
        "improvement": {
            "success_rate_delta": filtered_success - baseline_success,
            "reward_delta": filtered_reward - baseline_reward,
        }
    }

    with open(os.path.join(output_dir, "filtering_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}", flush=True)

    env.close()
    return results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
