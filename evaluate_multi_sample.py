#!/usr/bin/env python3
"""
Multi-Sample Observation Evaluation Script

Key Insight: Take multiple noisy observations and average them within a single
timestep. This reduces aleatoric uncertainty by sqrt(N) without adding latency.

This simulates having multiple sensors or taking multiple readings before acting.

For IROS 2026: Uncertainty Decomposition for Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate multi-sample observation averaging")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--num_samples", type=int, default=5, help="Number of observation samples to average")
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
from typing import Dict, List, Optional
from dataclasses import dataclass

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


def obs_to_tensor(obs) -> torch.Tensor:
    """Convert observation (TensorDict or Tensor) to tensor."""
    if hasattr(obs, 'get'):
        # It's a TensorDict - get the policy observation
        return obs.get('policy', obs)
    elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
        # Try to get 'policy' key
        try:
            return obs['policy']
        except (KeyError, TypeError):
            pass
    return obs


class MultiSamplePolicy:
    """
    Policy that uses multiple observation samples to reduce noise.

    Simulates having multiple sensors or taking multiple readings.
    Noise reduces by sqrt(N) when averaging N samples.
    """

    def __init__(
        self,
        base_policy,
        policy_nn,
        env,
        num_samples: int = 5,
        noise_params: Dict[str, float] = None,
    ):
        self.base_policy = base_policy
        self.policy_nn = policy_nn
        self.env = env
        self.num_samples = num_samples
        self.noise_params = noise_params or {}
        self.device = env.unwrapped.device

        # Get direct access to actor network for raw tensor input
        self.actor = policy_nn.actor
        self.actor_obs_normalizer = policy_nn.actor_obs_normalizer

    def get_ground_truth_obs(self) -> Optional[torch.Tensor]:
        """
        Get ground truth observation from environment state.

        Reconstructs observation from actual robot and object state,
        bypassing the noisy observation manager.
        """
        try:
            unwrapped = self.env.unwrapped

            # Get robot state
            robot = unwrapped.scene.articulations["robot"]
            joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
            joint_vel = robot.data.joint_vel - robot.data.default_joint_vel

            # Get object state
            obj = unwrapped.scene.rigid_objects["object"]
            object_pos = obj.data.root_pos_w - unwrapped.scene.env_origins

            # Get target pose (from command manager)
            if hasattr(unwrapped, 'command_manager'):
                target_pose = unwrapped.command_manager.get_command("object_pose")
            else:
                # Fallback - get from scene
                target_pose = torch.zeros(joint_pos.shape[0], 7, device=self.device)
                target_pose[:, 3] = 1.0  # unit quaternion

            # Get previous actions
            if hasattr(unwrapped, 'action_manager'):
                prev_actions = unwrapped.action_manager.action
            else:
                prev_actions = torch.zeros(joint_pos.shape[0], 8, device=self.device)

            # Concatenate: [joint_pos(9), joint_vel(9), obj_pos(3), target(7), actions(8)] = 36
            ground_truth = torch.cat([
                joint_pos[:, :9],   # 9 joints
                joint_vel[:, :9],   # 9 velocities
                object_pos[:, :3],  # 3D position
                target_pose[:, :7], # 7D pose (pos + quat)
                prev_actions[:, :8] # 8 action dims
            ], dim=-1)

            return ground_truth

        except Exception as e:
            print(f"Warning: Could not get ground truth obs: {e}")
            return None

    def get_multi_sample_obs(self, noisy_obs: torch.Tensor) -> torch.Tensor:
        """
        Generate multiple noisy samples from ground truth and average.

        This simulates taking multiple sensor readings.
        The averaged observation has reduced noise (by sqrt(N)).
        """
        if self.num_samples <= 1:
            return noisy_obs

        # Try to get ground truth
        ground_truth = self.get_ground_truth_obs()

        if ground_truth is None:
            # Fallback: use noisy obs as base
            ground_truth = noisy_obs

        batch_size = ground_truth.shape[0]

        # Generate multiple noisy samples from ground truth
        samples = []

        for _ in range(self.num_samples):
            sample = ground_truth.clone()

            # Add noise to joint positions (indices 0-8)
            if self.noise_params.get("joint_pos_std", 0) > 0:
                noise = torch.randn(batch_size, 9, device=self.device) * self.noise_params["joint_pos_std"]
                sample[:, 0:9] = sample[:, 0:9] + noise

            # Add noise to joint velocities (indices 9-17)
            if self.noise_params.get("joint_vel_std", 0) > 0:
                noise = torch.randn(batch_size, 9, device=self.device) * self.noise_params["joint_vel_std"]
                sample[:, 9:18] = sample[:, 9:18] + noise

            # Add noise to object position (indices 18-20)
            if self.noise_params.get("object_pos_std", 0) > 0:
                noise = torch.randn(batch_size, 3, device=self.device) * self.noise_params["object_pos_std"]
                sample[:, 18:21] = sample[:, 18:21] + noise

            samples.append(sample)

        # Average all samples - this reduces noise by sqrt(N)
        averaged_obs = torch.stack(samples, dim=0).mean(dim=0)

        return averaged_obs

    def __call__(self, obs) -> torch.Tensor:
        """Get action from multi-sample averaged observation."""
        # Convert to tensor if needed
        obs_tensor = obs_to_tensor(obs)

        # Get multi-sample averaged observation
        averaged_obs = self.get_multi_sample_obs(obs_tensor)

        # Use actor network directly with raw tensor
        with torch.inference_mode():
            normalized_obs = self.actor_obs_normalizer(averaged_obs)
            return self.actor(normalized_obs)

    def reset(self, dones: torch.Tensor):
        """Reset policy for done environments."""
        if hasattr(self.policy_nn, 'reset'):
            self.policy_nn.reset(dones)


class BaselinePolicy:
    """Baseline policy using raw noisy observations."""

    def __init__(self, policy, policy_nn):
        self.policy = policy
        self.policy_nn = policy_nn

    def __call__(self, obs):
        with torch.inference_mode():
            return self.policy(obs)

    def reset(self, dones):
        if hasattr(self.policy_nn, 'reset'):
            self.policy_nn.reset(dones)


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


def run_evaluation(env, policy, num_episodes: int, label: str) -> List[EpisodeResult]:
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
                print(f"  [{label}] Episodes: {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    return results


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Compare baseline vs multi-sample averaging."""
    print("\n" + "="*70, flush=True)
    print("MULTI-SAMPLE OBSERVATION AVERAGING", flush=True)
    print("IROS 2026: Reducing Aleatoric Uncertainty via Multi-Sensor Fusion", flush=True)
    print("="*70, flush=True)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]
    num_samples = args_cli.num_samples

    print(f"\n[CONFIG]", flush=True)
    print(f"  Noise level: {noise_level}", flush=True)
    print(f"  Object noise: {noise_params['object_pos_std']*100:.1f} cm", flush=True)
    print(f"  Number of samples: {num_samples}", flush=True)
    print(f"  Expected noise reduction: {1/np.sqrt(num_samples):.1%} of original", flush=True)

    # Setup environment WITH noise for baseline
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

    # =========================================================================
    # Evaluate BASELINE (single noisy observation)
    # =========================================================================
    print(f"\n{'='*60}", flush=True)
    print("BASELINE EVALUATION (single noisy observation)", flush=True)
    print(f"{'='*60}", flush=True)

    baseline_policy = BaselinePolicy(base_policy, policy_nn)
    baseline_results = run_evaluation(env, baseline_policy, args_cli.num_episodes, "Baseline")

    baseline_success = sum(1 for r in baseline_results if r.success) / len(baseline_results)
    baseline_reward = np.mean([r.reward for r in baseline_results])
    baseline_height = np.mean([r.max_height for r in baseline_results])

    print(f"\nBaseline Results:", flush=True)
    print(f"  Success Rate: {baseline_success:.1%}", flush=True)
    print(f"  Avg Reward: {baseline_reward:.2f}", flush=True)
    print(f"  Avg Max Height: {baseline_height:.3f} m", flush=True)

    # =========================================================================
    # Evaluate MULTI-SAMPLE (averaged observations from ground truth)
    # =========================================================================
    print(f"\n{'='*60}", flush=True)
    print(f"MULTI-SAMPLE EVALUATION ({num_samples} samples averaged)", flush=True)
    print(f"{'='*60}", flush=True)

    env.reset()

    multi_policy = MultiSamplePolicy(
        base_policy=base_policy,
        policy_nn=policy_nn,
        env=env,
        num_samples=num_samples,
        noise_params=noise_params,
    )

    multi_results = run_evaluation(env, multi_policy, args_cli.num_episodes, "Multi-Sample")

    multi_success = sum(1 for r in multi_results if r.success) / len(multi_results)
    multi_reward = np.mean([r.reward for r in multi_results])
    multi_height = np.mean([r.max_height for r in multi_results])

    print(f"\nMulti-Sample Results:", flush=True)
    print(f"  Success Rate: {multi_success:.1%}", flush=True)
    print(f"  Avg Reward: {multi_reward:.2f}", flush=True)
    print(f"  Avg Max Height: {multi_height:.3f} m", flush=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Method':<25} {'Success Rate':<15} {'Avg Reward':<15} {'Max Height':<15}", flush=True)
    print("-"*70, flush=True)
    print(f"{'Baseline (1 sample)':<25} {baseline_success*100:>12.1f}% {baseline_reward:>13.2f} {baseline_height:>13.3f} m", flush=True)
    print(f"{'Multi-Sample ('+str(num_samples)+' avg)':<25} {multi_success*100:>12.1f}% {multi_reward:>13.2f} {multi_height:>13.3f} m", flush=True)
    print("-"*70, flush=True)

    sr_improve = (multi_success - baseline_success) * 100
    rw_improve = multi_reward - baseline_reward

    print(f"\nImprovement:", flush=True)
    print(f"  Success Rate: {sr_improve:+.1f}%", flush=True)
    print(f"  Reward: {rw_improve:+.2f}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"multi_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "noise_level": noise_level,
        "noise_params": noise_params,
        "num_samples": num_samples,
        "expected_noise_reduction": 1/np.sqrt(num_samples),
        "num_episodes": args_cli.num_episodes,
        "baseline": {
            "success_rate": baseline_success,
            "avg_reward": baseline_reward,
            "avg_max_height": baseline_height,
        },
        "multi_sample": {
            "success_rate": multi_success,
            "avg_reward": multi_reward,
            "avg_max_height": multi_height,
        },
        "improvement": {
            "success_rate_delta": multi_success - baseline_success,
            "reward_delta": multi_reward - baseline_reward,
        }
    }

    with open(os.path.join(output_dir, "multi_sample_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}", flush=True)

    env.close()
    return results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
