#!/usr/bin/env python3
"""
Side-by-Side Visualization: Baseline vs Multi-Sample

This script runs two sets of robots simultaneously:
- Left side: Baseline (single noisy observation)
- Right side: Multi-Sample (5 averaged observations)

Both experience the SAME noise level, so you can directly compare performance.

For IROS 2026: Uncertainty Decomposition for Robot Manipulation

Usage:
    # Watch with HIGH noise (10cm) - good for showing improvement
    conda run -n env_py311 python visualize_comparison.py --noise_level high

    # Watch with EXTREME noise (15cm) - dramatic difference
    conda run -n env_py311 python visualize_comparison.py --noise_level extreme

    # Fewer robots for clearer view
    conda run -n env_py311 python visualize_comparison.py --noise_level high --num_envs 2
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Side-by-side comparison visualization")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments PER METHOD (total = 2x this)")
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--num_samples", type=int, default=5, help="Number of samples for multi-sample method")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Remove headless if present - we want visualization!
if hasattr(args_cli, 'headless'):
    args_cli.headless = False

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports after Isaac Sim
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Optional

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
        return obs.get('policy', obs)
    elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
        try:
            return obs['policy']
        except (KeyError, TypeError):
            pass
    return obs


class SideBySidePolicy:
    """
    Policy that applies different methods to different environment indices.

    Left half (indices 0 to num_envs//2): Baseline (single observation)
    Right half (indices num_envs//2 to num_envs): Multi-Sample (averaged)
    """

    def __init__(
        self,
        base_policy,
        policy_nn,
        env,
        num_envs: int,
        num_samples: int = 5,
        noise_params: Dict[str, float] = None,
    ):
        self.base_policy = base_policy
        self.policy_nn = policy_nn
        self.env = env
        self.num_envs = num_envs
        self.num_samples = num_samples
        self.noise_params = noise_params or {}
        self.device = env.unwrapped.device

        # Split point
        self.split_idx = num_envs // 2

        # Get direct access to actor network
        self.actor = policy_nn.actor
        self.actor_obs_normalizer = policy_nn.actor_obs_normalizer

        # Track statistics
        self.baseline_successes = 0
        self.baseline_episodes = 0
        self.multisample_successes = 0
        self.multisample_episodes = 0

        print(f"\n{'='*60}")
        print("SIDE-BY-SIDE COMPARISON")
        print(f"{'='*60}")
        print(f"  Environments 0-{self.split_idx-1}: BASELINE (single noisy obs)")
        print(f"  Environments {self.split_idx}-{num_envs-1}: MULTI-SAMPLE ({num_samples} averaged)")
        print(f"  Noise level: {noise_params.get('object_pos_std', 0)*100:.1f} cm object position noise")
        print(f"{'='*60}\n")

    def get_ground_truth_obs(self) -> Optional[torch.Tensor]:
        """Get ground truth observation from environment state."""
        try:
            unwrapped = self.env.unwrapped
            robot = unwrapped.scene.articulations["robot"]
            joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
            joint_vel = robot.data.joint_vel - robot.data.default_joint_vel

            obj = unwrapped.scene.rigid_objects["object"]
            object_pos = obj.data.root_pos_w - unwrapped.scene.env_origins

            if hasattr(unwrapped, 'command_manager'):
                target_pose = unwrapped.command_manager.get_command("object_pose")
            else:
                target_pose = torch.zeros(joint_pos.shape[0], 7, device=self.device)
                target_pose[:, 3] = 1.0

            if hasattr(unwrapped, 'action_manager'):
                prev_actions = unwrapped.action_manager.action
            else:
                prev_actions = torch.zeros(joint_pos.shape[0], 8, device=self.device)

            ground_truth = torch.cat([
                joint_pos[:, :9],
                joint_vel[:, :9],
                object_pos[:, :3],
                target_pose[:, :7],
                prev_actions[:, :8]
            ], dim=-1)

            return ground_truth

        except Exception as e:
            return None

    def get_multi_sample_obs(self, ground_truth: torch.Tensor, env_indices: torch.Tensor) -> torch.Tensor:
        """Generate multi-sample averaged observation for specific environments."""
        batch_size = env_indices.shape[0]
        gt_subset = ground_truth[env_indices]

        samples = []
        for _ in range(self.num_samples):
            sample = gt_subset.clone()

            if self.noise_params.get("joint_pos_std", 0) > 0:
                noise = torch.randn(batch_size, 9, device=self.device) * self.noise_params["joint_pos_std"]
                sample[:, 0:9] = sample[:, 0:9] + noise

            if self.noise_params.get("joint_vel_std", 0) > 0:
                noise = torch.randn(batch_size, 9, device=self.device) * self.noise_params["joint_vel_std"]
                sample[:, 9:18] = sample[:, 9:18] + noise

            if self.noise_params.get("object_pos_std", 0) > 0:
                noise = torch.randn(batch_size, 3, device=self.device) * self.noise_params["object_pos_std"]
                sample[:, 18:21] = sample[:, 18:21] + noise

            samples.append(sample)

        averaged_obs = torch.stack(samples, dim=0).mean(dim=0)
        return averaged_obs

    def __call__(self, obs) -> torch.Tensor:
        """Get actions using different methods for different env indices."""
        obs_tensor = obs_to_tensor(obs)
        batch_size = obs_tensor.shape[0]

        # Get ground truth for multi-sample method
        ground_truth = self.get_ground_truth_obs()

        # Prepare output actions
        actions = torch.zeros(batch_size, 8, device=self.device)

        # BASELINE: Use raw noisy observation (first half)
        baseline_indices = torch.arange(0, self.split_idx, device=self.device)
        if len(baseline_indices) > 0:
            baseline_obs = obs_tensor[baseline_indices]
            with torch.inference_mode():
                normalized_obs = self.actor_obs_normalizer(baseline_obs)
                actions[baseline_indices] = self.actor(normalized_obs)

        # MULTI-SAMPLE: Use averaged observations (second half)
        multisample_indices = torch.arange(self.split_idx, batch_size, device=self.device)
        if len(multisample_indices) > 0 and ground_truth is not None:
            averaged_obs = self.get_multi_sample_obs(ground_truth, multisample_indices)
            with torch.inference_mode():
                normalized_obs = self.actor_obs_normalizer(averaged_obs)
                actions[multisample_indices] = self.actor(normalized_obs)

        return actions

    def reset(self, dones: torch.Tensor):
        """Track successes for done environments."""
        if hasattr(self.policy_nn, 'reset'):
            self.policy_nn.reset(dones)

    def update_stats(self, env_idx: int, success: bool):
        """Update success statistics."""
        if env_idx < self.split_idx:
            self.baseline_episodes += 1
            if success:
                self.baseline_successes += 1
        else:
            self.multisample_episodes += 1
            if success:
                self.multisample_successes += 1

    def print_stats(self):
        """Print current statistics."""
        baseline_sr = self.baseline_successes / max(1, self.baseline_episodes) * 100
        multi_sr = self.multisample_successes / max(1, self.multisample_episodes) * 100
        print(f"\n{'='*50}")
        print(f"LIVE STATISTICS")
        print(f"{'='*50}")
        print(f"  BASELINE:     {self.baseline_successes}/{self.baseline_episodes} = {baseline_sr:.1f}%")
        print(f"  MULTI-SAMPLE: {self.multisample_successes}/{self.multisample_episodes} = {multi_sr:.1f}%")
        print(f"  Improvement:  {multi_sr - baseline_sr:+.1f}%")
        print(f"{'='*50}\n")


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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run side-by-side visualization."""
    print("\n" + "="*70)
    print("SIDE-BY-SIDE VISUALIZATION: BASELINE vs MULTI-SAMPLE")
    print("IROS 2026: Uncertainty Decomposition for Robot Manipulation")
    print("="*70)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]
    num_envs = args_cli.num_envs * 2  # Double for side-by-side

    print(f"\n[CONFIG]")
    print(f"  Noise level: {noise_level}")
    print(f"  Object noise: {noise_params['object_pos_std']*100:.1f} cm")
    print(f"  Total environments: {num_envs} ({args_cli.num_envs} per method)")
    print(f"  Multi-sample count: {args_cli.num_samples}")

    # Setup environment
    env_cfg.scene.num_envs = num_envs
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

    # Create side-by-side policy
    policy = SideBySidePolicy(
        base_policy=base_policy,
        policy_nn=policy_nn,
        env=env,
        num_envs=num_envs,
        num_samples=args_cli.num_samples,
        noise_params=noise_params,
    )

    # Run visualization loop
    print("\n" + "="*70)
    print("STARTING VISUALIZATION - Press Ctrl+C to stop")
    print("="*70)
    print("\nWatch the robots:")
    print("  - LEFT side (baseline): Often misses cube, erratic movements")
    print("  - RIGHT side (multi-sample): Smoother, more accurate grasping")
    print("\nTip: Use Isaac Sim's camera controls to get a good view")
    print("     Window > Recorder to capture video\n")

    device = env.unwrapped.device
    obs = env.get_observations()

    env_steps = torch.zeros(num_envs, device=device)
    env_max_height = torch.zeros(num_envs, device=device)

    step_count = 0
    try:
        while True:
            with torch.inference_mode():
                actions = policy(obs)

            obs, rewards, dones, _ = env.step(actions)
            policy.reset(dones)

            env_steps += 1

            # Track height
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'scene') and hasattr(unwrapped.scene, 'rigid_objects'):
                heights = unwrapped.scene.rigid_objects["object"].data.root_pos_w[:, 2]
                env_max_height = torch.maximum(env_max_height, heights)

            # Process done episodes
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_idx:
                idx_val = idx.item()
                success = env_max_height[idx_val].item() > 0.2
                policy.update_stats(idx_val, success)

                # Reset trackers
                env_steps[idx_val] = 0
                env_max_height[idx_val] = 0

            step_count += 1
            if step_count % 500 == 0:
                policy.print_stats()

    except KeyboardInterrupt:
        print("\n\nStopping visualization...")
        policy.print_stats()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
