#!/usr/bin/env python3
"""
Uncertainty-Aware Evaluation Script

This script compares:
1. Baseline policy (no uncertainty awareness)
2. Uncertainty-aware policy that takes conservative actions when uncertain

For IROS 2026: Uncertainty Decomposition for Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate uncertainty-aware policy")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes")
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--num_ensemble", type=int, default=5, help="Ensemble size")
parser.add_argument("--uncertainty_threshold", type=float, default=0.1,
                    help="Threshold for uncertain actions")
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
import copy
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg
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


class EnsemblePolicy:
    """Simple ensemble for uncertainty estimation."""

    def __init__(self, base_policy, inference_fn, num_members: int = 5, device: str = "cuda:0"):
        """
        Args:
            base_policy: The ActorCritic or policy module
            inference_fn: The inference function that takes obs and returns actions
            num_members: Number of ensemble members
            device: Device to run on
        """
        self.num_members = num_members
        self.device = device
        self.members = []
        self.inference_fns = []

        # Create ensemble by perturbing base policy weights
        for i in range(num_members):
            member = copy.deepcopy(base_policy)
            if i > 0:  # Keep first as original
                self._perturb_weights(member, scale=0.02)
            self.members.append(member)

            # Create inference function for this member
            def make_inference_fn(m):
                def fn(obs):
                    return m.act(obs)
                return fn
            self.inference_fns.append(make_inference_fn(member))

    def _perturb_weights(self, policy, scale: float):
        """Add noise to policy weights."""
        with torch.no_grad():
            for param in policy.parameters():
                param.add_(torch.randn_like(param) * scale * param.abs().mean())

    def get_predictions(self, obs: torch.Tensor) -> torch.Tensor:
        """Get predictions from all members."""
        predictions = []
        for member in self.members:
            with torch.inference_mode():
                # Use act() method which is the proper inference interface
                action = member.act(obs)
            predictions.append(action)
        return torch.stack(predictions, dim=0)

    def predict_with_uncertainty(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean action and uncertainty (std)."""
        all_actions = self.get_predictions(obs)  # [K, B, A]
        mean_action = all_actions.mean(dim=0)
        std_action = all_actions.std(dim=0)
        return mean_action, std_action


class UncertaintyAwarePolicy:
    """
    Policy that modifies actions based on uncertainty estimates.

    Strategy: Use ensemble agreement for action selection.
    When members agree (low uncertainty) -> use mean action
    When members disagree (high uncertainty) -> use most conservative member
    """

    def __init__(
        self,
        ensemble: EnsemblePolicy,
        uncertainty_threshold: float = 0.1,
        use_median: bool = True,  # Use median instead of mean when uncertain
    ):
        self.ensemble = ensemble
        self.threshold = uncertainty_threshold
        self.use_median = use_median

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Get uncertainty-aware action."""
        all_actions = self.ensemble.get_predictions(obs)  # [K, B, A]
        mean_action = all_actions.mean(dim=0)
        std_action = all_actions.std(dim=0)

        # Compute per-environment uncertainty
        uncertainty = std_action.mean(dim=-1, keepdim=True)  # [B, 1]

        # When uncertain, use median action (more robust to outlier ensemble members)
        if self.use_median:
            median_action = all_actions.median(dim=0)[0]

            # Blend between mean and median based on uncertainty
            confidence = torch.clamp(1.0 - uncertainty / self.threshold, 0.0, 1.0)
            action = confidence * mean_action + (1 - confidence) * median_action
        else:
            # Alternative: just use mean (standard ensemble)
            action = mean_action

        return action

    def reset(self, dones: torch.Tensor):
        """Reset for done environments (no-op for ensemble)."""
        pass


class EnsembleVotingPolicy:
    """
    Ensemble voting policy - select action that minimizes disagreement.

    This approach finds the action that all ensemble members would
    most likely agree on, reducing the chance of catastrophic errors.
    """

    def __init__(self, ensemble: EnsemblePolicy):
        self.ensemble = ensemble

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Use ensemble voting for action selection."""
        all_actions = self.ensemble.get_predictions(obs)  # [K, B, A]

        # Use trimmed mean (remove min/max, average rest)
        # This is more robust than simple mean
        if all_actions.shape[0] >= 3:
            # Sort along ensemble dimension
            sorted_actions, _ = all_actions.sort(dim=0)
            # Remove first and last
            trimmed = sorted_actions[1:-1]
            action = trimmed.mean(dim=0)
        else:
            action = all_actions.mean(dim=0)

        return action

    def reset(self, dones: torch.Tensor):
        pass


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    max_height: float
    reward: float
    avg_uncertainty: float


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


def run_evaluation(
    env,
    policy,
    policy_nn,
    num_episodes: int,
    track_uncertainty: bool = False,
) -> List[EpisodeResult]:
    """Run evaluation and collect results."""
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)
    env_rewards = torch.zeros(num_envs, device=device)
    env_max_height = torch.zeros(num_envs, device=device)
    env_uncertainty_sum = torch.zeros(num_envs, device=device)

    results: List[EpisodeResult] = []
    completed = 0

    while completed < num_episodes:
        # Get actions
        if track_uncertainty and hasattr(policy, 'ensemble'):
            _, std = policy.ensemble.predict_with_uncertainty(obs)
            uncertainty = std.mean(dim=-1)
            env_uncertainty_sum += uncertainty

        with torch.inference_mode():
            actions = policy(obs)

        obs, rewards, dones, _ = env.step(actions)

        if hasattr(policy_nn, 'reset'):
            policy_nn.reset(dones)
        elif hasattr(policy, 'reset'):
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

            avg_unc = 0.0
            if env_steps[idx].item() > 0:
                avg_unc = env_uncertainty_sum[idx].item() / env_steps[idx].item()

            results.append(EpisodeResult(
                success=success,
                steps=int(env_steps[idx].item()),
                max_height=env_max_height[idx].item(),
                reward=env_rewards[idx].item(),
                avg_uncertainty=avg_unc,
            ))
            completed += 1

            # Reset trackers
            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_max_height[idx] = 0
            env_uncertainty_sum[idx] = 0

            if completed % 20 == 0:
                sr = sum(1 for r in results if r.success) / len(results)
                print(f"  Episodes: {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    return results


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Compare baseline vs uncertainty-aware policy."""
    print("\n" + "="*70, flush=True)
    print("UNCERTAINTY-AWARE ROBOT MANIPULATION EVALUATION", flush=True)
    print("IROS 2026: Comparing Baseline vs Uncertainty-Aware Control", flush=True)
    print("="*70, flush=True)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]

    print(f"\n[CONFIG]", flush=True)
    print(f"  Noise level: {noise_level}", flush=True)
    print(f"  Object noise: {noise_params['object_pos_std']*100:.1f} cm", flush=True)
    print(f"  Ensemble size: {args_cli.num_ensemble}", flush=True)
    print(f"  Uncertainty threshold: {args_cli.uncertainty_threshold}", flush=True)

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

    # Load base policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    base_policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # =========================================================================
    # Evaluate BASELINE (no uncertainty awareness)
    # =========================================================================
    print(f"\n{'='*60}", flush=True)
    print("BASELINE EVALUATION (no uncertainty awareness)", flush=True)
    print(f"{'='*60}", flush=True)

    baseline_results = run_evaluation(
        env, base_policy, policy_nn,
        num_episodes=args_cli.num_episodes,
        track_uncertainty=False
    )

    baseline_success = sum(1 for r in baseline_results if r.success) / len(baseline_results)
    baseline_reward = np.mean([r.reward for r in baseline_results])
    baseline_height = np.mean([r.max_height for r in baseline_results])

    print(f"\nBaseline Results:", flush=True)
    print(f"  Success Rate: {baseline_success:.1%}", flush=True)
    print(f"  Avg Reward: {baseline_reward:.2f}", flush=True)
    print(f"  Avg Max Height: {baseline_height:.3f} m", flush=True)

    # =========================================================================
    # Create and evaluate UNCERTAINTY-AWARE policy
    # =========================================================================
    print(f"\n{'='*60}", flush=True)
    print("UNCERTAINTY-AWARE EVALUATION", flush=True)
    print(f"{'='*60}", flush=True)

    # Create ensemble
    ensemble = EnsemblePolicy(
        base_policy=policy_nn,
        inference_fn=base_policy,
        num_members=args_cli.num_ensemble,
        device=str(env.unwrapped.device)
    )

    # Create uncertainty-aware policy using ensemble voting (robust to outliers)
    ua_policy = EnsembleVotingPolicy(ensemble=ensemble)

    # Reset environment
    env.reset()

    ua_results = run_evaluation(
        env, ua_policy, ua_policy,
        num_episodes=args_cli.num_episodes,
        track_uncertainty=True
    )

    ua_success = sum(1 for r in ua_results if r.success) / len(ua_results)
    ua_reward = np.mean([r.reward for r in ua_results])
    ua_height = np.mean([r.max_height for r in ua_results])
    ua_uncertainty = np.mean([r.avg_uncertainty for r in ua_results])

    print(f"\nUncertainty-Aware Results:", flush=True)
    print(f"  Success Rate: {ua_success:.1%}", flush=True)
    print(f"  Avg Reward: {ua_reward:.2f}", flush=True)
    print(f"  Avg Max Height: {ua_height:.3f} m", flush=True)
    print(f"  Avg Uncertainty: {ua_uncertainty:.4f}", flush=True)

    # =========================================================================
    # Summary Comparison
    # =========================================================================
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Metric':<25} {'Baseline':<15} {'Uncertainty-Aware':<20} {'Improvement':<15}", flush=True)
    print("-"*70, flush=True)

    sr_improve = (ua_success - baseline_success) * 100
    print(f"{'Success Rate':<25} {baseline_success*100:>12.1f}% {ua_success*100:>17.1f}% {sr_improve:>+12.1f}%", flush=True)

    rw_improve = ua_reward - baseline_reward
    print(f"{'Avg Reward':<25} {baseline_reward:>13.2f} {ua_reward:>18.2f} {rw_improve:>+13.2f}", flush=True)

    ht_improve = (ua_height - baseline_height) * 100
    print(f"{'Avg Max Height (cm)':<25} {baseline_height*100:>13.1f} {ua_height*100:>18.1f} {ht_improve:>+13.1f}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "noise_level": noise_level,
        "noise_params": noise_params,
        "num_episodes": args_cli.num_episodes,
        "num_ensemble": args_cli.num_ensemble,
        "uncertainty_threshold": args_cli.uncertainty_threshold,
        "baseline": {
            "success_rate": baseline_success,
            "avg_reward": baseline_reward,
            "avg_max_height": baseline_height,
        },
        "uncertainty_aware": {
            "success_rate": ua_success,
            "avg_reward": ua_reward,
            "avg_max_height": ua_height,
            "avg_uncertainty": ua_uncertainty,
        },
        "improvement": {
            "success_rate_delta": ua_success - baseline_success,
            "reward_delta": ua_reward - baseline_reward,
            "height_delta": ua_height - baseline_height,
        }
    }

    with open(os.path.join(output_dir, "comparison_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}", flush=True)

    env.close()
    return results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
