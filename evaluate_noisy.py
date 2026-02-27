#!/usr/bin/env python3
"""
Evaluation script for Franka Lift Cube with noisy observations.

This script evaluates the pretrained RSL-RL policy under various noise conditions
to demonstrate how observation uncertainty affects task performance.

For IROS 2026 paper: Uncertainty Decomposition for Robot Manipulation
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# Parse arguments before launching Isaac Sim
parser = argparse.ArgumentParser(description="Evaluate Franka Lift Cube with noisy observations")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
parser.add_argument("--noise_level", type=str, default="medium",
                    choices=["none", "low", "medium", "high", "extreme"],
                    help="Noise level for observations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# AppLauncher arguments (includes --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest after Isaac Sim is launched
import torch
import numpy as np
import gymnasium as gym
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import isaaclab_tasks  # Register tasks
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


# Noise level configurations
NOISE_LEVELS = {
    "none": {
        "joint_pos_std": 0.0,
        "joint_vel_std": 0.0,
        "object_pos_std": 0.0,
    },
    "low": {
        "joint_pos_std": 0.005,  # ~0.3 degrees
        "joint_vel_std": 0.02,
        "object_pos_std": 0.02,  # 2cm
    },
    "medium": {
        "joint_pos_std": 0.01,   # ~0.5 degrees
        "joint_vel_std": 0.05,
        "object_pos_std": 0.05,  # 5cm
    },
    "high": {
        "joint_pos_std": 0.02,   # ~1 degree
        "joint_vel_std": 0.1,
        "object_pos_std": 0.10,  # 10cm
    },
    "extreme": {
        "joint_pos_std": 0.05,   # ~3 degrees
        "joint_vel_std": 0.2,
        "object_pos_std": 0.15,  # 15cm
    },
}


def create_noisy_obs_cfg(noise_params: Dict[str, float]):
    """Create observation configuration with specified noise levels."""

    @configclass
    class NoisyPolicyCfg(ObsGroup):
        """Policy observations with configurable noise."""

        # Joint positions with noise
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"])
                  if noise_params["joint_pos_std"] > 0 else None,
        )

        # Joint velocities with noise
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"])
                  if noise_params["joint_vel_std"] > 0 else None,
        )

        # Object position with noise (most critical!)
        object_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            noise=GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"])
                  if noise_params["object_pos_std"] > 0 else None,
        )

        # Target position - no noise (known goal)
        target_object_position = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "object_pose"}
        )

        # Previous actions - no noise
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoisyObservationsCfg:
        """Observation specifications with noise."""
        policy: NoisyPolicyCfg = NoisyPolicyCfg()

    return NoisyObservationsCfg


@configclass
class EvalRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner config for evaluation (matching training config)."""
    num_steps_per_env = 24
    max_iterations = 1
    save_interval = 50
    experiment_name = "franka_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    success: bool
    steps: int
    final_object_height: float
    max_object_height: float
    cumulative_reward: float


class NoisyEvaluator:
    """Evaluator for Franka Lift Cube with noisy observations."""

    def __init__(
        self,
        num_envs: int = 64,
        noise_level: str = "medium",
        seed: int = 42,
    ):
        self.num_envs = num_envs
        self.noise_level = noise_level
        self.noise_params = NOISE_LEVELS[noise_level]
        self.seed = seed

        # Create environment config
        print("[DEBUG] Creating env config...", flush=True)
        self.env_cfg = self._create_env_cfg()

        # Create environment
        print("[DEBUG] Creating gym environment...", flush=True)
        self.env = gym.make(
            "Isaac-Lift-Cube-Franka-v0",
            cfg=self.env_cfg,
        )
        print(f"[DEBUG] Environment created: {self.env}", flush=True)

        # Wrap for RSL-RL (clip_actions should be a float or None)
        print("[DEBUG] Wrapping for RSL-RL...", flush=True)
        self.env = RslRlVecEnvWrapper(self.env, clip_actions=1.0)
        print("[DEBUG] Wrapper applied", flush=True)

        # Load pretrained policy
        print("[DEBUG] Loading pretrained policy...", flush=True)
        self.policy, self.policy_nn = self._load_policy()
        print("[DEBUG] Policy loaded", flush=True)

        # Stats tracking
        self.episode_stats: List[EpisodeStats] = []

    def _create_env_cfg(self) -> FrankaCubeLiftEnvCfg:
        """Create environment config with noisy observations."""
        env_cfg = FrankaCubeLiftEnvCfg()
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.seed = self.seed

        # Replace observations with noisy version
        if self.noise_level != "none":
            env_cfg.observations = create_noisy_obs_cfg(self.noise_params)()

        return env_cfg

    def _load_policy(self):
        """Load pretrained RSL-RL policy."""
        # Get pretrained checkpoint path
        checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", "Isaac-Lift-Cube-Franka-v0")

        if not checkpoint_path:
            raise RuntimeError("Pretrained checkpoint not found!")

        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

        # Create runner config
        agent_cfg = EvalRunnerCfg()
        agent_cfg.seed = self.seed
        agent_cfg.device = str(self.env.unwrapped.device)

        # Create runner and load weights
        runner = OnPolicyRunner(
            self.env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=str(self.env.unwrapped.device)
        )
        runner.load(checkpoint_path)

        # Get inference policy
        policy = runner.get_inference_policy(device=self.env.unwrapped.device)

        # Get policy network for reset
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        return policy, policy_nn

    def evaluate(self, num_episodes: int = 100) -> Dict:
        """Run evaluation for specified number of episodes."""
        print(f"\n{'='*60}")
        print(f"Evaluating with noise level: {self.noise_level}")
        print(f"Noise parameters: {self.noise_params}")
        print(f"Number of episodes: {num_episodes}")
        print(f"{'='*60}\n")

        # Reset environment
        obs = self.env.get_observations()

        # Track per-environment stats
        env_steps = torch.zeros(self.num_envs, device=self.env.unwrapped.device)
        env_rewards = torch.zeros(self.num_envs, device=self.env.unwrapped.device)
        env_max_height = torch.zeros(self.num_envs, device=self.env.unwrapped.device)

        completed_episodes = 0
        step_count = 0
        max_steps = num_episodes * 500  # Safety limit

        while completed_episodes < num_episodes and step_count < max_steps:
            step_count += 1

            # Get actions from policy
            with torch.inference_mode():
                actions = self.policy(obs)

            # Step environment
            obs, rewards, dones, infos = self.env.step(actions)

            # Reset policy for done envs
            self.policy_nn.reset(dones)

            # Update stats
            env_steps += 1
            env_rewards += rewards

            # Track object height (for success metric)
            unwrapped_env = self.env.unwrapped
            if hasattr(unwrapped_env, 'scene') and hasattr(unwrapped_env.scene, 'rigid_objects'):
                obj_pos = unwrapped_env.scene.rigid_objects["object"].data.root_pos_w
                heights = obj_pos[:, 2]  # Z coordinate
                env_max_height = torch.maximum(env_max_height, heights)

            # Process completed episodes
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_indices:
                idx = idx.item()

                # Check success (object lifted above threshold)
                success = env_max_height[idx].item() > 0.15  # 15cm threshold

                stats = EpisodeStats(
                    success=success,
                    steps=int(env_steps[idx].item()),
                    final_object_height=heights[idx].item() if 'heights' in dir() else 0.0,
                    max_object_height=env_max_height[idx].item(),
                    cumulative_reward=env_rewards[idx].item(),
                )
                self.episode_stats.append(stats)
                completed_episodes += 1

                # Reset per-env stats
                env_steps[idx] = 0
                env_rewards[idx] = 0
                env_max_height[idx] = 0

                if completed_episodes % 10 == 0:
                    success_rate = sum(1 for s in self.episode_stats if s.success) / len(self.episode_stats)
                    print(f"Completed {completed_episodes}/{num_episodes} episodes | "
                          f"Success rate: {success_rate:.1%}")

        # Compute final statistics
        results = self._compute_statistics()
        return results

    def _compute_statistics(self) -> Dict:
        """Compute evaluation statistics."""
        if not self.episode_stats:
            return {}

        successes = [s.success for s in self.episode_stats]
        steps = [s.steps for s in self.episode_stats]
        rewards = [s.cumulative_reward for s in self.episode_stats]
        max_heights = [s.max_object_height for s in self.episode_stats]

        results = {
            "noise_level": self.noise_level,
            "noise_params": self.noise_params,
            "num_episodes": len(self.episode_stats),
            "success_rate": float(np.mean(successes)),
            "success_rate_std": float(np.std(successes)),
            "avg_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_max_height": float(np.mean(max_heights)),
            "std_max_height": float(np.std(max_heights)),
        }

        return results

    def close(self):
        """Close environment."""
        self.env.close()


def main():
    """Main evaluation function."""
    import sys
    print("\n" + "="*70, flush=True)
    print("FRANKA LIFT CUBE - NOISY OBSERVATION EVALUATION", flush=True)
    print("For IROS 2026: Uncertainty Decomposition for Robot Manipulation", flush=True)
    print("="*70 + "\n", flush=True)
    sys.stdout.flush()

    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEBUG] Output dir: {output_dir}", flush=True)

    # Evaluate single noise level
    print(f"[DEBUG] Creating evaluator with noise_level={args_cli.noise_level}", flush=True)
    try:
        evaluator = NoisyEvaluator(
            num_envs=args_cli.num_envs,
            noise_level=args_cli.noise_level,
            seed=args_cli.seed,
        )
        print("[DEBUG] Evaluator created successfully", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create evaluator: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    print("[DEBUG] Starting evaluation...", flush=True)
    try:
        results = evaluator.evaluate(num_episodes=args_cli.num_episodes)
        print("[DEBUG] Evaluation completed", flush=True)
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Noise Level: {results['noise_level']}")
    print(f"Object Position Noise: {results['noise_params']['object_pos_std']*100:.1f} cm")
    print(f"Episodes: {results['num_episodes']}")
    print(f"Success Rate: {results['success_rate']:.1%} ± {results['success_rate_std']:.1%}")
    print(f"Avg Steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"Avg Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Avg Max Height: {results['avg_max_height']:.3f} m ± {results['std_max_height']:.3f} m")
    print("="*60 + "\n")

    # Save results
    results_file = os.path.join(output_dir, f"results_{args_cli.noise_level}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    evaluator.close()

    return results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
