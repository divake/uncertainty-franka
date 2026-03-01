#!/usr/bin/env python3
"""
Full Decomposed Uncertainty-Aware Evaluation Pipeline

This is the main evaluation script for the paper. It:
  1. Loads calibration data (or collects it)
  2. Fits aleatoric + epistemic uncertainty estimators
  3. Optionally runs conformal calibration
  4. Evaluates: Vanilla, Multi-Sample, Total Uncertainty, Decomposed (Ours)
  5. Reports results with orthogonality verification

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Decomposed uncertainty-aware evaluation")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--perturbation", type=str, default=None,
                    help="Perturbation preset name (e.g., A1_high, E1_medium)")
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--cal_data_dir", type=str, default=None,
                    help="Path to calibration data (X_cal.npy). If None, collects new data.")
parser.add_argument("--cal_episodes", type=int, default=200,
                    help="Number of calibration episodes if collecting new data")
parser.add_argument("--tau_a", type=float, default=0.3, help="Aleatoric threshold")
parser.add_argument("--tau_e", type=float, default=0.7, help="Epistemic threshold")
parser.add_argument("--beta", type=float, default=0.3, help="Conservative scaling factor")
parser.add_argument("--skip_baselines", action="store_true",
                    help="Skip baseline evaluations, only run decomposed")

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
from dataclasses import dataclass, asdict

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# Our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty.aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator
from uncertainty.epistemic_cvpr import CVPREpistemicEstimator, ZERO_VAR_DIMS_LIFT
from uncertainty.intervention import (
    InterventionController, DecomposedPolicy, TotalUncertaintyPolicy,
    DeepEnsemblePolicy, MCDropoutPolicy,
)
from uncertainty.perturbations import (
    ObservationPerturbation, PerturbationType,
    get_perturbation_config, PERTURBATION_PRESETS
)
from uncertainty.orthogonality import OrthogonalityAnalyzer
from uncertainty.task_config import (
    get_task_config, get_ground_truth_obs as gt_obs_func,
    add_noise_to_samples, check_success, is_episode_success,
)

# Noise configurations (same as evaluate_multi_sample.py)
NOISE_LEVELS = {
    "none": {"joint_pos_std": 0.0, "joint_vel_std": 0.0, "object_pos_std": 0.0},
    "low": {"joint_pos_std": 0.005, "joint_vel_std": 0.02, "object_pos_std": 0.02},
    "medium": {"joint_pos_std": 0.01, "joint_vel_std": 0.05, "object_pos_std": 0.05},
    "high": {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10},
    "extreme": {"joint_pos_std": 0.05, "joint_vel_std": 0.2, "object_pos_std": 0.15},
}


def obs_to_tensor(obs) -> torch.Tensor:
    if hasattr(obs, 'get'):
        return obs.get('policy', obs)
    elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
        try:
            return obs['policy']
        except (KeyError, TypeError):
            pass
    return obs


# Task-specific config (set in main())
_task_cfg = None

def get_ground_truth_obs(env) -> Optional[torch.Tensor]:
    """Get ground truth observation from environment state (task-aware)."""
    return gt_obs_func(env, _task_cfg)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    max_height: float
    reward: float


class BaselinePolicy:
    """Vanilla policy using raw noisy observations."""
    def __init__(self, policy, policy_nn):
        self.policy = policy
        self.policy_nn = policy_nn

    def __call__(self, obs):
        with torch.inference_mode():
            return self.policy(obs)

    def reset(self, dones):
        if hasattr(self.policy_nn, 'reset'):
            self.policy_nn.reset(dones)


class MultiSamplePolicy:
    """Multi-sample averaging policy (no decomposition)."""
    def __init__(self, policy_nn, env, num_samples, noise_params, task_cfg=None):
        self.actor = policy_nn.actor
        self.obs_normalizer = policy_nn.actor_obs_normalizer
        self.env = env
        self.num_samples = num_samples
        self.noise_params = noise_params
        self.device = env.unwrapped.device
        self.task_cfg = task_cfg or _task_cfg

    def __call__(self, obs):
        obs_tensor = obs_to_tensor(obs)
        gt_obs = get_ground_truth_obs(self.env)
        if gt_obs is None:
            gt_obs = obs_tensor

        samples = []
        for _ in range(self.num_samples):
            sample = gt_obs.clone()
            sample = add_noise_to_samples(sample, self.noise_params, self.task_cfg, self.device)
            samples.append(sample)

        averaged = torch.stack(samples, dim=0).mean(dim=0)
        with torch.inference_mode():
            normalized = self.obs_normalizer(averaged)
            return self.actor(normalized)

    def reset(self, dones):
        pass


def run_evaluation(env, policy, num_episodes: int, label: str, task_cfg=None) -> List[EpisodeResult]:
    """Run evaluation and collect results (task-agnostic)."""
    cfg = task_cfg or _task_cfg
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)
    env_rewards = torch.zeros(num_envs, device=device)
    env_metric = torch.zeros(num_envs, device=device)
    if cfg.success_type == "tracking":
        env_metric.fill_(-float('inf'))  # Track best (negative) distance

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

        env_metric = check_success(env, env_metric, cfg)

        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()
            success = is_episode_success(env_metric[idx].item(), cfg)

            results.append(EpisodeResult(
                success=success,
                steps=int(env_steps[idx].item()),
                max_height=env_metric[idx].item(),
                reward=env_rewards[idx].item(),
            ))
            completed += 1

            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_metric[idx] = -float('inf') if cfg.success_type == "tracking" else 0.0

            if completed % 20 == 0:
                sr = sum(1 for r in results if r.success) / len(results)
                print(f"  [{label}] {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    return results


def collect_calibration_data(env, policy, num_episodes: int, task_cfg=None) -> np.ndarray:
    """Collect calibration data from clean environment (task-agnostic)."""
    cfg = task_cfg or _task_cfg
    print(f"\n  Collecting calibration data ({num_episodes} episodes)...")
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    all_obs = []
    completed = 0
    episode_obs = {i: [] for i in range(num_envs)}
    env_metric = torch.zeros(num_envs, device=device)
    if cfg.success_type == "tracking":
        env_metric.fill_(-float('inf'))

    obs = env.get_observations()

    while completed < num_episodes:
        gt_obs = get_ground_truth_obs(env)
        if gt_obs is not None:
            for i in range(num_envs):
                episode_obs[i].append(gt_obs[i].cpu().numpy())

        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)

        env_metric = check_success(env, env_metric, cfg)

        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()
            if is_episode_success(env_metric[idx].item(), cfg):
                all_obs.extend(episode_obs[idx])
            completed += 1
            episode_obs[idx] = []
            env_metric[idx] = -float('inf') if cfg.success_type == "tracking" else 0.0

            if completed % 50 == 0:
                print(f"    Calibration: {completed}/{num_episodes} episodes, {len(all_obs)} observations")

    X_cal = np.array(all_obs)
    print(f"  Calibration data: {X_cal.shape}")
    return X_cal


def add_noise_to_observations(env_cfg, noise_params):
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        policy_cfg = env_cfg.observations.policy
        if hasattr(policy_cfg, 'joint_pos') and noise_params["joint_pos_std"] > 0:
            policy_cfg.joint_pos.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"])
        if hasattr(policy_cfg, 'joint_vel') and noise_params["joint_vel_std"] > 0:
            policy_cfg.joint_vel.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"])
        # Lift has object_position, Reach does not
        if hasattr(policy_cfg, 'object_position') and noise_params.get("object_pos_std", 0) > 0:
            policy_cfg.object_position.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"])
        policy_cfg.enable_corruption = True
    return env_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Main evaluation with decomposed uncertainty."""
    global _task_cfg
    _task_cfg = get_task_config(args_cli.task)

    print("\n" + "=" * 70)
    print("DECOMPOSED UNCERTAINTY-AWARE EVALUATION")
    print("IROS 2026: Decomposed Uncertainty-Aware Control")
    print("=" * 70)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]

    print(f"\n[CONFIG]")
    print(f"  Task: {args_cli.task}")
    print(f"  Noise: {noise_level} (obj: {noise_params['object_pos_std']*100:.1f} cm)")
    print(f"  Samples: {args_cli.num_samples}")
    print(f"  Thresholds: tau_a={args_cli.tau_a}, tau_e={args_cli.tau_e}")
    print(f"  Beta: {args_cli.beta}")

    # =====================================================================
    # PHASE 1: Setup environment
    # =====================================================================
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    if noise_level != "none":
        env_cfg = add_noise_to_observations(env_cfg, noise_params)

    task_name = args_cli.task.split(":")[-1]
    checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", task_name)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    base_policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # =====================================================================
    # PHASE 2: Calibration — fit uncertainty estimators
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: CALIBRATION")
    print(f"{'='*60}")

    if args_cli.cal_data_dir and os.path.exists(os.path.join(args_cli.cal_data_dir, "X_cal.npy")):
        print(f"  Loading calibration data from: {args_cli.cal_data_dir}")
        X_cal = np.load(os.path.join(args_cli.cal_data_dir, "X_cal.npy"))
    else:
        # Collect calibration data on-the-fly
        X_cal = collect_calibration_data(env, base_policy, args_cli.cal_episodes)

        # Save for reuse
        cal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "calibration_data", f"inline_{task_name}")
        os.makedirs(cal_dir, exist_ok=True)
        np.save(os.path.join(cal_dir, "X_cal.npy"), X_cal)
        print(f"  Saved calibration data to: {cal_dir}")

    print(f"  Calibration data shape: {X_cal.shape}")

    # --- σ_alea = Mahalanobis distance (CVPR Sec 3.2) ---
    # Measures distance of observation from calibration distribution
    # High distance = degraded observation (noise, occlusion) = high aleatoric
    alea_est = AleatoricEstimator(reg_lambda=1e-4)
    alea_est.fit(X_cal, verbose=True)
    alea_est.to_torch(env.unwrapped.device)

    # --- σ_epis = ε_knn + ε_rank (CVPR Sec 3.1 adapted) ---
    # Measures state novelty via k-NN distance + spectral entropy
    # High = unfamiliar state (dynamics shift, OOD) = high epistemic
    # Computed on GROUND TRUTH, immune to sensor noise by design
    zero_var_dims = ZERO_VAR_DIMS_LIFT if "Lift" in args_cli.task else None
    epis_est = CVPREpistemicEstimator(k_knn=20, k_rank=50, zero_var_dims=zero_var_dims)
    epis_est.fit(X_cal, verbose=True)

    # MSV is the denoising MECHANISM (not a signal)
    # Used to average N noisy readings when σ_alea triggers filtering
    msv_est = MultiSampleVarianceEstimator()
    msv_est.calibrate(X_cal, noise_params, n_samples=args_cli.num_samples,
                      n_trials=500, verbose=True)

    # Create intervention controller
    controller = InterventionController(
        tau_a=args_cli.tau_a,
        tau_e=args_cli.tau_e,
        beta=args_cli.beta,
    )

    # =====================================================================
    # PHASE 3: Quick orthogonality check
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: ORTHOGONALITY CHECK")
    print(f"{'='*60}")

    # Test orthogonality via behavioral isolation
    # Key: noise → σ_alea rises, σ_epis stays flat (by design: epis uses GT)
    #      OOD  → σ_epis rises, σ_alea may also rise (Mahal detects any shift)
    n_check = min(300, len(X_cal))
    analyzer = OrthogonalityAnalyzer()

    # --- Noise sweep: varying noise on same GT states ---
    # σ_alea = Mahalanobis(noisy obs) → should INCREASE with noise
    # σ_epis = ε_knn+ε_rank(GT) → should stay FLAT (GT doesn't change)
    noise_levels_test = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15]
    noise_u_a_means = []
    noise_u_e_means = []
    gt_noise = X_cal[np.random.choice(len(X_cal), n_check, replace=False)]

    for noise_std in noise_levels_test:
        noisy_obs = gt_noise.copy()
        if noise_std > 0:
            noisy_obs[:, 18:21] += np.random.randn(n_check, 3) * noise_std
            noisy_obs[:, 0:9] += np.random.randn(n_check, 9) * (noise_std * 0.2)
        # σ_alea on noisy observation
        noise_u_a_means.append(alea_est.predict_normalized(noisy_obs).mean())
        # σ_epis on GROUND TRUTH (unchanged!) — must stay flat
        noise_u_e_means.append(epis_est.predict(gt_noise).mean())

    # --- OOD sweep: varying state shift, NO noise ---
    # σ_epis = ε_knn+ε_rank(shifted GT) → should INCREASE with shift
    # σ_alea = Mahalanobis(shifted GT) → also increases (expected, same obs)
    ood_shifts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    ood_u_a_means = []
    ood_u_e_means = []
    gt_ood = X_cal[np.random.choice(len(X_cal), n_check, replace=False)]

    for shift in ood_shifts:
        shifted = gt_ood.copy()
        shifted[:, 18:21] += np.array([shift, shift, 0])
        # In OOD-only condition, obs = GT (no noise)
        ood_u_a_means.append(alea_est.predict_normalized(shifted).mean())
        ood_u_e_means.append(epis_est.predict(shifted).mean())

    # Print behavioral isolation results
    print(f"\n  --- Noise Sweep (σ_alea ↑, σ_epis flat — epis uses GT) ---")
    print(f"  {'Noise':>8} | {'σ_alea (Mahal)':>14} | {'σ_epis (kNN+rank)':>18}")
    print(f"  {'-'*46}")
    for i, ns in enumerate(noise_levels_test):
        print(f"  {ns:>8.3f} | {noise_u_a_means[i]:>14.4f} | {noise_u_e_means[i]:>18.4f}")

    print(f"\n  --- OOD Sweep (σ_epis ↑, σ_alea also ↑ expected) ---")
    print(f"  {'Shift':>8} | {'σ_alea (Mahal)':>14} | {'σ_epis (kNN+rank)':>18}")
    print(f"  {'-'*46}")
    for i, sh in enumerate(ood_shifts):
        print(f"  {sh:>8.3f} | {ood_u_a_means[i]:>14.4f} | {ood_u_e_means[i]:>18.4f}")

    # Behavioral isolation metrics
    noise_u_a_range = noise_u_a_means[-1] - noise_u_a_means[0]
    noise_u_e_range = abs(noise_u_e_means[-1] - noise_u_e_means[0])
    ood_u_e_range = ood_u_e_means[-1] - ood_u_e_means[0]
    ood_u_a_range = abs(ood_u_a_means[-1] - ood_u_a_means[0])

    # Key metric: noise should NOT affect σ_epis (by design: epis uses GT)
    noise_isolation = noise_u_a_range / max(noise_u_e_range, 1e-6)

    print(f"\n  Behavioral Isolation:")
    print(f"    Noise sweep: σ_alea range={noise_u_a_range:.4f}, σ_epis range={noise_u_e_range:.6f}")
    print(f"    Noise isolation ratio: {noise_isolation:.1f}x (σ_alea/σ_epis change)")
    print(f"    OOD sweep:   σ_epis range={ood_u_e_range:.4f}, σ_alea range={ood_u_a_range:.4f}")

    # σ_epis should be EXACTLY flat under noise (Δ=0) because epis uses GT
    noise_pass = noise_u_e_range < 0.01  # Should be ~0
    ood_pass = ood_u_e_range > 0.1  # σ_epis should respond to OOD
    overall_pass = noise_pass and ood_pass

    print(f"    Noise→σ_epis flat: {'PASS' if noise_pass else 'FAIL'} (σ_epis Δ < 0.01)")
    print(f"    OOD→σ_epis rises:  {'PASS' if ood_pass else 'FAIL'} (σ_epis Δ > 0.1)")
    print(f"    OVERALL:           {'PASS' if overall_pass else 'FAIL'}")

    # Standard orthogonality on mixed conditions
    all_u_a, all_u_e = [], []
    for noise_std in noise_levels_test:
        noisy_obs = gt_noise.copy()
        if noise_std > 0:
            noisy_obs[:, 18:21] += np.random.randn(n_check, 3) * noise_std
            noisy_obs[:, 0:9] += np.random.randn(n_check, 9) * (noise_std * 0.2)
        all_u_a.extend(alea_est.predict_normalized(noisy_obs).tolist())
        all_u_e.extend(epis_est.predict(gt_noise).tolist())

    ortho_results = analyzer.analyze(np.array(all_u_a), np.array(all_u_e), verbose=True)
    ortho_results['behavioral_isolation'] = {
        'noise_isolation_ratio': float(noise_isolation),
        'noise_epis_range': float(noise_u_e_range),
        'ood_epis_range': float(ood_u_e_range),
        'noise_pass': noise_pass,
        'ood_pass': ood_pass,
        'overall_pass': overall_pass,
    }

    # =====================================================================
    # PHASE 4: Evaluate methods
    # =====================================================================
    all_results = {}

    # --- Baseline (Vanilla) ---
    if not args_cli.skip_baselines:
        print(f"\n{'='*60}")
        print("EVALUATING: VANILLA (Baseline)")
        print(f"{'='*60}")
        env.reset()

        vanilla_policy = BaselinePolicy(base_policy, policy_nn)
        vanilla_results = run_evaluation(env, vanilla_policy, args_cli.num_episodes, "Vanilla")

        vanilla_sr = sum(1 for r in vanilla_results if r.success) / len(vanilla_results)
        vanilla_rw = np.mean([r.reward for r in vanilla_results])
        all_results['vanilla'] = {'success_rate': vanilla_sr, 'avg_reward': vanilla_rw}
        print(f"  Vanilla: {vanilla_sr:.1%} success, {vanilla_rw:.2f} avg reward")

        # --- Multi-Sample Only ---
        print(f"\n{'='*60}")
        print(f"EVALUATING: MULTI-SAMPLE ONLY (N={args_cli.num_samples})")
        print(f"{'='*60}")
        env.reset()

        ms_policy = MultiSamplePolicy(policy_nn, env, args_cli.num_samples, noise_params)
        ms_results = run_evaluation(env, ms_policy, args_cli.num_episodes, "Multi-Sample")

        ms_sr = sum(1 for r in ms_results if r.success) / len(ms_results)
        ms_rw = np.mean([r.reward for r in ms_results])
        all_results['multi_sample'] = {'success_rate': ms_sr, 'avg_reward': ms_rw}
        print(f"  Multi-Sample: {ms_sr:.1%} success, {ms_rw:.2f} avg reward")

        # --- Deep Ensemble (B3) ---
        print(f"\n{'='*60}")
        print("EVALUATING: DEEP ENSEMBLE (B3)")
        print(f"{'='*60}")
        env.reset()

        de_policy = DeepEnsemblePolicy(
            policy_actor=policy_nn.actor,
            policy_obs_normalizer=policy_nn.actor_obs_normalizer,
            env=env,
            num_members=5,
            perturbation_std=0.02,
            uncertainty_threshold=0.1,
            beta=args_cli.beta,
        )
        de_results = run_evaluation(env, de_policy, args_cli.num_episodes, "DeepEnsemble")

        de_sr = sum(1 for r in de_results if r.success) / len(de_results)
        de_rw = np.mean([r.reward for r in de_results])
        all_results['deep_ensemble'] = {'success_rate': de_sr, 'avg_reward': de_rw}
        de_stats = de_policy.get_stats()
        print(f"  Deep Ensemble: {de_sr:.1%} success, {de_rw:.2f} avg reward")
        print(f"  Intervention: {de_stats['intervene']['fraction']:.1%} conservative, {de_stats['normal']['fraction']:.1%} normal")

        # --- MC Dropout (B4) ---
        print(f"\n{'='*60}")
        print("EVALUATING: MC DROPOUT (B4)")
        print(f"{'='*60}")
        env.reset()

        mcd_policy = MCDropoutPolicy(
            policy_actor=policy_nn.actor,
            policy_obs_normalizer=policy_nn.actor_obs_normalizer,
            env=env,
            dropout_p=0.1,
            num_passes=10,
            uncertainty_threshold=0.1,
            beta=args_cli.beta,
        )
        mcd_results = run_evaluation(env, mcd_policy, args_cli.num_episodes, "MCDropout")

        mcd_sr = sum(1 for r in mcd_results if r.success) / len(mcd_results)
        mcd_rw = np.mean([r.reward for r in mcd_results])
        all_results['mc_dropout'] = {'success_rate': mcd_sr, 'avg_reward': mcd_rw}
        mcd_stats = mcd_policy.get_stats()
        print(f"  MC Dropout: {mcd_sr:.1%} success, {mcd_rw:.2f} avg reward")
        print(f"  Intervention: {mcd_stats['intervene']['fraction']:.1%} conservative, {mcd_stats['normal']['fraction']:.1%} normal")

        # --- Total Uncertainty (B5) ---
        print(f"\n{'='*60}")
        print("EVALUATING: TOTAL UNCERTAINTY (B5)")
        print(f"{'='*60}")
        env.reset()

        tu_policy = TotalUncertaintyPolicy(
            policy_actor=policy_nn.actor,
            policy_obs_normalizer=policy_nn.actor_obs_normalizer,
            alea_estimator=alea_est,
            epis_estimator=epis_est,
            env=env,
            tau_total=args_cli.tau_a,
            beta=args_cli.beta,
            num_samples=args_cli.num_samples,
            noise_params=noise_params,
            task_cfg=_task_cfg,
        )
        tu_results = run_evaluation(env, tu_policy, args_cli.num_episodes, "TotalUncert")

        tu_sr = sum(1 for r in tu_results if r.success) / len(tu_results)
        tu_rw = np.mean([r.reward for r in tu_results])
        all_results['total_uncertainty'] = {'success_rate': tu_sr, 'avg_reward': tu_rw}
        tu_stats = tu_policy.get_stats()
        print(f"  Total Uncertainty: {tu_sr:.1%} success, {tu_rw:.2f} avg reward")
        print(f"  Intervention: {tu_stats['intervene']['fraction']:.1%} intervene, {tu_stats['normal']['fraction']:.1%} normal")

    # --- Decomposed (Ours) ---
    print(f"\n{'='*60}")
    print("EVALUATING: DECOMPOSED (Ours)")
    print(f"{'='*60}")
    env.reset()
    controller.reset_stats()

    decomposed_policy = DecomposedPolicy(
        policy_actor=policy_nn.actor,
        policy_obs_normalizer=policy_nn.actor_obs_normalizer,
        alea_estimator=alea_est,
        epis_estimator=epis_est,
        controller=controller,
        env=env,
        num_samples=args_cli.num_samples,
        noise_params=noise_params,
        task_cfg=_task_cfg,
    )

    decomposed_results = run_evaluation(env, decomposed_policy, args_cli.num_episodes, "Decomposed")

    decomposed_sr = sum(1 for r in decomposed_results if r.success) / len(decomposed_results)
    decomposed_rw = np.mean([r.reward for r in decomposed_results])
    all_results['decomposed'] = {'success_rate': decomposed_sr, 'avg_reward': decomposed_rw}
    print(f"  Decomposed: {decomposed_sr:.1%} success, {decomposed_rw:.2f} avg reward")

    # Intervention statistics
    intervention_stats = controller.get_stats()
    print(f"\n  Intervention breakdown:")
    for name, stats in intervention_stats.items():
        print(f"    {name}: {stats['count']} ({stats['fraction']:.1%})")

    # =====================================================================
    # PHASE 5: Summary
    # =====================================================================
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Success Rate':<15} {'Avg Reward':<15}")
    print("-" * 55)

    for method, res in all_results.items():
        print(f"{method:<25} {res['success_rate']*100:>12.1f}% {res['avg_reward']:>13.2f}")

    print("-" * 55)

    if 'vanilla' in all_results:
        improvement = (all_results['decomposed']['success_rate'] -
                      all_results['vanilla']['success_rate']) * 100
        print(f"\n  Decomposed vs Vanilla: {improvement:+.1f}% success rate")

    if 'multi_sample' in all_results:
        improvement = (all_results['decomposed']['success_rate'] -
                      all_results['multi_sample']['success_rate']) * 100
        print(f"  Decomposed vs Multi-Sample: {improvement:+.1f}% success rate")

    bi = ortho_results.get('behavioral_isolation', {})
    bi_pass = bi.get('overall_pass', False)
    print(f"\n  Behavioral Isolation: {'PASS' if bi_pass else 'FAIL'}")
    if bi:
        print(f"    Noise isolation ratio: {bi['noise_isolation_ratio']:.1f}x")
        print(f"    Noise→σ_epis flat: {'PASS' if bi['noise_pass'] else 'FAIL'} (Δ = {bi.get('noise_epis_range', 0):.6f})")
        print(f"    OOD→σ_epis rises: {'PASS' if bi['ood_pass'] else 'FAIL'} (Δ = {bi.get('ood_epis_range', 0):.4f})")
    print(f"  Statistical Orthogonality (noise sweep only):")
    print(f"    Pearson |r|: {ortho_results['pearson_abs_r']:.4f}")
    print(f"    Spearman |rho|: {ortho_results['spearman_abs_rho']:.4f}")
    print(f"    CKA: {ortho_results['cka']:.4f}")

    print(f"{'='*70}\n")

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"decomposed_{noise_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    save_data = {
        "config": {
            "task": args_cli.task,
            "noise_level": noise_level,
            "noise_params": noise_params,
            "num_samples": args_cli.num_samples,
            "tau_a": args_cli.tau_a,
            "tau_e": args_cli.tau_e,
            "beta": args_cli.beta,
            "num_episodes": args_cli.num_episodes,
            "seed": args_cli.seed,
        },
        "results": all_results,
        "orthogonality": {k: v for k, v in ortho_results.items()
                         if not isinstance(v, np.ndarray)},
        "intervention_stats": {k.value if hasattr(k, 'value') else str(k): v
                              for k, v in intervention_stats.items()},
    }

    def json_safe(obj):
        """Convert numpy types to Python natives for JSON serialization."""
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        return obj

    save_data = json_safe(save_data)

    with open(os.path.join(output_dir, "decomposed_results.json"), 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to: {output_dir}")

    env.close()
    return all_results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
