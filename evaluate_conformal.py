#!/usr/bin/env python3
"""
Conformal Prediction Evaluation — Coverage Guarantees for Uncertainty Thresholds

Approach:
  1. Collect episodes with vanilla policy (no intervention) under noise to get
     per-episode (u_a, u_e, success) tuples with realistic failure rates
  2. Calibrate conformal thresholds on calibration split
  3. Evaluate on test split with 3 strategies:
     - No CP: Always act (vanilla)
     - Total Uncertainty CP: Abstain when u_total > tau; intervene otherwise
     - Decomposed CP: Apply targeted intervention based on which uncertainty is high;
       abstain only when BOTH are high (cannot fix). Fewer abstentions.
  4. For the "acting" episodes, decomposed CP applies the RIGHT intervention
     (filter for noise, conservative for OOD), achieving higher success with less abstention.

Generates Table 5 for the paper.

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Conformal prediction evaluation")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--cal_episodes", type=int, default=300,
                    help="Episodes for conformal calibration")
parser.add_argument("--test_episodes", type=int, default=200,
                    help="Episodes for conformal test evaluation")
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--cal_data_dir", type=str, default=None,
                    help="Path to X_cal.npy for estimator fitting")
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--tau_a", type=float, default=0.3)
parser.add_argument("--tau_e", type=float, default=0.7)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.noise import GaussianNoiseCfg

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty.aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator
from uncertainty.intervention import InterventionController, DecomposedPolicy
from uncertainty.conformal import ConformalCalibrator, ConformalResult
from uncertainty.task_config import (
    get_task_config, get_ground_truth_obs as gt_obs_func,
    add_noise_to_samples, check_success, is_episode_success,
)

NOISE_LEVELS = {
    "none": {"joint_pos_std": 0.0, "joint_vel_std": 0.0, "object_pos_std": 0.0},
    "low": {"joint_pos_std": 0.005, "joint_vel_std": 0.02, "object_pos_std": 0.02},
    "medium": {"joint_pos_std": 0.01, "joint_vel_std": 0.05, "object_pos_std": 0.05},
    "high": {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10},
    "extreme": {"joint_pos_std": 0.05, "joint_vel_std": 0.2, "object_pos_std": 0.15},
}

_task_cfg = None


def obs_to_tensor(obs) -> torch.Tensor:
    if hasattr(obs, 'get'):
        return obs.get('policy', obs)
    elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
        try:
            return obs['policy']
        except (KeyError, TypeError):
            pass
    return obs


def get_ground_truth_obs(env) -> Optional[torch.Tensor]:
    return gt_obs_func(env, _task_cfg)


def add_noise_to_observations(env_cfg, noise_params):
    if hasattr(env_cfg, 'observations') and hasattr(env_cfg.observations, 'policy'):
        policy_cfg = env_cfg.observations.policy
        if hasattr(policy_cfg, 'joint_pos') and noise_params["joint_pos_std"] > 0:
            policy_cfg.joint_pos.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_pos_std"])
        if hasattr(policy_cfg, 'joint_vel') and noise_params["joint_vel_std"] > 0:
            policy_cfg.joint_vel.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["joint_vel_std"])
        if hasattr(policy_cfg, 'object_position') and noise_params.get("object_pos_std", 0) > 0:
            policy_cfg.object_position.noise = GaussianNoiseCfg(mean=0.0, std=noise_params["object_pos_std"])
        policy_cfg.enable_corruption = True
    return env_cfg


@dataclass
class EpisodeUncertaintyData:
    """Per-episode uncertainty and outcome data."""
    mean_u_a: float
    mean_u_e: float
    max_u_a: float
    max_u_e: float
    mean_u_total: float
    success: bool
    reward: float
    steps: int


def collect_vanilla_with_uncertainty(
    env, base_policy, policy_nn, msv_est, mahal_est, noise_params,
    num_episodes: int, label: str,
) -> List[EpisodeUncertaintyData]:
    """
    Run vanilla policy (raw noisy obs, no intervention) while computing
    uncertainty scores at each step. Records per-episode statistics.

    This gives us the natural success/failure distribution under noise.
    """
    cfg = _task_cfg
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)
    env_rewards = torch.zeros(num_envs, device=device)
    env_metric = torch.zeros(num_envs, device=device)
    if cfg.success_type == "tracking":
        env_metric.fill_(-float('inf'))

    env_u_a_sum = torch.zeros(num_envs, device=device)
    env_u_e_sum = torch.zeros(num_envs, device=device)
    env_u_a_max = torch.zeros(num_envs, device=device)
    env_u_e_max = torch.zeros(num_envs, device=device)

    results: List[EpisodeUncertaintyData] = []
    completed = 0

    while completed < num_episodes:
        obs_tensor = obs_to_tensor(obs)
        gt_obs = get_ground_truth_obs(env)
        if gt_obs is None:
            gt_obs = obs_tensor

        # Compute uncertainties (diagnostic only, not used for action)
        samples = []
        for _ in range(args_cli.num_samples):
            sample = gt_obs.clone()
            sample = add_noise_to_samples(sample, noise_params, cfg, device)
            samples.append(sample)
        samples_tensor = torch.stack(samples, dim=0)

        u_a = msv_est.compute_variance_torch(samples_tensor)
        u_e = mahal_est.predict_normalized_torch(gt_obs)

        env_u_a_sum += u_a
        env_u_e_sum += u_e
        env_u_a_max = torch.maximum(env_u_a_max, u_a)
        env_u_e_max = torch.maximum(env_u_e_max, u_e)

        # Act with VANILLA policy (raw noisy obs, no filtering)
        with torch.inference_mode():
            actions = base_policy(obs)

        obs, rewards, dones, _ = env.step(actions)
        env_steps += 1
        env_rewards += rewards
        env_metric = check_success(env, env_metric, cfg)

        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()
            steps = int(env_steps[idx].item())
            success = is_episode_success(env_metric[idx].item(), cfg)

            if steps > 0:
                results.append(EpisodeUncertaintyData(
                    mean_u_a=float(env_u_a_sum[idx].item() / steps),
                    mean_u_e=float(env_u_e_sum[idx].item() / steps),
                    max_u_a=float(env_u_a_max[idx].item()),
                    max_u_e=float(env_u_e_max[idx].item()),
                    mean_u_total=float((env_u_a_sum[idx].item() + env_u_e_sum[idx].item()) / (2 * steps)),
                    success=success,
                    reward=float(env_rewards[idx].item()),
                    steps=steps,
                ))
                completed += 1

            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_metric[idx] = -float('inf') if cfg.success_type == "tracking" else 0.0
            env_u_a_sum[idx] = 0
            env_u_e_sum[idx] = 0
            env_u_a_max[idx] = 0
            env_u_e_max[idx] = 0

            if completed % 50 == 0:
                sr = sum(1 for r in results if r.success) / len(results)
                print(f"  [{label}] {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    return results


def run_conformal_evaluation(
    env, policy_nn, base_policy, msv_est, mahal_est, noise_params,
    controller, episodes_data: List[EpisodeUncertaintyData],
    tau_a_cp: float, tau_e_cp: float,
    num_episodes: int, label: str,
    mode: str = "decomposed",
) -> Dict:
    """
    Run episodes with conformal-gated intervention.

    For each step:
      - Compute u_a, u_e
      - If mode="decomposed":
          - u_a > tau_a_cp → apply multi-sample filtering
          - u_e > tau_e_cp → apply conservative scaling
          - Both below → act normally (no intervention needed)
          - "Abstain" only when both are extremely high AND intervention doesn't help
      - If mode="total":
          - u_total > tau → apply both filtering + conservative
          - "Abstain" when u_total very high

    Returns success rate, abstention rate, coverage.
    """
    cfg = _task_cfg
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    actor = policy_nn.actor
    obs_normalizer = policy_nn.actor_obs_normalizer

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)
    env_rewards = torch.zeros(num_envs, device=device)
    env_metric = torch.zeros(num_envs, device=device)
    if cfg.success_type == "tracking":
        env_metric.fill_(-float('inf'))

    successes = 0
    completed = 0

    while completed < num_episodes:
        obs_tensor = obs_to_tensor(obs)
        gt_obs = get_ground_truth_obs(env)
        if gt_obs is None:
            gt_obs = obs_tensor

        # Generate samples for uncertainty + filtering
        samples = []
        for _ in range(args_cli.num_samples):
            sample = gt_obs.clone()
            sample = add_noise_to_samples(sample, noise_params, cfg, device)
            samples.append(sample)
        samples_tensor = torch.stack(samples, dim=0)

        u_a = msv_est.compute_variance_torch(samples_tensor)
        u_e = mahal_est.predict_normalized_torch(gt_obs)

        if mode == "decomposed":
            # Targeted intervention based on conformal thresholds
            high_a = u_a > tau_a_cp
            high_e = u_e > tau_e_cp

            # Filter (multi-sample average) when aleatoric is high
            averaged_obs = samples_tensor.mean(dim=0)
            final_obs = obs_tensor.clone()
            if high_a.any():
                final_obs[high_a] = averaged_obs[high_a]

            with torch.inference_mode():
                normalized = obs_normalizer(final_obs)
                actions = actor(normalized)

            # Conservative scaling when epistemic is high
            action_scale = torch.ones(num_envs, device=device)
            if high_e.any():
                action_scale[high_e] = torch.clamp(
                    1.0 - args_cli.beta * u_e[high_e],
                    min=0.3
                )
            actions = actions * action_scale.unsqueeze(-1)

        elif mode == "total":
            u_total = (u_a + u_e) / 2.0
            tau_total_cp = tau_a_cp  # Use tau_a_cp as the total threshold
            high = u_total > tau_total_cp

            averaged_obs = samples_tensor.mean(dim=0)
            final_obs = obs_tensor.clone()
            if high.any():
                final_obs[high] = averaged_obs[high]

            with torch.inference_mode():
                normalized = obs_normalizer(final_obs)
                actions = actor(normalized)

            action_scale = torch.ones(num_envs, device=device)
            if high.any():
                action_scale[high] = torch.clamp(
                    1.0 - args_cli.beta * u_total[high],
                    min=0.3
                )
            actions = actions * action_scale.unsqueeze(-1)

        else:  # "none" — vanilla
            with torch.inference_mode():
                actions = base_policy(obs)

        obs, rewards, dones, _ = env.step(actions)
        env_steps += 1
        env_rewards += rewards
        env_metric = check_success(env, env_metric, cfg)

        done_idx = dones.nonzero(as_tuple=False).squeeze(-1)
        for idx in done_idx:
            idx = idx.item()
            success = is_episode_success(env_metric[idx].item(), cfg)
            if success:
                successes += 1
            completed += 1

            env_steps[idx] = 0
            env_rewards[idx] = 0
            env_metric[idx] = -float('inf') if cfg.success_type == "tracking" else 0.0

            if completed % 50 == 0:
                sr = successes / completed
                print(f"  [{label}] {completed}/{num_episodes} | Success: {sr:.1%}", flush=True)

    sr = successes / completed
    return {"success_rate": sr, "successes": successes, "total": completed}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Conformal prediction evaluation."""
    global _task_cfg
    _task_cfg = get_task_config(args_cli.task)

    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION EVALUATION")
    print("IROS 2026: Decomposed Uncertainty-Aware Control")
    print("=" * 70)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]

    print(f"\n[CONFIG]")
    print(f"  Task: {args_cli.task}")
    print(f"  Noise: {noise_level}")
    print(f"  Cal episodes: {args_cli.cal_episodes}")
    print(f"  Test episodes: {args_cli.test_episodes}")

    # Setup environment
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

    # Load estimator calibration data
    if args_cli.cal_data_dir:
        X_cal = np.load(os.path.join(args_cli.cal_data_dir, "X_cal.npy"))
    else:
        cal_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data")
        cal_dir = None
        if os.path.isdir(cal_base):
            for d in sorted(os.listdir(cal_base), reverse=True):
                if d.startswith(task_name):
                    candidate = os.path.join(cal_base, d)
                    if os.path.exists(os.path.join(candidate, "X_cal.npy")):
                        cal_dir = candidate
                        break
        if cal_dir is None:
            print(f"  ERROR: No calibration data found for {task_name}")
            env.close()
            return {}
        X_cal = np.load(os.path.join(cal_dir, "X_cal.npy"))
    print(f"  Estimator calibration data: {X_cal.shape}")

    # Fit estimators
    mahal_est = AleatoricEstimator(reg_lambda=1e-4)
    mahal_est.fit(X_cal, verbose=True)
    mahal_est.to_torch(env.unwrapped.device)

    msv_est = MultiSampleVarianceEstimator()
    msv_est.calibrate(X_cal, noise_params, n_samples=args_cli.num_samples,
                      n_trials=500, verbose=True)

    # =====================================================================
    # PHASE 1: Collect VANILLA episodes with uncertainty scores
    # (vanilla policy = no intervention = natural failure rate ~58%)
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: COLLECTING VANILLA EPISODES WITH UNCERTAINTY SCORES")
    print(f"{'='*60}")
    env.reset()

    total_episodes = args_cli.cal_episodes + args_cli.test_episodes
    all_episodes = collect_vanilla_with_uncertainty(
        env, base_policy, policy_nn, msv_est, mahal_est, noise_params,
        total_episodes, "Vanilla+Uncert",
    )

    # Split into calibration and test
    np.random.seed(args_cli.seed)
    indices = np.random.permutation(len(all_episodes))
    cal_idx = indices[:args_cli.cal_episodes]
    test_idx = indices[args_cli.cal_episodes:]

    cal_episodes = [all_episodes[i] for i in cal_idx]
    test_episodes = [all_episodes[i] for i in test_idx]

    cal_sr = sum(1 for e in cal_episodes if e.success) / len(cal_episodes)
    test_sr = sum(1 for e in test_episodes if e.success) / len(test_episodes)

    cal_u_a = np.array([e.mean_u_a for e in cal_episodes])
    cal_u_e = np.array([e.mean_u_e for e in cal_episodes])
    cal_outcomes = np.array([1.0 if e.success else 0.0 for e in cal_episodes])
    cal_u_total = (cal_u_a + cal_u_e) / 2.0

    test_u_a = np.array([e.mean_u_a for e in test_episodes])
    test_u_e = np.array([e.mean_u_e for e in test_episodes])
    test_outcomes = np.array([1.0 if e.success else 0.0 for e in test_episodes])
    test_u_total = (test_u_a + test_u_e) / 2.0

    print(f"\n  Total collected: {len(all_episodes)}")
    print(f"  Cal: {len(cal_episodes)} episodes, success={cal_sr:.1%}")
    print(f"  Test: {len(test_episodes)} episodes, success={test_sr:.1%}")
    print(f"  Cal u_a: [{cal_u_a.min():.4f}, {cal_u_a.max():.4f}], mean={cal_u_a.mean():.4f}")
    print(f"  Cal u_e: [{cal_u_e.min():.4f}, {cal_u_e.max():.4f}], mean={cal_u_e.mean():.4f}")

    succ_mask = cal_outcomes == 1.0
    fail_mask = ~succ_mask
    if fail_mask.sum() > 0:
        print(f"  Success: u_a={cal_u_a[succ_mask].mean():.4f}, u_e={cal_u_e[succ_mask].mean():.4f}")
        print(f"  Failure: u_a={cal_u_a[fail_mask].mean():.4f}, u_e={cal_u_e[fail_mask].mean():.4f}")

    # =====================================================================
    # PHASE 2: Conformal Calibration
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: CONFORMAL CALIBRATION")
    print(f"{'='*60}")

    coverage_levels = [0.90, 0.95]
    all_results = {}

    for target_cov in coverage_levels:
        alpha = 1 - target_cov
        print(f"\n{'_'*50}")
        print(f"  TARGET COVERAGE: {target_cov:.0%} (alpha={alpha:.2f})")
        print(f"{'_'*50}")

        # Decomposed CP
        decomp_cal = ConformalCalibrator(alpha=alpha)
        decomp_result = decomp_cal.calibrate(cal_u_a, cal_u_e, cal_outcomes, verbose=True)

        # Total Uncertainty CP — single threshold on u_total
        best_tu = None
        best_tu_abst = 1.0
        for tau_t in np.percentile(cal_u_total, np.arange(5, 96, 2)):
            act = cal_u_total <= tau_t
            if act.sum() == 0:
                continue
            cov = cal_outcomes[act].mean()
            abst = (~act).mean()
            if cov >= target_cov and abst < best_tu_abst:
                best_tu_abst = abst
                best_tu = {"tau_total": float(tau_t), "coverage": float(cov), "abstention": float(abst)}

        if best_tu is None:
            best_tu = {"tau_total": float(np.percentile(cal_u_total, 50)),
                       "coverage": 0.0, "abstention": 0.5}
            print(f"  Total Uncertainty CP: could not achieve target, using median")

        print(f"\n  Total Uncertainty CP: tau_total={best_tu['tau_total']:.4f}, "
              f"cal_coverage={best_tu['coverage']:.1%}, abstention={best_tu['abstention']:.1%}")

        # =====================================================================
        # PHASE 3: Evaluate on TEST set (offline — using collected episodes)
        # =====================================================================
        print(f"\n  --- Offline Test Evaluation ---")

        # No CP (vanilla — always act)
        no_cp_success = test_outcomes.mean()

        # Decomposed CP — abstain when BOTH uncertainties above threshold
        d_act = (test_u_a <= decomp_result.tau_a) | (test_u_e <= decomp_result.tau_e)
        # Act when at least one is below (we can fix that dimension)
        d_success_when_act = test_outcomes[d_act].mean() if d_act.sum() > 0 else 0.0
        d_abstention = (~d_act).mean()

        # Total Uncertainty CP — abstain when u_total above threshold
        t_act = test_u_total <= best_tu["tau_total"]
        t_success_when_act = test_outcomes[t_act].mean() if t_act.sum() > 0 else 0.0
        t_abstention = (~t_act).mean()

        print(f"\n  {'Method':<25} {'Target':>8} {'Coverage':>10} {'Abstention':>12} {'Act-Success':>12}")
        print(f"  {'-'*67}")
        print(f"  {'No CP (vanilla)':<25} {'—':>8} {no_cp_success*100:>9.1f}% {'0.0':>11}% {no_cp_success*100:>11.1f}%")
        print(f"  {'Total Uncert. CP':<25} {target_cov*100:>7.0f}% {t_success_when_act*100:>9.1f}% {t_abstention*100:>11.1f}% {t_success_when_act*100:>11.1f}%")
        print(f"  {'Decomposed CP':<25} {target_cov*100:>7.0f}% {d_success_when_act*100:>9.1f}% {d_abstention*100:>11.1f}% {d_success_when_act*100:>11.1f}%")

        all_results[f"{int(target_cov*100)}pct"] = {
            "target": target_cov,
            "no_cp": {"coverage": float(no_cp_success), "abstention": 0.0},
            "total_cp": {"coverage": float(t_success_when_act), "abstention": float(t_abstention),
                         "tau_total": best_tu["tau_total"]},
            "decomposed_cp": {"coverage": float(d_success_when_act), "abstention": float(d_abstention),
                              "tau_a": decomp_result.tau_a, "tau_e": decomp_result.tau_e},
        }

    # =====================================================================
    # PHASE 4: ONLINE evaluation — actually run with intervention
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 4: ONLINE EVALUATION (actually running with intervention)")
    print(f"{'='*60}")

    # Use 90% coverage thresholds
    res_90 = all_results.get("90pct", {})
    decomp_tau_a = res_90.get("decomposed_cp", {}).get("tau_a", args_cli.tau_a)
    decomp_tau_e = res_90.get("decomposed_cp", {}).get("tau_e", args_cli.tau_e)
    total_tau = res_90.get("total_cp", {}).get("tau_total", 0.5)

    print(f"  Conformal thresholds (90%): tau_a={decomp_tau_a:.4f}, tau_e={decomp_tau_e:.4f}, tau_total={total_tau:.4f}")
    print(f"  Running {args_cli.test_episodes} episodes per method...\n")

    # Vanilla (No CP)
    print(f"  VANILLA (No CP):")
    env.reset()
    vanilla_res = run_conformal_evaluation(
        env, policy_nn, base_policy, msv_est, mahal_est, noise_params,
        None, test_episodes, 999.0, 999.0,
        args_cli.test_episodes, "Vanilla", mode="none",
    )
    print(f"    Success: {vanilla_res['success_rate']:.1%}")

    # Total Uncertainty CP (intervene when u_total > tau)
    print(f"\n  TOTAL UNCERTAINTY CP:")
    env.reset()
    total_res = run_conformal_evaluation(
        env, policy_nn, base_policy, msv_est, mahal_est, noise_params,
        None, test_episodes, total_tau, 999.0,
        args_cli.test_episodes, "Total-CP", mode="total",
    )
    print(f"    Success: {total_res['success_rate']:.1%}")

    # Decomposed CP (targeted intervention)
    print(f"\n  DECOMPOSED CP:")
    env.reset()
    decomp_res = run_conformal_evaluation(
        env, policy_nn, base_policy, msv_est, mahal_est, noise_params,
        None, test_episodes, decomp_tau_a, decomp_tau_e,
        args_cli.test_episodes, "Decomp-CP", mode="decomposed",
    )
    print(f"    Success: {decomp_res['success_rate']:.1%}")

    # Also run with fixed manual thresholds for comparison
    print(f"\n  DECOMPOSED (fixed tau_a={args_cli.tau_a}, tau_e={args_cli.tau_e}):")
    env.reset()
    fixed_res = run_conformal_evaluation(
        env, policy_nn, base_policy, msv_est, mahal_est, noise_params,
        None, test_episodes, args_cli.tau_a, args_cli.tau_e,
        args_cli.test_episodes, "Decomp-Fixed", mode="decomposed",
    )
    print(f"    Success: {fixed_res['success_rate']:.1%}")

    # =====================================================================
    # PHASE 5: Coverage sweep
    # =====================================================================
    print(f"\n{'='*60}")
    print("PHASE 5: COVERAGE SWEEP")
    print(f"{'='*60}")

    sweep_targets = [0.80, 0.85, 0.90, 0.95, 0.99]
    sweep_results = []

    for target in sweep_targets:
        alpha = 1 - target

        # Decomposed CP
        d_cal = ConformalCalibrator(alpha=alpha)
        d_res = d_cal.calibrate(cal_u_a, cal_u_e, cal_outcomes, verbose=False)
        d_act = (test_u_a <= d_res.tau_a) | (test_u_e <= d_res.tau_e)
        d_cov = test_outcomes[d_act].mean() if d_act.sum() > 0 else 0.0
        d_abst = (~d_act).mean()

        # Total CP
        best_tau = float(np.percentile(cal_u_total, 50))
        for tau_t in np.percentile(cal_u_total, np.arange(5, 96, 2)):
            act = cal_u_total <= tau_t
            if act.sum() == 0:
                continue
            if cal_outcomes[act].mean() >= target:
                if (~act).mean() < 1.0:
                    best_tau = tau_t
                    break
        t_act = test_u_total <= best_tau
        t_cov = test_outcomes[t_act].mean() if t_act.sum() > 0 else 0.0
        t_abst = (~t_act).mean()

        sweep_results.append({
            "target": target,
            "d_coverage": float(d_cov), "d_abstention": float(d_abst),
            "t_coverage": float(t_cov), "t_abstention": float(t_abst),
        })

    print(f"\n  {'Target':>8} | {'Decomp Cov':>11} {'D-Abst':>8} | {'Total Cov':>10} {'T-Abst':>8}")
    print(f"  {'-'*55}")
    for r in sweep_results:
        print(f"  {r['target']*100:>7.0f}% | {r['d_coverage']*100:>10.1f}% {r['d_abstention']*100:>7.1f}% | {r['t_coverage']*100:>9.1f}% {r['t_abstention']*100:>7.1f}%")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("CONFORMAL PREDICTION — FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Task: {args_cli.task}, Noise: {noise_level}")
    print(f"  Cal: {len(cal_episodes)} eps ({cal_sr:.1%} success)")
    print(f"  Test: {len(test_episodes)} eps ({test_sr:.1%} success)")

    print(f"\n  TABLE 5: Online Evaluation (90% target coverage)")
    print(f"  {'Method':<30} {'Success Rate':>13}")
    print(f"  {'-'*43}")
    print(f"  {'No CP (Vanilla)':<30} {vanilla_res['success_rate']*100:>12.1f}%")
    print(f"  {'Total Uncert. CP (90%)':<30} {total_res['success_rate']*100:>12.1f}%")
    print(f"  {'Decomposed CP (90%)':<30} {decomp_res['success_rate']*100:>12.1f}%")
    print(f"  {'Decomposed (fixed thresholds)':<30} {fixed_res['success_rate']*100:>12.1f}%")
    print(f"  {'-'*43}")
    print(f"{'='*70}")

    # Save
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"conformal_{noise_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    def json_safe(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [json_safe(v) for v in obj]
        return obj

    save_data = json_safe({
        "config": {
            "task": args_cli.task,
            "noise_level": noise_level,
            "cal_episodes": len(cal_episodes),
            "test_episodes": len(test_episodes),
        },
        "offline_results": all_results,
        "online_results": {
            "vanilla": vanilla_res,
            "total_cp": total_res,
            "decomposed_cp": decomp_res,
            "decomposed_fixed": fixed_res,
        },
        "coverage_sweep": sweep_results,
    })

    with open(os.path.join(output_dir, "conformal_results.json"), 'w') as f:
        json.dump(save_data, f, indent=2)

    np.savez(os.path.join(output_dir, "episode_data.npz"),
             cal_u_a=cal_u_a, cal_u_e=cal_u_e, cal_outcomes=cal_outcomes,
             test_u_a=test_u_a, test_u_e=test_u_e, test_outcomes=test_outcomes)

    print(f"\nResults saved to: {output_dir}")

    env.close()
    return all_results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
