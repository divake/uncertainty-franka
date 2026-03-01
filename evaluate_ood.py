#!/usr/bin/env python3
"""
OOD Perturbation Evaluation — Tests Decomposed vs Multi-Sample under epistemic shifts.

This is the KEY experiment for the paper: under pure noise, both methods perform
similarly. Under OOD perturbations, decomposed should BEAT multi-sample because
conservative scaling helps when the robot is in unfamiliar states.

Uses a SINGLE environment with PhysX property restoration between scenarios
to avoid the ~60s cost of gym.make() per scenario.

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OOD perturbation evaluation")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_episodes", type=int, default=100)
parser.add_argument("--noise_level", type=str, default="high",
                    choices=["none", "low", "medium", "high", "extreme"])
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--cal_data_dir", type=str, default=None)
parser.add_argument("--tau_a", type=float, default=0.3)
parser.add_argument("--tau_e", type=float, default=0.7)
parser.add_argument("--beta", type=float, default=0.3)
parser.add_argument("--tau_total", type=float, default=0.3,
                    help="Threshold for Total Uncertainty baseline")

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
from typing import Dict, List, Optional
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
from uncertainty.intervention import InterventionController, DecomposedPolicy, TotalUncertaintyPolicy
from uncertainty.task_config import (
    get_task_config, get_ground_truth_obs as gt_obs_func,
    add_noise_to_samples, check_success, is_episode_success,
    get_ood_scenarios,
)

# Task-specific config (set in main())
_task_cfg = None

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


def get_ground_truth_obs(env) -> Optional[torch.Tensor]:
    """Get ground truth observation (task-aware)."""
    return gt_obs_func(env, _task_cfg)


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    max_height: float
    reward: float


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


class MultiSamplePolicy:
    def __init__(self, policy_nn, env, num_samples, noise_params):
        self.actor = policy_nn.actor
        self.obs_normalizer = policy_nn.actor_obs_normalizer
        self.env = env
        self.num_samples = num_samples
        self.noise_params = noise_params
        self.device = env.unwrapped.device

    def __call__(self, obs):
        obs_tensor = obs_to_tensor(obs)
        gt_obs = get_ground_truth_obs(self.env)
        if gt_obs is None:
            gt_obs = obs_tensor
        samples = []
        for _ in range(self.num_samples):
            sample = gt_obs.clone()
            sample = add_noise_to_samples(sample, self.noise_params, _task_cfg, self.device)
            samples.append(sample)
        averaged = torch.stack(samples, dim=0).mean(dim=0)
        with torch.inference_mode():
            normalized = self.obs_normalizer(averaged)
            return self.actor(normalized)

    def reset(self, dones):
        pass


def run_ood_evaluation(env, policy, num_episodes: int, label: str) -> List[EpisodeResult]:
    """Run evaluation — OOD perturbations already applied to the env (task-agnostic)."""
    cfg = _task_cfg
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    obs = env.get_observations()
    env_steps = torch.zeros(num_envs, device=device)
    env_rewards = torch.zeros(num_envs, device=device)
    env_metric = torch.zeros(num_envs, device=device)
    if cfg.success_type == "tracking":
        env_metric.fill_(-float('inf'))

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


def add_noise_to_observations(env_cfg, noise_params):
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


# =========================================================================
# PhysX property snapshot / restore — avoids creating new environments
# =========================================================================

def snapshot_physx_properties(env, task_cfg):
    """Capture original PhysX properties so we can restore between scenarios."""
    import carb
    from isaaclab.sim import SimulationContext

    unwrapped = env.unwrapped
    snapshot = {}
    num_envs = unwrapped.num_envs
    snapshot["physx_indices"] = torch.arange(num_envs, dtype=torch.int32, device="cpu")

    if task_cfg.has_object:
        try:
            obj = unwrapped.scene.rigid_objects["object"]
            snapshot["object_masses"] = obj.root_physx_view.get_masses().clone()
            snapshot["object_materials"] = obj.root_physx_view.get_material_properties().clone()
        except Exception as e:
            print(f"  Warning: could not snapshot object properties: {e}")

    # Robot joint properties (for Reach OOD: damping changes)
    try:
        robot = unwrapped.scene.articulations["robot"]
        snapshot["robot_joint_damping"] = robot.root_physx_view.get_dof_dampings().clone()
    except Exception as e:
        print(f"  Warning: could not snapshot robot joint properties: {e}")

    try:
        sim_ctx = SimulationContext.instance()
        snapshot["gravity"] = list(sim_ctx.cfg.gravity)
    except Exception as e:
        print(f"  Warning: could not snapshot gravity: {e}")
    return snapshot


def restore_physx_properties(env, snapshot, task_cfg):
    """Restore PhysX properties to their original values."""
    import carb
    from isaaclab.sim import SimulationContext

    unwrapped = env.unwrapped
    idx = snapshot["physx_indices"]

    if task_cfg.has_object:
        try:
            obj = unwrapped.scene.rigid_objects["object"]
            if "object_masses" in snapshot:
                obj.root_physx_view.set_masses(snapshot["object_masses"].clone(), idx)
            if "object_materials" in snapshot:
                obj.root_physx_view.set_material_properties(snapshot["object_materials"].clone(), idx)
        except Exception as e:
            print(f"  Warning: could not restore object properties: {e}")

    try:
        robot = unwrapped.scene.articulations["robot"]
        if "robot_joint_damping" in snapshot:
            robot.root_physx_view.set_dof_dampings(snapshot["robot_joint_damping"].clone(), idx)
    except Exception as e:
        print(f"  Warning: could not restore robot joint properties: {e}")

    try:
        if "gravity" in snapshot:
            sim_ctx = SimulationContext.instance()
            sim_ctx.physics_sim_view.set_gravity(carb.Float3(*snapshot["gravity"]))
    except Exception as e:
        print(f"  Warning: could not restore gravity: {e}")


def apply_ood_perturbation(env, ood_type, ood_params, snapshot, task_cfg):
    """Apply a single OOD perturbation from original baseline (not cumulative)."""
    import carb
    from isaaclab.sim import SimulationContext

    restore_physx_properties(env, snapshot, task_cfg)

    unwrapped = env.unwrapped
    idx = snapshot["physx_indices"]

    if ood_type == "mass" and task_cfg.has_object:
        scale = ood_params.get("scale", 5.0)
        obj = unwrapped.scene.rigid_objects["object"]
        obj.root_physx_view.set_masses(snapshot["object_masses"].clone() * scale, idx)
        print(f"  Applied mass change: {scale}x")
    elif ood_type == "friction" and task_cfg.has_object:
        scale = ood_params.get("scale", 0.2)
        obj = unwrapped.scene.rigid_objects["object"]
        materials = snapshot["object_materials"].clone()
        materials[:, :, 0] *= scale  # static friction
        materials[:, :, 1] *= scale  # dynamic friction
        obj.root_physx_view.set_material_properties(materials, idx)
        print(f"  Applied friction change: {scale}x")
    elif ood_type == "gravity":
        scale = ood_params.get("scale", 1.5)
        original_gravity = snapshot["gravity"]
        new_gravity = [g * scale for g in original_gravity]
        sim_ctx = SimulationContext.instance()
        sim_ctx.physics_sim_view.set_gravity(carb.Float3(*new_gravity))
        print(f"  Applied gravity change: {scale}x ({new_gravity})")
    elif ood_type == "joint_damping":
        scale = ood_params.get("scale", 3.0)
        robot = unwrapped.scene.articulations["robot"]
        robot.root_physx_view.set_dof_dampings(snapshot["robot_joint_damping"].clone() * scale, idx)
        print(f"  Applied joint damping change: {scale}x")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """OOD perturbation evaluation using a SINGLE environment."""
    global _task_cfg
    _task_cfg = get_task_config(args_cli.task)

    # Get task-appropriate OOD scenarios
    OOD_SCENARIOS = get_ood_scenarios(_task_cfg)

    print("\n" + "=" * 70)
    print("OOD PERTURBATION EVALUATION")
    print("IROS 2026: Decomposed Uncertainty-Aware Control")
    print("=" * 70)

    noise_level = args_cli.noise_level
    noise_params = NOISE_LEVELS[noise_level]

    print(f"\n[CONFIG]")
    print(f"  Task: {args_cli.task}")
    print(f"  Noise: {noise_level}")
    print(f"  Thresholds: tau_a={args_cli.tau_a}, tau_e={args_cli.tau_e}, beta={args_cli.beta}")
    print(f"  Episodes per scenario: {args_cli.num_episodes}")
    print(f"  Scenarios: {len(OOD_SCENARIOS)}")

    # Load calibration data
    task_name = args_cli.task.split(":")[-1]
    if args_cli.cal_data_dir:
        X_cal = np.load(os.path.join(args_cli.cal_data_dir, "X_cal.npy"))
    else:
        # Search for calibration data matching the task
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
            print(f"  Run: python collect_calibration_data.py --task {args_cli.task} --headless --num_envs 32")
            env.close()
            return {}
        X_cal = np.load(os.path.join(cal_dir, "X_cal.npy"))
    print(f"  Calibration data: {X_cal.shape}")

    # Fit estimators
    mahal_est = AleatoricEstimator(reg_lambda=1e-4)
    mahal_est.fit(X_cal, verbose=False)

    msv_est = MultiSampleVarianceEstimator()
    msv_est.calibrate(X_cal, noise_params, n_samples=args_cli.num_samples, n_trials=500, verbose=False)

    # =====================================================================
    # Create ONE environment (the expensive part — only done once)
    # =====================================================================
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    if noise_level != "none":
        env_cfg = add_noise_to_observations(env_cfg, noise_params)

    task_name = args_cli.task.split(":")[-1]
    checkpoint_path = get_published_pretrained_checkpoint("rsl_rl", task_name)

    print(f"\n  Creating environment (one-time cost)...")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    base_policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    mahal_est.to_torch(env.unwrapped.device)

    # Snapshot original PhysX properties BEFORE any perturbations
    env.reset()
    physx_snapshot = snapshot_physx_properties(env, _task_cfg)
    print(f"  PhysX snapshot captured: {list(physx_snapshot.keys())}")

    # =====================================================================
    # Run all scenarios using the SAME environment
    # =====================================================================
    all_scenario_results = {}

    for scenario_name, scenario_cfg in OOD_SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name} — {scenario_cfg['description']}")
        print(f"{'='*70}")

        # Apply perturbation (restores originals first, then applies)
        apply_ood_perturbation(env, scenario_cfg["ood_type"], scenario_cfg["ood_params"], physx_snapshot, _task_cfg)

        scenario_results = {}

        # --- Vanilla ---
        print(f"\n  VANILLA:")
        env.reset()
        vanilla_policy = BaselinePolicy(base_policy, policy_nn)
        vanilla_res = run_ood_evaluation(
            env, vanilla_policy, args_cli.num_episodes, f"Vanilla-{scenario_name}"
        )
        v_sr = sum(1 for r in vanilla_res if r.success) / len(vanilla_res)
        v_rw = np.mean([r.reward for r in vanilla_res])
        scenario_results['vanilla'] = {'success_rate': v_sr, 'avg_reward': v_rw}
        print(f"    Vanilla: {v_sr:.1%} success, {v_rw:.2f} reward")

        # --- Multi-Sample ---
        print(f"\n  MULTI-SAMPLE:")
        env.reset()
        ms_policy = MultiSamplePolicy(policy_nn, env, args_cli.num_samples, noise_params)
        ms_res = run_ood_evaluation(
            env, ms_policy, args_cli.num_episodes, f"MS-{scenario_name}"
        )
        ms_sr = sum(1 for r in ms_res if r.success) / len(ms_res)
        ms_rw = np.mean([r.reward for r in ms_res])
        scenario_results['multi_sample'] = {'success_rate': ms_sr, 'avg_reward': ms_rw}
        print(f"    Multi-Sample: {ms_sr:.1%} success, {ms_rw:.2f} reward")

        # --- Decomposed ---
        print(f"\n  DECOMPOSED:")
        env.reset()
        controller = InterventionController(
            tau_a=args_cli.tau_a, tau_e=args_cli.tau_e, beta=args_cli.beta
        )
        decomp_policy = DecomposedPolicy(
            policy_actor=policy_nn.actor,
            policy_obs_normalizer=policy_nn.actor_obs_normalizer,
            msv_estimator=msv_est,
            mahal_estimator=mahal_est,
            controller=controller,
            env=env,
            num_samples=args_cli.num_samples,
            noise_params=noise_params,
            task_cfg=_task_cfg,
        )
        decomp_res = run_ood_evaluation(
            env, decomp_policy, args_cli.num_episodes, f"Decomp-{scenario_name}"
        )
        d_sr = sum(1 for r in decomp_res if r.success) / len(decomp_res)
        d_rw = np.mean([r.reward for r in decomp_res])
        scenario_results['decomposed'] = {'success_rate': d_sr, 'avg_reward': d_rw}

        intervention_stats = controller.get_stats()
        scenario_results['intervention_stats'] = {
            str(k) if not hasattr(k, 'value') else k.value: v
            for k, v in intervention_stats.items()
        }
        print(f"    Decomposed: {d_sr:.1%} success, {d_rw:.2f} reward")
        print(f"    Interventions: ", end="")
        for name, stats in intervention_stats.items():
            n = name.value if hasattr(name, 'value') else name
            print(f"{n}={stats['fraction']:.1%} ", end="")
        print()

        # --- Total Uncertainty (monolithic baseline) ---
        print(f"\n  TOTAL UNCERTAINTY:")
        env.reset()
        total_policy = TotalUncertaintyPolicy(
            policy_actor=policy_nn.actor,
            policy_obs_normalizer=policy_nn.actor_obs_normalizer,
            msv_estimator=msv_est,
            mahal_estimator=mahal_est,
            env=env,
            tau_total=args_cli.tau_total,
            beta=args_cli.beta,
            num_samples=args_cli.num_samples,
            noise_params=noise_params,
            task_cfg=_task_cfg,
        )
        total_res = run_ood_evaluation(
            env, total_policy, args_cli.num_episodes, f"Total-{scenario_name}"
        )
        t_sr = sum(1 for r in total_res if r.success) / len(total_res)
        t_rw = np.mean([r.reward for r in total_res])
        total_stats = total_policy.get_stats()
        scenario_results['total_uncertainty'] = {
            'success_rate': t_sr, 'avg_reward': t_rw,
            'intervention_stats': total_stats
        }
        print(f"    Total Uncertainty: {t_sr:.1%} success, {t_rw:.2f} reward")
        print(f"    Intervene: {total_stats['intervene']['fraction']:.1%}, "
              f"Normal: {total_stats['normal']['fraction']:.1%}")

        all_scenario_results[scenario_name] = scenario_results

    # Restore originals before closing
    restore_physx_properties(env, physx_snapshot, _task_cfg)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("OOD PERTURBATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Noise: {noise_level}")
    print(f"  Thresholds: tau_a={args_cli.tau_a}, tau_e={args_cli.tau_e}, beta={args_cli.beta}, tau_total={args_cli.tau_total}")
    print()
    print(f"{'Scenario':<22} {'Vanilla':>8} {'Multi-S':>8} {'Total-U':>8} {'Decomp':>8} {'D-MS':>7} {'D-TU':>7}")
    print("-" * 72)

    for scenario_name, res in all_scenario_results.items():
        v = res['vanilla']['success_rate']
        m = res['multi_sample']['success_rate']
        t = res['total_uncertainty']['success_rate']
        d = res['decomposed']['success_rate']
        d_ms = (d - m) * 100
        d_tu = (d - t) * 100
        marker = ""
        if d_ms > 0 or d_tu > 0:
            marker = " *"
        print(f"  {scenario_name:<20} {v*100:>7.1f}% {m*100:>7.1f}% {t*100:>7.1f}% {d*100:>7.1f}% {d_ms:>+6.1f}% {d_tu:>+6.1f}%{marker}")

    print("-" * 72)
    print("  (* = decomposed beats multi-sample or total uncertainty)")

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        f"ood_{noise_level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            "noise_params": noise_params,
            "num_samples": args_cli.num_samples,
            "tau_a": args_cli.tau_a,
            "tau_e": args_cli.tau_e,
            "beta": args_cli.beta,
            "tau_total": args_cli.tau_total,
            "num_episodes": args_cli.num_episodes,
        },
        "scenarios": all_scenario_results,
    })

    with open(os.path.join(output_dir, "ood_results.json"), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_dir}")

    env.close()
    return all_scenario_results


if __name__ == "__main__":
    results = main()
    simulation_app.close()
