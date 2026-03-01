"""
Uncertainty Decomposition and Targeted Intervention Controller

CVPR-consistent decomposition (arXiv:2511.12389):
  - σ_alea = Mahalanobis distance from calibration distribution (SAME as CVPR Sec 3.2)
  - σ_epis = ε_knn + ε_rank (local state-space statistics, adapted from CVPR Sec 3.1)

Targeted intervention (IROS contribution):
  - High σ_alea (noisy/degraded observation): Multi-sample averaging to filter noise
  - High σ_epis (unfamiliar state): Conservative action scaling to reduce risk
  - Both high: Filter + conservative (maximum caution)
  - Both low: Normal action

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .task_config import TaskObsConfig, get_task_config, get_ground_truth_obs as _get_gt_obs, add_noise_to_samples


class InterventionType(Enum):
    NORMAL = "normal"           # Both low: act normally
    FILTER = "filter"           # High aleatoric: multi-sample averaging
    CONSERVATIVE = "conservative"  # High epistemic: scale down actions
    FILTER_AND_CONSERVATIVE = "filter_conservative"  # Both high


@dataclass
class InterventionResult:
    """Result of a single intervention decision."""
    intervention: InterventionType
    u_aleatoric: float
    u_epistemic: float
    action_scale: float
    used_filtering: bool


class UncertaintyDecomposer:
    """
    Wraps aleatoric and epistemic estimators to provide decomposed uncertainty.
    """

    def __init__(self, aleatoric_estimator, epistemic_estimator):
        """
        Args:
            aleatoric_estimator: AleatoricEstimator (fitted)
            epistemic_estimator: CombinedEpistemicEstimator (fitted)
        """
        self.aleatoric = aleatoric_estimator
        self.epistemic = epistemic_estimator

    def decompose_np(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose uncertainty (NumPy).

        Returns:
            (u_aleatoric, u_epistemic) both in [0, 1], shape [N]
        """
        u_a = self.aleatoric.predict_normalized(obs)
        u_e = self.epistemic.predict(obs)
        return u_a, u_e

    def decompose_torch(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose uncertainty (PyTorch).

        Returns:
            (u_aleatoric, u_epistemic) both in [0, 1], shape [batch]
        """
        u_a = self.aleatoric.predict_normalized_torch(obs)
        u_e = self.epistemic.predict_torch(obs)
        return u_a, u_e


class InterventionController:
    """
    Decides which intervention to apply based on decomposed uncertainty.

    The decision logic:
      IF u_a > tau_a AND u_e <= tau_e: FILTER (multi-sample averaging)
      IF u_e > tau_e AND u_a <= tau_a: CONSERVATIVE (scale down actions)
      IF u_a > tau_a AND u_e > tau_e: FILTER + CONSERVATIVE
      ELSE: NORMAL
    """

    def __init__(self,
                 tau_a: float = 0.5,
                 tau_e: float = 0.5,
                 beta: float = 0.5,
                 min_action_scale: float = 0.3):
        """
        Args:
            tau_a: Aleatoric threshold (above = high aleatoric)
            tau_e: Epistemic threshold (above = high epistemic)
            beta: Conservative scaling factor (action *= 1 - beta * u_e)
            min_action_scale: Minimum action scale (prevent zero actions)
        """
        self.tau_a = tau_a
        self.tau_e = tau_e
        self.beta = beta
        self.min_action_scale = min_action_scale

        # Statistics tracking
        self.intervention_counts = {t: 0 for t in InterventionType}
        self.total_steps = 0

    def decide(self, u_a: torch.Tensor, u_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decide intervention for each environment in the batch.

        Args:
            u_a: Aleatoric uncertainty [batch]
            u_e: Epistemic uncertainty [batch]

        Returns:
            need_filter: Bool mask — which envs need multi-sample filtering [batch]
            action_scale: Scaling factor for actions [batch]
            intervention_type: Integer codes [batch] (0=normal, 1=filter, 2=conservative, 3=both)
        """
        batch_size = u_a.shape[0]
        device = u_a.device

        high_a = u_a > self.tau_a
        high_e = u_e > self.tau_e

        # Filter when aleatoric is high
        need_filter = high_a

        # Conservative scaling when epistemic is high
        # action_scale = 1 - beta * u_e (clamped to min)
        action_scale = torch.ones(batch_size, device=device)
        action_scale[high_e] = torch.clamp(
            1.0 - self.beta * u_e[high_e],
            min=self.min_action_scale
        )

        # Intervention type codes
        intervention_type = torch.zeros(batch_size, dtype=torch.long, device=device)
        intervention_type[high_a & ~high_e] = 1  # Filter only
        intervention_type[~high_a & high_e] = 2  # Conservative only
        intervention_type[high_a & high_e] = 3   # Both

        # Track statistics
        self.total_steps += batch_size
        self.intervention_counts[InterventionType.NORMAL] += (~high_a & ~high_e).sum().item()
        self.intervention_counts[InterventionType.FILTER] += (high_a & ~high_e).sum().item()
        self.intervention_counts[InterventionType.CONSERVATIVE] += (~high_a & high_e).sum().item()
        self.intervention_counts[InterventionType.FILTER_AND_CONSERVATIVE] += (high_a & high_e).sum().item()

        return need_filter, action_scale, intervention_type

    def get_stats(self) -> Dict:
        """Get intervention statistics."""
        if self.total_steps == 0:
            return {}
        return {
            name.value: {
                'count': count,
                'fraction': count / self.total_steps
            }
            for name, count in self.intervention_counts.items()
        }

    def reset_stats(self):
        """Reset intervention statistics."""
        self.intervention_counts = {t: 0 for t in InterventionType}
        self.total_steps = 0

    def set_thresholds(self, tau_a: float, tau_e: float):
        """Update thresholds (e.g., from conformal prediction)."""
        self.tau_a = tau_a
        self.tau_e = tau_e


class DecomposedPolicy:
    """
    Full decomposed uncertainty-aware policy (CVPR-consistent).

    Pipeline per step:
      1. Get ground truth observation from simulator
      2. Generate N noisy readings by adding sensor noise to ground truth
      3. σ_alea = Mahalanobis(noisy_obs) — distance of noisy observation from
         calibration distribution (SAME formula as CVPR Sec 3.2)
      4. σ_epis = ε_knn + ε_rank(gt_obs) — state novelty on GROUND TRUTH,
         immune to sensor noise (adapted from CVPR Sec 3.1)
      5. Controller decides targeted intervention:
         - High σ_alea: use averaged observation (multi-sample filtering = denoising)
         - High σ_epis: scale down actions (conservative behavior)
    """

    def __init__(self,
                 policy_actor,
                 policy_obs_normalizer,
                 alea_estimator,
                 epis_estimator,
                 controller: InterventionController,
                 env,
                 num_samples: int = 5,
                 noise_params: Dict = None,
                 task_cfg: TaskObsConfig = None):
        """
        Args:
            alea_estimator: AleatoricEstimator (Mahalanobis) — fitted on calibration data
            epis_estimator: CVPREpistemicEstimator (ε_knn + ε_rank) — fitted on calibration data
        """
        self.actor = policy_actor
        self.obs_normalizer = policy_obs_normalizer
        self.alea_estimator = alea_estimator
        self.epis_estimator = epis_estimator
        self.controller = controller
        self.env = env
        self.num_samples = num_samples
        self.noise_params = noise_params or {}
        self.device = env.unwrapped.device
        self.task_cfg = task_cfg or get_task_config("Isaac-Lift-Cube-Franka-v0")
        self.step_log = []

    def __call__(self, obs) -> torch.Tensor:
        obs_tensor = self._obs_to_tensor(obs)

        gt_obs = _get_gt_obs(self.env, self.task_cfg)
        if gt_obs is None:
            gt_obs = obs_tensor

        # Generate N noisy samples from ground truth
        samples = []
        for _ in range(self.num_samples):
            sample = gt_obs.clone()
            sample = add_noise_to_samples(sample, self.noise_params, self.task_cfg, self.device)
            samples.append(sample)

        samples_tensor = torch.stack(samples, dim=0)  # [N, batch, obs_dim]

        # σ_alea = Mahalanobis on the noisy observation (CVPR Sec 3.2)
        # High when observation is far from calibration distribution (noise, occlusion, etc.)
        u_a = self.alea_estimator.predict_normalized_torch(obs_tensor)

        # σ_epis = ε_knn + ε_rank on GROUND TRUTH (CVPR Sec 3.1 adapted)
        # High when the true state is novel/OOD. Immune to sensor noise by design.
        u_e = self.epis_estimator.predict_torch(gt_obs)

        need_filter, action_scale, intervention_type = self.controller.decide(u_a, u_e)

        # Multi-sample averaging = denoising mechanism (triggered by high σ_alea)
        averaged_obs = samples_tensor.mean(dim=0)
        final_obs = obs_tensor.clone()
        if need_filter.any():
            final_obs[need_filter] = averaged_obs[need_filter]

        with torch.inference_mode():
            normalized_obs = self.obs_normalizer(final_obs)
            actions = self.actor(normalized_obs)

        # Conservative scaling (triggered by high σ_epis)
        actions = actions * action_scale.unsqueeze(-1)

        return actions

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        if hasattr(obs, 'get'):
            return obs.get('policy', obs)
        elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
            try:
                return obs['policy']
            except (KeyError, TypeError):
                pass
        return obs

    def reset(self, dones: torch.Tensor):
        """Reset for done environments."""
        pass

    def get_intervention_stats(self) -> Dict:
        """Get intervention statistics."""
        return self.controller.get_stats()


class TotalUncertaintyPolicy:
    """
    Baseline: Total (monolithic) uncertainty — no decomposition.

    Uses the SAME signals as Decomposed (σ_alea + σ_epis) but combines them:
      σ_total = sqrt(σ_alea² + σ_epis²)
    When σ_total > τ_total: applies BOTH filtering AND conservative scaling.
    Cannot distinguish between noise and OOD, so it over-intervenes.

    This baseline proves that DECOMPOSITION adds value beyond just detecting uncertainty.
    """

    def __init__(self,
                 policy_actor,
                 policy_obs_normalizer,
                 alea_estimator,
                 epis_estimator,
                 env,
                 tau_total: float = 0.5,
                 beta: float = 0.3,
                 num_samples: int = 5,
                 noise_params: Dict = None,
                 min_action_scale: float = 0.3,
                 task_cfg: TaskObsConfig = None):
        self.actor = policy_actor
        self.obs_normalizer = policy_obs_normalizer
        self.alea_estimator = alea_estimator
        self.epis_estimator = epis_estimator
        self.env = env
        self.tau_total = tau_total
        self.beta = beta
        self.num_samples = num_samples
        self.noise_params = noise_params or {}
        self.min_action_scale = min_action_scale
        self.device = env.unwrapped.device
        self.task_cfg = task_cfg or get_task_config("Isaac-Lift-Cube-Franka-v0")
        self.intervene_count = 0
        self.normal_count = 0

    def __call__(self, obs) -> torch.Tensor:
        obs_tensor = self._obs_to_tensor(obs)

        gt_obs = _get_gt_obs(self.env, self.task_cfg)
        if gt_obs is None:
            gt_obs = obs_tensor

        batch_size = gt_obs.shape[0]
        samples = []
        for _ in range(self.num_samples):
            sample = gt_obs.clone()
            sample = add_noise_to_samples(sample, self.noise_params, self.task_cfg, self.device)
            samples.append(sample)

        samples_tensor = torch.stack(samples, dim=0)

        # Same signals as Decomposed
        u_a = self.alea_estimator.predict_normalized_torch(obs_tensor)
        u_e = self.epis_estimator.predict_torch(gt_obs)

        # Combine into single σ_total (same as CVPR's σ_comb)
        u_total = torch.sqrt(u_a ** 2 + u_e ** 2)
        high_total = u_total > self.tau_total

        # When high: apply BOTH interventions (cannot distinguish)
        averaged_obs = samples_tensor.mean(dim=0)
        final_obs = obs_tensor.clone()
        if high_total.any():
            final_obs[high_total] = averaged_obs[high_total]

        with torch.inference_mode():
            normalized_obs = self.obs_normalizer(final_obs)
            actions = self.actor(normalized_obs)

        action_scale = torch.ones(batch_size, device=self.device)
        action_scale[high_total] = torch.clamp(
            1.0 - self.beta * u_total[high_total],
            min=self.min_action_scale
        )
        actions = actions * action_scale.unsqueeze(-1)

        self.intervene_count += high_total.sum().item()
        self.normal_count += (~high_total).sum().item()

        return actions

    def get_stats(self) -> Dict:
        total = self.intervene_count + self.normal_count
        if total == 0:
            return {"intervene": {"count": 0, "fraction": 0.0},
                    "normal": {"count": 0, "fraction": 0.0}}
        return {
            "intervene": {"count": self.intervene_count,
                          "fraction": self.intervene_count / total},
            "normal": {"count": self.normal_count,
                       "fraction": self.normal_count / total},
        }

    def reset(self, dones: torch.Tensor):
        pass

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        if hasattr(obs, 'get'):
            return obs.get('policy', obs)
        elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
            try:
                return obs['policy']
            except (KeyError, TypeError):
                pass
        return obs


def _obs_to_tensor_fn(obs) -> torch.Tensor:
    """Shared helper for obs conversion."""
    if hasattr(obs, 'get'):
        return obs.get('policy', obs)
    elif hasattr(obs, '__getitem__') and not isinstance(obs, torch.Tensor):
        try:
            return obs['policy']
        except (KeyError, TypeError):
            pass
    return obs


class DeepEnsemblePolicy:
    """
    Baseline B3: Deep Ensemble — M weight-perturbed actor copies.

    Creates M copies of the pretrained actor with Gaussian weight perturbations.
    Action variance across ensemble members measures uncertainty.
    When uncertainty is high, applies conservative scaling (since ensemble
    cannot decompose into aleatoric/epistemic).
    """

    def __init__(self,
                 policy_actor,
                 policy_obs_normalizer,
                 env,
                 num_members: int = 5,
                 perturbation_std: float = 0.02,
                 uncertainty_threshold: float = 0.1,
                 beta: float = 0.3,
                 min_action_scale: float = 0.3):
        import copy
        self.obs_normalizer = policy_obs_normalizer
        self.env = env
        self.device = env.unwrapped.device
        self.num_members = num_members
        self.threshold = uncertainty_threshold
        self.beta = beta
        self.min_action_scale = min_action_scale

        # Create ensemble: M copies with perturbed weights
        self.ensemble = []
        for i in range(num_members):
            member = copy.deepcopy(policy_actor)
            if i > 0:  # Keep first member as original
                with torch.no_grad():
                    for param in member.parameters():
                        param.add_(torch.randn_like(param) * perturbation_std)
            member.eval()
            self.ensemble.append(member)

        self.intervene_count = 0
        self.normal_count = 0

    def __call__(self, obs) -> torch.Tensor:
        obs_tensor = _obs_to_tensor_fn(obs)

        with torch.inference_mode():
            normalized = self.obs_normalizer(obs_tensor)
            actions_list = [member(normalized) for member in self.ensemble]
            actions_stack = torch.stack(actions_list, dim=0)  # [M, batch, action_dim]

        mean_action = actions_stack.mean(dim=0)
        action_var = actions_stack.var(dim=0).mean(dim=-1)  # [batch]

        high_uncertainty = action_var > self.threshold
        action_scale = torch.ones(obs_tensor.shape[0], device=self.device)
        if high_uncertainty.any():
            max_var = action_var[high_uncertainty].max().clamp(min=1e-6)
            action_scale[high_uncertainty] = torch.clamp(
                1.0 - self.beta * (action_var[high_uncertainty] / max_var),
                min=self.min_action_scale
            )
        mean_action = mean_action * action_scale.unsqueeze(-1)

        self.intervene_count += high_uncertainty.sum().item()
        self.normal_count += (~high_uncertainty).sum().item()
        return mean_action

    def get_stats(self) -> Dict:
        total = self.intervene_count + self.normal_count
        if total == 0:
            return {"intervene": {"count": 0, "fraction": 0.0},
                    "normal": {"count": 0, "fraction": 0.0}}
        return {
            "intervene": {"count": self.intervene_count,
                          "fraction": self.intervene_count / total},
            "normal": {"count": self.normal_count,
                       "fraction": self.normal_count / total},
        }

    def reset(self, dones: torch.Tensor):
        pass


class MCDropoutPolicy:
    """
    Baseline B4: MC Dropout — dropout at inference time.

    Inserts dropout layers into the actor MLP and runs M forward passes
    with dropout enabled. Action variance across passes measures uncertainty.
    When uncertainty is high, applies conservative scaling.
    """

    def __init__(self,
                 policy_actor,
                 policy_obs_normalizer,
                 env,
                 dropout_p: float = 0.1,
                 num_passes: int = 10,
                 uncertainty_threshold: float = 0.1,
                 beta: float = 0.3,
                 min_action_scale: float = 0.3):
        import copy
        self.obs_normalizer = policy_obs_normalizer
        self.env = env
        self.device = env.unwrapped.device
        self.num_passes = num_passes
        self.threshold = uncertainty_threshold
        self.beta = beta
        self.min_action_scale = min_action_scale

        # Create actor copy with dropout inserted after each activation
        self.dropout_actor = self._add_dropout(copy.deepcopy(policy_actor), dropout_p)

        self.intervene_count = 0
        self.normal_count = 0

    def _add_dropout(self, actor, p: float) -> torch.nn.Module:
        """Insert Dropout layers after each activation in the MLP."""
        new_layers = []
        for module in actor:
            new_layers.append(module)
            if isinstance(module, (torch.nn.ELU, torch.nn.ReLU, torch.nn.LeakyReLU,
                                   torch.nn.GELU, torch.nn.Tanh, torch.nn.Sigmoid)):
                new_layers.append(torch.nn.Dropout(p=p))
        return torch.nn.Sequential(*new_layers).to(self.device)

    def __call__(self, obs) -> torch.Tensor:
        obs_tensor = _obs_to_tensor_fn(obs)
        normalized = self.obs_normalizer(obs_tensor)

        # Enable dropout for inference (key MC Dropout trick)
        self.dropout_actor.train()

        actions_list = []
        with torch.no_grad():
            for _ in range(self.num_passes):
                actions_list.append(self.dropout_actor(normalized))
        actions_stack = torch.stack(actions_list, dim=0)  # [M, batch, action_dim]

        mean_action = actions_stack.mean(dim=0)
        action_var = actions_stack.var(dim=0).mean(dim=-1)  # [batch]

        high_uncertainty = action_var > self.threshold
        action_scale = torch.ones(obs_tensor.shape[0], device=self.device)
        if high_uncertainty.any():
            max_var = action_var[high_uncertainty].max().clamp(min=1e-6)
            action_scale[high_uncertainty] = torch.clamp(
                1.0 - self.beta * (action_var[high_uncertainty] / max_var),
                min=self.min_action_scale
            )
        mean_action = mean_action * action_scale.unsqueeze(-1)

        self.intervene_count += high_uncertainty.sum().item()
        self.normal_count += (~high_uncertainty).sum().item()
        return mean_action

    def get_stats(self) -> Dict:
        total = self.intervene_count + self.normal_count
        if total == 0:
            return {"intervene": {"count": 0, "fraction": 0.0},
                    "normal": {"count": 0, "fraction": 0.0}}
        return {
            "intervene": {"count": self.intervene_count,
                          "fraction": self.intervene_count / total},
            "normal": {"count": self.normal_count,
                       "fraction": self.normal_count / total},
        }

    def reset(self, dones: torch.Tensor):
        pass
