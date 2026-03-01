"""
Uncertainty Decomposition and Targeted Intervention Controller

The core of the paper: different uncertainty types get different fixes.

- High aleatoric (noisy observation): Multi-sample averaging
- High epistemic (unfamiliar state): Conservative action scaling
- Both high: Filter + conservative
- Both low: Normal action

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


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
    Full decomposed uncertainty-aware policy.

    Pipeline per step:
      1. Take N noisy readings from ground truth
      2. u_aleatoric = variance across N samples (pure noise signal)
      3. u_epistemic = Mahalanobis distance of ground truth from calibration
      4. Controller decides intervention based on (u_a, u_e)
      5. If high aleatoric: use averaged observation (filtered)
      6. If high epistemic: scale down actions (conservative)
    """

    def __init__(self,
                 policy_actor,
                 policy_obs_normalizer,
                 msv_estimator,
                 mahal_estimator,
                 controller: InterventionController,
                 env,
                 num_samples: int = 5,
                 noise_params: Dict = None):
        """
        Args:
            policy_actor: Direct actor network (MLP)
            policy_obs_normalizer: Observation normalizer
            msv_estimator: MultiSampleVarianceEstimator (for aleatoric)
            mahal_estimator: AleatoricEstimator (used as epistemic — Mahalanobis)
            controller: InterventionController (thresholds set)
            env: Isaac Lab environment (for ground truth access)
            num_samples: Number of samples for multi-sample averaging
            noise_params: Noise parameters for generating samples
        """
        self.actor = policy_actor
        self.obs_normalizer = policy_obs_normalizer
        self.msv_estimator = msv_estimator
        self.mahal_estimator = mahal_estimator
        self.controller = controller
        self.env = env
        self.num_samples = num_samples
        self.noise_params = noise_params or {}
        self.device = env.unwrapped.device

        # Logging
        self.step_log = []

    def __call__(self, obs) -> torch.Tensor:
        """
        Get action with decomposed uncertainty-aware control.

        Pipeline:
          1. Get ground truth observation
          2. Generate N noisy samples
          3. u_aleatoric = variance across samples (pure noise signal)
          4. u_epistemic = Mahalanobis distance of ground truth from calibration
          5. Controller decides intervention
          6. Apply filtering and/or conservative scaling

        Args:
            obs: Observation (TensorDict or Tensor)

        Returns:
            actions [batch, action_dim]
        """
        obs_tensor = self._obs_to_tensor(obs)

        # Step 1: Get ground truth observation
        gt_obs = self._get_ground_truth_obs()
        if gt_obs is None:
            gt_obs = obs_tensor

        # Step 2: Generate N noisy samples from ground truth
        batch_size = gt_obs.shape[0]
        samples = []
        for _ in range(self.num_samples):
            sample = gt_obs.clone()
            if self.noise_params.get("joint_pos_std", 0) > 0:
                sample[:, 0:9] += torch.randn(batch_size, 9, device=self.device) * self.noise_params["joint_pos_std"]
            if self.noise_params.get("joint_vel_std", 0) > 0:
                sample[:, 9:18] += torch.randn(batch_size, 9, device=self.device) * self.noise_params["joint_vel_std"]
            if self.noise_params.get("object_pos_std", 0) > 0:
                sample[:, 18:21] += torch.randn(batch_size, 3, device=self.device) * self.noise_params["object_pos_std"]
            samples.append(sample)

        samples_tensor = torch.stack(samples, dim=0)  # [N, batch, 36]

        # Step 3: u_aleatoric = multi-sample variance (pure noise signal)
        u_a = self.msv_estimator.compute_variance_torch(samples_tensor)  # [batch]

        # Step 4: u_epistemic = Mahalanobis distance of ground truth from calibration
        u_e = self.mahal_estimator.predict_normalized_torch(gt_obs)  # [batch]

        # Step 5: Decide intervention
        need_filter, action_scale, intervention_type = self.controller.decide(u_a, u_e)

        # Step 6: Prepare observation
        # If high aleatoric: use averaged samples (filtered)
        # Otherwise: use the noisy observation as-is
        averaged_obs = samples_tensor.mean(dim=0)  # [batch, 36]
        final_obs = obs_tensor.clone()
        if need_filter.any():
            final_obs[need_filter] = averaged_obs[need_filter]

        # Step 7: Get action from actor
        with torch.inference_mode():
            normalized_obs = self.obs_normalizer(final_obs)
            actions = self.actor(normalized_obs)

        # Step 8: Apply conservative scaling for high epistemic
        actions = actions * action_scale.unsqueeze(-1)

        return actions

    def _get_ground_truth_obs(self) -> Optional[torch.Tensor]:
        """Get ground truth observation from environment."""
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

            return torch.cat([
                joint_pos[:, :9], joint_vel[:, :9],
                object_pos[:, :3], target_pose[:, :7],
                prev_actions[:, :8]
            ], dim=-1)
        except Exception:
            return None

    def _obs_to_tensor(self, obs) -> torch.Tensor:
        """Convert observation to tensor."""
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
