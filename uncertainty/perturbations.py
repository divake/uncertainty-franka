"""
Perturbation Factory

Implements all aleatoric (A1-A6) and epistemic (E1-E6) perturbation types
for the observation space.

Aleatoric perturbations corrupt the observation quality.
Epistemic perturbations put the robot in states it hasn't trained on.

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PerturbationType(Enum):
    # Aleatoric (corrupt observation)
    GAUSSIAN_NOISE = "A1_gaussian_noise"
    OCCLUSION = "A2_occlusion"
    SENSOR_DROPOUT = "A3_sensor_dropout"
    MOTION_BLUR = "A4_motion_blur"
    BIAS_OFFSET = "A5_bias_offset"
    SALT_PEPPER = "A6_salt_pepper"

    # Epistemic (unfamiliar state)
    OOD_OBJECT_POS = "E1_ood_object_pos"
    OOD_JOINT_CONFIG = "E2_ood_joint_config"
    MASS_CHANGE = "E3_mass_change"
    FRICTION_CHANGE = "E4_friction_change"
    OBJECT_SIZE_CHANGE = "E5_object_size"
    GRAVITY_CHANGE = "E6_gravity_change"

    # Combined
    NOISE_PLUS_OOD = "C1_noise_ood"
    OCCLUSION_PLUS_MASS = "C2_occlusion_mass"
    DROPOUT_PLUS_NOVEL = "C3_dropout_novel"


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation."""
    ptype: PerturbationType
    severity: str  # "low", "medium", "high"
    params: Dict


# Observation indices
OBS_JOINT_POS = slice(0, 9)
OBS_JOINT_VEL = slice(9, 18)
OBS_OBJECT_POS = slice(18, 21)
OBS_TARGET_POSE = slice(21, 28)
OBS_PREV_ACTIONS = slice(28, 36)


class ObservationPerturbation:
    """
    Applies aleatoric perturbations to observations at runtime.
    These corrupt the observation but do NOT change the true environment state.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.obs_history = []  # For motion blur
        self.last_known_obs = None  # For occlusion

    def apply(self, obs: torch.Tensor, ptype: PerturbationType,
              params: Dict) -> torch.Tensor:
        """
        Apply observation perturbation.

        Args:
            obs: Clean observation [batch, 36]
            ptype: Perturbation type
            params: Perturbation parameters

        Returns:
            Perturbed observation [batch, 36]
        """
        self.last_known_obs = obs.clone() if self.last_known_obs is None else self.last_known_obs

        if ptype == PerturbationType.GAUSSIAN_NOISE:
            return self._gaussian_noise(obs, params)
        elif ptype == PerturbationType.OCCLUSION:
            return self._occlusion(obs, params)
        elif ptype == PerturbationType.SENSOR_DROPOUT:
            return self._sensor_dropout(obs, params)
        elif ptype == PerturbationType.MOTION_BLUR:
            return self._motion_blur(obs, params)
        elif ptype == PerturbationType.BIAS_OFFSET:
            return self._bias_offset(obs, params)
        elif ptype == PerturbationType.SALT_PEPPER:
            return self._salt_pepper(obs, params)
        else:
            return obs

    def _gaussian_noise(self, obs: torch.Tensor, params: Dict) -> torch.Tensor:
        """A1: Add Gaussian noise to observation."""
        noisy = obs.clone()
        batch_size = obs.shape[0]

        # Per-group noise levels
        joint_pos_std = params.get("joint_pos_std", 0.02)
        joint_vel_std = params.get("joint_vel_std", 0.1)
        object_pos_std = params.get("object_pos_std", 0.1)

        noisy[:, OBS_JOINT_POS] += torch.randn(batch_size, 9, device=self.device) * joint_pos_std
        noisy[:, OBS_JOINT_VEL] += torch.randn(batch_size, 9, device=self.device) * joint_vel_std
        noisy[:, OBS_OBJECT_POS] += torch.randn(batch_size, 3, device=self.device) * object_pos_std

        return noisy

    def _occlusion(self, obs: torch.Tensor, params: Dict) -> torch.Tensor:
        """A2: Simulate object position occlusion (set to last-known or zero)."""
        occluded = obs.clone()
        p_occlude = params.get("p_occlude", 0.3)
        batch_size = obs.shape[0]

        # Random mask: which envs have occluded object
        mask = torch.rand(batch_size, device=self.device) < p_occlude

        if mask.any():
            # Replace object position with last-known position (or zero)
            if self.last_known_obs is not None:
                occluded[mask, 18:21] = self.last_known_obs[mask, 18:21]
            else:
                occluded[mask, 18:21] = 0.0

        # Update last known for non-occluded envs
        if (~mask).any():
            if self.last_known_obs is None:
                self.last_known_obs = obs.clone()
            self.last_known_obs[~mask] = obs[~mask]

        return occluded

    def _sensor_dropout(self, obs: torch.Tensor, params: Dict) -> torch.Tensor:
        """A3: Random dimensions set to 0 (partial sensor failure)."""
        dropped = obs.clone()
        p_dropout = params.get("p_dropout", 0.1)

        # Create dropout mask (per element)
        mask = torch.rand_like(obs) < p_dropout

        # Don't drop target pose or previous actions (indices 21:36)
        # Only drop sensor readings
        mask[:, 21:] = False

        dropped[mask] = 0.0
        return dropped

    def _motion_blur(self, obs: torch.Tensor, params: Dict) -> torch.Tensor:
        """A4: Temporal average of last K observations (simulates camera motion blur)."""
        K = params.get("K", 5)

        self.obs_history.append(obs.clone())
        if len(self.obs_history) > K:
            self.obs_history = self.obs_history[-K:]

        # Average over history
        blurred = torch.stack(self.obs_history, dim=0).mean(dim=0)
        return blurred

    def _bias_offset(self, obs: torch.Tensor, params: Dict) -> torch.Tensor:
        """A5: Add constant offset (miscalibrated sensor)."""
        biased = obs.clone()
        offset = params.get("offset", 0.1)

        # Add offset to sensor readings (joint pos, vel, object pos)
        biased[:, OBS_JOINT_POS] += offset
        biased[:, OBS_OBJECT_POS] += offset

        return biased

    def _salt_pepper(self, obs: torch.Tensor, params: Dict) -> torch.Tensor:
        """A6: Random dimensions set to extreme values (sensor spikes)."""
        spiked = obs.clone()
        p_spike = params.get("p_spike", 0.05)
        spike_val = params.get("spike_val", 5.0)

        # Create spike mask (sensor dims only)
        mask = torch.rand_like(obs[:, :21]) < p_spike

        # Random sign
        signs = (torch.rand_like(obs[:, :21]) > 0.5).float() * 2 - 1

        spiked[:, :21][mask] = spike_val * signs[mask]
        return spiked

    def reset(self):
        """Reset internal state (call on episode reset)."""
        self.obs_history = []
        self.last_known_obs = None


class EnvironmentPerturbation:
    """
    Applies epistemic perturbations to the environment.
    These change the true environment dynamics/state (OOD scenarios).
    Must be applied before environment reset/during setup.
    """

    @staticmethod
    def get_ood_object_position(env, shift: float = 0.2) -> None:
        """
        E1: Move object to unusual position outside training distribution.
        Must be called after env.reset().
        """
        unwrapped = env.unwrapped
        obj = unwrapped.scene.rigid_objects["object"]
        num_envs = unwrapped.num_envs
        device = unwrapped.device

        # Add random shift to object position
        shift_vec = (torch.rand(num_envs, 3, device=device) - 0.5) * 2 * shift
        obj.data.root_pos_w[:, :3] += shift_vec

    @staticmethod
    def get_ood_joint_config(env, deviation: float = 0.5) -> None:
        """
        E2: Start robot in unusual joint configuration.
        Must be called after env.reset().
        """
        unwrapped = env.unwrapped
        robot = unwrapped.scene.articulations["robot"]
        num_envs = unwrapped.num_envs
        device = unwrapped.device

        # Add deviation to default joint positions
        joint_noise = torch.randn(num_envs, robot.data.joint_pos.shape[1], device=device) * deviation
        robot.data.joint_pos[:] = robot.data.default_joint_pos + joint_noise

    @staticmethod
    def modify_mass(env, scale: float = 5.0) -> None:
        """E3: Modify object mass (heavier/lighter object)."""
        unwrapped = env.unwrapped
        obj = unwrapped.scene.rigid_objects["object"]
        try:
            masses = obj.root_physx_view.get_masses()
            indices = torch.arange(masses.shape[0], dtype=torch.int32, device="cpu")
            obj.root_physx_view.set_masses(masses * scale, indices)
        except Exception as e:
            print(f"Warning: Could not modify mass: {e}")

    @staticmethod
    def modify_friction(env, scale: float = 0.2) -> None:
        """E4: Modify surface friction (slippery object)."""
        unwrapped = env.unwrapped
        try:
            # Access material properties
            obj = unwrapped.scene.rigid_objects["object"]
            materials = obj.root_physx_view.get_material_properties()
            materials[:, :, 0] *= scale  # static friction
            materials[:, :, 1] *= scale  # dynamic friction
            indices = torch.arange(materials.shape[0], dtype=torch.int32, device="cpu")
            obj.root_physx_view.set_material_properties(materials, indices)
        except Exception as e:
            print(f"Warning: Could not modify friction: {e}")

    @staticmethod
    def modify_gravity(env, scale: float = 1.5) -> None:
        """E6: Modify gravity vector."""
        unwrapped = env.unwrapped
        try:
            import carb
            from isaaclab.sim import SimulationContext
            sim_ctx = SimulationContext.instance()
            gravity = sim_ctx.cfg.gravity
            new_gravity = [g * scale for g in gravity]
            sim_ctx.physics_sim_view.set_gravity(carb.Float3(*new_gravity))
        except Exception as e:
            print(f"Warning: Could not modify gravity: {e}")


# Preset perturbation configurations
PERTURBATION_PRESETS = {
    # Aleatoric presets
    "A1_low": PerturbationConfig(PerturbationType.GAUSSIAN_NOISE, "low",
                                  {"joint_pos_std": 0.005, "joint_vel_std": 0.02, "object_pos_std": 0.02}),
    "A1_medium": PerturbationConfig(PerturbationType.GAUSSIAN_NOISE, "medium",
                                     {"joint_pos_std": 0.01, "joint_vel_std": 0.05, "object_pos_std": 0.05}),
    "A1_high": PerturbationConfig(PerturbationType.GAUSSIAN_NOISE, "high",
                                   {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10}),
    "A1_extreme": PerturbationConfig(PerturbationType.GAUSSIAN_NOISE, "extreme",
                                      {"joint_pos_std": 0.05, "joint_vel_std": 0.2, "object_pos_std": 0.15}),

    "A2_low": PerturbationConfig(PerturbationType.OCCLUSION, "low", {"p_occlude": 0.1}),
    "A2_medium": PerturbationConfig(PerturbationType.OCCLUSION, "medium", {"p_occlude": 0.3}),
    "A2_high": PerturbationConfig(PerturbationType.OCCLUSION, "high", {"p_occlude": 0.5}),

    "A3_low": PerturbationConfig(PerturbationType.SENSOR_DROPOUT, "low", {"p_dropout": 0.05}),
    "A3_medium": PerturbationConfig(PerturbationType.SENSOR_DROPOUT, "medium", {"p_dropout": 0.1}),
    "A3_high": PerturbationConfig(PerturbationType.SENSOR_DROPOUT, "high", {"p_dropout": 0.2}),

    "A4_low": PerturbationConfig(PerturbationType.MOTION_BLUR, "low", {"K": 3}),
    "A4_medium": PerturbationConfig(PerturbationType.MOTION_BLUR, "medium", {"K": 5}),
    "A4_high": PerturbationConfig(PerturbationType.MOTION_BLUR, "high", {"K": 10}),

    "A5_low": PerturbationConfig(PerturbationType.BIAS_OFFSET, "low", {"offset": 0.05}),
    "A5_medium": PerturbationConfig(PerturbationType.BIAS_OFFSET, "medium", {"offset": 0.1}),
    "A5_high": PerturbationConfig(PerturbationType.BIAS_OFFSET, "high", {"offset": 0.2}),

    "A6_low": PerturbationConfig(PerturbationType.SALT_PEPPER, "low", {"p_spike": 0.05, "spike_val": 3.0}),
    "A6_medium": PerturbationConfig(PerturbationType.SALT_PEPPER, "medium", {"p_spike": 0.1, "spike_val": 5.0}),
    "A6_high": PerturbationConfig(PerturbationType.SALT_PEPPER, "high", {"p_spike": 0.15, "spike_val": 5.0}),

    # Epistemic presets
    "E1_low": PerturbationConfig(PerturbationType.OOD_OBJECT_POS, "low", {"shift": 0.1}),
    "E1_medium": PerturbationConfig(PerturbationType.OOD_OBJECT_POS, "medium", {"shift": 0.2}),
    "E1_high": PerturbationConfig(PerturbationType.OOD_OBJECT_POS, "high", {"shift": 0.3}),

    "E2_low": PerturbationConfig(PerturbationType.OOD_JOINT_CONFIG, "low", {"deviation": 0.2}),
    "E2_medium": PerturbationConfig(PerturbationType.OOD_JOINT_CONFIG, "medium", {"deviation": 0.5}),
    "E2_high": PerturbationConfig(PerturbationType.OOD_JOINT_CONFIG, "high", {"deviation": 1.0}),

    "E3_low": PerturbationConfig(PerturbationType.MASS_CHANGE, "low", {"scale": 2.0}),
    "E3_medium": PerturbationConfig(PerturbationType.MASS_CHANGE, "medium", {"scale": 5.0}),
    "E3_high": PerturbationConfig(PerturbationType.MASS_CHANGE, "high", {"scale": 10.0}),

    "E4_low": PerturbationConfig(PerturbationType.FRICTION_CHANGE, "low", {"scale": 0.5}),
    "E4_medium": PerturbationConfig(PerturbationType.FRICTION_CHANGE, "medium", {"scale": 0.2}),
    "E4_high": PerturbationConfig(PerturbationType.FRICTION_CHANGE, "high", {"scale": 0.1}),

    "E6_low": PerturbationConfig(PerturbationType.GRAVITY_CHANGE, "low", {"scale": 1.2}),
    "E6_medium": PerturbationConfig(PerturbationType.GRAVITY_CHANGE, "medium", {"scale": 1.5}),
    "E6_high": PerturbationConfig(PerturbationType.GRAVITY_CHANGE, "high", {"scale": 2.0}),

    # Combined presets
    "C1_medium": PerturbationConfig(PerturbationType.NOISE_PLUS_OOD, "medium",
                                     {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10,
                                      "shift": 0.2}),
    "C2_medium": PerturbationConfig(PerturbationType.OCCLUSION_PLUS_MASS, "medium",
                                     {"p_occlude": 0.3, "scale": 5.0}),
}


def get_perturbation_config(name: str) -> PerturbationConfig:
    """Get a preset perturbation configuration by name."""
    if name not in PERTURBATION_PRESETS:
        raise ValueError(f"Unknown perturbation: {name}. Available: {list(PERTURBATION_PRESETS.keys())}")
    return PERTURBATION_PRESETS[name]


def is_aleatoric(ptype: PerturbationType) -> bool:
    """Check if perturbation type is aleatoric."""
    return ptype.value.startswith("A")


def is_epistemic(ptype: PerturbationType) -> bool:
    """Check if perturbation type is epistemic."""
    return ptype.value.startswith("E")


def is_combined(ptype: PerturbationType) -> bool:
    """Check if perturbation type is combined."""
    return ptype.value.startswith("C")
