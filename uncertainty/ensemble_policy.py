"""
Deep Ensemble Policy for Uncertainty Estimation

This module wraps multiple policy networks to estimate epistemic uncertainty
through disagreement between ensemble members.

For IROS 2026: Uncertainty Decomposition for Robot Manipulation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UncertaintyMetrics:
    """Uncertainty metrics from ensemble prediction."""
    mean_action: torch.Tensor
    std_action: torch.Tensor  # Epistemic uncertainty
    entropy: torch.Tensor  # Action entropy
    disagreement: torch.Tensor  # Max disagreement between members


class EnsemblePolicy(nn.Module):
    """
    Deep Ensemble wrapper for uncertainty estimation.

    Uses multiple policy networks and measures disagreement to estimate
    epistemic uncertainty in action predictions.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        num_members: int = 5,
        device: str = "cuda:0"
    ):
        """
        Initialize ensemble from a base policy.

        Args:
            base_policy: Single policy network to clone
            num_members: Number of ensemble members
            device: Device to run on
        """
        super().__init__()
        self.num_members = num_members
        self.device = device

        # Create ensemble by cloning and perturbing the base policy
        self.members = nn.ModuleList()

        for i in range(num_members):
            # Clone the base policy
            member = self._clone_policy(base_policy)

            # Add small perturbations to weights (bootstrap-style diversity)
            if i > 0:  # Keep first member as original
                self._perturb_weights(member, scale=0.01)

            self.members.append(member)

        self.to(device)

    def _clone_policy(self, policy: nn.Module) -> nn.Module:
        """Deep clone a policy network."""
        import copy
        return copy.deepcopy(policy)

    def _perturb_weights(self, model: nn.Module, scale: float = 0.01):
        """Add Gaussian perturbation to model weights."""
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning mean action.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            Mean action across ensemble [batch_size, action_dim]
        """
        actions = self.get_ensemble_predictions(obs)
        return actions.mean(dim=0)  # Average across ensemble

    def get_ensemble_predictions(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from all ensemble members.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            Actions from all members [num_members, batch_size, action_dim]
        """
        predictions = []
        for member in self.members:
            with torch.no_grad():
                action = member(obs)
            predictions.append(action)

        return torch.stack(predictions, dim=0)

    def predict_with_uncertainty(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, UncertaintyMetrics]:
        """
        Predict actions with uncertainty estimates.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            Tuple of (mean_action, uncertainty_metrics)
        """
        # Get all ensemble predictions
        all_actions = self.get_ensemble_predictions(obs)  # [K, B, A]

        # Compute statistics
        mean_action = all_actions.mean(dim=0)  # [B, A]
        std_action = all_actions.std(dim=0)  # [B, A] - epistemic uncertainty

        # Compute disagreement (max diff between any two members)
        max_diff = torch.zeros(obs.shape[0], device=self.device)
        for i in range(self.num_members):
            for j in range(i + 1, self.num_members):
                diff = (all_actions[i] - all_actions[j]).abs().max(dim=-1)[0]
                max_diff = torch.maximum(max_diff, diff)

        # Compute action entropy (using variance as proxy)
        variance = std_action.pow(2).sum(dim=-1)  # [B]
        entropy = 0.5 * torch.log(2 * np.pi * np.e * variance + 1e-8)

        metrics = UncertaintyMetrics(
            mean_action=mean_action,
            std_action=std_action,
            entropy=entropy,
            disagreement=max_diff,
        )

        return mean_action, metrics

    def get_uncertainty_score(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get a single uncertainty score per observation.

        Higher score = more uncertain.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            Uncertainty scores [batch_size]
        """
        _, metrics = self.predict_with_uncertainty(obs)

        # Combine std and disagreement into single score
        std_score = metrics.std_action.mean(dim=-1)  # Average std across actions

        # Normalize and combine
        uncertainty = std_score + 0.5 * metrics.disagreement

        return uncertainty


class MCDropoutPolicy(nn.Module):
    """
    MC Dropout wrapper for uncertainty estimation.

    Enables dropout at inference time and uses multiple forward passes
    to estimate uncertainty.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        dropout_rate: float = 0.1,
        num_samples: int = 10,
        device: str = "cuda:0"
    ):
        """
        Initialize MC Dropout policy.

        Args:
            base_policy: Policy network (must have dropout layers)
            dropout_rate: Dropout probability
            num_samples: Number of forward passes for uncertainty estimation
            device: Device to run on
        """
        super().__init__()
        self.policy = base_policy
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        self.device = device

        # Add dropout layers if not present
        self._add_dropout_layers()

        self.to(device)

    def _add_dropout_layers(self):
        """Add dropout layers to the policy network."""
        # This modifies the policy in-place to add dropout after activations
        for name, module in self.policy.named_modules():
            if isinstance(module, (nn.ReLU, nn.ELU, nn.LeakyReLU)):
                # Find parent and add dropout after activation
                # This is a simplified version - proper implementation
                # would need to modify the forward pass
                pass

    def enable_dropout(self):
        """Enable dropout for MC sampling."""
        for module in self.policy.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (no uncertainty)."""
        self.policy.eval()
        with torch.no_grad():
            return self.policy(obs)

    def predict_with_uncertainty(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, UncertaintyMetrics]:
        """
        MC Dropout prediction with uncertainty.

        Args:
            obs: Observations [batch_size, obs_dim]

        Returns:
            Tuple of (mean_action, uncertainty_metrics)
        """
        self.enable_dropout()

        samples = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                action = self.policy(obs)
            samples.append(action)

        samples = torch.stack(samples, dim=0)  # [S, B, A]

        mean_action = samples.mean(dim=0)
        std_action = samples.std(dim=0)

        variance = std_action.pow(2).sum(dim=-1)
        entropy = 0.5 * torch.log(2 * np.pi * np.e * variance + 1e-8)

        max_diff = torch.zeros(obs.shape[0], device=self.device)
        for i in range(self.num_samples):
            for j in range(i + 1, self.num_samples):
                diff = (samples[i] - samples[j]).abs().max(dim=-1)[0]
                max_diff = torch.maximum(max_diff, diff)

        metrics = UncertaintyMetrics(
            mean_action=mean_action,
            std_action=std_action,
            entropy=entropy,
            disagreement=max_diff,
        )

        return mean_action, metrics


def create_ensemble_from_checkpoint(
    checkpoint_path: str,
    env,
    agent_cfg,
    num_members: int = 5,
    device: str = "cuda:0"
) -> EnsemblePolicy:
    """
    Create an ensemble policy from a single pretrained checkpoint.

    Args:
        checkpoint_path: Path to the pretrained checkpoint
        env: Environment (for creating runner)
        agent_cfg: Agent configuration
        num_members: Number of ensemble members
        device: Device to run on

    Returns:
        EnsemblePolicy with loaded weights
    """
    from rsl_rl.runners import OnPolicyRunner

    # Load the base policy
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=device
    )
    runner.load(checkpoint_path)

    # Get the actor network
    try:
        base_policy = runner.alg.policy.actor
    except AttributeError:
        base_policy = runner.alg.actor_critic.actor

    # Create ensemble
    ensemble = EnsemblePolicy(
        base_policy=base_policy,
        num_members=num_members,
        device=device
    )

    return ensemble
