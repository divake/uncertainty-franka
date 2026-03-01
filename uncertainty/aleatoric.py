"""
Aleatoric Uncertainty Estimation

Two complementary approaches:
  1. MahalanobisEstimator: Distance from calibration distribution (total uncertainty)
  2. MultiSampleVarianceEstimator: Variance across multiple noisy readings (pure aleatoric)

KEY INSIGHT: Mahalanobis distance captures BOTH noise and OOD shifts.
Multi-sample variance captures ONLY noise, because multiple readings of the
same ground-truth state will have zero variance even if the state is OOD.
This makes multi-sample variance perfectly orthogonal to epistemic uncertainty.

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
import json


class MultiSampleVarianceEstimator:
    """
    Aleatoric uncertainty via multi-sample observation variance.

    Takes N noisy readings from the same ground truth state and computes
    their variance. High variance = high sensor noise = high aleatoric.

    This is orthogonal to epistemic uncertainty BY CONSTRUCTION:
    - OOD state with NO noise → zero variance → zero aleatoric
    - Normal state with noise → high variance → high aleatoric

    Calibration: records the typical variance magnitude under known noise
    levels for normalization.
    """

    def __init__(self, noise_params: Dict = None):
        self.noise_params = noise_params or {}
        self.cal_mean_var = None
        self.cal_max_var = None
        self.is_calibrated = False

    def calibrate(self, X_cal: np.ndarray, noise_params: Dict,
                  n_samples: int = 5, n_trials: int = 500,
                  verbose: bool = True) -> 'MultiSampleVarianceEstimator':
        """
        Calibrate normalization by computing expected variance under noise.

        Args:
            X_cal: Calibration observations [N_cal, D]
            noise_params: Noise std dict (joint_pos_std, joint_vel_std, object_pos_std)
            n_samples: Number of samples per variance computation
            n_trials: Number of trials for calibration
        """
        self.noise_params = noise_params

        trial_idx = np.random.choice(len(X_cal), min(n_trials, len(X_cal)), replace=False)
        variances = []

        for idx in trial_idx:
            gt = X_cal[idx]
            samples = []
            for _ in range(n_samples):
                s = gt.copy()
                if noise_params.get("joint_pos_std", 0) > 0:
                    s[0:9] += np.random.randn(9) * noise_params["joint_pos_std"]
                if noise_params.get("joint_vel_std", 0) > 0:
                    s[9:18] += np.random.randn(9) * noise_params["joint_vel_std"]
                if noise_params.get("object_pos_std", 0) > 0:
                    s[18:21] += np.random.randn(3) * noise_params["object_pos_std"]
                samples.append(s)
            samples = np.array(samples)
            var = samples.var(axis=0).mean()
            variances.append(var)

        variances = np.array(variances)
        self.cal_mean_var = np.mean(variances)
        self.cal_max_var = np.percentile(variances, 95)
        self.is_calibrated = True

        if verbose:
            print(f"\n  Multi-Sample Variance Calibration:")
            print(f"    Mean variance under noise: {self.cal_mean_var:.6f}")
            print(f"    95th percentile:          {self.cal_max_var:.6f}")

        return self

    def compute_variance_torch(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute aleatoric uncertainty from multi-sample observations.

        Args:
            samples: [n_samples, batch, D] — multiple noisy readings

        Returns:
            u_aleatoric: [batch] normalized aleatoric uncertainty
        """
        # Variance across samples, mean across dimensions
        var = samples.var(dim=0).mean(dim=-1)  # [batch]

        # Normalize by calibration statistics
        if self.cal_max_var is not None and self.cal_max_var > 0:
            u_a = var / self.cal_max_var
        else:
            u_a = var

        return torch.clamp(u_a, 0.0, 1.0)

    def compute_variance_np(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute aleatoric from samples (NumPy).

        Args:
            samples: [n_samples, N, D]

        Returns:
            u_aleatoric: [N]
        """
        var = samples.var(axis=0).mean(axis=-1)
        if self.cal_max_var is not None and self.cal_max_var > 0:
            u_a = var / self.cal_max_var
        else:
            u_a = var
        return np.clip(u_a, 0.0, 1.0)


class AleatoricEstimator:

    def __init__(self, reg_lambda: float = 1e-4, eps: float = 1e-10):
        self.reg_lambda = reg_lambda
        self.eps = eps

        # NumPy parameters (fitted offline)
        self.mean_ = None           # [D]
        self.cov_ = None            # [D, D]
        self.cov_inv_ = None        # [D, D]
        self.log_M_min_ = None
        self.log_M_max_ = None

        # PyTorch parameters (for GPU inference)
        self.mean_torch_ = None
        self.cov_inv_torch_ = None

        self.is_fitted_ = False
        self.feature_dim_ = None
        self.cal_mahal_distances_ = None

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'AleatoricEstimator':
        """
        Fit multivariate Gaussian to calibration observations.

        Args:
            X_cal: Calibration observations [N_cal, 36]
            verbose: Print statistics
        """
        N, D = X_cal.shape
        self.feature_dim_ = D

        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING ALEATORIC (MAHALANOBIS) UNCERTAINTY")
            print(f"{'='*60}")
            print(f"Calibration samples: {N}")
            print(f"Feature dimension: {D}")

        # Step 1: Mean vector
        self.mean_ = np.mean(X_cal, axis=0)

        # Step 2: Covariance matrix
        self.cov_ = np.cov(X_cal, rowvar=False)

        # Handle 1D case
        if self.cov_.ndim == 0:
            self.cov_ = np.array([[self.cov_]])

        if verbose:
            print(f"  Cov condition number (before reg): {np.linalg.cond(self.cov_):.2e}")

        # Step 3: Regularize
        trace_val = np.trace(self.cov_)
        reg_val = self.reg_lambda * (trace_val / D)
        self.cov_ = self.cov_ + reg_val * np.eye(D)

        if verbose:
            print(f"  Cov condition number (after reg): {np.linalg.cond(self.cov_):.2e}")

        # Step 4: Invert covariance
        self.cov_inv_ = np.linalg.inv(self.cov_)

        # Step 5: Calibration distances for normalization
        self.cal_mahal_distances_ = self._compute_mahalanobis_np(X_cal)

        log_M_cal = np.log(self.cal_mahal_distances_ + self.eps)
        self.log_M_min_ = np.min(log_M_cal)
        self.log_M_max_ = np.max(log_M_cal)

        if verbose:
            print(f"\n  Calibration Mahalanobis distances:")
            print(f"    Min:    {np.min(self.cal_mahal_distances_):.4f}")
            print(f"    Mean:   {np.mean(self.cal_mahal_distances_):.4f}")
            print(f"    Median: {np.median(self.cal_mahal_distances_):.4f}")
            print(f"    Max:    {np.max(self.cal_mahal_distances_):.4f}")
            print(f"    Std:    {np.std(self.cal_mahal_distances_):.4f}")
            print(f"  Log range: [{self.log_M_min_:.4f}, {self.log_M_max_:.4f}]")

        self.is_fitted_ = True

        if verbose:
            print(f"\n  ALEATORIC FITTING COMPLETE")
            print(f"{'='*60}\n")

        return self

    def to_torch(self, device: torch.device) -> 'AleatoricEstimator':
        """Convert fitted parameters to PyTorch tensors for GPU inference."""
        if not self.is_fitted_:
            raise ValueError("Must fit before converting to torch")

        self.mean_torch_ = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self.cov_inv_torch_ = torch.tensor(self.cov_inv_, dtype=torch.float32, device=device)
        return self

    def _compute_mahalanobis_np(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance (NumPy). Returns [N]."""
        X_centered = X - self.mean_
        M_squared = np.sum((X_centered @ self.cov_inv_) * X_centered, axis=1)
        M_squared = np.maximum(M_squared, 0.0)
        return np.sqrt(M_squared)

    def compute_mahalanobis_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance (PyTorch, GPU).

        Args:
            obs: Observations [batch, D] on GPU

        Returns:
            Mahalanobis distances [batch]
        """
        if self.mean_torch_ is None:
            self.to_torch(obs.device)

        centered = obs - self.mean_torch_
        M_squared = torch.sum((centered @ self.cov_inv_torch_) * centered, dim=1)
        M_squared = torch.clamp(M_squared, min=0.0)
        return torch.sqrt(M_squared)

    def predict_normalized_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict normalized aleatoric uncertainty [0, 1] on GPU.

        Args:
            obs: Observations [batch, D]

        Returns:
            Normalized aleatoric uncertainty [batch]
        """
        M = self.compute_mahalanobis_torch(obs)
        log_M = torch.log(M + self.eps)
        d_norm = (log_M - self.log_M_min_) / (self.log_M_max_ - self.log_M_min_ + self.eps)
        return torch.clamp(d_norm, 0.0, 1.0)

    def predict_raw(self, X_test: np.ndarray) -> np.ndarray:
        """Predict raw Mahalanobis distance (NumPy)."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        return self._compute_mahalanobis_np(X_test)

    def predict_normalized(self, X_test: np.ndarray) -> np.ndarray:
        """Predict normalized uncertainty [0, 1] (NumPy)."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted")
        M = self._compute_mahalanobis_np(X_test)
        log_M = np.log(M + self.eps)
        d_norm = (log_M - self.log_M_min_) / (self.log_M_max_ - self.log_M_min_ + self.eps)
        return np.clip(d_norm, 0.0, 1.0)

    def save(self, path: str):
        """Save fitted model."""
        save_dict = {
            'mean': self.mean_.tolist(),
            'cov': self.cov_.tolist(),
            'cov_inv': self.cov_inv_.tolist(),
            'log_M_min': float(self.log_M_min_),
            'log_M_max': float(self.log_M_max_),
            'reg_lambda': self.reg_lambda,
            'feature_dim': self.feature_dim_,
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)

    def load(self, path: str) -> 'AleatoricEstimator':
        """Load fitted model."""
        with open(path, 'r') as f:
            d = json.load(f)
        self.mean_ = np.array(d['mean'])
        self.cov_ = np.array(d['cov'])
        self.cov_inv_ = np.array(d['cov_inv'])
        self.log_M_min_ = d['log_M_min']
        self.log_M_max_ = d['log_M_max']
        self.reg_lambda = d['reg_lambda']
        self.feature_dim_ = d['feature_dim']
        self.is_fitted_ = True
        return self
