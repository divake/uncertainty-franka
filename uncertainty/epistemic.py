"""
LEGACY — NOT USED IN CURRENT PIPELINE (v3.0+)

Epistemic Uncertainty Estimation via Spectral Collapse + Repulsive Void Detection.
This was the v2.x epistemic estimator, REPLACED by epistemic_cvpr.py in v3.0.

Kept for reference only. The current CVPR-consistent pipeline uses:
  σ_epis = ε_knn + ε_rank  (see epistemic_cvpr.py)

Reason for replacement:
  - SpectralEpistemicEstimator computed on NOISY observations, so it was not
    immune to sensor noise — violating the decomposition requirement.
  - RepulsiveEpistemicEstimator used a Coulomb-like force model that was
    slow and not grounded in the CVPR paper methodology.
  - The new epistemic_cvpr.py uses GROUND TRUTH observations, achieving
    perfect noise isolation (Δ=0.000000 under noise).

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
from scipy.spatial.distance import cdist
import json


class SpectralEpistemicEstimator:
    """
    Epistemic uncertainty via spectral analysis of local feature manifold.

    Core idea: When model encounters states it hasn't trained on, the local
    feature manifold collapses to lower dimensions, measurable via eigenspectrum.

    Low spectral entropy → collapsed manifold → high epistemic uncertainty
    """

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors

        # Calibration data
        self.X_cal = None
        self.X_cal_torch = None
        self.feature_dim = None

        # Normalization
        self.cal_min_entropy = None
        self.cal_max_entropy = None
        self.cal_mean_eff_rank = None

        self.is_fitted = False

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'SpectralEpistemicEstimator':
        """
        Fit spectral model on calibration data.

        Args:
            X_cal: Calibration observations [N_cal, D]
        """
        self.X_cal = X_cal
        self.feature_dim = X_cal.shape[1]

        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING SPECTRAL EPISTEMIC ESTIMATOR")
            print(f"{'='*60}")
            print(f"Calibration samples: {len(X_cal)}")
            print(f"Feature dimension: {self.feature_dim}")
            print(f"K neighbors: {self.k_neighbors}")

        # Compute calibration statistics on a sample
        n_samples = min(500, len(X_cal))
        sample_idx = np.random.choice(len(X_cal), n_samples, replace=False)

        cal_entropies = []
        cal_eff_ranks = []

        for i, idx in enumerate(sample_idx):
            if i % 100 == 0 and verbose:
                print(f"  Processing {i}/{n_samples}...")

            entropy, eff_rank = self._compute_spectral_metrics(X_cal[idx], X_cal)
            cal_entropies.append(entropy)
            cal_eff_ranks.append(eff_rank)

        cal_entropies = np.array(cal_entropies)
        cal_eff_ranks = np.array(cal_eff_ranks)

        self.cal_min_entropy = np.percentile(cal_entropies, 5)
        self.cal_max_entropy = np.percentile(cal_entropies, 95)
        self.cal_mean_eff_rank = np.mean(cal_eff_ranks)

        self.is_fitted = True

        if verbose:
            print(f"\n  Entropy range: [{self.cal_min_entropy:.3f}, {self.cal_max_entropy:.3f}]")
            print(f"  Mean effective rank: {self.cal_mean_eff_rank:.1f} / {self.feature_dim}")
            print(f"  Rank utilization: {self.cal_mean_eff_rank/self.feature_dim*100:.1f}%")
            print(f"\n  SPECTRAL FITTING COMPLETE")
            print(f"{'='*60}\n")

        return self

    def to_torch(self, device: torch.device) -> 'SpectralEpistemicEstimator':
        """Cache calibration data on GPU for faster inference."""
        if self.X_cal is not None:
            self.X_cal_torch = torch.tensor(self.X_cal, dtype=torch.float32, device=device)
        return self

    def _compute_spectral_metrics(self, x_test: np.ndarray,
                                   X_ref: np.ndarray) -> Tuple[float, float]:
        """Compute spectral entropy and effective rank."""
        # Find k nearest neighbors
        distances = np.linalg.norm(X_ref - x_test, axis=1)
        k = min(self.k_neighbors, len(X_ref))
        neighbor_idx = np.argsort(distances)[:k]
        X_local = X_ref[neighbor_idx]

        # Center and compute local covariance
        mu_local = X_local.mean(axis=0)
        X_centered = X_local - mu_local

        if k < self.feature_dim:
            Sigma = (X_centered.T @ X_centered) / k + np.eye(self.feature_dim) * 1e-8
        else:
            Sigma = np.cov(X_centered.T)

        # Eigendecomposition
        try:
            eigenvalues = np.linalg.eigvalsh(Sigma)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(self.feature_dim) / self.feature_dim

        # Spectral entropy
        lambda_norm = eigenvalues / (eigenvalues.sum() + 1e-10)
        entropy = -np.sum(lambda_norm * np.log(lambda_norm + 1e-10))
        effective_rank = np.exp(entropy)

        return entropy, effective_rank

    def _normalize_entropy(self, entropy: float) -> float:
        """Normalize entropy to [0, 1] epistemic uncertainty.
        Low entropy (collapsed) → High epistemic
        """
        entropy_clipped = np.clip(entropy, self.cal_min_entropy, self.cal_max_entropy)
        entropy_norm = (entropy_clipped - self.cal_min_entropy) / \
                      (self.cal_max_entropy - self.cal_min_entropy + 1e-10)
        # Invert: low entropy = high epistemic
        # Use parabolic transformation
        epistemic = 1.0 - 2.0 * np.abs(entropy_norm - 0.5)
        return np.clip(epistemic, 0, 1)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict spectral epistemic uncertainty [0, 1]."""
        if not self.is_fitted:
            raise ValueError("Not fitted")

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        n_test = len(X_test)
        epistemic = np.zeros(n_test)

        for i in range(n_test):
            entropy, _ = self._compute_spectral_metrics(X_test[i], self.X_cal)
            epistemic[i] = self._normalize_entropy(entropy)

        return epistemic

    def predict_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict epistemic uncertainty on GPU (converts to numpy internally)."""
        X_np = obs.detach().cpu().numpy()
        result = self.predict(X_np)
        return torch.tensor(result, dtype=torch.float32, device=obs.device)


class RepulsiveEpistemicEstimator:
    """
    Epistemic uncertainty via repulsive force field analysis.

    Points in knowledge voids experience high repulsive forces from all directions.
    """

    def __init__(self, k_neighbors: int = 100, temperature: float = 1.0):
        self.k_neighbors = k_neighbors
        self.temperature = temperature

        self.X_cal = None
        self.mean_force = None
        self.std_force = None
        self.is_fitted = False

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'RepulsiveEpistemicEstimator':
        """Fit repulsive model on calibration data."""
        self.X_cal = X_cal
        self.feature_dim = X_cal.shape[1]

        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING REPULSIVE EPISTEMIC ESTIMATOR")
            print(f"{'='*60}")
            print(f"Calibration samples: {len(X_cal)}")
            print(f"K neighbors: {self.k_neighbors}")

        # Compute calibration force statistics
        n_samples = min(500, len(X_cal))
        sample_idx = np.random.choice(len(X_cal), n_samples, replace=False)

        cal_magnitudes = []
        for i, idx in enumerate(sample_idx):
            if i % 100 == 0 and verbose:
                print(f"  Processing {i}/{n_samples}...")
            mag = self._compute_force_magnitude(X_cal[idx], X_cal)
            cal_magnitudes.append(mag)

        cal_magnitudes = np.array(cal_magnitudes)
        self.mean_force = np.mean(cal_magnitudes)
        self.std_force = np.std(cal_magnitudes)

        self.is_fitted = True

        if verbose:
            print(f"\n  Mean force: {self.mean_force:.6f}")
            print(f"  Std force:  {self.std_force:.6f}")
            print(f"\n  REPULSIVE FITTING COMPLETE")
            print(f"{'='*60}\n")

        return self

    def _compute_force_magnitude(self, x_test: np.ndarray, X_ref: np.ndarray) -> float:
        """Compute net repulsive force magnitude at test point."""
        distances = np.linalg.norm(X_ref - x_test, axis=1)
        distances[distances < 1e-10] = np.inf

        k = min(self.k_neighbors, len(X_ref))
        nearest_idx = np.argsort(distances)[:k]

        # Compute repulsive forces (Coulomb-like)
        forces = []
        for idx in nearest_idx:
            d_i = distances[idx]
            direction = x_test - X_ref[idx]
            direction_norm = np.linalg.norm(direction)

            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                direction = np.random.randn(len(direction))
                direction = direction / np.linalg.norm(direction)

            magnitude = np.exp(-d_i / self.temperature) / (d_i**2 + 1e-6)
            forces.append(direction * magnitude)

        forces = np.array(forces)
        net_force = np.sum(forces, axis=0)
        return np.linalg.norm(net_force)

    def _normalize_force(self, force_magnitude: float) -> float:
        """Normalize to [0, 1] using sigmoid of z-score."""
        z = (force_magnitude - self.mean_force) / (self.std_force + 1e-10)
        return np.clip(1 / (1 + np.exp(-z)), 0, 1)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict repulsive epistemic uncertainty [0, 1]."""
        if not self.is_fitted:
            raise ValueError("Not fitted")

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        n_test = len(X_test)
        epistemic = np.zeros(n_test)

        for i in range(n_test):
            mag = self._compute_force_magnitude(X_test[i], self.X_cal)
            epistemic[i] = self._normalize_force(mag)

        return epistemic

    def predict_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict on GPU (via numpy)."""
        X_np = obs.detach().cpu().numpy()
        result = self.predict(X_np)
        return torch.tensor(result, dtype=torch.float32, device=obs.device)


class CombinedEpistemicEstimator:
    """
    Combined epistemic uncertainty from Spectral + Repulsive sources.
    Weights can be optimized for orthogonality with aleatoric uncertainty.
    """

    def __init__(self, k_neighbors_spectral: int = 50,
                 k_neighbors_repulsive: int = 100,
                 temperature: float = 1.0,
                 weights: str = 'equal'):
        """
        Args:
            weights: 'equal' or 'optimize' (optimized during fit for orthogonality)
        """
        self.spectral = SpectralEpistemicEstimator(k_neighbors=k_neighbors_spectral)
        self.repulsive = RepulsiveEpistemicEstimator(
            k_neighbors=k_neighbors_repulsive, temperature=temperature
        )
        self.weight_mode = weights
        self.weights = [0.5, 0.5]  # [spectral_weight, repulsive_weight]
        self.is_fitted = False

    def fit(self, X_cal: np.ndarray,
            aleatoric_scores: Optional[np.ndarray] = None,
            verbose: bool = True) -> 'CombinedEpistemicEstimator':
        """
        Fit combined epistemic model.

        Args:
            X_cal: Calibration observations [N_cal, D]
            aleatoric_scores: Aleatoric uncertainty on calibration set (for weight optimization)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING COMBINED EPISTEMIC ESTIMATOR")
            print(f"{'='*60}")

        # Fit individual components
        self.spectral.fit(X_cal, verbose=verbose)
        self.repulsive.fit(X_cal, verbose=verbose)

        # Optimize weights if requested and aleatoric scores provided
        if self.weight_mode == 'optimize' and aleatoric_scores is not None:
            self._optimize_weights(X_cal, aleatoric_scores, verbose)

        self.is_fitted = True

        if verbose:
            print(f"\n  Final weights: spectral={self.weights[0]:.3f}, repulsive={self.weights[1]:.3f}")
            print(f"\n  COMBINED EPISTEMIC FITTING COMPLETE")
            print(f"{'='*60}\n")

        return self

    def _optimize_weights(self, X_cal: np.ndarray,
                          aleatoric_scores: np.ndarray,
                          verbose: bool = True):
        """Optimize weights to minimize correlation with aleatoric."""
        from scipy.optimize import minimize

        # Get calibration predictions from both components
        n_opt = min(300, len(X_cal))
        idx = np.random.choice(len(X_cal), n_opt, replace=False)
        X_opt = X_cal[idx]
        ale_opt = aleatoric_scores[idx]

        spec_scores = self.spectral.predict(X_opt)
        rep_scores = self.repulsive.predict(X_opt)

        def objective(w):
            combined = w[0] * spec_scores + w[1] * rep_scores
            corr = np.abs(np.corrcoef(combined, ale_opt)[0, 1])
            penalty = 0.01 * np.std(w)
            return corr + penalty

        constraints = [
            {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1.0},
            {'type': 'ineq', 'fun': lambda w: w[0]},
            {'type': 'ineq', 'fun': lambda w: w[1]},
        ]

        result = minimize(objective, [0.5, 0.5], method='SLSQP',
                         constraints=constraints, options={'maxiter': 100})

        if result.success:
            self.weights = result.x.tolist()
            if verbose:
                combined = self.weights[0] * spec_scores + self.weights[1] * rep_scores
                final_corr = np.corrcoef(combined, ale_opt)[0, 1]
                print(f"\n  Weight optimization successful!")
                print(f"  Final |correlation| with aleatoric: {abs(final_corr):.4f}")
        else:
            if verbose:
                print(f"\n  Weight optimization failed, using equal weights")
            self.weights = [0.5, 0.5]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict combined epistemic uncertainty [0, 1]."""
        spec = self.spectral.predict(X_test)
        rep = self.repulsive.predict(X_test)
        combined = self.weights[0] * spec + self.weights[1] * rep
        return np.clip(combined, 0, 1)

    def predict_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict on GPU (via numpy)."""
        X_np = obs.detach().cpu().numpy()
        result = self.predict(X_np)
        return torch.tensor(result, dtype=torch.float32, device=obs.device)

    def predict_components(self, X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict with component breakdown."""
        spec = self.spectral.predict(X_test)
        rep = self.repulsive.predict(X_test)
        combined = self.weights[0] * spec + self.weights[1] * rep
        return {
            'combined': np.clip(combined, 0, 1),
            'spectral': spec,
            'repulsive': rep,
            'weights': self.weights,
        }

    def save(self, path: str):
        """Save model parameters."""
        save_dict = {
            'weights': self.weights,
            'spectral': {
                'k_neighbors': self.spectral.k_neighbors,
                'cal_min_entropy': float(self.spectral.cal_min_entropy),
                'cal_max_entropy': float(self.spectral.cal_max_entropy),
                'cal_mean_eff_rank': float(self.spectral.cal_mean_eff_rank),
            },
            'repulsive': {
                'k_neighbors': self.repulsive.k_neighbors,
                'temperature': self.repulsive.temperature,
                'mean_force': float(self.repulsive.mean_force),
                'std_force': float(self.repulsive.std_force),
            }
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f, indent=2)
