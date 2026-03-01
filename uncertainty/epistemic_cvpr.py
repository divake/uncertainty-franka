"""
CVPR-Consistent Epistemic Uncertainty for Robotics

Two components matching the CVPR paper's methodology:
  1. ε_knn: k-NN distance in standardized state space (analogous to CVPR ε_supp)
  2. ε_rank: Spectral entropy of local covariance (SAME formula as CVPR Sec 3.1)

All signals are post-hoc, require zero training, and use only calibration data statistics.
This is consistent with the CVPR paper's philosophy: "no sampling, no ensembling,
and no additional forward passes."

For IROS 2026: Extension of arXiv:2511.12389 to robot manipulation.
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from scipy.spatial import cKDTree
import json


# Dimensions to EXCLUDE from distance/covariance computations.
# Dims 24-27 in Lift (target quaternion) have zero variance and break computations.
ZERO_VAR_DIMS_LIFT = [24, 25, 26, 27]


def _get_active_dims(obs_dim: int, zero_var_dims: list = None) -> np.ndarray:
    """Get indices of active (non-zero-variance) dimensions."""
    all_dims = np.arange(obs_dim)
    if zero_var_dims is None or len(zero_var_dims) == 0:
        return all_dims
    return np.array([d for d in all_dims if d not in zero_var_dims])


class KNNEpistemicEstimator:
    """
    ε_knn: k-th nearest neighbor distance in standardized state space.

    Analogous to CVPR Section 3.1 ε_supp (local support deficiency).
    States far from any calibration point receive high epistemic scores.

    Reference: Sun et al. (ICML 2022) "Out-of-Distribution Detection with
    Deep Nearest Neighbors"
    """

    def __init__(self, k: int = 20, zero_var_dims: list = None):
        """
        Args:
            k: Number of nearest neighbors.
            zero_var_dims: Dimension indices to exclude (zero variance).
        """
        self.k = k
        self.zero_var_dims = zero_var_dims or []
        self.active_dims = None

        # Standardization parameters
        self.mean_ = None     # [D_active]
        self.std_ = None      # [D_active]

        # k-NN index
        self.tree_ = None

        # Normalization (from calibration distances)
        self.cal_dist_min_ = None
        self.cal_dist_max_ = None

        # PyTorch cached
        self.X_cal_std_torch_ = None
        self.mean_torch_ = None
        self.std_torch_ = None

        self.is_fitted = False

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'KNNEpistemicEstimator':
        """
        Fit k-NN index on calibration data.

        Args:
            X_cal: Calibration observations [N_cal, D]
        """
        N, D = X_cal.shape
        self.active_dims = _get_active_dims(D, self.zero_var_dims)
        D_active = len(self.active_dims)

        X_active = X_cal[:, self.active_dims]

        # Z-score standardization (heterogeneous units: radians, m/s, meters)
        self.mean_ = np.mean(X_active, axis=0)
        self.std_ = np.std(X_active, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # prevent division by zero

        X_std = (X_active - self.mean_) / self.std_

        # Build k-d tree
        self.tree_ = cKDTree(X_std)

        # Compute calibration distances for normalization
        # Use a subsample for speed
        n_sample = min(5000, N)
        idx = np.random.choice(N, n_sample, replace=False)
        dists, _ = self.tree_.query(X_std[idx], k=self.k + 1)  # +1 because self is included
        cal_knn_dist = dists[:, -1]  # k-th neighbor distance (excluding self)

        # Use log transform for heavy-tailed distribution + wider percentiles
        log_cal_dist = np.log(cal_knn_dist + 1e-8)
        self.cal_log_dist_median_ = np.median(log_cal_dist)
        self.cal_log_dist_iqr_ = np.percentile(log_cal_dist, 75) - np.percentile(log_cal_dist, 25)
        if self.cal_log_dist_iqr_ < 1e-8:
            self.cal_log_dist_iqr_ = 1.0

        # Also store raw percentiles for verbose output
        self.cal_dist_min_ = np.percentile(cal_knn_dist, 5)
        self.cal_dist_max_ = np.percentile(cal_knn_dist, 95)

        self.is_fitted = True

        if verbose:
            print(f"\n  ε_knn (k-NN Epistemic) fitted:")
            print(f"    Calibration points: {N}")
            print(f"    Active dimensions: {D_active} (excluded: {self.zero_var_dims})")
            print(f"    k = {self.k}")
            print(f"    Cal k-NN distance: [{self.cal_dist_min_:.4f}, {self.cal_dist_max_:.4f}]")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ε_knn for test observations (NumPy).

        IMPORTANT: This should be called on GROUND TRUTH observations,
        not noisy observations. The epistemic signal measures state novelty,
        which is independent of sensor noise. Noise is handled by σ_alea.

        Args:
            X: Test observations [N_test, D] — ground truth or best estimate

        Returns:
            ε_knn: [N_test] in [0, 1]
        """
        X_active = X[:, self.active_dims]
        X_std = (X_active - self.mean_) / self.std_

        dists, _ = self.tree_.query(X_std, k=self.k)
        knn_dist = dists[:, -1]  # k-th neighbor distance

        # Log-space normalization (matches CVPR's approach for Mahalanobis)
        # This handles the heavy-tailed distance distribution gracefully
        log_dist = np.log(knn_dist + 1e-8)
        log_min = np.log(self.cal_dist_min_ + 1e-8)
        log_max = np.log(self.cal_dist_max_ + 1e-8)
        eps_knn = (log_dist - log_min) / (log_max - log_min + 1e-8)
        return np.clip(eps_knn, 0.0, 1.0)

    def predict_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict ε_knn on GPU. Falls back to CPU for k-NN lookup.

        Args:
            obs: [batch, D] on GPU

        Returns:
            ε_knn: [batch] on same device
        """
        device = obs.device
        X_np = obs.detach().cpu().numpy()
        eps = self.predict(X_np)
        return torch.tensor(eps, dtype=torch.float32, device=device)

    def save(self, path: str):
        d = {
            'k': self.k,
            'zero_var_dims': self.zero_var_dims,
            'active_dims': self.active_dims.tolist(),
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'cal_dist_min': float(self.cal_dist_min_),
            'cal_dist_max': float(self.cal_dist_max_),
            'cal_log_dist_median': float(self.cal_log_dist_median_),
            'cal_log_dist_iqr': float(self.cal_log_dist_iqr_),
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    def load(self, path: str, X_cal: np.ndarray) -> 'KNNEpistemicEstimator':
        """Load parameters and rebuild k-d tree from calibration data."""
        with open(path, 'r') as f:
            d = json.load(f)
        self.k = d['k']
        self.zero_var_dims = d['zero_var_dims']
        self.active_dims = np.array(d['active_dims'])
        self.mean_ = np.array(d['mean'])
        self.std_ = np.array(d['std'])
        self.cal_dist_min_ = d['cal_dist_min']
        self.cal_dist_max_ = d['cal_dist_max']
        self.cal_log_dist_median_ = d['cal_log_dist_median']
        self.cal_log_dist_iqr_ = d['cal_log_dist_iqr']

        # Rebuild tree
        X_active = X_cal[:, self.active_dims]
        X_std = (X_active - self.mean_) / self.std_
        self.tree_ = cKDTree(X_std)
        self.is_fitted = True
        return self


class RankEpistemicEstimator:
    """
    ε_rank: Spectral entropy of local covariance — SAME formula as CVPR Sec 3.1.

    Measures geometric collapse in the local neighborhood of calibration data.
    When the robot enters a region where calibration data has degenerate local
    structure (low effective rank), epistemic uncertainty is high.

    Formula (identical to CVPR):
      Σ_loc = (1/k) X_loc^T X_loc
      H = -Σ p_i log p_i,  where p_i = λ_i / Σ λ_j
      r_eff = exp(H)
      ε_rank = 1 - (r_eff - 1) / (d - 1)
    """

    def __init__(self, k: int = 50, zero_var_dims: list = None):
        """
        Args:
            k: Number of nearest neighbors for local covariance.
            zero_var_dims: Dimension indices to exclude.
        """
        self.k = k
        self.zero_var_dims = zero_var_dims or []
        self.active_dims = None
        self.d_active = None

        # Standardization
        self.mean_ = None
        self.std_ = None

        # k-NN index and standardized calibration data
        self.tree_ = None
        self.X_cal_std_ = None

        # Normalization from calibration
        self.cal_rank_min_ = None
        self.cal_rank_max_ = None

        self.is_fitted = False

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'RankEpistemicEstimator':
        """
        Fit on calibration data.

        Args:
            X_cal: Calibration observations [N_cal, D]
        """
        N, D = X_cal.shape
        self.active_dims = _get_active_dims(D, self.zero_var_dims)
        self.d_active = len(self.active_dims)

        X_active = X_cal[:, self.active_dims]

        # Standardize
        self.mean_ = np.mean(X_active, axis=0)
        self.std_ = np.std(X_active, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0

        self.X_cal_std_ = (X_active - self.mean_) / self.std_

        # Build tree
        self.tree_ = cKDTree(self.X_cal_std_)

        # Calibrate normalization on subsample
        n_sample = min(1000, N)
        idx = np.random.choice(N, n_sample, replace=False)
        cal_ranks = np.array([self._compute_rank_score(self.X_cal_std_[i]) for i in idx])

        self.cal_rank_min_ = np.percentile(cal_ranks, 5)
        self.cal_rank_max_ = np.percentile(cal_ranks, 95)

        self.is_fitted = True

        if verbose:
            print(f"\n  ε_rank (Spectral Entropy) fitted:")
            print(f"    Calibration points: {N}")
            print(f"    Active dimensions: {self.d_active}")
            print(f"    k = {self.k}")
            print(f"    Cal ε_rank range: [{self.cal_rank_min_:.4f}, {self.cal_rank_max_:.4f}]")

        return self

    def _compute_rank_score(self, x_std: np.ndarray) -> float:
        """
        Compute ε_rank for a single standardized observation.

        SAME formula as CVPR Section 3.1 (geometric collapse / spectral entropy).
        """
        # Find k nearest neighbors
        dists, indices = self.tree_.query(x_std, k=self.k)
        neighbors = self.X_cal_std_[indices]

        # Center neighbors around their mean
        X_loc = neighbors - neighbors.mean(axis=0)

        # Local covariance
        Sigma_loc = (X_loc.T @ X_loc) / self.k

        # Eigendecomposition
        eigenvalues = np.linalg.eigvalsh(Sigma_loc)
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # numerical stability

        # Spectral entropy (CVPR formula)
        p = eigenvalues / eigenvalues.sum()
        H = -np.sum(p * np.log(p + 1e-12))

        # Effective rank
        r_eff = np.exp(H)

        # ε_rank (CVPR formula): high when geometry is collapsed
        d = self.d_active
        eps_rank = 1.0 - (r_eff - 1.0) / (d - 1.0)

        return eps_rank

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ε_rank for test observations (NumPy).

        Args:
            X: [N_test, D]

        Returns:
            ε_rank: [N_test] in [0, 1]
        """
        X_active = X[:, self.active_dims]
        X_std = (X_active - self.mean_) / self.std_

        scores = np.array([self._compute_rank_score(x) for x in X_std])

        # Normalize to [0, 1]
        eps_rank = (scores - self.cal_rank_min_) / (self.cal_rank_max_ - self.cal_rank_min_ + 1e-8)
        return np.clip(eps_rank, 0.0, 1.0)

    def predict_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict ε_rank on GPU. Falls back to CPU for k-NN + eigendecomposition.
        """
        device = obs.device
        X_np = obs.detach().cpu().numpy()
        eps = self.predict(X_np)
        return torch.tensor(eps, dtype=torch.float32, device=device)

    def save(self, path: str):
        d = {
            'k': self.k,
            'zero_var_dims': self.zero_var_dims,
            'active_dims': self.active_dims.tolist(),
            'd_active': self.d_active,
            'mean': self.mean_.tolist(),
            'std': self.std_.tolist(),
            'cal_rank_min': float(self.cal_rank_min_),
            'cal_rank_max': float(self.cal_rank_max_),
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    def load(self, path: str, X_cal: np.ndarray) -> 'RankEpistemicEstimator':
        with open(path, 'r') as f:
            d = json.load(f)
        self.k = d['k']
        self.zero_var_dims = d['zero_var_dims']
        self.active_dims = np.array(d['active_dims'])
        self.d_active = d['d_active']
        self.mean_ = np.array(d['mean'])
        self.std_ = np.array(d['std'])
        self.cal_rank_min_ = d['cal_rank_min']
        self.cal_rank_max_ = d['cal_rank_max']

        X_active = X_cal[:, self.active_dims]
        self.X_cal_std_ = (X_active - self.mean_) / self.std_
        self.tree_ = cKDTree(self.X_cal_std_)
        self.is_fitted = True
        return self


class CVPREpistemicEstimator:
    """
    Combined epistemic uncertainty: σ_epis = w₁·ε_knn + w₂·ε_rank

    Weights optimized for orthogonality with σ_alea (same methodology as CVPR).
    """

    def __init__(self, k_knn: int = 20, k_rank: int = 50, zero_var_dims: list = None):
        self.knn_estimator = KNNEpistemicEstimator(k=k_knn, zero_var_dims=zero_var_dims)
        self.rank_estimator = RankEpistemicEstimator(k=k_rank, zero_var_dims=zero_var_dims)

        self.w_knn = 0.5
        self.w_rank = 0.5

        self.is_fitted = False

    def fit(self, X_cal: np.ndarray, verbose: bool = True) -> 'CVPREpistemicEstimator':
        """Fit both components on calibration data."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"FITTING CVPR-CONSISTENT EPISTEMIC UNCERTAINTY")
            print(f"{'='*60}")

        self.knn_estimator.fit(X_cal, verbose=verbose)
        self.rank_estimator.fit(X_cal, verbose=verbose)
        self.is_fitted = True

        if verbose:
            print(f"\n  Combined σ_epis = {self.w_knn:.2f}·ε_knn + {self.w_rank:.2f}·ε_rank")
            print(f"{'='*60}\n")

        return self

    def optimize_weights(self, X_test: np.ndarray, sigma_alea: np.ndarray,
                         verbose: bool = True):
        """
        Optimize weights to minimize correlation with σ_alea.
        Same methodology as CVPR: choose weights that maximize orthogonality.

        Args:
            X_test: Test observations [N, D]
            sigma_alea: Corresponding aleatoric uncertainty [N]
        """
        from scipy.optimize import minimize

        eps_knn = self.knn_estimator.predict(X_test)
        eps_rank = self.rank_estimator.predict(X_test)

        def neg_orthogonality(w):
            w1, w2 = w[0], 1.0 - w[0]
            sigma_epis = w1 * eps_knn + w2 * eps_rank
            corr = np.abs(np.corrcoef(sigma_alea, sigma_epis)[0, 1])
            return corr

        result = minimize(neg_orthogonality, x0=[0.5], bounds=[(0.01, 0.99)],
                          method='L-BFGS-B')
        self.w_knn = result.x[0]
        self.w_rank = 1.0 - self.w_knn

        if verbose:
            final_corr = result.fun
            print(f"\n  Weight optimization: w_knn={self.w_knn:.3f}, w_rank={self.w_rank:.3f}")
            print(f"  |corr(σ_alea, σ_epis)| = {final_corr:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict σ_epis [0, 1] (NumPy)."""
        eps_knn = self.knn_estimator.predict(X)
        eps_rank = self.rank_estimator.predict(X)
        sigma_epis = self.w_knn * eps_knn + self.w_rank * eps_rank
        return np.clip(sigma_epis, 0.0, 1.0)

    def predict_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict σ_epis [0, 1] (PyTorch)."""
        eps_knn = self.knn_estimator.predict_torch(obs)
        eps_rank = self.rank_estimator.predict_torch(obs)
        sigma_epis = self.w_knn * eps_knn + self.w_rank * eps_rank
        return torch.clamp(sigma_epis, 0.0, 1.0)

    def predict_components(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (σ_epis, ε_knn, ε_rank) for analysis."""
        eps_knn = self.knn_estimator.predict(X)
        eps_rank = self.rank_estimator.predict(X)
        sigma_epis = self.w_knn * eps_knn + self.w_rank * eps_rank
        return np.clip(sigma_epis, 0.0, 1.0), eps_knn, eps_rank

    def save(self, directory: str):
        from pathlib import Path
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.knn_estimator.save(str(d / 'knn_params.json'))
        self.rank_estimator.save(str(d / 'rank_params.json'))
        with open(str(d / 'combined_params.json'), 'w') as f:
            json.dump({'w_knn': self.w_knn, 'w_rank': self.w_rank}, f, indent=2)

    def load(self, directory: str, X_cal: np.ndarray) -> 'CVPREpistemicEstimator':
        from pathlib import Path
        d = Path(directory)
        self.knn_estimator.load(str(d / 'knn_params.json'), X_cal)
        self.rank_estimator.load(str(d / 'rank_params.json'), X_cal)
        with open(str(d / 'combined_params.json'), 'r') as f:
            params = json.load(f)
        self.w_knn = params['w_knn']
        self.w_rank = params['w_rank']
        self.is_fitted = True
        return self
