"""
Orthogonality Verification for Uncertainty Decomposition

Verifies that aleatoric and epistemic uncertainty estimates are orthogonal
(statistically independent), which is the core theoretical requirement.

Metrics:
  - Pearson |r| < 0.30    (linear independence)
  - Spearman |rho| < 0.20 (monotonic independence)
  - HSIC p-value > 0.05   (general independence)
  - CKA < 0.20            (representation similarity)

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats


class OrthogonalityAnalyzer:
    """Analyze orthogonality between aleatoric and epistemic uncertainty estimates."""

    # Thresholds for passing
    THRESHOLDS = {
        'pearson': 0.30,
        'spearman': 0.20,
        'hsic_pvalue': 0.05,
        'cka': 0.20,
    }

    def analyze(self, u_aleatoric: np.ndarray,
                u_epistemic: np.ndarray,
                verbose: bool = True) -> Dict:
        """
        Run full orthogonality analysis.

        Args:
            u_aleatoric: Aleatoric uncertainty scores [N]
            u_epistemic: Epistemic uncertainty scores [N]

        Returns:
            Dictionary with all metrics and pass/fail status
        """
        results = {}

        # 1. Pearson correlation (linear)
        r, p_pearson = stats.pearsonr(u_aleatoric, u_epistemic)
        results['pearson_r'] = float(r)
        results['pearson_abs_r'] = float(abs(r))
        results['pearson_p'] = float(p_pearson)
        results['pearson_pass'] = bool(abs(r) < self.THRESHOLDS['pearson'])

        # 2. Spearman rank correlation (monotonic)
        rho, p_spearman = stats.spearmanr(u_aleatoric, u_epistemic)
        results['spearman_rho'] = float(rho)
        results['spearman_abs_rho'] = float(abs(rho))
        results['spearman_p'] = float(p_spearman)
        results['spearman_pass'] = bool(abs(rho) < self.THRESHOLDS['spearman'])

        # 3. HSIC (Hilbert-Schmidt Independence Criterion)
        hsic_stat, hsic_p = self._compute_hsic(u_aleatoric, u_epistemic)
        results['hsic_stat'] = float(hsic_stat)
        results['hsic_p'] = float(hsic_p)
        results['hsic_pass'] = bool(hsic_p > self.THRESHOLDS['hsic_pvalue'])

        # 4. CKA (Centered Kernel Alignment)
        cka = self._compute_cka(u_aleatoric, u_epistemic)
        results['cka'] = float(cka)
        results['cka_pass'] = bool(cka < self.THRESHOLDS['cka'])

        # Overall pass
        results['all_pass'] = all([
            results['pearson_pass'],
            results['spearman_pass'],
            results['hsic_pass'],
            results['cka_pass'],
        ])

        if verbose:
            self._print_results(results)

        return results

    def behavioral_test(self,
                        u_a_noise_sweep: Dict[str, np.ndarray],
                        u_e_noise_sweep: Dict[str, np.ndarray],
                        u_a_ood_sweep: Dict[str, np.ndarray],
                        u_e_ood_sweep: Dict[str, np.ndarray],
                        verbose: bool = True) -> Dict:
        """
        Behavioral isolation test.

        Tests that:
          - Varying noise levels affects u_a but NOT u_e
          - Varying OOD levels affects u_e but NOT u_a

        Args:
            u_a_noise_sweep: {noise_level: u_a_values} for noise sweep
            u_e_noise_sweep: {noise_level: u_e_values} for noise sweep
            u_a_ood_sweep: {ood_level: u_a_values} for OOD sweep
            u_e_ood_sweep: {ood_level: u_e_values} for OOD sweep

        Returns:
            Dictionary with behavioral test results
        """
        results = {}

        # Test 1: u_a should increase with noise
        noise_levels = sorted(u_a_noise_sweep.keys())
        u_a_means = [np.mean(u_a_noise_sweep[k]) for k in noise_levels]
        u_e_means_noise = [np.mean(u_e_noise_sweep[k]) for k in noise_levels]

        # u_a should correlate positively with noise level
        if len(noise_levels) >= 3:
            r_a_noise, _ = stats.spearmanr(range(len(noise_levels)), u_a_means)
            r_e_noise, _ = stats.spearmanr(range(len(noise_levels)), u_e_means_noise)
        else:
            r_a_noise = 1.0 if u_a_means[-1] > u_a_means[0] else 0.0
            r_e_noise = 0.0

        results['noise_sweep'] = {
            'u_a_correlation_with_noise': float(r_a_noise),
            'u_e_correlation_with_noise': float(r_e_noise),
            'u_a_responsive': r_a_noise > 0.5,  # u_a should respond to noise
            'u_e_invariant': abs(r_e_noise) < 0.3,  # u_e should NOT respond to noise
            'u_a_means': {str(k): float(v) for k, v in zip(noise_levels, u_a_means)},
            'u_e_means': {str(k): float(v) for k, v in zip(noise_levels, u_e_means_noise)},
        }

        # Test 2: u_e should increase with OOD severity
        ood_levels = sorted(u_a_ood_sweep.keys())
        u_a_means_ood = [np.mean(u_a_ood_sweep[k]) for k in ood_levels]
        u_e_means_ood = [np.mean(u_e_ood_sweep[k]) for k in ood_levels]

        if len(ood_levels) >= 3:
            r_a_ood, _ = stats.spearmanr(range(len(ood_levels)), u_a_means_ood)
            r_e_ood, _ = stats.spearmanr(range(len(ood_levels)), u_e_means_ood)
        else:
            r_a_ood = 0.0
            r_e_ood = 1.0 if u_e_means_ood[-1] > u_e_means_ood[0] else 0.0

        results['ood_sweep'] = {
            'u_a_correlation_with_ood': float(r_a_ood),
            'u_e_correlation_with_ood': float(r_e_ood),
            'u_a_invariant': abs(r_a_ood) < 0.3,  # u_a should NOT respond to OOD
            'u_e_responsive': r_e_ood > 0.5,  # u_e should respond to OOD
            'u_a_means': {str(k): float(v) for k, v in zip(ood_levels, u_a_means_ood)},
            'u_e_means': {str(k): float(v) for k, v in zip(ood_levels, u_e_means_ood)},
        }

        # Overall behavioral pass
        results['behavioral_pass'] = all([
            results['noise_sweep']['u_a_responsive'],
            results['noise_sweep']['u_e_invariant'],
            results['ood_sweep']['u_a_invariant'],
            results['ood_sweep']['u_e_responsive'],
        ])

        if verbose:
            print(f"\n{'='*60}")
            print("BEHAVIORAL ISOLATION TEST")
            print(f"{'='*60}")
            print(f"\nNoise Sweep:")
            print(f"  u_a correlation with noise: {r_a_noise:.3f} ({'PASS' if results['noise_sweep']['u_a_responsive'] else 'FAIL'})")
            print(f"  u_e correlation with noise: {r_e_noise:.3f} ({'PASS' if results['noise_sweep']['u_e_invariant'] else 'FAIL'})")
            print(f"\nOOD Sweep:")
            print(f"  u_a correlation with OOD:   {r_a_ood:.3f} ({'PASS' if results['ood_sweep']['u_a_invariant'] else 'FAIL'})")
            print(f"  u_e correlation with OOD:   {r_e_ood:.3f} ({'PASS' if results['ood_sweep']['u_e_responsive'] else 'FAIL'})")
            print(f"\nOverall: {'PASS' if results['behavioral_pass'] else 'FAIL'}")
            print(f"{'='*60}")

        return results

    def _compute_hsic(self, x: np.ndarray, y: np.ndarray,
                      sigma: float = None) -> Tuple[float, float]:
        """
        Compute HSIC (Hilbert-Schmidt Independence Criterion).
        Uses RBF kernel. P-value from permutation test.
        """
        n = len(x)
        if sigma is None:
            sigma_x = np.median(np.abs(x[:, None] - x[None, :]))
            sigma_y = np.median(np.abs(y[:, None] - y[None, :]))
            sigma_x = max(sigma_x, 1e-6)
            sigma_y = max(sigma_y, 1e-6)
        else:
            sigma_x = sigma_y = sigma

        # RBF kernels
        K = np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / sigma_x ** 2)
        L = np.exp(-0.5 * (y[:, None] - y[None, :]) ** 2 / sigma_y ** 2)

        # Center kernels
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ K @ H
        Lc = H @ L @ H

        # HSIC statistic
        hsic = np.trace(Kc @ Lc) / (n ** 2)

        # Permutation test for p-value
        n_perm = 200
        perm_hsics = []
        for _ in range(n_perm):
            perm_idx = np.random.permutation(n)
            L_perm = L[perm_idx][:, perm_idx]
            Lc_perm = H @ L_perm @ H
            perm_hsic = np.trace(Kc @ Lc_perm) / (n ** 2)
            perm_hsics.append(perm_hsic)

        p_value = np.mean(np.array(perm_hsics) >= hsic)
        return hsic, p_value

    def _compute_cka(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute CKA (Centered Kernel Alignment).
        Linear CKA for efficiency.
        """
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y

        # Linear CKA
        xx = x.T @ x
        yy = y.T @ y
        xy = x.T @ y

        hsic_xy = np.trace(xy @ xy.T)
        hsic_xx = np.trace(xx @ xx)
        hsic_yy = np.trace(yy @ yy)

        denom = np.sqrt(hsic_xx * hsic_yy)
        if denom < 1e-10:
            return 0.0

        return float(hsic_xy / denom)

    def _print_results(self, results: Dict):
        """Print orthogonality results."""
        print(f"\n{'='*60}")
        print("ORTHOGONALITY VERIFICATION")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Value':>10} {'Threshold':>12} {'Status':>8}")
        print("-" * 60)
        print(f"{'Pearson |r|':<25} {results['pearson_abs_r']:>10.4f} {'< ' + str(self.THRESHOLDS['pearson']):>12} "
              f"{'PASS' if results['pearson_pass'] else 'FAIL':>8}")
        print(f"{'Spearman |rho|':<25} {results['spearman_abs_rho']:>10.4f} {'< ' + str(self.THRESHOLDS['spearman']):>12} "
              f"{'PASS' if results['spearman_pass'] else 'FAIL':>8}")
        print(f"{'HSIC p-value':<25} {results['hsic_p']:>10.4f} {'> ' + str(self.THRESHOLDS['hsic_pvalue']):>12} "
              f"{'PASS' if results['hsic_pass'] else 'FAIL':>8}")
        print(f"{'CKA':<25} {results['cka']:>10.4f} {'< ' + str(self.THRESHOLDS['cka']):>12} "
              f"{'PASS' if results['cka_pass'] else 'FAIL':>8}")
        print("-" * 60)
        print(f"{'OVERALL':<25} {'':>10} {'':>12} {'PASS' if results['all_pass'] else 'FAIL':>8}")
        print(f"{'='*60}")
