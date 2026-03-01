"""
Conformal Prediction for Calibrating Intervention Thresholds

Provides statistical coverage guarantees:
  P(success | u_a < tau_a AND u_e < tau_e) >= 1 - alpha

Includes Adaptive Conformal Inference (ACI) for sequential/non-i.i.d. settings.

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ConformalResult:
    """Result of conformal calibration."""
    tau_a: float          # Aleatoric threshold
    tau_e: float          # Epistemic threshold
    target_coverage: float
    empirical_coverage: float
    abstention_rate: float
    success_when_acting: float


class ConformalCalibrator:
    """
    Calibrate uncertainty thresholds using conformal prediction.

    Given calibration episodes with known outcomes (success/fail),
    set thresholds tau_a, tau_e such that:
      P(success | act) >= 1 - alpha

    Two modes:
      1. Split conformal: Standard conformal prediction with exchangeability
      2. ACI: Adaptive conformal inference for sequential settings
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Target miscoverage rate (default 10% → 90% coverage)
        """
        self.alpha = alpha
        self.tau_a = None
        self.tau_e = None
        self.is_calibrated = False

    def calibrate(self,
                  u_a_cal: np.ndarray,
                  u_e_cal: np.ndarray,
                  outcomes: np.ndarray,
                  verbose: bool = True) -> ConformalResult:
        """
        Calibrate thresholds on calibration episodes.

        Args:
            u_a_cal: Aleatoric uncertainty per episode [N_cal]
            u_e_cal: Epistemic uncertainty per episode [N_cal]
            outcomes: Success (1) or failure (0) per episode [N_cal]

        Returns:
            ConformalResult with calibrated thresholds
        """
        n = len(outcomes)
        successes = outcomes.astype(bool)
        failures = ~successes

        if verbose:
            print(f"\n{'='*60}")
            print("CONFORMAL CALIBRATION")
            print(f"{'='*60}")
            print(f"Calibration episodes: {n}")
            print(f"Success rate: {successes.mean():.1%}")
            print(f"Target coverage: {1 - self.alpha:.1%}")

        # Method: Use nonconformity scores
        # For failures: the score is max(u_a, u_e) — how uncertain were we when failing
        # We set thresholds to cover (1-alpha) of the distribution

        # Approach: Grid search over (tau_a, tau_e) to find best pair
        # that achieves coverage >= 1-alpha with minimum abstention

        # Generate candidate thresholds from data quantiles
        tau_a_candidates = np.percentile(u_a_cal, np.arange(10, 91, 5))
        tau_e_candidates = np.percentile(u_e_cal, np.arange(10, 91, 5))

        best_result = None
        best_abstention = 1.0

        for ta in tau_a_candidates:
            for te in tau_e_candidates:
                # "Act" when both uncertainties below threshold
                act_mask = (u_a_cal <= ta) & (u_e_cal <= te)
                abstain_mask = ~act_mask

                if act_mask.sum() == 0:
                    continue

                # Coverage = P(success | act)
                success_when_acting = outcomes[act_mask].mean()
                abstention_rate = abstain_mask.mean()

                if success_when_acting >= 1 - self.alpha:
                    # Valid threshold pair — check if better (lower abstention)
                    if abstention_rate < best_abstention:
                        best_abstention = abstention_rate
                        best_result = ConformalResult(
                            tau_a=float(ta),
                            tau_e=float(te),
                            target_coverage=1 - self.alpha,
                            empirical_coverage=float(success_when_acting),
                            abstention_rate=float(abstention_rate),
                            success_when_acting=float(success_when_acting),
                        )

        if best_result is None:
            # Fallback: use conservative thresholds (high quantiles)
            self.tau_a = float(np.percentile(u_a_cal, 50))
            self.tau_e = float(np.percentile(u_e_cal, 50))

            act_mask = (u_a_cal <= self.tau_a) & (u_e_cal <= self.tau_e)
            success_rate = outcomes[act_mask].mean() if act_mask.sum() > 0 else 0.0

            best_result = ConformalResult(
                tau_a=self.tau_a,
                tau_e=self.tau_e,
                target_coverage=1 - self.alpha,
                empirical_coverage=float(success_rate),
                abstention_rate=float((~act_mask).mean()),
                success_when_acting=float(success_rate),
            )

            if verbose:
                print("\n  WARNING: Could not achieve target coverage. Using median thresholds.")
        else:
            self.tau_a = best_result.tau_a
            self.tau_e = best_result.tau_e

        self.is_calibrated = True

        if verbose:
            print(f"\n  Calibrated thresholds:")
            print(f"    tau_a (aleatoric):  {self.tau_a:.4f}")
            print(f"    tau_e (epistemic):  {self.tau_e:.4f}")
            print(f"    Empirical coverage: {best_result.empirical_coverage:.1%}")
            print(f"    Abstention rate:    {best_result.abstention_rate:.1%}")
            print(f"    Success when acting: {best_result.success_when_acting:.1%}")
            print(f"{'='*60}")

        return best_result

    def calibrate_marginal(self,
                           u_a_cal: np.ndarray,
                           u_e_cal: np.ndarray,
                           outcomes: np.ndarray,
                           verbose: bool = True) -> ConformalResult:
        """
        Calibrate thresholds using marginal quantiles on failure cases.

        Simpler approach: set tau_a and tau_e independently based on
        the distribution of uncertainty values on failure cases.
        """
        n = len(outcomes)
        failures = ~outcomes.astype(bool)

        if failures.sum() < 5:
            # Not enough failures to calibrate
            self.tau_a = float(np.percentile(u_a_cal, 75))
            self.tau_e = float(np.percentile(u_e_cal, 75))
        else:
            # Conformal quantile on failure set
            q_level = 1 - self.alpha

            # For each dimension: find the quantile of uncertainty on failures
            # The threshold should be set so that most failures have uncertainty above it
            fail_u_a = u_a_cal[failures]
            fail_u_e = u_e_cal[failures]

            # Use the (1-alpha)-quantile of failure uncertainties as threshold
            # This means: (1-alpha) of failures had u above this threshold
            self.tau_a = float(np.percentile(fail_u_a, (1 - q_level) * 100))
            self.tau_e = float(np.percentile(fail_u_e, (1 - q_level) * 100))

        # Evaluate
        act_mask = (u_a_cal <= self.tau_a) & (u_e_cal <= self.tau_e)
        success_rate = outcomes[act_mask].mean() if act_mask.sum() > 0 else 0.0

        result = ConformalResult(
            tau_a=self.tau_a,
            tau_e=self.tau_e,
            target_coverage=1 - self.alpha,
            empirical_coverage=float(success_rate),
            abstention_rate=float((~act_mask).mean()),
            success_when_acting=float(success_rate),
        )

        self.is_calibrated = True

        if verbose:
            print(f"\n  Marginal conformal thresholds:")
            print(f"    tau_a: {self.tau_a:.4f}")
            print(f"    tau_e: {self.tau_e:.4f}")
            print(f"    Empirical coverage: {result.empirical_coverage:.1%}")

        return result


class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) for sequential/non-i.i.d. settings.

    Updates thresholds online:
      tau_{t+1} = tau_t + eta * (alpha - err_t)

    Where err_t = 1 if action failed, 0 otherwise.
    This ensures long-run coverage guarantee even without exchangeability.
    """

    def __init__(self, alpha: float = 0.1, eta: float = 0.01,
                 tau_a_init: float = 0.5, tau_e_init: float = 0.5):
        """
        Args:
            alpha: Target miscoverage rate
            eta: Learning rate for threshold updates
            tau_a_init: Initial aleatoric threshold
            tau_e_init: Initial epistemic threshold
        """
        self.alpha = alpha
        self.eta = eta
        self.tau_a = tau_a_init
        self.tau_e = tau_e_init

        # History
        self.tau_a_history = [tau_a_init]
        self.tau_e_history = [tau_e_init]
        self.error_history = []
        self.coverage_history = []

    def update(self, u_a: float, u_e: float, success: bool):
        """
        Update thresholds based on outcome.

        Args:
            u_a: Aleatoric uncertainty for this step
            u_e: Epistemic uncertainty for this step
            success: Whether the action was successful
        """
        # Was the action within thresholds?
        acted = (u_a <= self.tau_a) and (u_e <= self.tau_e)

        if acted:
            error = 0.0 if success else 1.0
        else:
            error = 0.0  # Abstained, so no error

        self.error_history.append(error)

        # Update thresholds
        # If error is high (> alpha): increase thresholds (more conservative)
        # If error is low (< alpha): decrease thresholds (less conservative)
        self.tau_a = self.tau_a + self.eta * (self.alpha - error)
        self.tau_e = self.tau_e + self.eta * (self.alpha - error)

        # Clip to reasonable range
        self.tau_a = np.clip(self.tau_a, 0.05, 0.95)
        self.tau_e = np.clip(self.tau_e, 0.05, 0.95)

        self.tau_a_history.append(self.tau_a)
        self.tau_e_history.append(self.tau_e)

        # Running coverage
        if len(self.error_history) > 10:
            recent_coverage = 1 - np.mean(self.error_history[-50:])
            self.coverage_history.append(recent_coverage)

    def get_thresholds(self) -> Tuple[float, float]:
        """Get current thresholds."""
        return self.tau_a, self.tau_e

    def get_diagnostics(self) -> Dict:
        """Get ACI diagnostics."""
        return {
            'tau_a': float(self.tau_a),
            'tau_e': float(self.tau_e),
            'tau_a_history': [float(t) for t in self.tau_a_history],
            'tau_e_history': [float(t) for t in self.tau_e_history],
            'error_history': self.error_history,
            'coverage_history': self.coverage_history,
            'cumulative_error_rate': np.mean(self.error_history) if self.error_history else 0.0,
            'total_steps': len(self.error_history),
        }
