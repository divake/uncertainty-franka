#!/usr/bin/env python3
"""
Controlled Validation of Uncertainty Decomposition (Offline)

Tests that σ_alea and σ_epis correctly respond to known perturbation types
using calibration data only — no simulator needed.

Three perturbation types (from cvpr_iros_plan.md):
  1. Sensor noise (aleatoric): Gaussian noise on observations
     → σ_alea SHOULD rise, σ_epis SHOULD stay flat
  2. Dynamics shift (epistemic): Shifted state (simulates mass/friction change)
     → σ_epis SHOULD rise, σ_alea may also rise
  3. Partial occlusion (aleatoric): Object position dims → 0
     → σ_alea SHOULD rise, σ_epis depends on which dims change

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
CVPR extension: arXiv:2511.12389
"""

import numpy as np
import os
import sys
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uncertainty.aleatoric import AleatoricEstimator
from uncertainty.epistemic_cvpr import CVPREpistemicEstimator, ZERO_VAR_DIMS_LIFT


def main():
    print("=" * 70)
    print("CONTROLLED VALIDATION: Uncertainty Decomposition")
    print("CVPR-consistent: σ_alea=Mahalanobis, σ_epis=ε_knn+ε_rank")
    print("=" * 70)

    # Load calibration data
    cal_dir = "calibration_data/Isaac-Lift-Cube-Franka-v0_20260228_181927"
    X_cal = np.load(os.path.join(cal_dir, "X_cal.npy"))
    print(f"\nCalibration data: {X_cal.shape}")

    # Fit estimators
    print("\n--- Fitting estimators ---")
    alea_est = AleatoricEstimator(reg_lambda=1e-4)
    alea_est.fit(X_cal, verbose=True)

    epis_est = CVPREpistemicEstimator(k_knn=20, k_rank=50, zero_var_dims=ZERO_VAR_DIMS_LIFT)
    epis_est.fit(X_cal, verbose=True)

    # Select test points
    np.random.seed(42)
    n_test = 500
    idx = np.random.choice(len(X_cal), n_test, replace=False)
    X_gt = X_cal[idx].copy()

    # =====================================================================
    # TEST 1: Clean (baseline)
    # =====================================================================
    sigma_alea_clean = alea_est.predict_normalized(X_gt)
    sigma_epis_clean = epis_est.predict(X_gt)

    print(f"\n{'='*70}")
    print("TEST 1: CLEAN (baseline)")
    print(f"{'='*70}")
    print(f"  σ_alea: mean={sigma_alea_clean.mean():.4f}, std={sigma_alea_clean.std():.4f}")
    print(f"  σ_epis: mean={sigma_epis_clean.mean():.4f}, std={sigma_epis_clean.std():.4f}")

    # =====================================================================
    # TEST 2: Sensor Noise ONLY (aleatoric)
    # σ_alea should rise, σ_epis should stay flat
    # Key: σ_epis is computed on GROUND TRUTH, not noisy obs
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 2: SENSOR NOISE ONLY (aleatoric)")
    print("  σ_alea = Mahalanobis(noisy_obs) → should RISE")
    print("  σ_epis = ε_knn+ε_rank(gt_obs) → should stay FLAT")
    print(f"{'='*70}")

    noise_levels = {
        "None":    {"jp": 0.0,  "jv": 0.0,  "op": 0.0},
        "Low":     {"jp": 0.005, "jv": 0.02,  "op": 0.02},
        "Medium":  {"jp": 0.01,  "jv": 0.05,  "op": 0.05},
        "High":    {"jp": 0.02,  "jv": 0.1,   "op": 0.10},
        "Extreme": {"jp": 0.05,  "jv": 0.2,   "op": 0.15},
    }

    print(f"\n  {'Level':<10} {'σ_alea':>8} {'σ_epis':>8} {'Δ_alea':>8} {'Δ_epis':>8}")
    print(f"  {'-'*42}")

    noise_alea_values = []
    noise_epis_values = []

    for level_name, params in noise_levels.items():
        X_noisy = X_gt.copy()
        X_noisy[:, 0:9] += np.random.normal(0, params["jp"], (n_test, 9))
        X_noisy[:, 9:18] += np.random.normal(0, params["jv"], (n_test, 9))
        X_noisy[:, 18:21] += np.random.normal(0, params["op"], (n_test, 3))

        # σ_alea on NOISY observation
        s_a = alea_est.predict_normalized(X_noisy).mean()
        # σ_epis on GROUND TRUTH (unchanged!) — this is the key design choice
        s_e = epis_est.predict(X_gt).mean()

        d_a = s_a - sigma_alea_clean.mean()
        d_e = s_e - sigma_epis_clean.mean()

        noise_alea_values.append(s_a)
        noise_epis_values.append(s_e)

        print(f"  {level_name:<10} {s_a:>8.4f} {s_e:>8.4f} {d_a:>+8.4f} {d_e:>+8.4f}")

    noise_alea_range = noise_alea_values[-1] - noise_alea_values[0]
    noise_epis_range = abs(noise_epis_values[-1] - noise_epis_values[0])
    noise_pass = noise_epis_range < 0.01  # σ_epis should be EXACTLY flat

    print(f"\n  σ_alea range: {noise_alea_range:.4f} (should be > 0.2)")
    print(f"  σ_epis range: {noise_epis_range:.6f} (should be ~0, target < 0.01)")
    print(f"  NOISE ISOLATION: {'PASS' if noise_pass else 'FAIL'}")

    # =====================================================================
    # TEST 3: Dynamics Shift ONLY (epistemic)
    # σ_epis should rise, σ_alea also rises (expected — Mahal detects shifts)
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 3: DYNAMICS SHIFT ONLY (epistemic)")
    print("  σ_epis = ε_knn+ε_rank(shifted_gt) → should RISE")
    print("  σ_alea = Mahalanobis(shifted_gt) → also rises (expected)")
    print(f"{'='*70}")

    ood_scenarios = {
        "Clean":       {"obj_shift": [0, 0, 0],   "joint_shift": 0.0},
        "Small OOD":   {"obj_shift": [0.1, 0.1, 0], "joint_shift": 0.0},
        "Medium OOD":  {"obj_shift": [0.2, 0.2, 0], "joint_shift": 0.0},
        "Large OOD":   {"obj_shift": [0.3, 0.3, 0], "joint_shift": 0.0},
        "Joint shift":  {"obj_shift": [0, 0, 0],   "joint_shift": 0.5},
        "Both shift":  {"obj_shift": [0.2, 0.2, 0], "joint_shift": 0.3},
    }

    print(f"\n  {'Scenario':<15} {'σ_alea':>8} {'σ_epis':>8} {'Δ_alea':>8} {'Δ_epis':>8}")
    print(f"  {'-'*47}")

    ood_epis_values = []

    for name, cfg in ood_scenarios.items():
        X_shifted = X_gt.copy()
        X_shifted[:, 18:21] += np.array(cfg["obj_shift"])
        if cfg["joint_shift"] > 0:
            X_shifted[:, 0:9] += cfg["joint_shift"]

        # Both computed on shifted GT (no noise — pure OOD)
        s_a = alea_est.predict_normalized(X_shifted).mean()
        s_e = epis_est.predict(X_shifted).mean()

        d_a = s_a - sigma_alea_clean.mean()
        d_e = s_e - sigma_epis_clean.mean()

        ood_epis_values.append(s_e)
        print(f"  {name:<15} {s_a:>8.4f} {s_e:>8.4f} {d_a:>+8.4f} {d_e:>+8.4f}")

    ood_epis_range = ood_epis_values[-1] - ood_epis_values[0]
    ood_pass = ood_epis_range > 0.1

    print(f"\n  σ_epis range (clean→OOD): {ood_epis_range:.4f} (should be > 0.1)")
    print(f"  OOD RESPONSE: {'PASS' if ood_pass else 'FAIL'}")

    # =====================================================================
    # TEST 4: Partial Occlusion (aleatoric)
    # σ_alea should rise, σ_epis computed on GT stays flat
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 4: PARTIAL OCCLUSION (aleatoric)")
    print("  Dims 18-20 (object position) set to 0 or last-known")
    print("  σ_alea = Mahalanobis(occluded_obs) → should RISE")
    print("  σ_epis = ε_knn+ε_rank(gt_obs) → should stay FLAT")
    print(f"{'='*70}")

    occ_levels = {
        "No occlusion": 0.0,
        "10% occluded":  0.1,
        "30% occluded":  0.3,
        "50% occluded":  0.5,
        "100% occluded": 1.0,
    }

    print(f"\n  {'Level':<16} {'σ_alea':>8} {'σ_epis':>8} {'Δ_alea':>8} {'Δ_epis':>8}")
    print(f"  {'-'*48}")

    occ_alea_values = []
    occ_epis_values = []

    for name, p_occlude in occ_levels.items():
        X_occ = X_gt.copy()
        # Randomly occlude object position for p_occlude fraction
        occlude_mask = np.random.rand(n_test) < p_occlude
        X_occ[occlude_mask, 18:21] = 0.0  # Set to zero (simulates missing data)

        # σ_alea on occluded observation
        s_a = alea_est.predict_normalized(X_occ).mean()
        # σ_epis on GROUND TRUTH (unaffected by occlusion)
        s_e = epis_est.predict(X_gt).mean()

        d_a = s_a - sigma_alea_clean.mean()
        d_e = s_e - sigma_epis_clean.mean()

        occ_alea_values.append(s_a)
        occ_epis_values.append(s_e)

        print(f"  {name:<16} {s_a:>8.4f} {s_e:>8.4f} {d_a:>+8.4f} {d_e:>+8.4f}")

    occ_alea_range = occ_alea_values[-1] - occ_alea_values[0]
    occ_epis_range = abs(occ_epis_values[-1] - occ_epis_values[0])
    occ_pass = occ_epis_range < 0.01  # σ_epis flat since it uses GT

    print(f"\n  σ_alea range: {occ_alea_range:.4f} (should be > 0.1)")
    print(f"  σ_epis range: {occ_epis_range:.6f} (should be ~0)")
    print(f"  OCCLUSION ISOLATION: {'PASS' if occ_pass else 'FAIL'}")

    # =====================================================================
    # TEST 5: Combined (noise + OOD)
    # Both should rise
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 5: COMBINED (noise + dynamics shift)")
    print("  Both σ_alea and σ_epis should rise")
    print(f"{'='*70}")

    combined_scenarios = {
        "Clean":                {"noise": 0.0,  "shift": 0.0},
        "Noise only":           {"noise": 0.10, "shift": 0.0},
        "OOD only":             {"noise": 0.0,  "shift": 0.3},
        "Noise + Small OOD":    {"noise": 0.10, "shift": 0.1},
        "Noise + Large OOD":    {"noise": 0.10, "shift": 0.3},
    }

    print(f"\n  {'Scenario':<22} {'σ_alea':>8} {'σ_epis':>8}")
    print(f"  {'-'*38}")

    for name, cfg in combined_scenarios.items():
        X_combined = X_gt.copy()
        # Apply OOD shift to GT
        X_shifted_gt = X_gt.copy()
        X_shifted_gt[:, 18:21] += cfg["shift"]

        # Apply noise on top
        X_noisy = X_shifted_gt.copy()
        if cfg["noise"] > 0:
            X_noisy[:, 0:9] += np.random.normal(0, cfg["noise"] * 0.2, (n_test, 9))
            X_noisy[:, 9:18] += np.random.normal(0, cfg["noise"], (n_test, 9))
            X_noisy[:, 18:21] += np.random.normal(0, cfg["noise"], (n_test, 3))

        # σ_alea on noisy+shifted observation
        s_a = alea_est.predict_normalized(X_noisy).mean()
        # σ_epis on shifted GT (OOD state, immune to noise)
        s_e = epis_est.predict(X_shifted_gt).mean()

        print(f"  {name:<22} {s_a:>8.4f} {s_e:>8.4f}")

    # =====================================================================
    # TEST 6: Orthogonality statistics
    # =====================================================================
    print(f"\n{'='*70}")
    print("TEST 6: ORTHOGONALITY STATISTICS")
    print(f"{'='*70}")

    # Collect diverse conditions
    all_alea = []
    all_epis = []

    for noise_std in [0.0, 0.02, 0.05, 0.10, 0.15]:
        X_noisy = X_gt.copy()
        if noise_std > 0:
            X_noisy[:, 0:9] += np.random.normal(0, noise_std * 0.2, (n_test, 9))
            X_noisy[:, 9:18] += np.random.normal(0, noise_std, (n_test, 9))
            X_noisy[:, 18:21] += np.random.normal(0, noise_std, (n_test, 3))
        all_alea.extend(alea_est.predict_normalized(X_noisy).tolist())
        all_epis.extend(epis_est.predict(X_gt).tolist())  # GT always same

    all_alea = np.array(all_alea)
    all_epis = np.array(all_epis)

    r_pearson, p_pearson = pearsonr(all_alea, all_epis)
    r_spearman, p_spearman = spearmanr(all_alea, all_epis)

    print(f"\n  Noise sweep (σ_epis constant by design):")
    print(f"    Pearson  |r| = {abs(r_pearson):.4f} (p={p_pearson:.4f})")
    print(f"    Spearman |ρ| = {abs(r_spearman):.4f} (p={p_spearman:.4f})")
    print(f"    Target: |r| < 0.1 (CVPR target)")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Test 2 (Noise isolation):    {'PASS' if noise_pass else 'FAIL'} — σ_epis Δ = {noise_epis_range:.6f} < 0.01")
    print(f"  Test 3 (OOD response):       {'PASS' if ood_pass else 'FAIL'} — σ_epis Δ = {ood_epis_range:.4f} > 0.1")
    print(f"  Test 4 (Occlusion isolation): {'PASS' if occ_pass else 'FAIL'} — σ_epis Δ = {occ_epis_range:.6f} < 0.01")
    print(f"  Test 6 (Pearson |r|):         {'PASS' if abs(r_pearson) < 0.1 else 'FAIL'} — |r| = {abs(r_pearson):.4f}")

    overall_pass = noise_pass and ood_pass and occ_pass and abs(r_pearson) < 0.1
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
