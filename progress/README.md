# Progress Journal

**Paper:** Decomposed Uncertainty-Aware Control for Robust Robot Manipulation (IROS 2026)

This folder tracks every major milestone with full parameters, results, and paper-ready values.

## Version History

| Version | Title | Date | Key Result |
|---------|-------|------|------------|
| [v1.0](v1.0_baseline_noise_impact.md) | Baseline: Noise Impact | 2026-02-27 | High noise drops success 100% → 58% |
| [v1.1](v1.1_observation_filtering.md) | EMA Observation Filtering | 2026-02-27 | EMA barely helps, can hurt (-12.5%) |
| [v1.2](v1.2_multi_sample_averaging.md) | Multi-Sample Averaging | 2026-02-27 | 58% → 96.1% under high noise |
| [v2.0](v2.0_calibration_data_collection.md) | Calibration Data Collection | 2026-02-28 | 56K obs from 224 successful episodes |
| [v2.1](v2.1_uncertainty_estimator_exploration.md) | Estimator Exploration | 2026-02-28 | MSV + Mahalanobis = orthogonal by construction |
| [v2.2](v2.2_decomposed_evaluation.md) | Full Decomposed Evaluation | 2026-02-28 | 96.1% success, 120.68 reward (matches MS, higher reward) |
| [v2.3](v2.3_noise_level_sweep.md) | Noise Level Sweep (Table 2) | 2026-02-28 | Decomposed beats MS at extreme noise (+3.2%) |
| [v2.4](v2.4_ood_perturbation_testing.md) | OOD Perturbation Testing (Table 3) | 2026-02-28 | Decomposed beats MS at mass 5x (+9.0%) |
| [v2.5](v2.5_total_uncertainty_baseline.md) | Total Uncertainty Baseline (P2) | 2026-02-28 | Decomposed beats TU in 4/6 scenarios |
| [v2.6](v2.6_multi_task_reach.md) | Multi-Task: Reach (P3) | 2026-02-28 | Orthogonality holds, D beats TU in 4/4 Reach scenarios |
| [v2.7](v2.7_deep_ensemble_mc_dropout.md) | Deep Ensemble & MC Dropout (P4) | 2026-02-28 | DE 45%, MCD 53% — action variance fails under noise |

## Current Best Results (HIGH noise, 100 episodes, Lift)

| Method              | Success Rate | Avg Reward |
|---------------------|-------------|------------|
| Vanilla             | 58.0%       | 45.85      |
| Deep Ensemble (B3)  | 45.0%       | 25.62      |
| MC Dropout (B4)     | 53.0%       | 29.67      |
| Multi-Sample Only   | 96.1%       | 119.16     |
| Total Uncertainty   | 93.0%       | 102.74     |
| **Decomposed (Ours)** | **93.0%** | **117.11** |

## Current Best Parameters
```
tau_a = 0.3, tau_e = 0.7, beta = 0.3, N = 5
Aleatoric: Multi-Sample Variance (MSV)
Epistemic: Mahalanobis distance (reg_lambda=1e-4)
```
