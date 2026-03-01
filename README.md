# Decomposed Uncertainty-Aware Control for Robust Robot Manipulation

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.3.2-green)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

**IROS 2026 Paper:** Decomposed Uncertainty-Aware Control for Robust Robot Manipulation

## Overview

We decompose observation uncertainty into **aleatoric** (sensor noise) and **epistemic** (out-of-distribution) components and apply targeted interventions:
- **Aleatoric** (high sensor noise): Multi-sample averaging to filter noise
- **Epistemic** (OOD state): Conservative action scaling to reduce risk
- **Both**: Combined filtering + scaling

Tested on Isaac Lab's Franka tasks: **Lift Cube** and **Reach**.

## Key Results

### Table 1: All Methods Comparison (HIGH noise, 100 episodes, Lift)

| Method | Success Rate | Avg Reward |
|---|---|---|
| Vanilla | 58.0% | 45.85 |
| Deep Ensemble (B3) | 45.0% | 25.62 |
| MC Dropout (B4) | 53.0% | 29.67 |
| Multi-Sample Only | 96.1% | 119.16 |
| Total Uncertainty | 93.0% | 102.74 |
| **Decomposed (Ours)** | **93.0%** | **117.11** |

Deep Ensemble and MC Dropout perform **worse than vanilla** because action variance conflates noise with uncertainty, causing ~100% false conservative scaling.

### Table 2: Noise Level Ablation

| Noise Level | Vanilla | Multi-Sample | **Decomposed** |
|---|---|---|---|
| None (0 cm) | 100.0% | 100.0% | **100.0%** |
| Low (2 cm) | 100.0% | 100.0% | **100.0%** |
| Medium (5 cm) | 95.3% | 100.0% | **100.0%** |
| High (10 cm) | 58.0% | 96.1% | **96.1%** |
| Extreme (15 cm) | 29.0% | 82.0% | **85.2%** |

### Table 3: OOD Perturbation — Lift (HIGH noise + OOD)

| Scenario | Vanilla | DeepEns | MCDrop | Multi-S | Total-U | **Decomposed** | D-TU |
|---|---|---|---|---|---|---|---|
| Mass 2x | 52.0% | 40.0% | 51.0% | 98.0% | 87.0% | **95.3%** | **+8.3%** |
| Mass 5x | 35.0% | 19.0% | 32.0% | 82.2% | 62.0% | **76.0%** | **+14.0%** |
| Mass 10x | 2.0% | 2.0% | 2.0% | 18.4% | 14.0% | 15.0% | +1.0% |
| Friction 0.5x | 48.0% | 56.0% | 44.0% | 96.1% | 88.0% | **95.0%** | **+7.0%** |
| Friction 0.2x | 49.0% | 38.6% | 47.0% | 94.0% | 92.0% | 90.6% | -1.4% |
| Gravity 1.5x | 58.0% | 55.5% | 55.5% | 96.1% | 89.1% | **97.7%** | **+8.6%** |

### Table 4: Cross-Task — Reach (HIGH noise + OOD)

| Scenario | Vanilla | Multi-Sample | Total Uncert. | **Decomposed** | D-TU |
|---|---|---|---|---|---|
| Gravity 1.5x | 92.2% | 86.7% | 57.8% | **87.5%** | **+29.7%** |
| Gravity 2.0x | 50.0% | 44.5% | 30.5% | **44.5%** | **+14.1%** |
| Damping 3x | 96.9% | 96.1% | 92.2% | 93.8% | +1.6% |
| Damping 5x | 96.9% | 96.1% | 93.0% | **98.4%** | **+5.5%** |

### Table 5: Conformal Prediction Coverage (HIGH noise, 200 test episodes, Lift)

| Method | Success Rate |
|---|---|
| No CP (Vanilla) | 66.5% |
| Total Uncertainty CP (90%) | 79.0% |
| **Decomposed CP (90%)** | **80.0%** |
| **Decomposed (fixed thresholds)** | **95.1%** |

Conformal prediction provides principled threshold selection with statistical coverage guarantees. Decomposed CP achieves higher success with fewer abstentions than Total Uncertainty CP.

## Quick Start

```bash
conda activate env_py311

# Lift Cube (default task)
python collect_calibration_data.py --headless --num_envs 32
python evaluate_decomposed.py --headless --num_envs 32 --noise_level high
python evaluate_ood.py --headless --num_envs 32 --noise_level high
python evaluate_conformal.py --headless --num_envs 32 --noise_level high

# Reach (multi-task)
python collect_calibration_data.py --headless --num_envs 32 --task Isaac-Reach-Franka-v0
python evaluate_decomposed.py --headless --num_envs 32 --noise_level high --task Isaac-Reach-Franka-v0
python evaluate_ood.py --headless --num_envs 32 --noise_level high --task Isaac-Reach-Franka-v0
```

## Uncertainty Decomposition

- **Aleatoric signal**: Multi-Sample Variance (MSV) — variance across N noisy readings of the same ground truth
- **Epistemic signal**: Mahalanobis distance from calibration distribution
- **Orthogonal by construction**: MSV=0 for OOD without noise; Mahalanobis unaffected by sample variance
- **Parameters**: `tau_a=0.3, tau_e=0.7, beta=0.3, N=5`

## Project Structure

```
uncertainty_franka/
├── evaluate_decomposed.py       # Main evaluation: calibration + orthogonality + all methods
├── evaluate_ood.py              # OOD perturbation evaluation (mass/friction/gravity/damping)
├── evaluate_conformal.py        # Conformal prediction coverage evaluation
├── collect_calibration_data.py  # Collect X_cal from clean environment
├── plot_figures.py              # Generate all paper figures and LaTeX tables
├── MASTER_PLAN.md               # Full paper plan with experiments and baselines
├── uncertainty/                 # Core uncertainty module
│   ├── aleatoric.py             # MSV + Mahalanobis estimators
│   ├── epistemic.py             # Spectral/Repulsive estimators (not used — 36-dim too low)
│   ├── intervention.py          # All policies: Decomposed, TotalUncertainty, DeepEnsemble, MCDropout
│   ├── task_config.py           # Task-specific configs (Lift 36D, Reach 32D)
│   ├── orthogonality.py         # OrthogonalityAnalyzer (Pearson, Spearman, HSIC, CKA)
│   ├── perturbations.py         # Observation + Environment perturbations
│   └── conformal.py             # Conformal prediction calibration + ACI
├── figures/                     # Generated paper figures (PDF + PNG)
├── progress/                    # Versioned experiment journal
│   ├── v1.0 — v1.2             # Baseline → EMA → Multi-sample
│   └── v2.0 — v2.9             # Calibration → Estimators → Decomposed → Sweep → OOD → Multi-task → Baselines → Conformal → Figures
└── README.md
```

## Requirements

- Isaac Lab v2.3.2 / Isaac Sim 5.1.0
- Python 3.11 (conda env: `env_py311`)
- RSL-RL, PyTorch

## License

MIT License
