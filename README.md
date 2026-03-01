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

### Table 1: Noise Robustness (HIGH noise, 100 episodes)

| Method | Success Rate | Avg Reward |
|---|---|---|
| Vanilla | 58.0% | 45.85 |
| Multi-Sample Only | 96.1% | 119.16 |
| **Decomposed (Ours)** | **96.1%** | **120.68** |

### Table 2: Noise Level Ablation

| Noise Level | Vanilla | Multi-Sample | **Decomposed** |
|---|---|---|---|
| None (0 cm) | 100.0% | 100.0% | **100.0%** |
| Low (2 cm) | 100.0% | 100.0% | **100.0%** |
| Medium (5 cm) | 95.3% | 100.0% | **100.0%** |
| High (10 cm) | 58.0% | 96.1% | **96.1%** |
| Extreme (15 cm) | 29.0% | 82.0% | **85.2%** |

### Table 3: OOD Perturbation — Lift (HIGH noise + OOD)

| Scenario | Vanilla | Multi-Sample | Total Uncert. | **Decomposed** | D-TU |
|---|---|---|---|---|---|
| Mass 2x | 52.0% | 98.0% | 95.0% | **97.0%** | **+2.0%** |
| Mass 5x | 41.2% | 78.2% | 78.0% | 77.7% | -0.3% |
| Friction 0.5x | 42.0% | 95.3% | 86.0% | **96.0%** | **+10.0%** |
| Gravity 1.5x | 67.0% | 95.3% | 92.0% | **96.1%** | **+4.1%** |

### Table 4: Cross-Task — Reach (HIGH noise + OOD)

| Scenario | Vanilla | Multi-Sample | Total Uncert. | **Decomposed** | D-TU |
|---|---|---|---|---|---|
| Gravity 1.5x | 92.2% | 86.7% | 57.8% | **87.5%** | **+29.7%** |
| Gravity 2.0x | 50.0% | 44.5% | 30.5% | **44.5%** | **+14.1%** |
| Damping 3x | 96.9% | 96.1% | 92.2% | 93.8% | +1.6% |
| Damping 5x | 96.9% | 96.1% | 93.0% | **98.4%** | **+5.5%** |

## Quick Start

```bash
conda activate env_py311

# Lift Cube (default task)
python collect_calibration_data.py --headless --num_envs 32
python evaluate_decomposed.py --headless --num_envs 32 --noise_level high
python evaluate_ood.py --headless --num_envs 32 --noise_level high

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
├── collect_calibration_data.py  # Collect X_cal from clean environment
├── MASTER_PLAN.md               # Full paper plan with experiments and baselines
├── uncertainty/                 # Core uncertainty module
│   ├── aleatoric.py             # MSV + Mahalanobis estimators
│   ├── epistemic.py             # Spectral/Repulsive estimators (not used — 36-dim too low)
│   ├── intervention.py          # InterventionController + DecomposedPolicy + TotalUncertaintyPolicy
│   ├── task_config.py           # Task-specific configs (Lift 36D, Reach 32D)
│   ├── orthogonality.py         # OrthogonalityAnalyzer (Pearson, Spearman, HSIC, CKA)
│   ├── perturbations.py         # Observation + Environment perturbations
│   └── conformal.py             # Conformal prediction (planned)
├── progress/                    # Versioned experiment journal
│   ├── v1.0 — v1.2             # Baseline → EMA → Multi-sample
│   └── v2.0 — v2.6             # Calibration → Estimators → Decomposed → Sweep → OOD → Multi-task
└── README.md
```

## Requirements

- Isaac Lab v2.3.2 / Isaac Sim 5.1.0
- Python 3.11 (conda env: `env_py311`)
- RSL-RL, PyTorch

## License

MIT License
