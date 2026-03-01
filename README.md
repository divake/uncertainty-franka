# Decomposed Uncertainty-Aware Control for Robust Robot Manipulation

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.3.2-green)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

**IROS 2026 Paper:** Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
**CVPR Extension:** arXiv:2511.12389

## Overview

We decompose observation uncertainty into **aleatoric** (sensor noise) and **epistemic** (out-of-distribution) components using CVPR-consistent methodology, then apply **targeted interventions**:

- **High aleatoric** (noisy/degraded observation): Multi-sample averaging to filter noise
- **High epistemic** (unfamiliar state): Conservative action scaling to reduce risk
- **Both high**: Combined filtering + conservative scaling
- **Both low**: Normal action execution

All signals are **post-hoc** (zero training required) and computed from calibration data statistics only.

Tested on Isaac Lab's Franka tasks: **Lift Cube** and **Reach**.

## Uncertainty Decomposition (v3.0 — CVPR-consistent)

| Signal | Formula | Input | Detects |
|---|---|---|---|
| **σ_alea** | Mahalanobis distance (CVPR Sec 3.2) | Noisy observation | Sensor noise, occlusion |
| **σ_epis** | 0.5·ε_knn + 0.5·ε_rank (CVPR Sec 3.1) | Ground truth state | OOD / dynamics shift |

- **ε_knn**: k-NN distance in standardized state space (analogous to CVPR ε_supp)
- **ε_rank**: Spectral entropy of local covariance (SAME formula as CVPR)
- **MSV**: Multi-sample variance — used as a denoising MECHANISM (not a signal)

**Key design choice:** σ_epis is computed on **ground truth** observations, making it perfectly immune to sensor noise (Δ=0.000000 under noise). This achieves pipeline-level orthogonality |r| = 0.054 (CVPR target: < 0.1).

## Key Results

### Table 1: All Methods (HIGH noise, 100 episodes, Lift)

| Method | Success Rate | Avg Reward |
|---|---|---|
| Vanilla | 58.0% | 45.85 |
| Deep Ensemble (B3) | 45.0% | 25.62 |
| MC Dropout (B4) | 53.0% | 29.67 |
| Multi-Sample Only | 96.1% | 119.16 |
| Total Uncertainty | 91.4% | 101.20 |
| **Decomposed (Ours)** | **96.1%** | **118.40** |

Decomposed matches Multi-Sample under noise and beats Total Uncertainty by +4.7%. Total-U over-intervenes (100% intervention rate) because it cannot distinguish noise from OOD.

### Table 2: OOD Perturbation (HIGH noise + OOD, 100 episodes, Lift)

| Scenario | Vanilla | Multi-S | DeepEns | MCDrop | Total-U | **Decomposed** | D-TU |
|---|---|---|---|---|---|---|---|
| Mass 2x | 52.0% | 98.0% | 40.0% | 51.0% | 91.0% | **94.0%** | **+3.0%** |
| Mass 5x | 35.0% | 74.8% | 39.0% | 31.0% | 70.0% | **82.0%** | **+12.0%** |
| Mass 10x | 1.0% | 14.0% | 0.0% | 3.0% | 12.0% | **14.0%** | **+2.0%** |
| Friction 0.5x | 51.0% | 93.8% | 58.0% | 43.0% | 92.0% | **93.0%** | **+1.0%** |
| Friction 0.2x | 48.0% | 89.0% | 35.0% | 39.6% | 89.0% | **91.5%** | **+2.5%** |
| Gravity 1.5x | 55.0% | 96.1% | 64.1% | 51.6% | 89.8% | **94.5%** | **+4.7%** |

**Decomposed beats Total Uncertainty in ALL 6 OOD scenarios** (+1.0% to +12.0%).

### Validation (Offline — All PASS)

| Test | Metric | Value | Status |
|---|---|---|---|
| Noise isolation | σ_epis Δ under noise | 0.000000 | PASS |
| OOD response | σ_epis Δ under OOD | +0.20 | PASS |
| Occlusion isolation | σ_epis Δ under occlusion | 0.000000 | PASS |
| Orthogonality | Pearson \|r\| noise sweep | 0.054 | PASS (< 0.1) |

## Quick Start

```bash
conda activate env_py311

# Collect calibration data
python collect_calibration_data.py --headless --num_envs 32

# Main evaluation (noise sweep + all methods)
python evaluate_decomposed.py --headless --num_envs 32 --noise_level high

# OOD evaluation (dynamics shift)
python evaluate_ood.py --headless --num_envs 32 --noise_level high

# Conformal prediction
python evaluate_conformal.py --headless --num_envs 32 --noise_level high

# Offline validation (no simulator needed)
python validate_decomposition.py
```

## Parameters

```
tau_a=0.3, tau_e=0.7, beta=0.3, N=5
k_knn=20, k_rank=50
w_knn=0.50, w_rank=0.50
reg_lambda=1e-4
zero_var_dims=[24,25,26,27]
```

## Project Structure

```
uncertainty_franka/
├── evaluate_decomposed.py       # Main evaluation: calibration + orthogonality + all methods
├── evaluate_ood.py              # OOD perturbation evaluation (mass/friction/gravity)
├── evaluate_conformal.py        # Conformal prediction coverage evaluation
├── validate_decomposition.py    # Offline controlled validation (no simulator needed)
├── collect_calibration_data.py  # Collect X_cal from clean environment
├── plot_figures.py              # Generate all paper figures and LaTeX tables
├── run_parallel_eval.sh         # Parallel evaluation on multiple GPUs via tmux
├── cvpr_iros_plan.md            # CVPR→IROS extension plan
├── uncertainty/                 # Core uncertainty module
│   ├── aleatoric.py             # σ_alea: Mahalanobis (CVPR Sec 3.2) + MSV denoising
│   ├── epistemic_cvpr.py        # σ_epis: ε_knn + ε_rank (CVPR Sec 3.1) [NEW in v3.0]
│   ├── intervention.py          # DecomposedPolicy, TotalUncertaintyPolicy, baselines
│   ├── task_config.py           # Task-specific configs (Lift 36D, Reach 32D)
│   ├── perturbations.py         # Observation + Environment perturbations
│   ├── conformal.py             # Conformal prediction calibration + ACI
│   ├── orthogonality.py         # OrthogonalityAnalyzer (Pearson, Spearman, HSIC, CKA)
│   └── epistemic.py             # LEGACY v2.x spectral/repulsive estimators (reference only)
├── figures/                     # Generated paper figures (PDF + PNG)
├── progress/                    # Versioned development journal (v1.0 — v3.0)
└── README.md
```

## Requirements

- Isaac Lab v2.3.2 / Isaac Sim 5.1.0
- Python 3.11 (conda env: `env_py311`)
- RSL-RL, PyTorch

## License

MIT License
