# Results Summary: Uncertainty-Aware Robot Manipulation

**Project:** IROS 2026 - Uncertainty Decomposition for Robot Manipulation
**Date:** 2026-02-27
**GitHub:** https://github.com/divake/uncertainty-franka

---

## 1. Baseline Ablation Study

**Task:** Franka Panda lifts a cube from table to target position

### Impact of Observation Noise on Task Success

| Noise Level | Object Noise (cm) | Success Rate | Avg Reward | Avg Max Height (m) |
|-------------|-------------------|--------------|------------|-------------------|
| none        | 0                 | **100.0%**   | 158.80     | 0.358             |
| low         | 2                 | **100.0%**   | 150.64     | 0.377             |
| medium      | 5                 | **87.5%**    | 104.59     | 0.348             |
| high        | 10                | **83.3%**    | 57.41      | 0.326             |
| extreme     | 15                | **24.0%**    | 10.02      | 0.123             |

### Key Observations
1. Policy is robust to small noise (≤2cm object position uncertainty)
2. Performance degrades gradually between 5-10cm noise
3. Catastrophic failure at 15cm noise (76% success rate drop)
4. Reward correlates strongly with success (94% reward reduction at extreme noise)

---

## 2. Observation Filtering Results

**Approach:** Exponential Moving Average (EMA) filter on observations to reduce sensor noise

### High Noise (10cm object position noise)

| Method | Success Rate | Avg Reward | Improvement |
|--------|--------------|------------|-------------|
| Baseline (raw obs) | 48.0% | 37.55 | - |
| EMA Filtered | **56.0%** | **50.12** | +8% success, +33% reward |

### Extreme Noise (15cm object position noise)

| Method | Success Rate | Avg Reward | Improvement |
|--------|--------------|------------|-------------|
| Baseline (raw obs) | 17.1% | 7.01 | - |
| EMA Filtered | **18.1%** | **11.47** | +1% success, +63% reward |

### Key Findings
1. **Filtering significantly improves reward** even when success rate gains are modest
2. Reward improvement indicates better partial task completion
3. The EMA filter reduces aleatoric (sensor) noise effectively
4. Trade-off: filtering adds latency, which can hurt fast dynamic tasks

---

## 3. Experimental Setup

### Environment
- **Simulator:** Isaac Lab v2.3.2 + Isaac Sim 5.1.0
- **Robot:** Franka Panda (7 DoF arm + 2 finger gripper)
- **Task:** Lift cube from table (0.055m) to target (~0.4m)
- **Policy:** Pretrained RSL-RL PPO (MLP: 256-128-64)

### Observation Space (36 dimensions)
- Joint positions: 9 dims
- Joint velocities: 9 dims
- Object position: 3 dims (noisy)
- Target pose: 7 dims
- Previous actions: 8 dims

### Noise Model
- **Joint position noise:** Gaussian, 0.005-0.05 rad
- **Joint velocity noise:** Gaussian, 0.02-0.2 rad/s
- **Object position noise:** Gaussian, 0.02-0.15 m (most impactful)

---

## 4. Methods Tested

### 4.1 Deep Ensemble (Epistemic Uncertainty)
- Create ensemble by perturbing pretrained weights
- Measure disagreement between members
- **Result:** Did not improve performance (perturbing trained weights breaks learned behavior)

### 4.2 Observation Filtering (Aleatoric Uncertainty)
- EMA filter on raw observations
- α = 2/(N+1) where N = filter window
- **Result:** Significant improvement in reward (+33-63%)

---

## 5. Conclusions

1. **Observation uncertainty significantly impacts manipulation success**
   - 15cm position noise causes 76% success rate drop

2. **Simple filtering can reduce aleatoric uncertainty effects**
   - EMA filtering improves reward by 33-63% depending on noise level

3. **Epistemic uncertainty (ensemble) requires proper training**
   - Post-hoc weight perturbation is insufficient

4. **Reward is a better metric than success for high-noise scenarios**
   - Captures partial progress even when task fails

---

## 6. Future Work

1. **Train ensemble from scratch** with diverse initializations
2. **Active perception** - move robot to better observation positions
3. **Uncertainty-conditioned policy** - train policy aware of observation confidence
4. **Multi-modal sensing** - combine vision + proprioception for robustness

---

## Repository Structure

```
uncertainty_franka/
├── evaluate_noisy_v2.py           # Baseline noisy evaluation
├── evaluate_observation_filtering.py  # EMA filtering comparison
├── evaluate_uncertainty_aware.py  # Ensemble evaluation
├── run_full_ablation.py           # Run all noise levels
├── uncertainty/
│   ├── __init__.py
│   └── ensemble_policy.py         # Ensemble implementation
├── configs/
│   └── noisy_env_cfg.py
├── results/                       # JSON results
├── BASELINE_RESULTS.md
└── README.md
```

## Citation

```bibtex
@inproceedings{uncertainty_manipulation_2026,
  title={Uncertainty Decomposition for Robot Manipulation},
  author={...},
  booktitle={IROS},
  year={2026}
}
```
