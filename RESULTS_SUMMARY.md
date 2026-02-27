# Results Summary: Uncertainty-Aware Robot Manipulation

**Project:** IROS 2026 - Uncertainty Decomposition for Robot Manipulation
**Date:** 2026-02-27
**GitHub:** https://github.com/divake/uncertainty-franka

---

## 1. Main Result: Multi-Sample Observation Averaging

**Key Insight:** Averaging multiple noisy sensor readings reduces aleatoric uncertainty by sqrt(N), dramatically improving manipulation success without retraining.

### Performance Comparison (5 Samples)

| Noise Level | Object Noise | Baseline | Multi-Sample | Improvement |
|-------------|--------------|----------|--------------|-------------|
| Medium      | 5 cm         | 92.5%    | **100.0%**   | **+7.5%**   |
| High        | 10 cm        | 63.7%    | **92.6%**    | **+28.9%**  |
| Extreme     | 15 cm        | 33.3%    | **87.5%**    | **+54.2%**  |

### Reward Improvement

| Noise Level | Baseline Reward | Multi-Sample Reward | Improvement |
|-------------|-----------------|---------------------|-------------|
| Medium      | 116.70          | **150.58**          | +29.0%      |
| High        | 42.43           | **118.36**          | +179.0%     |
| Extreme     | 11.63           | **97.84**           | +741.4%     |

### Why This Works
- **Noise reduction:** Averaging N independent noisy observations reduces variance by factor of N
- **Ground truth recovery:** By averaging, we approximate the true (unobservable) state
- **No retraining required:** Works with any pretrained policy
- **No latency:** All samples taken at same timestep (simulates multiple sensors)

---

## 2. Baseline Ablation Study

**Task:** Franka Panda lifts a cube from table to target position

### Impact of Observation Noise on Task Success

| Noise Level | Object Noise (cm) | Success Rate | Avg Reward | Avg Max Height (m) |
|-------------|-------------------|--------------|------------|-------------------|
| none        | 0                 | **100.0%**   | 158.80     | 0.358             |
| low         | 2                 | **100.0%**   | 150.64     | 0.377             |
| medium      | 5                 | **92.5%**    | 116.70     | 0.381             |
| high        | 10                | **63.7%**    | 42.43      | 0.265             |
| extreme     | 15                | **33.3%**    | 11.63      | 0.166             |

### Key Observations
1. Policy is robust to small noise (≤2cm object position uncertainty)
2. Performance degrades gradually between 5-10cm noise
3. Catastrophic failure at 15cm noise (67% success rate drop)
4. Reward correlates strongly with success (93% reward reduction at extreme noise)

---

## 3. Methods Comparison

### 3.1 Multi-Sample Averaging (Main Contribution)
- Average N noisy observations from ground truth
- Simulates multiple sensors or repeated measurements
- **Result:** **+54.2% success rate at extreme noise**

### 3.2 Deep Ensemble (Epistemic Uncertainty)
- Create ensemble by perturbing pretrained weights
- Measure disagreement between members
- **Result:** Did not improve performance (perturbing trained weights breaks learned behavior)

### 3.3 EMA Filtering (Temporal)
- EMA filter on raw observations over time
- α = 2/(N+1) where N = filter window
- **Result:** Modest success improvement (+8%), better reward (+33-63%)
- **Limitation:** Adds temporal latency

---

## 4. Experimental Setup

### Environment
- **Simulator:** Isaac Lab v2.3.2 + Isaac Sim 5.1.0
- **Robot:** Franka Panda (7 DoF arm + 2 finger gripper)
- **Task:** Lift cube from table (0.055m) to target (~0.4m)
- **Policy:** Pretrained RSL-RL PPO (MLP: 256-128-64)
- **Episodes:** 80 per experiment

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

## 5. Conclusions

1. **Multi-sample averaging dramatically improves robustness to observation noise**
   - +54.2% success rate improvement at 15cm noise
   - Works without any policy retraining

2. **Observation uncertainty significantly impacts manipulation success**
   - 15cm position noise causes 67% success rate drop from baseline

3. **Epistemic uncertainty (ensemble) requires proper training**
   - Post-hoc weight perturbation is insufficient
   - Need to train ensemble from scratch with diverse initializations

4. **Multi-sensor fusion is highly effective for aleatoric uncertainty**
   - Simple averaging of multiple observations approximates ground truth
   - Practical for real robots with redundant sensors

---

## 6. Future Work

1. **Varying sample counts:** Ablation on N=2,3,5,10,20 samples
2. **Heterogeneous sensors:** Combine different sensor modalities
3. **Uncertainty-aware action selection:** Use uncertainty to modulate actions
4. **Real robot validation:** Deploy on physical Franka Panda

---

## Repository Structure

```
uncertainty_franka/
├── evaluate_multi_sample.py       # Multi-sample averaging (MAIN)
├── evaluate_noisy_v2.py           # Baseline noisy evaluation
├── evaluate_observation_filtering.py  # EMA filtering comparison
├── evaluate_uncertainty_aware.py  # Ensemble evaluation
├── run_full_ablation.py           # Run all noise levels
├── uncertainty/
│   ├── __init__.py
│   └── ensemble_policy.py         # Ensemble implementation
├── results/                       # JSON results
├── RESULTS_SUMMARY.md            # This file
└── README.md
```

## Citation

```bibtex
@inproceedings{uncertainty_manipulation_2026,
  title={Multi-Sample Observation Averaging for Robust Robot Manipulation},
  author={...},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```
