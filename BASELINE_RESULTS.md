# Baseline Results: Impact of Observation Noise on Robot Manipulation

**Project:** Uncertainty Decomposition for Robot Manipulation
**Conference:** IROS 2026
**Date:** 2026-02-27
**Task:** Franka Lift Cube (Isaac Lab)

## Experimental Setup

- **Environment:** Isaac Lab v2.3.2 with Isaac Sim 5.1.0
- **Robot:** Franka Panda (7 DoF arm + parallel gripper)
- **Task:** Lift a cube from table to target height (~0.4m)
- **Policy:** Pretrained RSL-RL PPO (MLP: 256-128-64)
- **Observation Space:** 36 dimensions
  - Joint positions: 9 dims (7 arm + 2 gripper)
  - Joint velocities: 9 dims
  - Object position: 3 dims
  - Target pose: 7 dims (pos + quat)
  - Previous actions: 8 dims

## Noise Configuration

We add Gaussian noise to simulate sensor uncertainty:

| Noise Level | Joint Pos (rad) | Joint Vel (rad/s) | Object Pos (m) |
|-------------|-----------------|-------------------|----------------|
| none        | 0.0             | 0.0               | 0.0            |
| low         | 0.005 (~0.3°)   | 0.02              | 0.02 (2cm)     |
| medium      | 0.01 (~0.5°)    | 0.05              | 0.05 (5cm)     |
| high        | 0.02 (~1°)      | 0.1               | 0.10 (10cm)    |
| extreme     | 0.05 (~3°)      | 0.2               | 0.15 (15cm)    |

## Results

### Summary Table

| Noise Level | Object Noise | Success Rate | Avg Reward | Avg Max Height | Episodes |
|-------------|--------------|--------------|------------|----------------|----------|
| none        | 0 cm         | **100.0%**   | 158.80     | 0.358 m        | 24       |
| low         | 2 cm         | **100.0%**   | 150.64     | 0.377 m        | 24       |
| medium      | 5 cm         | **87.5%**    | 104.59     | 0.348 m        | 24       |
| high        | 10 cm        | **83.3%**    | 57.41      | 0.326 m        | 24       |
| extreme     | 15 cm        | **24.0%**    | 10.02      | 0.123 m        | 25       |

### Key Observations

1. **Robustness to Small Noise:** The pretrained policy maintains 100% success with up to 2cm object position noise.

2. **Graceful Degradation:** Performance drops gradually from 100% to ~85% as noise increases from 2cm to 10cm.

3. **Catastrophic Failure at Extreme Noise:** At 15cm noise, success rate plummets to 24%, with average max height (0.123m) barely above table level (0.055m).

4. **Reward Correlation:** Average reward strongly correlates with success rate:
   - No noise: 158.80
   - Extreme noise: 10.02 (94% reduction)

### Performance Degradation Curve

```
Success Rate vs Object Position Noise

100% |  *----*
     |        \
 80% |         *---*
     |
 60% |
     |
 40% |
     |
 20% |                   *
     |
  0% +----+----+----+----+----+
     0    2    5   10   15
         Object Noise (cm)
```

## Implications for Uncertainty-Aware Control

These results demonstrate that:

1. **Observation uncertainty significantly impacts manipulation success**
2. **There's a critical threshold (~5cm) where performance starts degrading**
3. **Standard policies lack uncertainty awareness** - they don't adapt behavior based on observation confidence

## Next Steps

1. **Implement Uncertainty Estimation:** Add MC Dropout or Deep Ensembles to quantify prediction uncertainty
2. **Active Perception:** When uncertainty is high, move robot to gather better observations before acting
3. **Uncertainty-Aware Policy:** Train policy that considers observation uncertainty in decision-making

## Reproduction

```bash
# Run single evaluation
conda activate env_py311
cd /mnt/ssd1/divake/robo_uncertain/uncertainty_franka
python evaluate_noisy_v2.py --headless --num_envs 32 --num_episodes 100 --noise_level medium

# Run full ablation
python run_full_ablation.py --num_envs 32 --num_episodes 100
```

## Hardware

- GPU: 2x NVIDIA RTX 6000 Ada Generation (48GB each)
- CPU: Intel Xeon w5-3423 (12 cores)
- RAM: 128GB
