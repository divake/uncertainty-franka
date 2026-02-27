# Uncertainty-Aware Robot Manipulation

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.3.2-green)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

**IROS 2026 Paper:** Uncertainty Decomposition for Robot Manipulation

## Overview

This project investigates how observation uncertainty affects robot manipulation success and develops uncertainty-aware control strategies. We use Isaac Lab's Franka Lift Cube task as a testbed to:

1. **Quantify** the impact of sensor noise on task success
2. **Estimate** observation uncertainty using ensemble methods
3. **Implement** active perception to reduce uncertainty before acting

## Quick Start

```bash
# Activate environment
conda activate env_py311

# Run single evaluation
cd /mnt/ssd1/divake/robo_uncertain/uncertainty_franka
python evaluate_noisy_v2.py --headless --num_envs 32 --num_episodes 100 --noise_level medium

# Run full ablation study
python run_full_ablation.py --num_envs 32 --num_episodes 100
```

## Baseline Results

| Noise Level | Object Noise | Success Rate | Avg Reward | Avg Max Height |
|-------------|--------------|--------------|------------|----------------|
| none        | 0 cm         | **100.0%**   | 158.80     | 0.358 m        |
| low         | 2 cm         | **100.0%**   | 150.64     | 0.377 m        |
| medium      | 5 cm         | **87.5%**    | 104.59     | 0.348 m        |
| high        | 10 cm        | **83.3%**    | 57.41      | 0.326 m        |
| extreme     | 15 cm        | **24.0%**    | 10.02      | 0.123 m        |

See [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for detailed analysis.

## Key Findings

1. **Robustness Threshold**: Policy maintains 100% success with up to 2cm noise
2. **Graceful Degradation**: Performance drops to ~85% at 5-10cm noise
3. **Catastrophic Failure**: 15cm noise reduces success to only 24%
4. **Reward Correlation**: 94% reward reduction at extreme noise

## Noise Configuration

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `joint_pos_std` | Proprioceptive noise (rad) | Minor |
| `joint_vel_std` | Velocity noise (rad/s) | Minor |
| `object_pos_std` | Object position noise (m) | **Critical** |

## Project Structure

```
uncertainty_franka/
├── evaluate_noisy_v2.py      # Main evaluation script (hydra-based)
├── evaluate_noisy.py         # Alternative evaluation script
├── run_full_ablation.py      # Run all noise levels
├── configs/
│   ├── __init__.py
│   └── noisy_env_cfg.py      # Environment configurations
├── results/                   # Evaluation results (JSON)
├── BASELINE_RESULTS.md       # Detailed results analysis
└── README.md
```

## Requirements

- Isaac Lab v2.3.2
- Isaac Sim 5.1.0
- Python 3.11 (conda env: env_py311)
- RSL-RL
- PyTorch

## Citation

```bibtex
@inproceedings{uncertainty_manipulation_2026,
  title={Uncertainty Decomposition for Robot Manipulation},
  author={...},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

## License

MIT License
