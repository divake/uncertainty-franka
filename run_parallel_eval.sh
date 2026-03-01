#!/bin/bash
# =============================================================================
# Parallel Evaluation Runner for IROS 2026
# Runs evaluations on BOTH GPUs simultaneously with increased num_envs
# Uses tmux for persistence (survives SSH disconnects)
#
# Usage:
#   ./run_parallel_eval.sh              # Run all evaluations
#   ./run_parallel_eval.sh noise_only   # Run noise sweep only
#   ./run_parallel_eval.sh ood_only     # Run OOD sweep only
#   tmux attach -t iros_eval            # Attach to see progress
# =============================================================================

set -e

# Configuration
NUM_ENVS=256          # 8x current (32). Safe for 47GB VRAM with Franka Lift.
NUM_EPISODES=100      # Per evaluation
CAL_DIR="calibration_data/Isaac-Lift-Cube-Franka-v0_20260228_181927"
CONDA_SETUP="source /home/divake/miniconda3/etc/profile.d/conda.sh && conda activate env_py311"
PROJECT_DIR="/mnt/ssd1/divake/robo_uncertain/uncertainty_franka"

# CPU thread limiting (24 cores / 2 GPUs = 12 per process)
export PXR_WORK_THREAD_LIMIT=12
export OPENBLAS_NUM_THREADS=12

MODE="${1:-all}"

echo "============================================="
echo "IROS 2026 Parallel Evaluation"
echo "============================================="
echo "  GPUs: 2x RTX 6000 Ada (47.4 GB each)"
echo "  num_envs: ${NUM_ENVS} (per GPU)"
echo "  num_episodes: ${NUM_EPISODES}"
echo "  Mode: ${MODE}"
echo "============================================="

# Kill existing session if any
tmux kill-session -t iros_eval 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s iros_eval -n main

# Function to create a tmux window and run a command
run_in_tmux() {
    local window_name="$1"
    local gpu_id="$2"
    local cmd="$3"

    tmux new-window -t iros_eval -n "${window_name}"
    tmux send-keys -t "iros_eval:${window_name}" \
        "cd ${PROJECT_DIR} && ${CONDA_SETUP} && PXR_WORK_THREAD_LIMIT=12 OPENBLAS_NUM_THREADS=12 ${cmd}" Enter
}

if [ "$MODE" = "all" ] || [ "$MODE" = "noise_only" ]; then
    echo ""
    echo ">>> Launching NOISE evaluations..."

    # GPU 0: Main decomposed eval (noise sweep, all methods)
    run_in_tmux "noise_eval" 0 \
        "python evaluate_decomposed.py \
            --task Isaac-Lift-Cube-Franka-v0 \
            --headless --device cuda:0 \
            --num_envs ${NUM_ENVS} \
            --noise_level high \
            --num_episodes ${NUM_EPISODES} \
            --cal_data_dir ${CAL_DIR} \
            --tau_a 0.3 --tau_e 0.7 --beta 0.3 \
        2>&1 | tee results/noise_eval_gpu0_\$(date +%Y%m%d_%H%M%S).log"

    echo "  [GPU 0] Noise evaluation launched"
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "ood_only" ]; then
    echo ""
    echo ">>> Launching OOD evaluations..."

    # GPU 1: OOD evaluation (dynamics shift, all methods)
    run_in_tmux "ood_eval" 1 \
        "python evaluate_ood.py \
            --task Isaac-Lift-Cube-Franka-v0 \
            --headless --device cuda:1 \
            --num_envs ${NUM_ENVS} \
            --noise_level high \
            --num_episodes ${NUM_EPISODES} \
            --cal_data_dir ${CAL_DIR} \
            --tau_a 0.3 --tau_e 0.7 --beta 0.3 \
        2>&1 | tee results/ood_eval_gpu1_\$(date +%Y%m%d_%H%M%S).log"

    echo "  [GPU 1] OOD evaluation launched"
fi

echo ""
echo "============================================="
echo "Evaluations launched in tmux session: iros_eval"
echo ""
echo "Commands:"
echo "  tmux attach -t iros_eval        # Attach to session"
echo "  tmux list-windows -t iros_eval  # List running evals"
echo "  Ctrl+B then N                   # Next window"
echo "  Ctrl+B then D                   # Detach (keeps running)"
echo "  nvidia-smi                      # Monitor GPU usage"
echo "============================================="
