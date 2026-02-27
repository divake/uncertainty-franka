#!/usr/bin/env python3
"""
Full ablation study script for Franka Lift Cube with noisy observations.

This script runs evaluations across all noise levels and generates
a comprehensive results table for the IROS 2026 paper.
"""

import argparse
import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

NOISE_LEVELS = ["none", "low", "medium", "high", "extreme"]

def run_evaluation(noise_level: str, num_envs: int, num_episodes: int, output_dir: str):
    """Run evaluation for a single noise level."""
    script_path = Path(__file__).parent / "evaluate_noisy_v2.py"

    cmd = [
        "conda", "run", "-n", "env_py311",
        "python", str(script_path),
        "--headless",
        "--num_envs", str(num_envs),
        "--num_episodes", str(num_episodes),
        "--noise_level", noise_level,
    ]

    print(f"\n{'='*60}")
    print(f"Running evaluation for noise level: {noise_level}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    # Parse results from output
    lines = result.stdout.split('\n')
    results = {}

    for line in lines:
        if "Success Rate:" in line:
            rate = line.split(":")[-1].strip().replace("%", "")
            results["success_rate"] = float(rate) / 100
        elif "Avg Reward:" in line:
            results["avg_reward"] = float(line.split(":")[-1].strip())
        elif "Avg Max Height:" in line:
            height_str = line.split(":")[-1].strip().replace(" m", "")
            results["avg_max_height"] = float(height_str)
        elif "Object Noise:" in line:
            noise_str = line.split(":")[-1].strip().replace(" cm", "")
            results["object_noise_cm"] = float(noise_str)
        elif "Episodes:" in line and "Success" not in line:
            results["num_episodes"] = int(line.split(":")[-1].strip())

    results["noise_level"] = noise_level

    return results


def main():
    parser = argparse.ArgumentParser(description="Run full ablation study")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of environments")
    parser.add_argument("--num_episodes", type=int, default=100, help="Episodes per noise level")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("FRANKA LIFT CUBE - FULL ABLATION STUDY")
    print("IROS 2026: Uncertainty Decomposition for Robot Manipulation")
    print("="*70)
    print(f"Environments: {args.num_envs}")
    print(f"Episodes per level: {args.num_episodes}")
    print("="*70 + "\n")

    # Output directory
    output_dir = Path(__file__).parent / "results" / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for noise_level in NOISE_LEVELS:
        try:
            results = run_evaluation(
                noise_level=noise_level,
                num_envs=args.num_envs,
                num_episodes=args.num_episodes,
                output_dir=str(output_dir),
            )
            all_results.append(results)

            # Save intermediate results
            with open(output_dir / f"results_{noise_level}.json", 'w') as f:
                json.dump(results, f, indent=2)

            print(f"\n[RESULT] {noise_level}: {results.get('success_rate', 0)*100:.1f}% success")

        except Exception as e:
            print(f"[ERROR] Failed for {noise_level}: {e}")
            all_results.append({"noise_level": noise_level, "error": str(e)})

    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*80)
    print(f"{'Noise Level':<12} {'Noise (cm)':<12} {'Success Rate':<15} {'Avg Reward':<12} {'Max Height':<12}")
    print("-"*80)
    for r in all_results:
        if "error" not in r:
            print(f"{r.get('noise_level', 'N/A'):<12} "
                  f"{r.get('object_noise_cm', 0):<12.1f} "
                  f"{r.get('success_rate', 0)*100:<15.1f}% "
                  f"{r.get('avg_reward', 0):<12.2f} "
                  f"{r.get('avg_max_height', 0):<12.3f} m")
        else:
            print(f"{r.get('noise_level', 'N/A'):<12} ERROR: {r.get('error', '')}")
    print("="*80)

    # Save combined results
    with open(output_dir / "ablation_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate LaTeX table
    latex_table = generate_latex_table(all_results)
    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex_table)

    print(f"\nResults saved to: {output_dir}")


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Impact of Observation Noise on Task Success Rate}
\label{tab:noise_ablation}
\begin{tabular}{lcccc}
\toprule
Noise Level & Object Noise (cm) & Success Rate (\%) & Avg Reward & Max Height (m) \\
\midrule
"""
    for r in results:
        if "error" not in r:
            latex += f"{r.get('noise_level', 'N/A').capitalize()} & "
            latex += f"{r.get('object_noise_cm', 0):.1f} & "
            latex += f"{r.get('success_rate', 0)*100:.1f} & "
            latex += f"{r.get('avg_reward', 0):.1f} & "
            latex += f"{r.get('avg_max_height', 0):.3f} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    main()
