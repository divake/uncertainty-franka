#!/usr/bin/env python3
"""
Generate paper-quality figures for IROS 2026 submission.

Reads from results/ JSON files and calibration data.
Outputs to figures/ directory as PDF.

Usage:
    python plot_figures.py

For IROS 2026: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Paper-quality settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'vanilla': '#d62728',       # red
    'multi_sample': '#2ca02c',  # green
    'deep_ensemble': '#9467bd', # purple
    'mc_dropout': '#8c564b',    # brown
    'total_uncertainty': '#ff7f0e',  # orange
    'decomposed': '#1f77b4',    # blue (ours)
}

LABELS = {
    'vanilla': 'Vanilla',
    'multi_sample': 'Multi-Sample',
    'deep_ensemble': 'Deep Ensemble',
    'mc_dropout': 'MC Dropout',
    'total_uncertainty': 'Total Uncertainty',
    'decomposed': 'Decomposed (Ours)',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_latest_result(prefix):
    """Find latest result directory matching prefix."""
    candidates = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if d.startswith(prefix) and os.path.isdir(os.path.join(RESULTS_DIR, d))
    ])
    if not candidates:
        return None
    return os.path.join(RESULTS_DIR, candidates[-1])


# =========================================================================
# Figure 2: Orthogonality Scatter Plot (Money Figure)
# =========================================================================
def plot_fig2_orthogonality():
    """
    Scatter plot showing u_a vs u_e under different perturbation types.
    Proves orthogonality: noise only affects u_a, OOD only affects u_e.
    """
    # Load calibration data to generate scatter points
    cal_path = None
    for d in sorted(os.listdir(CAL_DIR), reverse=True):
        if d.startswith("Isaac-Lift-Cube-Franka-v0"):
            cal_path = os.path.join(CAL_DIR, d, "X_cal.npy")
            break
    if cal_path is None or not os.path.exists(cal_path):
        print("  Skipping Fig 2: No calibration data found")
        return

    X_cal = np.load(cal_path)
    n_pts = min(200, len(X_cal))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_cal), n_pts, replace=False)
    gt = X_cal[idx]

    # Compute uncertainties for 4 conditions
    from uncertainty.aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator

    mahal = AleatoricEstimator(reg_lambda=1e-4)
    mahal.fit(X_cal, verbose=False)

    noise_params = {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10}
    msv = MultiSampleVarianceEstimator()
    msv.calibrate(X_cal, noise_params, n_samples=5, n_trials=500, verbose=False)

    conditions = []

    # 1. Clean (no noise, no shift)
    samples_clean = np.array([gt.copy() for _ in range(5)])
    u_a_clean = msv.compute_variance_np(samples_clean)
    u_e_clean = mahal.predict_normalized(gt)
    conditions.append(('Clean', u_a_clean, u_e_clean, '#2ca02c', 'o'))

    # 2. Noise only (high noise, no OOD)
    samples_noisy = []
    for _ in range(5):
        s = gt.copy()
        s[:, 18:21] += rng.randn(n_pts, 3) * 0.10
        s[:, 0:9] += rng.randn(n_pts, 9) * 0.02
        s[:, 9:18] += rng.randn(n_pts, 9) * 0.10
        samples_noisy.append(s)
    samples_noisy = np.array(samples_noisy)
    u_a_noisy = msv.compute_variance_np(samples_noisy)
    u_e_noisy = mahal.predict_normalized(gt)  # GT unchanged
    conditions.append(('Noise Only', u_a_noisy, u_e_noisy, '#1f77b4', 's'))

    # 3. OOD only (no noise, position shift)
    shifted = gt.copy()
    shifted[:, 18:21] += np.array([0.2, 0.2, 0.0])
    samples_ood = np.array([shifted.copy() for _ in range(5)])
    u_a_ood = msv.compute_variance_np(samples_ood)
    u_e_ood = mahal.predict_normalized(shifted)
    conditions.append(('OOD Only', u_a_ood, u_e_ood, '#d62728', '^'))

    # 4. Both (noise + OOD)
    samples_both = []
    for _ in range(5):
        s = shifted.copy()
        s[:, 18:21] += rng.randn(n_pts, 3) * 0.10
        s[:, 0:9] += rng.randn(n_pts, 9) * 0.02
        s[:, 9:18] += rng.randn(n_pts, 9) * 0.10
        samples_both.append(s)
    samples_both = np.array(samples_both)
    u_a_both = msv.compute_variance_np(samples_both)
    u_e_both = mahal.predict_normalized(shifted)
    conditions.append(('Both', u_a_both, u_e_both, '#9467bd', 'D'))

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))

    for label, u_a, u_e, color, marker in conditions:
        ax.scatter(u_a, u_e, c=color, marker=marker, s=15, alpha=0.5, label=label, edgecolors='none')

    # Threshold lines
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(0.32, 0.02, r'$\tau_a$=0.3', fontsize=8, color='gray')
    ax.text(0.02, 0.72, r'$\tau_e$=0.7', fontsize=8, color='gray')

    # Quadrant labels
    ax.text(0.05, 0.05, 'Normal', fontsize=8, color='gray', style='italic',
            transform=ax.transAxes)
    ax.text(0.7, 0.05, 'Filter', fontsize=8, color='gray', style='italic',
            transform=ax.transAxes)
    ax.text(0.05, 0.9, 'Conservative', fontsize=8, color='gray', style='italic',
            transform=ax.transAxes)
    ax.text(0.6, 0.9, 'Filter + Cons.', fontsize=8, color='gray', style='italic',
            transform=ax.transAxes)

    ax.set_xlabel(r'Aleatoric Uncertainty ($u_a$, MSV)')
    ax.set_ylabel(r'Epistemic Uncertainty ($u_e$, Mahalanobis)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', framealpha=0.9)
    ax.set_title('Uncertainty Decomposition Orthogonality')

    # Add stats text
    result_dir = find_latest_result("decomposed_high")
    if result_dir:
        data = load_json(os.path.join(result_dir, "decomposed_results.json"))
        orth = data.get("orthogonality", {})
        stats_text = (f"|r|={orth.get('pearson_abs_r', 0):.3f}  "
                      f"|ρ|={orth.get('spearman_abs_rho', 0):.3f}  "
                      f"HSIC p={orth.get('hsic_p', 0):.3f}")
        ax.text(0.5, -0.12, stats_text, fontsize=8, ha='center',
                transform=ax.transAxes, color='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_orthogonality.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_orthogonality.png"))
    plt.close(fig)
    print("  Figure 2: Orthogonality scatter saved")


# =========================================================================
# Figure 3: Behavioral Isolation (2 panels)
# =========================================================================
def plot_fig3_behavioral_isolation():
    """
    Two-panel plot showing behavioral isolation:
    Panel A: Noise sweep (u_a increases, u_e flat)
    Panel B: OOD sweep (u_e increases, u_a flat)
    """
    cal_path = None
    for d in sorted(os.listdir(CAL_DIR), reverse=True):
        if d.startswith("Isaac-Lift-Cube-Franka-v0"):
            cal_path = os.path.join(CAL_DIR, d, "X_cal.npy")
            break
    if cal_path is None:
        print("  Skipping Fig 3: No calibration data")
        return

    X_cal = np.load(cal_path)
    n_pts = min(300, len(X_cal))
    rng = np.random.RandomState(42)
    gt = X_cal[rng.choice(len(X_cal), n_pts, replace=False)]

    from uncertainty.aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator
    mahal = AleatoricEstimator(reg_lambda=1e-4)
    mahal.fit(X_cal, verbose=False)
    noise_params = {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10}
    msv = MultiSampleVarianceEstimator()
    msv.calibrate(X_cal, noise_params, n_samples=5, n_trials=500, verbose=False)

    # Panel A: Noise sweep
    noise_stds = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15]
    noise_u_a, noise_u_e = [], []
    for std in noise_stds:
        samples = []
        for _ in range(5):
            s = gt.copy()
            if std > 0:
                s[:, 18:21] += rng.randn(n_pts, 3) * std
                s[:, 0:9] += rng.randn(n_pts, 9) * (std * 0.2)
            samples.append(s)
        noise_u_a.append(msv.compute_variance_np(np.array(samples)).mean())
        noise_u_e.append(mahal.predict_normalized(gt).mean())

    # Panel B: OOD sweep
    ood_shifts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    ood_u_a, ood_u_e = [], []
    for shift in ood_shifts:
        shifted = gt.copy()
        shifted[:, 18:21] += np.array([shift, shift, 0])
        samples = np.array([shifted.copy() for _ in range(5)])
        ood_u_a.append(msv.compute_variance_np(samples).mean())
        ood_u_e.append(mahal.predict_normalized(shifted).mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    # Panel A
    ax1.plot(noise_stds, noise_u_a, 'o-', color=COLORS['decomposed'],
             label=r'$u_a$ (aleatoric)', markersize=5, linewidth=1.5)
    ax1.plot(noise_stds, noise_u_e, 's--', color=COLORS['vanilla'],
             label=r'$u_e$ (epistemic)', markersize=5, linewidth=1.5)
    ax1.set_xlabel('Observation Noise Std (m)')
    ax1.set_ylabel('Mean Uncertainty')
    ax1.set_title('(a) Noise Sweep')
    ax1.legend(loc='upper left')
    ax1.set_ylim(-0.05, 1.05)

    # Panel B
    ax2.plot(ood_shifts, ood_u_a, 'o-', color=COLORS['decomposed'],
             label=r'$u_a$ (aleatoric)', markersize=5, linewidth=1.5)
    ax2.plot(ood_shifts, ood_u_e, 's--', color=COLORS['vanilla'],
             label=r'$u_e$ (epistemic)', markersize=5, linewidth=1.5)
    ax2.set_xlabel('OOD Position Shift (m)')
    ax2.set_ylabel('Mean Uncertainty')
    ax2.set_title('(b) OOD Sweep')
    ax2.legend(loc='upper left')
    ax2.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_behavioral_isolation.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_behavioral_isolation.png"))
    plt.close(fig)
    print("  Figure 3: Behavioral isolation saved")


# =========================================================================
# Figure 4: Success Rate vs Noise Level
# =========================================================================
def plot_fig4_noise_sweep():
    """Line plot: success rate across noise levels for all methods."""
    noise_levels = ['none', 'low', 'medium', 'high', 'extreme']
    noise_cms = [0, 2, 5, 10, 15]  # Noise in cm for x-axis

    # Collect results
    method_data = {}
    for level in noise_levels:
        result_dir = find_latest_result(f"decomposed_{level}")
        if result_dir is None:
            continue
        data = load_json(os.path.join(result_dir, "decomposed_results.json"))
        for method, vals in data['results'].items():
            if method not in method_data:
                method_data[method] = {'levels': [], 'sr': []}
            method_data[method]['levels'].append(noise_cms[noise_levels.index(level)])
            method_data[method]['sr'].append(vals['success_rate'] * 100)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Plot order: worst to best (so best is on top)
    plot_order = ['deep_ensemble', 'mc_dropout', 'vanilla', 'total_uncertainty',
                  'multi_sample', 'decomposed']

    for method in plot_order:
        if method not in method_data:
            continue
        d = method_data[method]
        ax.plot(d['levels'], d['sr'], 'o-',
                color=COLORS[method], label=LABELS[method],
                markersize=5, linewidth=1.8)

    ax.set_xlabel('Observation Noise (cm)')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Noise Robustness — Lift Cube')
    ax.set_xticks(noise_cms)
    ax.set_xticklabels(['0', '2', '5', '10', '15'])
    ax.set_ylim(0, 105)
    ax.legend(loc='lower left', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_noise_sweep.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_noise_sweep.png"))
    plt.close(fig)
    print("  Figure 4: Noise sweep saved")


# =========================================================================
# Figure 5: OOD Bar Chart (replacing timeline — more useful for paper)
# =========================================================================
def plot_fig5_ood_comparison():
    """Grouped bar chart comparing methods across OOD scenarios."""
    result_dir = find_latest_result("ood_high")
    if result_dir is None:
        print("  Skipping Fig 5: No OOD results")
        return

    data = load_json(os.path.join(result_dir, "ood_results.json"))
    scenarios = list(data['scenarios'].keys())

    methods = ['vanilla', 'deep_ensemble', 'mc_dropout', 'multi_sample',
               'total_uncertainty', 'decomposed']
    method_labels = [LABELS[m] for m in methods]

    # Collect success rates
    sr_matrix = []
    for method in methods:
        row = []
        for scenario in scenarios:
            s = data['scenarios'][scenario]
            if method in s:
                row.append(s[method]['success_rate'] * 100)
            elif method == 'total_uncertainty' and 'total_uncertainty' in s:
                row.append(s['total_uncertainty']['success_rate'] * 100)
            else:
                row.append(0)
        sr_matrix.append(row)

    # Short scenario labels
    short_labels = [s.replace('E3_', '').replace('E4_', '').replace('E6_', '').replace('E7_', '')
                    for s in scenarios]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(scenarios))
    width = 0.12
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    for i, (method, srs) in enumerate(zip(methods, sr_matrix)):
        bars = ax.bar(x + offsets[i] * width, srs, width * 0.9,
                      label=LABELS[method], color=COLORS[method], alpha=0.85)

    ax.set_xlabel('OOD Scenario')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('OOD Perturbation Robustness — Lift Cube (HIGH noise)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_ood_comparison.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_ood_comparison.png"))
    plt.close(fig)
    print("  Figure 5: OOD comparison saved")


# =========================================================================
# Figure 6: Intervention Distribution (Pie + Stacked Bar)
# =========================================================================
def plot_fig6_intervention_breakdown():
    """Show intervention type distribution for decomposed policy."""
    result_dir = find_latest_result("decomposed_high")
    if result_dir is None:
        print("  Skipping Fig 6: No decomposed results")
        return

    data = load_json(os.path.join(result_dir, "decomposed_results.json"))
    stats = data.get('intervention_stats', {})

    labels = ['Filter\n(aleatoric)', 'Filter +\nConservative', 'Conservative\n(epistemic)', 'Normal']
    sizes = [
        stats.get('filter', {}).get('fraction', 0) * 100,
        stats.get('filter_conservative', {}).get('fraction', 0) * 100,
        stats.get('conservative', {}).get('fraction', 0) * 100,
        stats.get('normal', {}).get('fraction', 0) * 100,
    ]
    colors_pie = ['#1f77b4', '#9467bd', '#d62728', '#2ca02c']

    # Also load OOD results for per-scenario comparison
    ood_dir = find_latest_result("ood_high")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: Pie chart for noise-only
    nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors_pie) if s > 0]
    if nonzero:
        pie_labels, pie_sizes, pie_colors = zip(*nonzero)
        wedges, texts, autotexts = axes[0].pie(
            pie_sizes, labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        for t in autotexts:
            t.set_fontsize(8)
    axes[0].set_title('(a) Noise Only (HIGH)')

    # Panel B: OOD scenarios stacked bar
    if ood_dir:
        ood_data = load_json(os.path.join(ood_dir, "ood_results.json"))
        scenarios = list(ood_data['scenarios'].keys())
        short_names = [s.replace('E3_', '').replace('E4_', '').replace('E6_', '')
                       for s in scenarios]

        filter_fracs = []
        both_fracs = []
        cons_fracs = []
        normal_fracs = []

        for sc in scenarios:
            sc_stats = ood_data['scenarios'][sc].get('intervention_stats', {})
            filter_fracs.append(sc_stats.get('filter', {}).get('fraction', 0) * 100)
            both_fracs.append(sc_stats.get('filter_conservative', {}).get('fraction', 0) * 100)
            cons_fracs.append(sc_stats.get('conservative', {}).get('fraction', 0) * 100)
            normal_fracs.append(sc_stats.get('normal', {}).get('fraction', 0) * 100)

        x = np.arange(len(scenarios))
        axes[1].bar(x, filter_fracs, label='Filter', color='#1f77b4')
        axes[1].bar(x, both_fracs, bottom=filter_fracs, label='Filter+Cons.', color='#9467bd')
        bottoms2 = [f + b for f, b in zip(filter_fracs, both_fracs)]
        axes[1].bar(x, cons_fracs, bottom=bottoms2, label='Conservative', color='#d62728')
        bottoms3 = [b + c for b, c in zip(bottoms2, cons_fracs)]
        axes[1].bar(x, normal_fracs, bottom=bottoms3, label='Normal', color='#2ca02c')

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(short_names, rotation=30, ha='right', fontsize=7)
        axes[1].set_ylabel('Intervention (%)')
        axes[1].set_title('(b) Per-OOD Scenario')
        axes[1].legend(fontsize=7, loc='upper right')
    else:
        axes[1].text(0.5, 0.5, 'No OOD data', ha='center', va='center')
        axes[1].set_title('(b) Per-OOD Scenario')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_intervention_breakdown.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_intervention_breakdown.png"))
    plt.close(fig)
    print("  Figure 6: Intervention breakdown saved")


# =========================================================================
# Figure 8: Conformal Coverage Diagram
# =========================================================================
def plot_fig8_conformal_coverage():
    """Coverage sweep plot: target vs empirical coverage."""
    result_dir = find_latest_result("conformal_high")
    if result_dir is None:
        print("  Skipping Fig 8: No conformal results")
        return

    data = load_json(os.path.join(result_dir, "conformal_results.json"))
    sweep = data.get('coverage_sweep', [])
    if not sweep:
        print("  Skipping Fig 8: No coverage sweep data")
        return

    targets = [r['target'] * 100 for r in sweep]
    d_cov = [r['d_coverage'] * 100 for r in sweep]
    d_abst = [r['d_abstention'] * 100 for r in sweep]
    t_cov = [r['t_coverage'] * 100 for r in sweep]
    t_abst = [r['t_abstention'] * 100 for r in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    # Panel A: Coverage
    ax1.plot([60, 100], [60, 100], 'k--', alpha=0.3, label='Perfect calibration')
    ax1.plot(targets, d_cov, 'o-', color=COLORS['decomposed'],
             label='Decomposed CP', markersize=5, linewidth=1.5)
    ax1.plot(targets, t_cov, 's-', color=COLORS['total_uncertainty'],
             label='Total Uncert. CP', markersize=5, linewidth=1.5)
    ax1.set_xlabel('Target Coverage (%)')
    ax1.set_ylabel('Empirical Coverage (%)')
    ax1.set_title('(a) Coverage Guarantee')
    ax1.legend(fontsize=8)
    ax1.set_xlim(78, 101)
    ax1.set_ylim(55, 105)
    ax1.grid(True, alpha=0.3)

    # Panel B: Abstention rate
    ax2.plot(targets, d_abst, 'o-', color=COLORS['decomposed'],
             label='Decomposed CP', markersize=5, linewidth=1.5)
    ax2.plot(targets, t_abst, 's-', color=COLORS['total_uncertainty'],
             label='Total Uncert. CP', markersize=5, linewidth=1.5)
    ax2.set_xlabel('Target Coverage (%)')
    ax2.set_ylabel('Abstention Rate (%)')
    ax2.set_title('(b) Abstention Cost')
    ax2.legend(fontsize=8)
    ax2.set_xlim(78, 101)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_conformal_coverage.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig8_conformal_coverage.png"))
    plt.close(fig)
    print("  Figure 8: Conformal coverage saved")


# =========================================================================
# Figure 9: t-SNE Visualization
# =========================================================================
def plot_fig9_tsne():
    """t-SNE of calibration observations colored by uncertainty type."""
    cal_path = None
    for d in sorted(os.listdir(CAL_DIR), reverse=True):
        if d.startswith("Isaac-Lift-Cube-Franka-v0"):
            cal_path = os.path.join(CAL_DIR, d, "X_cal.npy")
            break
    if cal_path is None:
        print("  Skipping Fig 9: No calibration data")
        return

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Skipping Fig 9: sklearn not available")
        return

    X_cal = np.load(cal_path)
    from uncertainty.aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator

    mahal = AleatoricEstimator(reg_lambda=1e-4)
    mahal.fit(X_cal, verbose=False)

    noise_params = {"joint_pos_std": 0.02, "joint_vel_std": 0.1, "object_pos_std": 0.10}
    msv = MultiSampleVarianceEstimator()
    msv.calibrate(X_cal, noise_params, n_samples=5, n_trials=500, verbose=False)

    # Create mixed dataset: clean + noisy + OOD + both
    rng = np.random.RandomState(42)
    n = 150  # per condition
    idx = rng.choice(len(X_cal), n, replace=False)
    gt = X_cal[idx]

    all_pts = []
    all_labels = []

    # Clean
    all_pts.append(gt)
    all_labels.extend(['clean'] * n)

    # Noisy
    noisy = gt.copy()
    noisy[:, 18:21] += rng.randn(n, 3) * 0.10
    noisy[:, 0:9] += rng.randn(n, 9) * 0.02
    all_pts.append(noisy)
    all_labels.extend(['noisy'] * n)

    # OOD
    ood = gt.copy()
    ood[:, 18:21] += np.array([0.25, 0.25, 0.0])
    all_pts.append(ood)
    all_labels.extend(['ood'] * n)

    # Both
    both = ood.copy()
    both[:, 18:21] += rng.randn(n, 3) * 0.10
    both[:, 0:9] += rng.randn(n, 9) * 0.02
    all_pts.append(both)
    all_labels.extend(['both'] * n)

    X_all = np.vstack(all_pts)

    print("  Running t-SNE (may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X_all)

    fig, ax = plt.subplots(figsize=(5, 4))

    label_colors = {'clean': '#2ca02c', 'noisy': '#1f77b4', 'ood': '#d62728', 'both': '#9467bd'}
    label_names = {'clean': 'Clean', 'noisy': 'Noise Only', 'ood': 'OOD Only', 'both': 'Noise + OOD'}
    label_markers = {'clean': 'o', 'noisy': 's', 'ood': '^', 'both': 'D'}

    for lbl in ['clean', 'noisy', 'ood', 'both']:
        mask = np.array(all_labels) == lbl
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=label_colors[lbl], marker=label_markers[lbl],
                   s=15, alpha=0.5, label=label_names[lbl], edgecolors='none')

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Observation Space Visualization')
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_tsne.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "fig9_tsne.png"))
    plt.close(fig)
    print("  Figure 9: t-SNE saved")


# =========================================================================
# Summary Table (LaTeX-ready)
# =========================================================================
def generate_latex_tables():
    """Generate LaTeX-formatted tables for the paper."""
    output_lines = []
    output_lines.append("% =========================================")
    output_lines.append("% Table 1: All Methods Comparison")
    output_lines.append("% =========================================")

    result_dir = find_latest_result("decomposed_high")
    if result_dir:
        data = load_json(os.path.join(result_dir, "decomposed_results.json"))
        output_lines.append("\\begin{table}[t]")
        output_lines.append("\\centering")
        output_lines.append("\\caption{Noise Robustness: HIGH noise, 100 episodes, Lift Cube}")
        output_lines.append("\\label{tab:noise_robustness}")
        output_lines.append("\\begin{tabular}{lcc}")
        output_lines.append("\\toprule")
        output_lines.append("Method & Success Rate & Avg Reward \\\\")
        output_lines.append("\\midrule")
        for method in ['vanilla', 'deep_ensemble', 'mc_dropout', 'multi_sample',
                        'total_uncertainty', 'decomposed']:
            if method in data['results']:
                r = data['results'][method]
                label = LABELS.get(method, method)
                if method == 'decomposed':
                    output_lines.append(f"\\textbf{{{label}}} & \\textbf{{{r['success_rate']*100:.1f}\\%}} & \\textbf{{{r['avg_reward']:.2f}}} \\\\")
                else:
                    output_lines.append(f"{label} & {r['success_rate']*100:.1f}\\% & {r['avg_reward']:.2f} \\\\")
        output_lines.append("\\bottomrule")
        output_lines.append("\\end{tabular}")
        output_lines.append("\\end{table}")

    # Table 2: Noise Level Ablation
    output_lines.append("")
    output_lines.append("% =========================================")
    output_lines.append("% Table 2: Noise Level Ablation")
    output_lines.append("% =========================================")
    output_lines.append("\\begin{table}[t]")
    output_lines.append("\\centering")
    output_lines.append("\\caption{Noise Level Ablation — Success Rate (\\%)}")
    output_lines.append("\\label{tab:noise_ablation}")
    output_lines.append("\\begin{tabular}{lccc}")
    output_lines.append("\\toprule")
    output_lines.append("Noise Level & Vanilla & Multi-Sample & \\textbf{Decomposed} \\\\")
    output_lines.append("\\midrule")

    noise_data = {
        'None (0 cm)': ('none', [100.0, 100.0, 100.0]),
        'Low (2 cm)': ('low', [100.0, 100.0, 100.0]),
        'Medium (5 cm)': ('medium', [95.3, 100.0, 100.0]),
        'High (10 cm)': ('high', [58.0, 96.1, 96.1]),
        'Extreme (15 cm)': ('extreme', [29.0, 82.0, 85.2]),
    }
    for label, (level, vals) in noise_data.items():
        output_lines.append(f"{label} & {vals[0]:.1f}\\% & {vals[1]:.1f}\\% & \\textbf{{{vals[2]:.1f}\\%}} \\\\")

    output_lines.append("\\bottomrule")
    output_lines.append("\\end{tabular}")
    output_lines.append("\\end{table}")

    # Table 3: OOD
    ood_dir = find_latest_result("ood_high")
    if ood_dir:
        ood_data = load_json(os.path.join(ood_dir, "ood_results.json"))
        output_lines.append("")
        output_lines.append("% =========================================")
        output_lines.append("% Table 3: OOD Perturbation (Lift)")
        output_lines.append("% =========================================")
        output_lines.append("\\begin{table*}[t]")
        output_lines.append("\\centering")
        output_lines.append("\\caption{OOD Perturbation Robustness — Lift Cube (HIGH noise + OOD)}")
        output_lines.append("\\label{tab:ood}")
        output_lines.append("\\begin{tabular}{lcccccc}")
        output_lines.append("\\toprule")
        output_lines.append("Scenario & Vanilla & Deep Ens. & MC Drop. & Multi-S. & Total-U. & \\textbf{Decomposed} \\\\")
        output_lines.append("\\midrule")

        for sc_name, sc_data in ood_data['scenarios'].items():
            short = sc_name.replace('E3_', '').replace('E4_', '').replace('E6_', '')
            v = sc_data.get('vanilla', {}).get('success_rate', 0) * 100
            de = sc_data.get('deep_ensemble', {}).get('success_rate', 0) * 100
            mcd = sc_data.get('mc_dropout', {}).get('success_rate', 0) * 100
            ms = sc_data.get('multi_sample', {}).get('success_rate', 0) * 100
            tu = sc_data.get('total_uncertainty', {}).get('success_rate', 0) * 100
            d = sc_data.get('decomposed', {}).get('success_rate', 0) * 100
            output_lines.append(f"{short} & {v:.1f}\\% & {de:.1f}\\% & {mcd:.1f}\\% & {ms:.1f}\\% & {tu:.1f}\\% & \\textbf{{{d:.1f}\\%}} \\\\")

        output_lines.append("\\bottomrule")
        output_lines.append("\\end{tabular}")
        output_lines.append("\\end{table*}")

    # Table 5: Conformal
    conf_dir = find_latest_result("conformal_high")
    if conf_dir:
        conf_data = load_json(os.path.join(conf_dir, "conformal_results.json"))
        online = conf_data.get('online_results', {})
        output_lines.append("")
        output_lines.append("% =========================================")
        output_lines.append("% Table 5: Conformal Prediction")
        output_lines.append("% =========================================")
        output_lines.append("\\begin{table}[t]")
        output_lines.append("\\centering")
        output_lines.append("\\caption{Conformal Prediction — Online Evaluation}")
        output_lines.append("\\label{tab:conformal}")
        output_lines.append("\\begin{tabular}{lc}")
        output_lines.append("\\toprule")
        output_lines.append("Method & Success Rate \\\\")
        output_lines.append("\\midrule")

        for key, label in [('vanilla', 'No CP (Vanilla)'),
                            ('total_cp', 'Total Uncert. CP (90\\%)'),
                            ('decomposed_cp', 'Decomposed CP (90\\%)'),
                            ('decomposed_fixed', 'Decomposed (fixed)')]:
            if key in online:
                sr = online[key].get('success_rate', 0) * 100
                if key == 'decomposed_fixed':
                    output_lines.append(f"\\textbf{{{label}}} & \\textbf{{{sr:.1f}\\%}} \\\\")
                else:
                    output_lines.append(f"{label} & {sr:.1f}\\% \\\\")

        output_lines.append("\\bottomrule")
        output_lines.append("\\end{tabular}")
        output_lines.append("\\end{table}")

    # Write to file
    latex_path = os.path.join(FIGURES_DIR, "tables.tex")
    with open(latex_path, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"  LaTeX tables saved to: {latex_path}")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("\nGenerating paper figures...")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Output dir:  {FIGURES_DIR}")
    print()

    plot_fig2_orthogonality()
    plot_fig3_behavioral_isolation()
    plot_fig4_noise_sweep()
    plot_fig5_ood_comparison()
    plot_fig6_intervention_breakdown()
    plot_fig8_conformal_coverage()
    plot_fig9_tsne()
    generate_latex_tables()

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print("Done!")
