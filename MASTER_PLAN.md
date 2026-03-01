# MASTER PLAN: Uncertainty Decomposition for Robust Robot Manipulation

**Paper Title**: "Decomposed Uncertainty-Aware Control for Robust Robot Manipulation"
**Venue**: IROS 2026
**Repository**: https://github.com/divake/uncertainty-franka

---

## PAPER THESIS (One Sentence)

> Robots that know *why* they are uncertain — sensor noise vs. lack of experience — can respond with the right fix for each, achieving higher manipulation success than methods that treat all uncertainty the same.

---

## PART I: WHAT WE HAVE (Completed)

### 1.1 Infrastructure
- [x] Isaac Lab v2.3.2 + Isaac Sim 5.1.0 running
- [x] RSL-RL PPO policies loading and evaluating
- [x] Pretrained Franka Panda Lift Cube checkpoint working
- [x] Noise injection via GaussianNoiseCfg working
- [x] GPU-parallel evaluation (16-32 envs) working

### 1.2 Baseline Results (Lift Cube, Franka)
| Noise Level | Object Noise | Success Rate | Avg Reward |
|-------------|-------------|--------------|------------|
| None        | 0 cm        | 100%         | 158.80     |
| Low         | 2 cm        | 100%         | 150.64     |
| Medium      | 5 cm        | 92.5%        | 116.70     |
| High        | 10 cm       | 63.7%        | 42.43      |
| Extreme     | 15 cm       | 33.3%        | 11.63      |

### 1.3 Multi-Sample Averaging Results
| Noise Level | Baseline | Multi-Sample (N=5) | Improvement |
|-------------|----------|-------------------|-------------|
| Medium      | 92.5%    | 100.0%            | +7.5%       |
| High        | 63.7%    | 92.6%             | +28.9%      |
| Extreme     | 33.3%    | 87.5%             | +54.2%      |

### 1.4 Existing Uncertainty Code (From MOT/Tracking Work)
- [x] `mahalanobis.py` — Aleatoric uncertainty via Mahalanobis distance
- [x] `epistemic_spectral.py` — Epistemic via eigenspectrum analysis
- [x] `epistemic_repulsive.py` — Epistemic via repulsive void detection
- [x] `epistemic_gradient.py` — Epistemic via cross-layer divergence
- [x] `epistemic_combined.py` — Weighted fusion with orthogonality optimization

### 1.5 Working Scripts
- [x] `evaluate_noisy_v2.py` — Baseline noisy evaluation
- [x] `evaluate_multi_sample.py` — Multi-sample averaging
- [x] `evaluate_observation_filtering.py` — EMA temporal filtering
- [x] `visualize_comparison.py` — Side-by-side visualization

---

## PART II: THE METHOD

### 2.1 Overview Diagram

```
INPUT                    DECOMPOSITION                TARGETED INTERVENTION           OUTPUT
─────                    ─────────────                ─────────────────────           ──────

                     ┌─────────────────┐
                     │   CALIBRATION   │
                     │   PHASE (once)  │
                     │                 │
                     │ Run nominal     │
                     │ trajectories    │
                     │ → Fit Mahal.    │
                     │ → Fit Spectral  │
                     │ → Fit Conformal │
                     └────────┬────────┘
                              │
                              ▼
┌─────────┐          ┌─────────────────┐         ┌──────────────────────┐         ┌─────────┐
│  Noisy  │─────────▶│  UNCERTAINTY    │────────▶│  INTERVENTION        │────────▶│ Action  │
│  Obs    │          │  DECOMPOSITION  │         │  SELECTOR            │         │         │
└─────────┘          │                 │         │                      │         └─────────┘
                     │  ┌───────────┐  │         │  ┌────────────────┐  │
                     │  │Mahalanobis│──┼──u_a───▶│  │IF u_a > τ_a:   │  │
                     │  │(Aleatoric)│  │         │  │ MultiSample    │  │
                     │  └───────────┘  │         │  │ Averaging      │  │
                     │                 │         │  └────────────────┘  │
                     │  ┌───────────┐  │         │                      │
                     │  │Spectral + │──┼──u_e───▶│  ┌────────────────┐  │
                     │  │Repulsive  │  │         │  │IF u_e > τ_e:   │  │
                     │  │(Epistemic)│  │         │  │ Conservative   │  │
                     │  └───────────┘  │         │  │ Action Scaling │  │
                     │                 │         │  └────────────────┘  │
                     └─────────────────┘         │                      │
                                                 │  ┌────────────────┐  │
                              ┌──────────────────┼──│CONFORMAL PRED. │  │
                              │                  │  │Calibrate τ_a,  │  │
                              │  Coverage        │  │τ_e for coverage│  │
                              │  Guarantee       │  │guarantee       │  │
                              │                  │  └────────────────┘  │
                              │                  └──────────────────────┘
                              ▼
                     P(success | act) ≥ 1 - α
```

### 2.2 Phase A: Calibration (Offline, Once)

**Step 1**: Run the pretrained policy in clean environment (no noise) for 500+ episodes
**Step 2**: Collect all 36-dim observations from successful trajectories → calibration dataset X_cal
**Step 3**: Fit Mahalanobis model on X_cal (mean μ, covariance Σ)
**Step 4**: Fit Spectral + Repulsive epistemic model on X_cal
**Step 5**: Run calibration episodes with known perturbations → compute conformal thresholds

### 2.3 Phase B: Aleatoric Uncertainty Estimation

**Method**: Mahalanobis Distance (from our MOT work)

```
Given: observation obs (36-dim)
       calibration mean μ (36-dim)
       calibration covariance Σ (36×36)

u_aleatoric = sqrt((obs - μ)^T Σ^{-1} (obs - μ))

Interpretation:
  Low u_a  → observation is within normal operating range
  High u_a → observation is corrupted by noise/occlusion/blur
```

**Why Mahalanobis for aleatoric**:
- Captures feature correlations (joint positions are correlated)
- Computationally efficient for 36-dim
- Does NOT depend on model — purely data-driven
- High when observation is noisy/corrupted (far from clean distribution)

### 2.4 Phase C: Epistemic Uncertainty Estimation

**Method**: Spectral Collapse + Repulsive Void Detection (from our MOT work)

```
SPECTRAL:
  1. Find k nearest neighbors of obs in X_cal
  2. Compute local covariance matrix
  3. Eigendecomposition → eigenvalues λ_1...λ_D
  4. Spectral entropy H = -Σ p_i log(p_i) where p_i = λ_i / Σλ
  5. Low entropy = collapsed manifold = high epistemic

REPULSIVE:
  1. Compute "force" from nearest calibration points
  2. In dense regions → forces cancel → low magnitude
  3. In sparse/void regions → net force pushes away → high magnitude
  4. High force = in void = high epistemic

COMBINED:
  u_epistemic = w_s * spectral + w_r * repulsive
  (weights optimized for orthogonality with u_aleatoric)
```

**Why Spectral+Repulsive for epistemic**:
- Captures "has the model seen this state before?"
- Does NOT respond to noise (key for orthogonality)
- Responds to OOD states, novel configurations, domain shift
- Proven in our tracking work with |r| < 0.3 orthogonality

### 2.5 Phase D: Targeted Intervention

The core novelty — **different uncertainty types get different fixes**:

```
INTERVENTION LOGIC:

IF u_aleatoric > τ_a AND u_epistemic ≤ τ_e:
    # High noise, familiar state → FILTER THE OBSERVATION
    obs_clean = average(N noisy samples from ground truth)
    action = policy(obs_clean)

ELIF u_epistemic > τ_e AND u_aleatoric ≤ τ_a:
    # Clean observation, unfamiliar state → BE CONSERVATIVE
    action = policy(obs) * (1 - β * u_epistemic)
    # Reduce action magnitude → slower, safer movements

ELIF u_aleatoric > τ_a AND u_epistemic > τ_e:
    # Both high → FILTER + BE CONSERVATIVE
    obs_clean = average(N noisy samples)
    action = policy(obs_clean) * (1 - β * u_epistemic)

ELSE:
    # Both low → ACT NORMALLY
    action = policy(obs)
```

**Why targeted beats uniform**:
- Multi-sample averaging ONLY helps with aleatoric (noise)
  - Applying it during OOD states wastes computation and doesn't help
- Conservative scaling ONLY helps with epistemic (unfamiliar states)
  - Applying it during noisy-but-familiar states unnecessarily slows the robot
- Combined: Apply the right fix for the right problem

### 2.6 Phase E: Conformal Prediction

**Purpose**: Set thresholds τ_a and τ_e with statistical coverage guarantee.

```
CALIBRATION:
  1. Run 200-500 episodes with various perturbations
  2. For each episode: record u_a, u_e, and outcome (success/fail)
  3. Compute nonconformity scores on failures
  4. Set τ_a = quantile of u_a at level (1-α) on failure set
  5. Set τ_e = quantile of u_e at level (1-α) on failure set

GUARANTEE:
  P(success | u_a < τ_a AND u_e < τ_e) ≥ 1 - α

  For sequential settings, use Adaptive Conformal Inference (ACI):
  τ_{t+1} = τ_t + η(α - err_t)
```

---

## PART III: PERTURBATION TYPES

### 3.1 Aleatoric Perturbations (σ_alea)

These corrupt the observation quality. The true state is unchanged.

| # | Perturbation | Implementation | Parameters | Real-World Analog |
|---|-------------|----------------|------------|-------------------|
| A1 | **Gaussian Noise** | obs += N(0, σ) | σ ∈ {0.01, 0.05, 0.1, 0.2} | Sensor noise |
| A2 | **Occlusion** | Set object position to zero/last-known | p_occlude ∈ {0.1, 0.3, 0.5} | Object hidden behind hand |
| A3 | **Sensor Dropout** | Random dimensions → 0 | p_dropout ∈ {0.05, 0.1, 0.2} | Partial sensor failure |
| A4 | **Motion Blur** | Temporal average of last K obs | K ∈ {3, 5, 10} | Camera motion blur |
| A5 | **Bias/Offset** | obs += constant offset | offset ∈ {0.05, 0.1, 0.2} | Miscalibrated sensor |
| A6 | **Salt-and-Pepper** | Random dims → extreme values | p ∈ {0.05, 0.1} | Sensor spikes |

### 3.2 Epistemic Perturbations (σ_epis)

These put the robot in states it hasn't trained on. Observation is clean.

| # | Perturbation | Implementation | Parameters | Real-World Analog |
|---|-------------|----------------|------------|-------------------|
| E1 | **OOD Object Position** | Object outside training range | shift ∈ {0.1, 0.2, 0.3}m | Object in unusual location |
| E2 | **OOD Joint Config** | Start robot in unusual pose | deviation ∈ {0.2, 0.5}rad | Novel starting position |
| E3 | **Mass Change** | Modify object mass | scale ∈ {2x, 5x, 10x} | Different object weight |
| E4 | **Friction Change** | Modify surface friction | scale ∈ {0.5x, 0.2x, 0.1x} | Slippery object |
| E5 | **Object Size Change** | Scale object dimensions | scale ∈ {0.5x, 1.5x, 2x} | Different object shape |
| E6 | **Gravity Change** | Modify gravity vector | scale ∈ {0.8, 1.2, 1.5} | Tilted surface |

### 3.3 Combined Perturbations (Both)

| # | Combination | Why Important |
|---|-------------|---------------|
| C1 | Gaussian Noise + OOD Position | Realistic: noisy camera + unusual scene |
| C2 | Occlusion + Mass Change | Hardest: can't see properly + unfamiliar dynamics |
| C3 | Sensor Dropout + Novel Object | Multi-failure mode |

**Key Expectation**:
- For A1-A6: u_aleatoric should be HIGH, u_epistemic should stay LOW
- For E1-E6: u_epistemic should be HIGH, u_aleatoric should stay LOW
- For C1-C3: BOTH should be HIGH
- This is the orthogonality validation

---

## PART IV: TASKS AND ROBOTS

### 4.1 Tasks

| # | Task | Isaac Lab ID | Difficulty | Key Uncertainty Challenge |
|---|------|-------------|------------|---------------------------|
| T1 | **Reach Target** | Isaac-Reach-Franka-v0 | Easy | Position accuracy under noise |
| T2 | **Lift Cube** | Isaac-Lift-Cube-Franka-v0 | Medium | Grasp under object position noise |
| T3 | **Cube Stacking** | Isaac-Stack-Cube-Franka-v0 | Hard | Sequential reasoning under uncertainty |
| T4 | **Open Cabinet** | Isaac-Open-Drawer-Franka-v0 | Hard | Contact-rich, constraint following |

### 4.2 Why These Tasks

```
DIFFICULTY GRADIENT:

Reach      →  Lift       →  Stack      →  Cabinet
(easy)        (medium)      (hard)        (hard)

Only position  Grasp +      Multi-step    Contact-rich
accuracy       lift          sequence     articulated

Aleatoric     Both types    Both types    Epistemic
dominates     matter        matter        dominates
```

### 4.3 Expected Behavior Per Task

| Task | Aleatoric Impact | Epistemic Impact | Our Advantage |
|------|-----------------|------------------|---------------|
| Reach | High (position error → miss target) | Low (simple motion) | Multi-sample fixes it |
| Lift | High (miss cube) | Medium (drop if unfamiliar) | Both interventions help |
| Stack | High (alignment critical) | High (sequencing fragile) | Full decomposition needed |
| Cabinet | Medium (contact handles noise) | High (novel dynamics) | Conservative scaling critical |

---

## PART V: BASELINES

### 5.1 Baseline Methods

| # | Method | Description | Decomposition? | Configuration |
|---|--------|-------------|----------------|---------------|
| B0 | **Vanilla** | Raw policy, no uncertainty | None | Standard policy |
| B1 | **Multi-Sample Only** | Average N observations | No (our current approach) | N=5 |
| B2 | **EMA Filter** | Temporal exponential filter | No | α=0.3 |
| B3 | **Deep Ensemble** | 5 copies with perturbed weights | Yes (var of means/mean of vars) | M=5 |
| B4 | **MC Dropout** | Dropout at inference time | Indirect | p=0.1, M=10 |
| B5 | **Total Uncertainty** | Use u_total for uniform intervention | No decomposition | Threshold-based |
| B6 | **Oracle** | Ground truth observations | Upper bound | Perfect information |

### 5.2 What Each Baseline Proves

| Comparison | What It Proves |
|------------|---------------|
| Ours vs B0 (Vanilla) | Uncertainty awareness helps at all |
| Ours vs B1 (Multi-Sample) | Decomposition adds value beyond just filtering |
| Ours vs B3 (Ensemble) | Our decomposition is better calibrated |
| Ours vs B5 (Total Uncert.) | **Decomposition beats monolithic** (key claim) |
| Ours vs B6 (Oracle) | How close we get to perfect information |
| B5 vs B0 | Even total uncertainty helps some |

---

## PART VI: EXPERIMENTS AND TABLES

### Table 1: Main Results — Success Rate Across Tasks and Methods
*This is THE main table of the paper*

| Method | Reach | Lift | Stack | Cabinet | Average |
|--------|-------|------|-------|---------|---------|
| Vanilla (no uncertainty) | | | | | |
| Multi-Sample Only (N=5) | | | | | |
| EMA Filter | | | | | |
| Deep Ensemble (M=5) | | | | | |
| MC Dropout (p=0.1) | | | | | |
| Total Uncertainty Thresh. | | | | | |
| **Decomposed (Ours)** | **bold** | **bold** | **bold** | **bold** | **bold** |
| Oracle (upper bound) | | | | | |

*Run at HIGH noise (σ=0.1). Report mean ± std across 5 seeds. 500 episodes per seed.*

### Table 2: Noise Level Ablation (Lift Cube Task)

| Method | None | Low | Medium | High | Extreme |
|--------|------|-----|--------|------|---------|
| Vanilla | 100% | 100% | 92.5% | 63.7% | 33.3% |
| Multi-Sample | 100% | 100% | 100% | 92.6% | 87.5% |
| Total Uncertainty | | | | | |
| **Decomposed (Ours)** | | | | | |

### Table 3: Perturbation Type Analysis
*Shows that decomposition correctly identifies uncertainty source*

| Perturbation | Type | Vanilla | Aleatoric-Only Fix | Epistemic-Only Fix | **Decomposed** |
|-------------|------|---------|-------------------|-------------------|-------------|
| Gaussian Noise | Alea | | **improved** | no change | **improved** |
| Occlusion | Alea | | **improved** | no change | **improved** |
| Sensor Dropout | Alea | | **improved** | no change | **improved** |
| OOD Position | Epis | | no change | **improved** | **improved** |
| Mass Change | Epis | | no change | **improved** | **improved** |
| Friction Change | Epis | | no change | **improved** | **improved** |
| Combined (Noise+OOD) | Both | | partial | partial | **best** |

*Key insight this table shows: applying the WRONG fix doesn't help. Only decomposed applies the RIGHT fix.*

### Table 4: Orthogonality Verification

| Metric | Threshold | Deep Ensemble | MC Dropout | **Ours** |
|--------|-----------|---------------|------------|----------|
| Pearson \|r\| | < 0.30 | ~0.65 | ~0.72 | **< 0.20** |
| Spearman \|ρ\| | < 0.20 | ~0.58 | ~0.68 | **< 0.15** |
| HSIC (p-value) | > 0.05 | < 0.01 | < 0.01 | **> 0.10** |
| CKA | < 0.20 | ~0.50 | ~0.60 | **< 0.15** |
| Behavioral Test | Pass | Fail | Fail | **Pass** |

*Expected values based on our MOT results. Will update with actual numbers.*

### Table 5: Conformal Prediction Coverage

| Method | Target | Empirical Coverage | Abstention Rate | Success When Acting |
|--------|--------|-------------------|-----------------|---------------------|
| No CP | — | 45% | 0% | 45% |
| Total Uncert. CP (90%) | 90% | ~88% | ~30% | ~75% |
| **Decomposed CP (90%)** | **90%** | **~92%** | **~15%** | **~88%** |
| **Decomposed CP (95%)** | **95%** | **~96%** | **~22%** | **~90%** |

*Key insight: decomposed CP has FEWER abstentions because targeted intervention fixes issues instead of giving up.*

### Table 6: Ablation — Number of Multi-Sample Observations

| N (samples) | Noise Reduction | Success Rate (High) | Computation Time |
|-------------|----------------|---------------------|------------------|
| 1 (baseline) | 1.0x | 63.7% | 1.0x |
| 3 | 1.73x | ~82% | ~1.0x |
| 5 | 2.24x | 92.6% | ~1.0x |
| 10 | 3.16x | ~95% | ~1.0x |
| 20 | 4.47x | ~97% | ~1.0x |

### Table 7: Cross-Task Generalization

| Metric | Reach | Lift | Stack | Cabinet |
|--------|-------|------|-------|---------|
| Vanilla Success (high noise) | | 63.7% | | |
| Ours Success (high noise) | | 92.6% | | |
| Improvement | | +28.9% | | |
| u_a / u_e correlation | | | | |
| Orthogonal? | | | | |

---

## PART VII: FIGURES

### Figure 1: System Overview (Full Width, Top of Paper)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Noisy Obs ──▶ [Uncertainty Decomposition] ──▶ [Targeted Intervention] │
│                  ├─ Mahalanobis → u_alea         ├─ u_alea: Filter      │
│                  └─ Spectral   → u_epis          └─ u_epis: Conservative│
│                                                          │              │
│                                                   [Conformal Pred.]     │
│                                                   Coverage Guarantee    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
*Clean vector diagram. Use TikZ or draw.io.*

### Figure 2: Orthogonality Scatter Plot
- X-axis: Aleatoric uncertainty (u_a)
- Y-axis: Epistemic uncertainty (u_e)
- Points colored by perturbation type:
  - Blue points (Gaussian noise) → cluster on RIGHT (high u_a, low u_e)
  - Red points (OOD position) → cluster on TOP (low u_a, high u_e)
  - Purple points (combined) → top-right
  - Green points (clean) → bottom-left
- Annotate with r, ρ, HSIC values
- **This is the money figure that proves decomposition works**

### Figure 3: Behavioral Isolation Tests (2 panels)
```
Panel A: Vary Noise Level           Panel B: Vary Training Data
─────────────────────────           ──────────────────────────
u_a ↑                               u_a ↑
    │    ╱  u_aleatoric                  │──── u_aleatoric (FLAT)
    │   ╱                                │
    │  ╱                                 │
    │ ╱                                  │
    │╱                                   │
    │──── u_epistemic (FLAT)         u_e ↑
    │                                    │    ╱  u_epistemic
    └─────────────────▶                  │   ╱
      Noise Level                        │  ╱
                                         │ ╱
                                         └─────────────────▶
                                           Training Data Size
```
*Shows that noise only affects aleatoric, data only affects epistemic*

### Figure 4: Success Rate vs Noise Level (Line Plot)
```
Success Rate
    │
100%├─●────●
    │       \
 90%├        ●───────────── Decomposed (Ours)
    │         \
 80%├          ●─────────── Multi-Sample Only
    │           \
 70%├            ●
    │             \
 60%├              ●──────── Total Uncertainty
    │               \
 50%├                ●
    │                 \
 40%├                  ●──── Deep Ensemble
    │                   \
 30%├                    ●── Vanilla
    │
    └────────────────────────▶
     None  Low  Med  High  Ext
            Noise Level
```
*Multiple lines, one per method. Shaded confidence intervals.*

### Figure 5: Intervention Timeline (Single Episode)
```
Episode Timeline (Lift Cube, High Noise)
──────────────────────────────────────────────

u_a  1.0│         ╱╲    ╱╲
     0.5│    ╱╲  ╱  ╲  ╱  ╲   ╱╲
     0.0│───╱──╲╱    ╲╱    ╲─╱──╲───
        │   │    │         │
τ_a ----│---│----|---------|----------
        │   │    │         │
u_e  1.0│   │    │         │
     0.5│   │    │    ╱╲   │
     0.0│───────────╱──╲──────────────
        │              │
τ_e ----│--------------│--------------

Action  │   F    F      C    F
        │
        └──────────────────────────▶
          Step 0     100     200

F = Filtered (multi-sample)
C = Conservative (scaled action)
(blank) = Normal action
```
*Shows how the controller adapts in real-time*

### Figure 6: Confusion Matrix of Interventions
```
                    APPLIED INTERVENTION
                    ─────────────────────────
                    │ Filter   │ Conservative │
    ACTUAL    Noise │  ✓ GOOD  │   ✗ Wasteful │
    SOURCE    ──────│──────────│──────────────│
              OOD   │ ✗ Wrong  │   ✓ GOOD     │
                    ─────────────────────────

Shows: Decomposed applies right fix 90%+ of the time
       Total uncertainty applies wrong fix 50% of the time
```

### Figure 7: Qualitative Trajectories (Side-by-Side Frames)
- 3 rows: Vanilla | Total Uncertainty | Decomposed (Ours)
- 4 columns: t=0, t=50, t=100, t=150
- Color-coded uncertainty tubes around trajectories
- Blue tube = aleatoric dominant
- Red tube = epistemic dominant
- Annotations: "filtered", "conservative", "normal"

### Figure 8: Conformal Coverage Diagram
- X-axis: Target coverage level (80% to 99%)
- Y-axis: Empirical coverage
- Diagonal line = perfect calibration
- Our method should track the diagonal closely
- Baselines should deviate (under-coverage or over-conservative)

### Figure 9: t-SNE/UMAP Visualization
- Project 36-dim observations to 2D
- Color by dominant uncertainty type (blue=aleatoric, red=epistemic, gray=low)
- Show clear spatial separation
- Overlay intervention decisions

---

## PART VIII: IMPLEMENTATION STEPS

### Step 1: Calibration Data Collection
- [ ] Run pretrained policy in clean environment for 500 episodes
- [ ] Collect all observations → X_cal (numpy array, shape [N, 36])
- [ ] Save calibration dataset for reuse

### Step 2: Port Aleatoric Estimation
- [ ] Adapt MahalanobisUncertainty class for 36-dim robot observations
- [ ] Fit on calibration data
- [ ] Test: verify high u_a under Gaussian noise, low u_a under clean observations

### Step 3: Port Epistemic Estimation
- [ ] Adapt SpectralCollapseDetector for 36-dim observations
- [ ] Adapt RepulsiveVoidDetector for 36-dim observations
- [ ] Adapt EpistemicUncertainty (combined) with orthogonality optimization
- [ ] Fit on calibration data
- [ ] Test: verify high u_e under OOD positions, low u_e under clean observations

### Step 4: Orthogonality Verification
- [ ] Implement OrthogonalityAnalyzer (Pearson, Spearman, HSIC, CKA)
- [ ] Run on calibration data with various perturbations
- [ ] Verify |r| < 0.3, HSIC p > 0.05
- [ ] Generate orthogonality scatter plot (Figure 2)

### Step 5: Implement Perturbation Factory
- [ ] All aleatoric perturbations (A1-A6)
- [ ] All epistemic perturbations (E1-E6)
- [ ] Combined perturbations (C1-C3)
- [ ] Test each perturbation works correctly in Isaac Lab

### Step 6: Implement Targeted Intervention Controller
- [ ] UncertaintyDecomposer class (wraps aleatoric + epistemic)
- [ ] InterventionController class (decision logic)
- [ ] Integration with existing policy inference pipeline
- [ ] Test: correct intervention selected for each perturbation type

### Step 7: Implement Conformal Prediction
- [ ] ConformalCalibrator class
- [ ] Calibrate thresholds τ_a, τ_e on held-out data
- [ ] Implement Adaptive Conformal Inference (ACI) for sequential setting
- [ ] Verify coverage guarantee empirically

### Step 8: Implement Baselines
- [ ] Deep Ensemble (M=5 with weight perturbation)
- [ ] MC Dropout (p=0.1, M=10 forward passes)
- [ ] Total Uncertainty thresholding (no decomposition)
- [ ] Oracle (ground truth observations)

### Step 9: Run Multi-Task Experiments
- [ ] Reach task: baseline + all methods
- [ ] Lift Cube task: baseline + all methods (already have partial)
- [ ] Stack Cube task: baseline + all methods
- [ ] Cabinet task: baseline + all methods
- [ ] 5 seeds per experiment, 500 episodes per seed

### Step 10: Generate All Tables
- [ ] Table 1: Main results
- [ ] Table 2: Noise level ablation
- [ ] Table 3: Perturbation type analysis
- [ ] Table 4: Orthogonality verification
- [ ] Table 5: Conformal coverage
- [ ] Table 6: Multi-sample count ablation
- [ ] Table 7: Cross-task generalization

### Step 11: Generate All Figures
- [ ] Figure 1: System overview
- [ ] Figure 2: Orthogonality scatter
- [ ] Figure 3: Behavioral isolation
- [ ] Figure 4: Success vs noise curves
- [ ] Figure 5: Intervention timeline
- [ ] Figure 6: Intervention confusion matrix
- [ ] Figure 7: Qualitative trajectories
- [ ] Figure 8: Conformal coverage
- [ ] Figure 9: t-SNE visualization

### Step 12: Video
- [ ] Side-by-side comparison video using visualize_comparison.py
- [ ] Record for multiple noise levels
- [ ] Add text annotations

---

## PART IX: EXPECTED RESULTS AND STORY

### 9.1 What We Expect to Show

**Claim 1**: Observation noise degrades manipulation success by 30-67%.
*Evidence*: Table 2, baseline results (already proven)

**Claim 2**: Our decomposition correctly separates aleatoric from epistemic uncertainty.
*Evidence*: Table 4 (orthogonality), Figure 2 (scatter), Figure 3 (behavioral tests)

**Claim 3**: Targeted intervention outperforms uniform intervention.
*Evidence*: Table 3 (perturbation type analysis), Figure 6 (confusion matrix)

**Claim 4**: Decomposed uncertainty beats total uncertainty across all tasks.
*Evidence*: Table 1 (main results), Figure 4 (noise curves)

**Claim 5**: Conformal prediction provides valid coverage with fewer abstentions.
*Evidence*: Table 5 (coverage), Figure 8 (calibration diagram)

### 9.2 Expected Failure Modes and How We Handle Them

| Risk | Mitigation |
|------|------------|
| Orthogonality doesn't hold | Retune spectral/repulsive k_neighbors; use orthogonality optimization from epistemic_combined.py |
| Multi-sample doesn't help for some tasks | Expected for contact-rich tasks; report honestly |
| Conservative scaling hurts performance | Tune β parameter; report sensitivity analysis |
| Conformal coverage is loose | Use ACI for tighter bounds; report calibration set size sensitivity |
| Pretrained policies unavailable for some tasks | Train from scratch using RSL-RL (takes ~2-4 hours per task) |

---

## PART X: PAPER OUTLINE

### Title: Decomposed Uncertainty-Aware Control for Robust Robot Manipulation

### Abstract (~150 words)
Robot manipulation policies fail under observation uncertainty. We decompose uncertainty into aleatoric (sensor noise) and epistemic (unfamiliar states), apply targeted interventions — multi-sample averaging for aleatoric, conservative action scaling for epistemic — and validate with conformal prediction coverage guarantees. Experiments across 4 tasks and 12 perturbation types show +X% improvement over baselines.

### I. Introduction (1 page)
- Problem: uncertainty in manipulation
- Insight: different types need different fixes
- Contribution summary (3 bullets)

### II. Related Work (0.75 pages)
- Uncertainty in robot learning
- Aleatoric vs epistemic decomposition
- Conformal prediction in robotics

### III. Method (1.5 pages)
- A. Uncertainty Decomposition (Mahalanobis + Spectral/Repulsive)
- B. Targeted Intervention Logic
- C. Conformal Prediction Calibration

### IV. Experiments (2 pages)
- A. Setup (tasks, robots, perturbations, baselines)
- B. Main Results (Table 1)
- C. Decomposition Quality (Table 4, Figure 2)
- D. Perturbation Analysis (Table 3)
- E. Conformal Coverage (Table 5)
- F. Ablations (Tables 2, 6)

### V. Conclusion (0.25 pages)

### References (1+ pages)

---

## PART XI: KEY REFERENCES

### Must Cite
1. KnowNo (CoRL 2023) — CP for robot planners
2. LUCCa (WAFR 2024) — CP + decomposition for dynamics
3. Conformal Decision Theory (ICRA 2024) — CP without i.i.d.
4. Mucsányi et al. (NeurIPS 2024) — Benchmarking decomposition
5. THE COLOSSEUM (RSS 2024) — Robustness benchmark

### Methodological Foundations
6. Kendall & Gal (NeurIPS 2017) — Aleatoric + epistemic
7. Lakshminarayanan et al. (NeurIPS 2017) — Deep Ensembles
8. Gal & Ghahramani (ICML 2016) — MC Dropout
9. Angelopoulos & Bates (FnTML 2023) — Conformal prediction
10. Depeweg et al. (ICML 2018) — Uncertainty decomposition for RL

---

*This plan was created on 2026-02-28. Track progress by checking boxes above.*
*All code lives in: /mnt/ssd1/divake/robo_uncertain/uncertainty_franka/*
*GitHub: https://github.com/divake/uncertainty-franka*
