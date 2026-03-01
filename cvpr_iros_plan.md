# CVPR → IROS Extension: Final Agreed Plan

**CVPR Paper**: "Calibrated Decomposition of Aleatoric and Epistemic Uncertainty in Deep Features for Inference-Time Adaptation" (arXiv:2511.12389)

**IROS Extension**: Same decomposition framework applied to robot manipulation. Post-hoc, lightweight, zero training — just calibration data statistics.

**Status**: AGREED. Ready for implementation.

---

## 1. Core Principles (Non-Negotiable)

1. **Mahalanobis distance = σ_alea (aleatoric)** in BOTH papers. Same formula. Same name.
2. **Post-hoc, zero training, no ensembles** — same philosophy as CVPR ("no sampling, no ensembling, no additional forward passes"). All signals computed from calibration data statistics only.
3. **Scales to multiple robots and policies** — collect calibration data → fit statistics → deploy. No per-robot training.
4. **CVPR adapts COMPUTATION** (model switching). **IROS adapts BEHAVIOR** (targeted intervention). Complementary, not duplicative.

---

## 2. Three Perturbation Types

| # | Name | Type | Mechanism | Real-World Analogue |
|---|------|------|-----------|---------------------|
| 1 | **Sensor noise** | Aleatoric | Gaussian noise on joint positions, velocities, object position | Noisy encoders, imprecise camera |
| 2 | **Dynamics shift** | Epistemic | Changed object mass, surface friction, gravity | New object weight, worn surface, tilted platform |
| 3 | **Partial occlusion** | Aleatoric | Intermittent object position dropout (dims 18-20 → zero/last-known) | Hand/obstacle blocking camera view |

**Purpose**: Controlled validation. We inject KNOWN perturbation types and verify the decomposition correctly identifies the source. This provides causal evidence that CVPR could not — in vision, perturbation types are entangled in natural data. In robotics, we can isolate them.

**Experiment structure**:
- Inject noise ONLY → σ_alea should rise, σ_epis should stay flat (PROOF)
- Inject OOD ONLY → σ_epis should rise, σ_alea should stay flat (PROOF)
- Inject BOTH → both rise, robot must figure out which is which (APPLICATION)

---

## 3. Uncertainty Decomposition

### 3.1 Aleatoric: σ_alea = Mahalanobis Distance (IDENTICAL to CVPR Sec 3.2)

```
Calibration:
  μ = mean of clean calibration observations
  Σ_reg = covariance + λ·(tr(Σ)/d)·I

At test time:
  M(s) = sqrt( (s - μ)^T Σ_reg^{-1} (s - μ) )
  σ_alea = (log(M + ε) - log M_min) / (log M_max - log M_min)
```

- Same formula as CVPR. Same normalization. Same meaning.
- Under sensor noise: observation deviates from clean distribution → high σ_alea.
- Under occlusion: missing dimensions push observation far from distribution → high σ_alea.
- Code: `AleatoricEstimator` in `aleatoric.py` — already exists, already correctly named.

### 3.2 Epistemic: σ_epis = Two Local State-Space Statistics

Two components (adapted from CVPR's three; ε_grad dropped since there's no encoder):

**Component 1: ε_knn (State Novelty)**
```
ε_knn(s) = normalize( ||s - s^(k)||₂ )
```
- k-th nearest neighbor distance in calibration set
- Same spirit as CVPR's ε_supp (local support deficiency)
- No training. Just build k-NN index on calibration data.
- Well-cited: Sun et al. (ICML 2022), LOF (Breunig et al. 2000)
- OOD states → far from calibration → high ε_knn
- Sensor noise → robot visits SAME states (noisily measured) → ε_knn stays flat

**Component 2: ε_rank (Geometric Collapse) — SAME FORMULA as CVPR Sec 3.1**
```
Find k nearest calibration neighbors → local covariance Σ_loc
Eigendecomposition → spectral entropy H = -Σ p_i log p_i
Effective rank: r_eff = exp(H)
ε_rank(s) = 1 - (r_eff - 1) / (d_eff - 1)
```
- IDENTICAL formula to CVPR.
- No training. Just eigendecomposition of local covariance.
- Preprocessing: remove zero-variance dimensions (24-27) before computing.
- OOD regions → degenerate local geometry → high ε_rank

**Combined:**
```
σ_epis = w₁·ε_knn + w₂·ε_rank
Weights optimized for orthogonality with σ_alea (same methodology as CVPR).
```

### 3.3 Total Uncertainty (Baseline)
```
σ_total = sqrt(σ²_alea + σ²_epis)
```
Same as CVPR's σ_comb. Monolithic — cannot distinguish noise from OOD.

---

## 4. Targeted Intervention (IROS-Specific: Behavior Adaptation)

**CVPR**: uncertainty type → which MODEL to use (adaptive computation)
**IROS**: uncertainty type → how to ACT (adaptive behavior)

| Condition | σ_alea | σ_epis | Robot Behavioral Response |
|-----------|--------|--------|--------------------------|
| Normal | Low | Low | Full speed, normal control |
| Noisy observation | High | Low | **Observation filtering** (multi-sample averaging to denoise) |
| Unfamiliar state | Low | High | **Conservative action** (reduce action magnitude, move cautiously) |
| Both degraded | High | High | **Filter + conservative** (maximum caution) |

**Why decomposed beats total uncertainty**:
- Total Uncertainty applies BOTH filtering AND conservative action EVERY TIME → over-intervention, wasted capability
- Decomposed applies only the APPROPRIATE fix → right fix for right problem → higher success with less intervention

**MSV's role**: Not a separate uncertainty signal. It is the denoising mechanism used when σ_alea triggers the filtering intervention. In the paper: "When σ_alea indicates degraded observations, we suppress sensor noise by averaging N=5 independent state readings."

---

## 5. Calibration Dataset

**Procedure** (same as CVPR):
1. Run pretrained policy in nominal environment (no noise, no perturbations)
2. Collect all state observations from successful episodes
3. This is our calibration set V_cal

**What we fit from V_cal**:
- μ, Σ_reg for σ_alea (Mahalanobis) — global statistics
- k-NN index for ε_knn — neighbor lookup
- Local covariance structure for ε_rank — local statistics

**Existing data**: 56K observations from 224 successful Lift Cube episodes. Verify preprocessing (remove zero-variance dims, standardize).

---

## 6. Paper Deliverables (2-3 Pages)

### Table 1: Controlled Validation + Main Results

| Perturbation | Type | σ_alea ↑? | σ_epis ↑? | Vanilla | Total-U | **Decomposed** |
|-------------|------|-----------|-----------|---------|---------|----------------|
| Sensor noise | Alea | Yes | No | ~58% | ~85% | **~95%** |
| Occlusion | Alea | Yes | No | ~50% | ~80% | **~90%** |
| Mass 5x | Epis | No | Yes | ~35% | ~70% | **~80%** |
| Friction 0.2x | Epis | No | Yes | ~49% | ~85% | **~90%** |
| Noise + Mass 5x | Both | Yes | Yes | ~30% | ~65% | **~80%** |

### Table 2: Multi-Robot / Multi-Policy (Future — after single robot works)

| Robot | Task | Policy | Clean | Noise | OOD | Both | Decomposed |
|-------|------|--------|-------|-------|-----|------|-----------|
| Franka | Lift | PPO | 100% | ... | ... | ... | **...** |
| Franka | Reach | PPO | 94% | ... | ... | ... | **...** |
| Robot B | Task X | ... | ... | ... | ... | ... | **...** |

### Figure: Orthogonality + Behavioral Isolation

- Panel A: Scatter plot σ_alea vs σ_epis (4 conditions, matching CVPR Figure 1d)
- Panel B: Behavioral isolation curves (noise → σ_alea rises; OOD → σ_epis rises)

---

## 7. Implementation Order

### Phase 1: Core Framework (ONE robot, ONE policy — Franka Lift Cube)
1. Implement ε_knn (k-NN distance) for 36D state space
2. Implement ε_rank (spectral entropy) for 36D state space
3. Preprocess calibration data (remove zero-variance dims 24-27, standardize)
4. Rewire: σ_alea = Mahalanobis (existing), σ_epis = w₁·ε_knn + w₂·ε_rank (new)
5. Add partial occlusion perturbation type
6. Run controlled validation: inject each perturbation separately, verify decomposition

### Phase 2: Intervention + Comparison
7. Update intervention logic: σ_alea → filter, σ_epis → conservative
8. Run comparison: Vanilla vs Total Uncertainty vs Decomposed
9. Tune thresholds (τ_alea, τ_epis) and weights (w₁, w₂)

### Phase 3: Results + Figures
10. Generate orthogonality scatter plot
11. Generate behavioral isolation plots
12. Generate LaTeX tables
13. Verify results are solid

### Phase 4: Multi-Robot Expansion (AFTER Phase 3 is solid)
14. Franka Reach task
15. Additional robots/tasks as available
16. Multi-robot comparison table

---

## 8. CVPR ↔ IROS Consistency Summary

| Aspect | CVPR (Vision) | IROS (Robotics) |
|--------|--------------|-----------------|
| Philosophy | Post-hoc, no training | Post-hoc, no training |
| σ_alea | Mahalanobis (global density) | Mahalanobis (global density) — **SAME** |
| σ_epis component 1 | ε_supp (local support) | ε_knn (k-NN distance) — same concept |
| σ_epis component 2 | ε_rank (spectral entropy) | ε_rank (spectral entropy) — **SAME FORMULA** |
| σ_epis component 3 | ε_grad (cross-layer) | Dropped (no encoder layers) |
| Orthogonality | \|r\| = 0.048 | Target: \|r\| < 0.1 |
| Intervention | Adaptive model selection | Adaptive robot behavior |
| Scaling | 8 detector architectures | Multiple robots × policies |
| Conformal | Distribution-free thresholds | Same approach |
