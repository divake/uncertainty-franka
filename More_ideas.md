# Uncertainty decomposition for robust robot manipulation: a comprehensive research guide

**The proposed approach—decomposing policy uncertainty into aleatoric and epistemic components, applying type-specific interventions, and validating with conformal prediction—fills a genuine gap in the literature and is scientifically sound.** No existing work combines all three elements for robot manipulation. The closest precedents are LUCCa (WAFR 2024) for CP + decomposition in dynamics models and a 2020 quadcopter control paper that applies different responses per uncertainty type. However, the current evaluation (lift cube only, +54% with multi-sample averaging) needs significant expansion for a top-venue publication.

---

## 1. Perturbation types and noise models in the literature

Recent robotics papers converge on **four perturbation categories** for evaluating manipulation policy robustness, though no single universally adopted standard exists yet.

**Observation/state noise** is the most common evaluation axis. Gaussian noise dominates, with standard deviations of **σ = 0.1–0.3** on normalized action spaces (Liu et al., 2025, "RobustVLA") and **σ = 0.20** per pixel on normalized images (Ferdous et al., 2025, "CRT"). Beyond Gaussian, structured noise models include Bernoulli-style image masking/erasure (Liu et al., 2025), salt-and-pepper noise on trajectory data (Motamedi et al., ICRA 2024), horizontal line artifacts and water-drop lens contamination (Ferdous et al., 2025), and color jitter. For action noise, adversarial perturbations prove more damaging than random ones (Ayabe et al., 2024), and adversarial training objectives are increasingly used for VLA robustness (Guo et al., ICLR 2026).

**Dynamics perturbations** (mass, friction, damping) are standard in sim-to-real transfer work. DORAEMON (Tiboni et al., ICLR 2024) randomizes **17 dynamics parameters** using Beta distributions for a 7-DoF push task. This builds on foundational dynamics randomization work (Peng et al., ICRA 2018). Visual perturbations span lighting, textures, backgrounds, camera pose, and distractor objects—THE COLOSSEUM benchmark (Pumacay et al., RSS 2024 Workshop) defines **14 perturbation axes** with 213 textures and 20 colors, finding 30–50% success degradation per single perturbation and ≥75% degradation under combined perturbations.

**Occlusion simulation** takes three main forms: random rectangular image patches (Liu et al., 2025), distractor objects from the YCB dataset placed in-scene (COLOSSEUM), and point cloud incompleteness from single-view depth sensing. Some work uses 3D representations (point clouds) specifically to handle occlusion robustly (Ze et al., 2024; Ke et al., 2024).

**OOD scenarios** include novel objects (Physical Intelligence π0.5, 2025—deployed in entirely new homes), changed environments (SIMPLER, Li et al., CoRL 2024—5 distribution shift axes), sim-to-real transfer as an OOD problem (DORAEMON), and modified task instructions (LIBERO-Plus, Fei et al., 2025—found models largely ignore language instructions). LIBERO-Plus reports performance drops from **95% to below 30%** under modest camera viewpoint changes.

**Established benchmark suites** for the proposed work to use or reference:

- **THE COLOSSEUM** (Pumacay et al., RSS 2024): 14 perturbation axes × 20 tasks × 5 models; open-source with leaderboard; R² = 0.614 with real-world performance
- **SIMPLER** (Li et al., CoRL 2024): 5 distribution shift axes; evaluates RT-1, Octo; open-source Gym API
- **LIBERO-Plus** (Fei et al., 2025): 7 perturbation dimensions for VLAs; finer-grained per dimension
- **RL-ViGen** (Yuan et al., NeurIPS 2023): 5 generalization categories across manipulation, locomotion, navigation

For the proposed experiments, the recommended noise sweep should include Gaussian observation noise at σ ∈ {0.01, 0.05, 0.1, 0.2} on the 36-D state vector, action noise at σ ∈ {0.1, 0.2, 0.3}, dynamics randomization of mass and friction (±20–50%), and structured dropout/masking of observation channels to simulate sensor failures.

---

## 2. State-of-the-art uncertainty decomposition methods

The landscape of aleatoric vs. epistemic decomposition has shifted significantly in 2023–2025, with a critical finding from the NeurIPS 2024 Spotlight paper by Mucsányi, Kirchhof, and Oh ("Benchmarking Uncertainty Disentanglement"): **no existing method achieves truly disentangled uncertainty estimates in practice**, with most decomposition pairs showing rank correlation ≥ 0.78.

**Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017) remain the gold standard. The standard decomposition—variance of means (epistemic) + mean of variances (aleatoric)—provides the best calibration (ECE < 2%) and most reliable disentanglement among existing methods. **Five ensemble members** is the de facto standard, with diminishing returns beyond this. For 36-D robot states, small MLP ensembles are computationally cheap, making this the primary baseline.

**MC Dropout** (Gal & Ghahramani, ICML 2016) uses dropout rate p = 0.1 (standard) with M = 10–50 forward passes at test time. It suffers from mode collapse, overestimates epistemic uncertainty in training regions, and does not reliably capture aleatoric uncertainty. It should be included as a baseline but is not recommended as the primary method.

**Evidential Deep Learning** (Amini et al., NeurIPS 2020) predicts Normal-Inverse-Gamma parameters in a single forward pass. However, it has been strongly criticized: "Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?" (NeurIPS 2024) found EDL's epistemic estimates are **unreliable and non-vanishing even with infinite data**. Improved variants exist—R-EDL (Chen et al., ICLR 2024), Density-Aware EDL (Yoon & Kim, ICML 2024)—but the fundamental limitations remain.

**Newer approaches worth considering for 36-D state spaces:**

- **DEUP** (Lahlou et al., TMLR 2023): Trains a secondary "error predictor" on held-out residuals; captures model misspecification bias that variance-based methods miss. Outperforms ensembles and DUQ on OOD detection. Particularly suited for RL because it handles interactive learning settings.
- **SNGP** (Liu et al., NeurIPS 2020/JMLR 2022): Spectral normalization + GP output layer provides distance-aware uncertainty in a single model. Consistently outperforms MC Dropout and approaches ensemble performance. Works on MLPs processing robot states.
- **DDU** (Mukhoti et al., ICML 2022): Spectral normalization + GMD density in feature space for epistemic, softmax for aleatoric. The philosophy mirrors the proposed Mahalanobis + spectral approach.
- **Repulsive Last-Layer Ensembles** (Steger et al., ICML Workshop 2024): Function-space diversity with minimal compute (single multi-headed network). Successfully disentangles AU/EU on DirtyMNIST benchmarks.
- **CDRM** (An et al., L4DC 2025): Novel framework for learned dynamics models enabling Langevin-dynamics-based sampling; achieves **AUROC of 0.8876 (aleatoric) and 0.9981 (epistemic)** on mixed uncertainty test sets. Handles multimodal outputs common in contact-rich manipulation.

**Is Mahalanobis (aleatoric) + Spectral/Repulsive (epistemic) reasonable for 36-D?** Yes, with important caveats. The approach mirrors DDU's architecture (density estimation for one type + spectral features for the other), and Kumar et al. (2025) validate exactly this combination in deep feature spaces for multi-object tracking. For 36-D vectors, Mahalanobis distance is well-suited: it accounts for feature correlations, is computationally efficient, and the dimensionality is manageable for covariance estimation (need ≥360 training samples). Spectral normalization on MLPs processing robot states is straightforward and architecture-agnostic. However, **Mahalanobis assumes Gaussian structure**—reasonable for joint angles/poses with approximately Gaussian noise, but potentially limiting for contact-rich multimodal dynamics. For this regime, CDRM or heteroscedastic NLL heads may be more appropriate.

**Best alternatives for low-dimensional observations:**

- **Gaussian Processes**: Natural fit for ≤50 dimensions; principled decomposition through posterior variance (epistemic) and noise variance (aleatoric); computationally tractable
- **Heteroscedastic NLL Networks** (Kendall & Gal, NeurIPS 2017): Learn input-dependent aleatoric uncertainty via variance prediction head; combine with ensembles for epistemic
- **Small Deep Ensembles (3–5 members)**: Low computational cost for 36-D inputs; provides natural decomposition
- **Bayesian Last Layer** (2024–2025 variants): Single deterministic feature extractor + Bayesian output layer; efficient and well-calibrated for regression

---

## 3. Orthogonality verification metrics and thresholds

Verifying that aleatoric and epistemic uncertainties are properly disentangled requires multiple complementary metrics, since Pearson correlation alone captures only linear dependencies.

**Pearson |r| < 0.3** is the most widely used threshold. Kumar et al. (2025) validate this across 68,630 detections, 8 models, and 3 datasets, achieving |r| ≈ 0.05 between components for both CNN and Transformer architectures. This is the minimum standard reviewers will expect.

**Spearman rank correlation** is preferred by Mucsányi et al. (NeurIPS 2024) as the primary disentanglement metric because it handles non-Gaussian distributions. Their key finding: most methods show ρ ≥ 0.78—dramatically failing the independence test. Target ρ < 0.2 for strong disentanglement. However, Uncertainty Quantification Metrics for Deep Regression (2024) recommends against using Spearman correlation alone for uncertainty evaluation, suggesting AUSE (Area Under Sparsification Error) as a complement.

**HSIC** (Hilbert-Schmidt Independence Criterion; Gretton et al., 2005) is the most principled metric, capturing **nonlinear dependencies** that correlation measures miss. HSIC = 0 if and only if variables are independent under appropriate kernel choice. Use a permutation test (typically 1000 permutations) for significance at p > 0.05. Computational cost is O(n²), but fast approximations exist. This is strongly recommended for the proposed work.

**CKA** (Centered Kernel Alignment; Kornblith et al., ICML 2019) provides a normalized version of HSIC bounded in [0,1]. CKA ≈ 0 indicates structural independence. Use **debiased CKA** with unbiased U-statistics (Murphy et al., 2024) to avoid bias in the low-data, higher-dimensionality regime.

**Behavioral/ablation tests** (Valdenegro-Toro, 2024) are considered the gold standard for validation:

- **Aleatoric isolation test**: Vary observation noise while keeping model capacity constant; epistemic should NOT change
- **Epistemic isolation test**: Vary training data amount while keeping noise constant; aleatoric should NOT change
- **Cross-contamination test**: Measure correlation between predicted AU and changes in ground-truth EU (and vice versa)

Their findings are sobering: "aleatoric and epistemic uncertainty are not reliably separated" with current methods under these behavioral tests.

**Task-based evaluation** (Mucsányi et al., NeurIPS 2024): Evaluate epistemic uncertainty on OOD detection (AUROC on ID vs. OOD data) and aleatoric on predicting annotation disagreement. Proper disentanglement means epistemic significantly outperforms aleatoric on OOD detection AND aleatoric outperforms epistemic on noise-level prediction.

**Recommended verification protocol for the proposed work**: Report Pearson |r| and Spearman ρ (both < 0.3), HSIC permutation test (p > 0.05), behavioral isolation experiments, and temporal trajectory analysis showing flat epistemic trajectories (coefficient of variation ~0.08) versus monotonic aleatoric trajectories (~0.36) as validated by Kumar et al. (2025).

---

## 4. Conformal prediction for robotics manipulation

Conformal prediction has emerged as the dominant framework for distribution-free uncertainty quantification in robotics, with a rapid expansion of applications from 2023 to 2025.

**The foundational reference** is Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (Foundations and Trends in ML, 2023). The split conformal guarantee states that for n exchangeable calibration points, coverage satisfies **1 − α ≤ P(Y ∈ C(X)) ≤ 1 − α + 1/(n+1)**—tight up to O(1/n).

**The most relevant robotics CP papers:**

**KnowNo** (Ren, Dixit et al., CoRL 2023 Oral) applies CP to LLM-based robot planners, constructing prediction sets from softmax likelihoods. If the prediction set is a singleton, the robot acts autonomously; if multiple options remain, it asks for help. Uses **400 calibration tasks** and provides statistical guarantees on task completion rate. The nonconformity score is `s_i = 1 − f̂(X_i)_{Y_i}`.

**Conformal Decision Theory** (Lekeufack, Angelopoulos et al., ICRA 2024) extends CP from prediction sets to **calibrating decisions directly**. The conformal controller adapts λ online: `λ_{t+1} = λ_t + η(ε − ℓ_t)`. Critically, it requires no i.i.d. assumptions and works with adversarial observations—making it directly applicable to the proposed manipulation setting.

**LUCCa** (2024) is the most relevant paper for combining CP with uncertainty decomposition in robotics. It calibrates aleatoric uncertainty estimates (from MVN dynamics predictors) using local conformal prediction (LOCART decision tree partitions) to account for epistemic uncertainty in underrepresented state-action regions. This provides validity guarantees for any finite calibration set.

**Conformalized Teleoperation** (Zhao et al., RSS 2024) applies adaptive conformalized quantile regression to 7-DOF Kinova Jaco manipulation, using quantile regression residuals as nonconformity scores.

**For sequential/dynamic settings** where i.i.d. assumptions are violated:

- **Adaptive Conformal Inference (ACI)** (Gibbs & Candès, NeurIPS 2021): Recursively updates the miscoverage level; guarantees long-run average coverage converges to the target
- **Conformal PID Control** (Angelopoulos et al., NeurIPS 2023): PID-style dynamics for faster adaptation to distribution shifts
- **Bonferroni correction** for multi-step plans: Set per-step failure probability δ̄ = δ/T for T-step horizons (Lindemann et al., RA-L 2023)
- **AdaptNC** (Tumu, Lindemann et al., 2026): Jointly adapts nonconformity score parameters AND conformal threshold online—specifically designed for robotics with distribution shifts

**CP + uncertainty decomposition** is an emerging area with significant gaps. Sale, Javanmardi & Hüllermeier (CoPE 2025) provide the first rigorous analysis of how aleatoric and epistemic uncertainty enter vanilla CP. **EPICSCORE** (2025) models epistemic uncertainty of any conformal score via Bayesian processes (GP, BART, MC Dropout). However, **no paper currently applies CP separately for aleatoric vs. epistemic uncertainty in robotics manipulation**—this is a clear novelty opportunity.

Standard coverage levels in robotics are **90% (α = 0.1)** for typical applications and **95% (α = 0.05)** for safety-critical settings. Calibration sets range from as few as **1/ε data points** (Luo et al., WAFR 2022—10 points for 90% coverage in safety warning systems) to 400 tasks (KnowNo). For the proposed manipulation work, 200–500 calibration episodes should suffice.

---

## 5. Baselines to compare against

**Uncertainty-aware baselines (all should be included):**

| Method | Configuration | Decomposition | Key Reference |
|--------|--------------|---------------|---------------|
| MC Dropout | p = 0.1, M = 10–50 passes | Indirect; total uncertainty only without modifications | Gal & Ghahramani, ICML 2016 |
| Deep Ensembles | **M = 5 members** (de facto standard) | Variance of means = epistemic; mean of variances = aleatoric | Lakshminarayanan et al., NeurIPS 2017 |
| Evidential DL | Single forward pass, NIG parameters | Built-in but unreliable (Mucsányi et al., NeurIPS 2024) | Amini et al., NeurIPS 2020 |
| SNGP | Single model + GP output layer | Distance-aware; no direct decomposition | Liu et al., JMLR 2022 |
| Total uncertainty thresholding | Use combined uncertainty without decomposition | None (ablation of decomposition) | — |

**Non-uncertainty baselines:**

- **PPO/SAC without uncertainty** (Schulman et al., 2017 / Haarnoja et al., 2018): Standard RL baselines showing the performance floor
- **Domain Randomization**: Fixed uniform DR over observation and dynamics parameters; DORAEMON (Tiboni et al., ICLR 2024) for SOTA adaptive DR
- **Robust Domain Contraction** (CoRL 2024): Bi-level approach unifying DR and domain adaptation for contact-rich manipulation
- **Adversarial training**: Adding worst-case perturbations during policy training

For manipulation-specific uncertainty methods, cite AREPO (RA-L 2025) for uncertainty-aware ensemble RL under partial observability, and Safe Robotic Suturing (2025) for diffusion policy ensembles with CBF safety filters.

---

## 6. Experimental setup for a top-venue submission

**Task count and selection**: Top manipulation papers evaluate on **3–6 distinct tasks** spanning difficulty levels. A single task is insufficient. Using Isaac Lab's Franka environments, the minimum recommended set is:

- **Lift Cube** (Isaac-Lift-Cube-Franka-v0): Current task; simple baseline
- **Cube Stacking** (Isaac-Stack-Cube-Franka-IK-Rel-v0): Multi-step sequential reasoning under uncertainty
- **Cabinet Opening** (Isaac-Franka-Cabinet-Direct-v0): Articulated manipulation; different uncertainty profile
- **Peg Insertion** (Isaac-Forge-PegInsert-Direct-v0): Contact-rich precision; aleatoric uncertainty should dominate

**Seeds and episodes**: Use **5–10 random seeds** minimum (Colas et al., 2018 recommends 5 minimum for Welch's t-test; 10 is strong). Run **500–1000 episodes per condition** in simulation—Isaac Lab's GPU parallelism with 1024+ environments makes this feasible. Report compute resources: GPU type (RTX 3090/4090/A100), training time, number of parallel environments.

**Statistical tests**: Use **Welch's t-test** (default for comparing two algorithms; does not assume equal variances) and **bootstrap 95% confidence intervals** (non-parametric, fewer assumptions). Report mean ± standard deviation across seeds, bold best results, include per-task AND aggregated metrics. The "Robot Learning as an Empirical Science" position paper (Kress-Gazit et al., 2024, TRI + Cornell) recommends reporting explicit success criteria, initial conditions, number of evaluations, and narrative failure mode descriptions.

**Visualization requirements**: The following figures are considered essential or highly impactful for uncertainty papers at top robotics venues:

- **Uncertainty decomposition landscape**: Side-by-side heatmaps of aleatoric vs. epistemic uncertainty across the state/workspace; overlay on robot trajectories with color-coded uncertainty tubes (aleatoric = blue, epistemic = red)
- **Calibration/reliability diagrams**: Predicted confidence vs. empirical coverage; essential for conformal prediction validation; show prediction set sizes at 90%, 95%, 99% coverage levels
- **Intervention trigger timeline**: Episode-level traces showing uncertainty values over time with markers indicating when and which interventions were triggered
- **Ablation heatmap**: Grid of performance across noise levels × intervention strategies
- **Failure mode analysis**: Qualitative snapshots with uncertainty annotations; confusion-matrix-style figure (rows = uncertainty type, columns = intervention type)
- **t-SNE/UMAP embedding**: States colored by dominant uncertainty source, demonstrating decomposition quality

IROS uses IEEE two-column format (6 pages + references); CoRL uses single-column PMLR format (8 pages + limitations + references + appendix). CoRL strongly encourages supplementary videos (≤3 min) and code release.

---

## 7. Related work gap analysis and the novelty claim

The critical novelty gap is well-defined across three dimensions.

**Gap 1: No work applies type-specific action modifications in manipulation.** Existing papers that decompose uncertainty use it for data collection triggers (ICREATe, Celemin et al., 2022), help-seeking behavior (KnowNo), stopping/cautious navigation (IROS 2025 social navigation paper), or uniform conservatism. The closest precedent—"Deep Learning based Uncertainty Decomposition for Real-time Control" (2020)—applies different responses (epistemic → event-triggered data collection; aleatoric → controller gain adjustment) but only for quadcopter navigation, not manipulation, and without CP validation.

**Gap 2: No end-to-end framework combining decomposition + targeted intervention + CP validation for manipulation.** LUCCa (WAFR 2024) combines CP with decomposition for dynamics models but does not close the loop to manipulation policy interventions. KnowNo and other CP robotics papers treat total uncertainty without decomposition.

**Gap 3: Limited investigation of decomposition quality in manipulation settings.** Most decomposition evaluations use toy regression, classification, or simple control. Manipulation's unique challenges—contact dynamics, partial observability, multimodal demonstrations—remain underexplored.

**The closest competing approaches**, ranked by proximity:

1. **Uncertainty Decomposition for Real-time Control** (2020): Different interventions per type, quadcopter only, no CP
2. **LUCCa** (Marques & Berenson, WAFR 2024): CP + decomposition in robotics dynamics, no type-specific interventions
3. **Depeweg et al.** (ICML 2018): Foundational decomposition + risk-sensitive RL, not manipulation, no CP
4. **DADEE** (2024): Multiple uncertainty methods for CBF-based safe control, uniform safety treatment
5. **Disentangling Uncertainty for Safe Social Navigation** (IROS 2025): Decomposed uncertainty for DRL navigation, uses cautious stopping only

---

## 8. Top papers to cite and compare against

**Tier 1 — Must cite (directly relevant):**

1. **KnowNo** — Ren, Dixit, Bodrova, Singh et al. — CoRL 2023 Oral — CP for LLM robot planners
2. **LUCCa** — Marques & Berenson — WAFR 2024 — Local conformal calibration for aleatoric + epistemic dynamics uncertainty
3. **Conformal Decision Theory** — Lekeufack, Angelopoulos, Bajcsy, Jordan, Malik — ICRA 2024 — Decision-level CP without i.i.d.
4. **THE COLOSSEUM** — Pumacay, Singh et al. — RSS 2024 — 14-axis robustness benchmark
5. **Disentangling Uncertainty for Social Navigation** — IROS 2025 — Decomposed uncertainty for safe DRL

**Tier 2 — Should cite (methodological foundations):**

6. **Conformalized Teleoperation** — Zhao, Simmons, Admoni, Bajcsy — RSS 2024 — CP for 7-DOF manipulation
7. **DORAEMON** — Tiboni et al. — ICLR 2024 — Dynamics randomization baseline
8. **Robust Manipulation via Domain Contraction** — CoRL 2024 — Non-uncertainty robustness baseline
9. **AREPO** — RA-L 2025 — Uncertainty-aware ensemble RL for manipulation
10. **Benchmarking Uncertainty Disentanglement** — Mucsányi et al. — NeurIPS 2024 Spotlight — Comprehensive decomposition evaluation

**Foundational works (essential citations):**

- Kendall & Gal, NeurIPS 2017 — Aleatoric + epistemic decomposition for deep learning
- Lakshminarayanan et al., NeurIPS 2017 — Deep Ensembles
- Gal & Ghahramani, ICML 2016 — MC Dropout as Bayesian approximation
- Depeweg et al., ICML 2018 — Uncertainty decomposition for model-based RL
- Angelopoulos & Bates, FnTML 2023 — Conformal prediction tutorial
- Hüllermeier & Waegeman, Machine Learning 2021 — Uncertainty concepts survey

---

## 9. Figures and visual strategy

The paper should include **5–7 core figures** arranged to tell a clear narrative: "decompose → diagnose → intervene → validate."

**Figure 1 (System overview)**: Pipeline diagram showing observation → uncertainty decomposition (Mahalanobis aleatoric, spectral/repulsive epistemic) → decision logic (intervention selection) → action modification → conformal prediction validation loop. This is the "story" figure that frames the contribution.

**Figure 2 (Decomposition validation)**: Two panels — (a) scatter plot of aleatoric vs. epistemic uncertainty values across all evaluation episodes, colored by noise type, with Pearson/Spearman/HSIC statistics annotated; (b) behavioral isolation tests showing that varying noise changes aleatoric but not epistemic, and varying training data changes epistemic but not aleatoric.

**Figure 3 (Uncertainty landscapes)**: Side-by-side workspace heatmaps or trajectory overlays for each task, showing where each uncertainty type is elevated. For example, peg insertion should show high aleatoric near contact, while OOD object positions should show high epistemic.

**Figure 4 (Main results table + bar chart)**: Success rates across all tasks × noise conditions × methods (proposed vs. baselines), with 95% bootstrap CIs. Supplement with a bar chart showing relative improvement by uncertainty type.

**Figure 5 (Conformal calibration)**: Reliability diagram showing predicted vs. empirical coverage for the proposed method and baselines. Show that type-specific interventions produce tighter prediction sets while maintaining coverage guarantees.

**Figure 6 (Ablation heatmap)**: Grid of noise magnitude × intervention strategy showing that targeted intervention outperforms uniform intervention, and that decomposition outperforms total-uncertainty approaches.

**Figure 7 (Qualitative trajectories)**: Selected manipulation episodes showing trajectory with uncertainty overlays, intervention trigger points, and outcomes (success/failure). Include failure mode examples with annotations.

---

## 10. Strategic recommendations

**The approach is scientifically sound.** The insight that "robots should respond differently to noise they cannot reduce (aleatoric) versus gaps in their knowledge (epistemic)" is well-motivated by decision theory and has clear practical implications. Multi-sample averaging for aleatoric uncertainty (reducing variance of noisy observations) and conservative/fallback actions for epistemic uncertainty (reducing risk in unfamiliar states) are principled interventions.

**To elevate from acceptable to best-paper-worthy:**

1. **Expand to 3–4 tasks minimum** — lift cube alone will receive a desk-reject-level weakness. Cube stacking, cabinet opening, and peg insertion provide a compelling difficulty gradient.
2. **Real-robot proof-of-concept** — Even 10–20 trials per condition on a real Franka Panda for 1–2 tasks dramatically increases impact. For CoRL especially, this is increasingly expected.
3. **Formalize the decomposition→intervention mapping** with theoretical guarantees, connecting to Conformal Decision Theory (Lekeufack et al., ICRA 2024) for finite-time risk bounds.
4. **Open-source the uncertainty toolkit** as an Isaac Lab extension — community impact is a strong differentiator.
5. **Frame the narrative compellingly**: "Robots that know *why* they're uncertain can respond *appropriately*" is a powerful and memorable framing.

**Potential weaknesses reviewers will flag:**

- **Decomposition transferred from tracking to manipulation** — Mitigate by providing theoretical justification, running calibration experiments on manipulation data specifically, and showing competitive or superior disentanglement metrics (|r| < 0.3, HSIC p > 0.05, behavioral tests) compared to standard baselines.
- **Single task evaluation** — The most critical weakness. Must expand.
- **Mahalanobis assumes Gaussian structure** — Reasonable for joint angles/poses, but discuss limitations for contact-rich multimodal dynamics. Consider ablating against heteroscedastic NLL or CDRM alternatives.
- **Sequential CP violates exchangeability** — Use ACI (Gibbs & Candès, NeurIPS 2021) or Conformal Decision Theory for online settings; report empirical coverage verification; explicitly acknowledge the limitation.
- **+54% improvement needs strong baselines** — Multi-sample averaging is a known technique. Show improvement over MC Dropout (M=10), Deep Ensembles (M=5), EDL, SNGP, and total-uncertainty thresholding.

**Essential ablation studies:**

1. Total uncertainty only vs. aleatoric only vs. epistemic only vs. decomposed (shows decomposition is necessary)
2. Same intervention for all uncertainty vs. targeted (shows targeting matters)
3. Independent noise type sweeps: observation noise (aleatoric source) varied separately from domain shift (epistemic source)
4. Multi-sample count sweep: 1, 3, 5, 10, 20 samples
5. Intervention threshold sensitivity analysis
6. Conformal calibration set size: 50, 100, 200, 500 calibration episodes
7. Full baseline comparison: MC Dropout, Deep Ensembles, EDL, SNGP, domain randomization, robust control

**Critical deadline note**: IROS 2026 submission closes **March 2, 2026** (2 days from now). CoRL 2026 submissions are due **May 28, 2026**. Given the current state (lift cube only, no baselines beyond multi-sample averaging), **CoRL 2026 is the recommended target**. It provides three additional months, and the topic aligns perfectly with CoRL's explicitly listed scope areas: "Probabilistic learning and representation of uncertainty in robotics" and "Theoretical foundations of robot learning, including generalization theory and uncertainty quantification." The venue's single-column 8-page format also provides more space for the multi-section experimental evaluation this work requires.