# Epistemic uncertainty for 36-D robot states: five methods that work

> **NOTE (v3.0):** Early brainstorm document. From these ideas, we adopted ε_knn
> and spectral entropy (ε_rank) for the CVPR-consistent pipeline. See
> `uncertainty/epistemic_cvpr.py` for the final implementation. Kept for reference.

**For low-dimensional physical observation vectors in robot manipulation, the strongest literature-grounded approach combines five complementary epistemic signals: k-NN state distance, dynamics ensemble disagreement, Random Network Distillation, policy entropy, and normalizing-flow density.** Together these capture geometric state novelty, physics surprise, learned feature-space novelty, behavioral confidence, and calibrated distributional support — all computable within a combined **~3–4 ms budget** from a calibration dataset of successful trajectories. Each method draws on well-cited work (NeurIPS, ICML, ICLR, ICRA) validated in continuous control or manipulation, and each captures a genuinely distinct facet of "model ignorance" that would be defensible to IROS/CoRL reviewers.

The two candidate methods already proposed — k-NN state distance and Mahalanobis on transitions — are both reasonable starting points. The k-NN idea is strongly validated; the Mahalanobis-on-transitions idea should be **upgraded to dynamics ensemble disagreement**, which conditions on (s_t, a_t), cleanly separates epistemic from aleatoric uncertainty, and is the gold standard in model-based RL. Below are the five recommended components with full literature grounding.

---

## Method 1: k-NN distance in state space captures geometric novelty

**What it measures.** The Euclidean distance from a test state to its k-th nearest neighbor in the calibration set. States far from any calibration point receive high epistemic scores. This is a non-parametric, model-free signal that answers: *"Has the robot been in a state like this before?"*

**Literature grounding.** Sun et al. (ICML 2022, "Out-of-Distribution Detection with Deep Nearest Neighbors") demonstrated that k-NN distance in feature space is a state-of-the-art OOD detector, outperforming energy-based and softmax methods. While their paper operates on deep features, the core algorithm is purely distance-based and applies directly to raw 36-D vectors without any neural encoder. The closely related **Local Outlier Factor** (Breunig et al., 2000) extends this with local density ratios, handling multi-modal calibration distributions common in multi-phase manipulation tasks. In robotics, **FAIL-Detect** (Xu et al., RSS 2025) uses distance-based OOD scores for failure detection in robotic manipulation.

**Formula.** For a test state $s$ and calibration set $\mathcal{D}$:

$$\varepsilon_{\text{knn}}(s) = \|s - s^{(k)}\|_2$$

where $s^{(k)}$ is the k-th nearest neighbor in $\mathcal{D}$. Typically **k = 10–50** works well. Each state dimension should be z-score normalized given heterogeneous units (radians, m/s, meters).

**Computational cost.** With FAISS (Meta's approximate nearest neighbor library), brute-force search over **100K calibration points in 36-D takes ~0.1–0.5 ms** on CPU. KD-trees are also effective at d=36 (borderline for curse of dimensionality but workable). **Easily meets the <5 ms budget.**

**Why it works for 36-D robot states.** Unlike KDE, which suffers severely from the curse of dimensionality at d=36, k-NN distance degrades gracefully because it only requires local distance computations. The 36 dimensions are physically meaningful and relatively low-rank (joint angles are correlated with end-effector positions), so effective dimensionality is lower than 36. No training is needed — just store the calibration set and precompute an index.

**Limitation.** Sensitive to feature scaling; does not capture dynamics information. A state may be geometrically "close" to calibration data but exhibit novel dynamics (e.g., unexpected contact forces). This is why a dynamics-based signal is essential as a complement.

---

## Method 2: Probabilistic dynamics ensemble disagreement quantifies physics surprise

**What it measures.** The variance across an ensemble of independently trained dynamics models predicting the next state. High disagreement indicates the ensemble has not converged — a direct measure of epistemic uncertainty about the physics. This answers: *"Do our dynamics models agree on what should happen next?"*

**Literature grounding.** This is the most thoroughly validated approach for epistemic uncertainty in dynamics models, grounded in three foundational papers:

- **Lakshminarayanan et al. (NeurIPS 2017)** established deep ensembles as the gold standard for predictive uncertainty, showing that **B = 5 members** suffice for reliable estimates with diminishing returns beyond that.
- **Chua et al. (NeurIPS 2018, PETS)** applied probabilistic ensembles specifically to dynamics models for model-based RL, providing a clean **law-of-total-variance decomposition**: epistemic = Var_b[μ_b], aleatoric = E_b[Σ_b]. Validated on continuous control up to 20-D state spaces (HalfCheetah, Pusher with a real PR2 robot).
- **Pathak et al. (ICML 2019, "Self-Supervised Exploration via Disagreement")** showed ensemble disagreement is high for novel states and low for stochastic-but-familiar states — exactly the epistemic/aleatoric distinction needed. **Validated on a real robotic arm.**

More recently, **RWM-U (2025)** validates ensemble-based epistemic uncertainty on real ANYmal D and Unitree G1 robots for offline model-based RL.

**Formula.** Train B = 5 independently initialized MLPs, each mapping $(s_t, a_t) \to (\mu_b, \Sigma_b)$:

$$\varepsilon_{\text{ensemble}}(s_t, a_t) = \frac{1}{B}\sum_{b=1}^{B}\|\mu_b(s_t, a_t) - \bar{\mu}(s_t, a_t)\|^2$$

where $\bar{\mu} = \frac{1}{B}\sum_b \mu_b$. The aleatoric component is $\frac{1}{B}\sum_b \Sigma_b(s_t, a_t)$, trained with negative log-likelihood loss.

**Architecture.** Each member: MLP **256→128→64**, input $\mathbb{R}^{36+d_a}$, output $(\mu \in \mathbb{R}^{36}, \log\sigma^2 \in \mathbb{R}^{36})$. Train each member with different random initialization and optional bootstrap masking on the calibration dataset.

**Computational cost.** Five batched forward passes through small MLPs: **~1–2 ms on GPU**. Can be further parallelized via a single wide network with B output heads. **Comfortably meets <5 ms.**

**Why this replaces Mahalanobis on transitions.** The user's proposed ε_dyn (Mahalanobis distance of state deltas from a global Gaussian) has three critical weaknesses the ensemble approach resolves: (1) it is **state-independent** — a single global Gaussian cannot capture the fact that expected transitions vary enormously across the state space; (2) it **does not condition on the action taken**; (3) the **Gaussian assumption** is violated near contacts and task phase transitions. Ensemble disagreement is conditioned on $(s_t, a_t)$, handles multimodality, and provides a principled epistemic/aleatoric split. The Mahalanobis approach remains useful as an ultra-fast (<0.01 ms) supplementary sanity check, well-grounded in Lee et al. (NeurIPS 2018, >2200 citations).

---

## Method 3: Random Network Distillation provides a learned novelty detector

**What it measures.** The prediction error of a trainable network trying to match the outputs of a fixed, randomly initialized target network. States visited frequently during calibration have low error; novel states have high error. This answers: *"How novel is this state in a learned feature space?"*

**Literature grounding.** **Burda et al. (ICLR 2019, "Exploration by Random Network Distillation")** introduced RND as an exploration bonus for RL, explicitly designed to work with PPO. The key insight is that because the target network is fixed and deterministic, RND is **immune to the "noisy TV" problem** — it does not produce false novelty signals from stochastic dynamics (unlike forward prediction error methods like ICM). This makes it a **pure epistemic uncertainty signal** by construction. In robotics, RND has been used in **RND-DAgger (2024)** for OOD detection in robot imitation learning and in the **FIPER framework** for failure prediction in robotic manipulation.

**Formula.** Two small MLPs with the same architecture:

$$\varepsilon_{\text{rnd}}(s) = \|f_{\text{pred}}(s; \theta) - f_{\text{target}}(s; \theta_{\text{fixed}})\|^2$$

where $f_{\text{target}}$ has frozen random weights and $f_{\text{pred}}$ is trained to minimize this error on calibration states. Both networks map $\mathbb{R}^{36} \to \mathbb{R}^{64}$. **Observation normalization** (running mean/std whitening + clipping) is critical for stability.

**Computational cost.** Two forward passes through tiny networks: **~0.05–0.5 ms.** The fastest neural method available.

**Why it complements k-NN.** While k-NN captures geometric distance in raw state space, RND captures novelty in a **learned feature space** that may discover non-obvious distributional boundaries. k-NN treats all dimensions equally (after normalization); RND implicitly learns which state dimensions matter most for distinguishing in-distribution from OOD states. The combination is more robust than either alone — k-NN provides a geometry-grounded baseline while RND captures learned distributional structure.

**Why it complements ensemble disagreement.** RND captures state novelty independent of actions or dynamics; the ensemble captures dynamics surprise conditioned on actions. A state could be geometrically familiar (low RND) but exhibit surprising dynamics (high ensemble disagreement) due to changed object properties, or vice versa.

---

## Method 4: PPO policy entropy signals behavioral confidence

**What it measures.** The entropy of the policy's action distribution at a given state. High entropy means the policy is uncertain about which action to take. This answers: *"Does the policy know what to do here?"*

**Literature grounding.** Policy entropy is the standard exploration signal in PPO (Schulman et al., 2017) and is already computed during training. **Flögel et al. (IROS 2025, "Disentangling Uncertainty for Safe Social Navigation")** formalized the decomposition using observation-dependent variance (ODV) PPO with MC-Dropout: epistemic uncertainty = $\text{Var}_t[\mu_t(s)]$ across T dropout masks, aleatoric = $\frac{1}{T}\sum_t \sigma^2_t(s)$. This was validated on social robot navigation. **Charpentier et al. (2022)** compared multiple UQ methods for model-free RL and confirmed that MC-Dropout on policy networks provides useful epistemic signals for OOD environment detection.

**Formula.** For a Gaussian policy $\pi(a|s) = \mathcal{N}(\mu(s), \sigma^2(s))$ with $d_a$ action dimensions:

$$\varepsilon_{\pi}(s) = \frac{d_a}{2}\log(2\pi e) + \frac{1}{2}\sum_{i=1}^{d_a}\log\sigma_i^2(s)$$

For a more principled epistemic-only estimate using MC-Dropout (T = 10–20 passes with dropout active):

$$\varepsilon_{\pi\text{-epi}}(s) = \frac{1}{T}\sum_{t=1}^T \|\mu_t(s) - \bar{\mu}(s)\|^2$$

**Computational cost.** Raw entropy: **~0.01 ms** (already computed during inference). MC-Dropout with T = 20 on the 256→128→64 PPO MLP with 36-D input: **~1–2 ms.** Both feasible.

**Important caveat.** Policy entropy alone **cannot reliably distinguish epistemic from aleatoric uncertainty** — a well-trained policy may have high entropy in genuinely multimodal regions (aleatoric) or dangerously low entropy on OOD states (overconfident extrapolation). This is why it should be combined with, not substitute for, model-free novelty detectors (k-NN, RND). Its unique value is capturing a complementary signal: the policy's **own confidence**, which may detect situations where states look normal but the task requires unfamiliar behaviors.

---

## Method 5: Normalizing flow density estimation provides calibrated distributional support

**What it measures.** The exact log-likelihood of a state under a learned density model trained on calibration data. Low likelihood = the state lies in a region with sparse calibration support. This answers: *"How probable is this state under the calibration distribution?"*

**Literature grounding.** Normalizing flows provide **exact log-likelihood computation** via the change-of-variables formula, making them the most principled density estimators for continuous data. **Kirichenko et al. (NeurIPS 2020)** showed that flows fail for OOD detection on high-dimensional images due to learning local pixel correlations, but this failure mode is **largely mitigated in low-dimensional (36-D) state spaces** where flows can learn meaningful global density structure. Critically, **FAIL-Detect (Xu et al., RSS 2025)** demonstrated that normalizing-flow log-likelihood on observation space was the **best-performing OOD score** for failure detection in robotic manipulation, directly validating this approach for our setting. **Feng et al. (2023)** also validated normalizing flows for feasibility learning in robotic assembly.

**Formula.** Learn an invertible mapping $f: \mathbb{R}^{36} \to \mathbb{R}^{36}$ such that $f(s) \sim \mathcal{N}(0, I)$:

$$\varepsilon_{\text{nf}}(s) = -\log p(s) = -\log p_z(f(s)) - \log|\det(\partial f / \partial s)|$$

Use a **Masked Autoregressive Flow (MAF)** or **Real NVP** with 3–4 coupling layers, each containing small MLPs (2 hidden layers of 128 units).

**Computational cost.** Forward pass through a small MAF/RealNVP for 36-D input: **<0.5 ms** on GPU, **<1 ms** on CPU. Requires offline training (~minutes to hours on calibration data).

**Why it adds value beyond k-NN.** k-NN captures local distance to nearest neighbors but cannot estimate absolute density — a state equidistant from two calibration points could be in a high-density region (between clusters) or a low-density region (far from the manifold). Normalizing flows provide a calibrated **probability density** that captures the full shape of the calibration distribution, including its tails. The log-likelihood score is also more naturally interpretable and combinable with other probabilistic signals.

**Alternative: SNGP.** If modifying the PPO architecture is acceptable, **Spectral-Normalized Neural Gaussian Process** (Liu et al., NeurIPS 2020/JMLR 2023) offers an elegant single-pass alternative. It applies spectral normalization to hidden layers (enforcing bi-Lipschitz distance preservation) and replaces the output layer with a random-feature GP approximation. This produces **distance-aware uncertainty** in a single forward pass at ~0.3 ms, with uncertainty that provably increases as inputs move away from training data. However, it requires modifying the network architecture, whereas normalizing flows operate as a separate module.

---

## How the five methods combine into a unified epistemic score

The five components capture genuinely distinct aspects of epistemic uncertainty and can be combined via a weighted sum or learned aggregation:

| Component | Signal type | What it detects | Cost | Key citation |
|-----------|------------|-----------------|------|--------------|
| **ε_knn** | Geometric state novelty | States far from calibration manifold | ~0.3 ms | Sun et al., ICML 2022 |
| **ε_ensemble** | Dynamics model ignorance | Unpredictable transitions | ~1.5 ms | Chua et al., NeurIPS 2018 |
| **ε_rnd** | Learned feature novelty | Distributional shift in feature space | ~0.3 ms | Burda et al., ICLR 2019 |
| **ε_π** | Behavioral confidence | Policy uncertainty about action | ~0.01 ms | Flögel et al., IROS 2025 |
| **ε_nf** | Calibrated density support | Low-probability states | ~0.5 ms | Xu et al., RSS 2025 |
| **Total** | | | **~2.6 ms** | |

The combined epistemic uncertainty score:

$$\varepsilon_{\text{epistemic}} = w_1 \cdot \tilde{\varepsilon}_{\text{knn}} + w_2 \cdot \tilde{\varepsilon}_{\text{ensemble}} + w_3 \cdot \tilde{\varepsilon}_{\text{rnd}} + w_4 \cdot \tilde{\varepsilon}_{\pi} + w_5 \cdot \tilde{\varepsilon}_{\text{nf}}$$

where each $\tilde{\varepsilon}$ is min-max normalized to [0, 1] over a validation set. Weights can be set uniformly or tuned on held-out perturbation scenarios. **Total inference: ~2.6 ms, well within the 5 ms budget.**

If only 4 components are desired, dropping **ε_nf** (the normalizing flow) is most defensible since k-NN already captures geometric novelty — the normalizing flow adds calibrated density at the cost of training complexity. Alternatively, dropping **ε_π** (policy entropy) is reasonable if the PPO policy was not trained with dropout, since raw entropy conflates aleatoric and epistemic signals.

---

## Key design decisions and reviewer-facing justifications

**Why not just Mahalanobis on transitions?** The originally proposed ε_dyn (Mahalanobis on state deltas) fits a single global Gaussian to all calibration transitions. This ignores the critical dependence of expected transitions on the current state and action — free-space motion produces very different deltas than contact phases. Ensemble disagreement resolves this by conditioning on $(s_t, a_t)$ and learning state-dependent dynamics, providing a strictly more informative signal that cleanly separates epistemic from aleatoric uncertainty via the law of total variance.

**Why ensembles rather than MC-Dropout or Bayesian NNs?** Multiple comparative studies (Fort et al., NeurIPS 2019; Ovadia et al., NeurIPS 2019) show deep ensembles **consistently outperform** MC-Dropout and variational Bayesian methods for uncertainty quality. B = 5 members is sufficient (confirmed by PETS, DUDES, and PaiDEs). For a 256→128→64 MLP, five parallel forward passes are cheaper than 20 MC-Dropout passes.

**Why both k-NN and RND for state novelty?** k-NN operates on raw Euclidean geometry; RND operates in a learned feature space. They fail differently: k-NN struggles when the effective state manifold is curved or when irrelevant dimensions dominate; RND may miss novelty in dimensions the random target network happens to ignore. Their combination is strictly more robust. In ablation, either alone would be defensible — the paper could present both and show the marginal contribution of each.

**Scaling to 36-D.** At 36 dimensions, k-NN distance, normalizing flows, and ensemble MLPs all work well. The main method to **avoid** at d = 36 is kernel density estimation (KDE), which suffers catastrophically from the curse of dimensionality (bandwidth barely shrinks with data per Scott's rule: $h \propto n^{-1/40}$). Gaussian Process dynamics models (PILCO-style) also scale poorly to 36-D inputs — Chua et al. explicitly found GPs "infeasible" for higher-dimensional tasks.

**Training all components.** All five methods train offline from the same calibration dataset: k-NN requires only index construction (FAISS, seconds); the dynamics ensemble trains on $(s_t, a_t, s_{t+1})$ tuples (~minutes); RND trains the predictor on calibration states (~minutes); policy entropy requires no additional training; the normalizing flow trains on calibration states (~minutes to hours). No online adaptation is needed during deployment.