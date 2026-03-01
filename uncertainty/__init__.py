# Uncertainty estimation modules for robot manipulation
# IROS 2026: Decomposed Uncertainty-Aware Control
# CVPR-consistent: arXiv:2511.12389

# Core decomposition — CVPR-consistent naming
# σ_alea = Mahalanobis distance (CVPR Sec 3.2)
from .aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator
# σ_epis = ε_knn + ε_rank (CVPR Sec 3.1 adapted)
from .epistemic_cvpr import (
    KNNEpistemicEstimator,
    RankEpistemicEstimator,
    CVPREpistemicEstimator,
    ZERO_VAR_DIMS_LIFT,
)
# LEGACY epistemic (v2.x — NOT used in current pipeline, kept for reference only)
# See epistemic.py docstring for why these were replaced
from .epistemic import (
    SpectralEpistemicEstimator,
    RepulsiveEpistemicEstimator,
    CombinedEpistemicEstimator,
)

# Intervention
from .intervention import (
    InterventionType,
    InterventionController,
    UncertaintyDecomposer,
    DecomposedPolicy,
    TotalUncertaintyPolicy,
    DeepEnsemblePolicy,
    MCDropoutPolicy,
)

# Perturbations
from .perturbations import (
    PerturbationType,
    PerturbationConfig,
    ObservationPerturbation,
    EnvironmentPerturbation,
    PERTURBATION_PRESETS,
    get_perturbation_config,
)

# Task configuration
from .task_config import (
    TaskObsConfig,
    TASK_CONFIGS,
    get_task_config,
)

# Analysis
from .orthogonality import OrthogonalityAnalyzer
from .conformal import (
    ConformalCalibrator,
    AdaptiveConformalInference,
)

__all__ = [
    # Aleatoric (σ_alea = Mahalanobis)
    "AleatoricEstimator", "MultiSampleVarianceEstimator",
    # Epistemic — CVPR-consistent (σ_epis = ε_knn + ε_rank)
    "KNNEpistemicEstimator", "RankEpistemicEstimator",
    "CVPREpistemicEstimator", "ZERO_VAR_DIMS_LIFT",
    # Epistemic — legacy
    "SpectralEpistemicEstimator", "RepulsiveEpistemicEstimator",
    "CombinedEpistemicEstimator",
    # Intervention
    "InterventionType", "InterventionController",
    "UncertaintyDecomposer", "DecomposedPolicy", "TotalUncertaintyPolicy",
    "DeepEnsemblePolicy", "MCDropoutPolicy",
    # Perturbations
    "PerturbationType", "PerturbationConfig",
    "ObservationPerturbation", "EnvironmentPerturbation",
    "PERTURBATION_PRESETS", "get_perturbation_config",
    # Task configuration
    "TaskObsConfig", "TASK_CONFIGS", "get_task_config",
    # Analysis
    "OrthogonalityAnalyzer",
    "ConformalCalibrator", "AdaptiveConformalInference",
]
