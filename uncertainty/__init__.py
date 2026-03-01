# Uncertainty estimation modules for robot manipulation
# IROS 2026: Decomposed Uncertainty-Aware Control

# Core decomposition
from .aleatoric import AleatoricEstimator, MultiSampleVarianceEstimator
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
    # Aleatoric
    "AleatoricEstimator", "MultiSampleVarianceEstimator",
    # Epistemic
    "SpectralEpistemicEstimator", "RepulsiveEpistemicEstimator",
    "CombinedEpistemicEstimator",
    # Intervention
    "InterventionType", "InterventionController",
    "UncertaintyDecomposer", "DecomposedPolicy", "TotalUncertaintyPolicy",
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
