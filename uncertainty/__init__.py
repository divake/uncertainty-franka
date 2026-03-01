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
    "UncertaintyDecomposer", "DecomposedPolicy",
    # Perturbations
    "PerturbationType", "PerturbationConfig",
    "ObservationPerturbation", "EnvironmentPerturbation",
    "PERTURBATION_PRESETS", "get_perturbation_config",
    # Analysis
    "OrthogonalityAnalyzer",
    "ConformalCalibrator", "AdaptiveConformalInference",
]
