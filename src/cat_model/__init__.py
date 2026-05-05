"""Casualty catastrophe model — probabilistic loss simulation."""

from .distributions import (
    FrequencyParams,
    SeverityParams,
    sample_frequency,
    sample_severity,
    severity_quantile,
)
from .exposure import Policy, Portfolio, build_sample_portfolio
from .financial import apply_contract_terms, apply_contract_terms_vectorized
from .sensitivity import (
    SUPPORTED_PARAMETERS,
    SensitivityPoint,
    one_at_a_time,
    tornado,
)
from .simulation import (
    SimulationConfig,
    SimulationResult,
    bootstrap_metric_ci,
    run_simulation,
)

__all__ = [
    "FrequencyParams",
    "SeverityParams",
    "sample_frequency",
    "sample_severity",
    "severity_quantile",
    "Policy",
    "Portfolio",
    "build_sample_portfolio",
    "apply_contract_terms",
    "apply_contract_terms_vectorized",
    "SUPPORTED_PARAMETERS",
    "SensitivityPoint",
    "one_at_a_time",
    "tornado",
    "SimulationConfig",
    "SimulationResult",
    "bootstrap_metric_ci",
    "run_simulation",
]
