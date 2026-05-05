"""
Sensitivity analysis for the casualty cat model.

We support two types of sensitivity analysis:

  1. One-at-a-time (OAT) sensitivity: vary a single parameter while holding
     others fixed. Quick and interpretable; good for executive-level
     "what drives the result" questions.

  2. Tornado analysis: OAT sensitivity for each parameter, ranked by impact.
     The classic sensitivity output insurance executives expect.

For full Sobol indices or PCE-based sensitivity, see future work — those
are appropriate for production cat models but overkill here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .distributions import FrequencyParams, SeverityParams
from .simulation import SimulationConfig, SimulationResult, run_simulation


@dataclass(frozen=True)
class SensitivityPoint:
    """One point on a sensitivity curve."""

    parameter: str
    multiplier: float
    aal: float
    pml_100yr: float
    pml_250yr: float
    tvar_250yr: float


def _config_with_param(
    base: SimulationConfig, parameter: str, multiplier: float
) -> SimulationConfig:
    """Return a new config with the given parameter scaled by multiplier."""
    if parameter == "frequency_mean":
        new_freq = FrequencyParams(
            mean=base.frequency.mean * multiplier,
            dispersion=base.frequency.dispersion,
        )
        return replace(base, frequency=new_freq)
    elif parameter == "frequency_dispersion":
        # Dispersion has a hard floor of 1.0
        new_disp = max(1.0, base.frequency.dispersion * multiplier)
        new_freq = FrequencyParams(mean=base.frequency.mean, dispersion=new_disp)
        return replace(base, frequency=new_freq)
    elif parameter == "severity_mean":
        # Scale the lognormal mean while preserving CV
        old_mean = base.severity.mean
        old_cv = float(np.sqrt(np.exp(base.severity.sigma**2) - 1))
        new_sev = SeverityParams.from_mean_cv(old_mean * multiplier, old_cv)
        return replace(base, severity=new_sev)
    elif parameter == "severity_cv":
        # Scale the CV while preserving the mean
        old_mean = base.severity.mean
        old_cv = float(np.sqrt(np.exp(base.severity.sigma**2) - 1))
        new_sev = SeverityParams.from_mean_cv(old_mean, old_cv * multiplier)
        return replace(base, severity=new_sev)
    else:
        raise ValueError(f"Unknown parameter: {parameter}")


def _summarize_result(
    parameter: str, multiplier: float, result: SimulationResult
) -> SensitivityPoint:
    return SensitivityPoint(
        parameter=parameter,
        multiplier=multiplier,
        aal=result.aal,
        pml_100yr=result.pml(100),
        pml_250yr=result.pml(250),
        tvar_250yr=result.tvar(250),
    )


SUPPORTED_PARAMETERS = (
    "frequency_mean",
    "frequency_dispersion",
    "severity_mean",
    "severity_cv",
)


def one_at_a_time(
    base_config: SimulationConfig,
    parameter: str,
    multipliers: np.ndarray | None = None,
) -> list[SensitivityPoint]:
    """Run OAT sensitivity for a single parameter.

    Args:
        base_config: baseline simulation configuration
        parameter: one of SUPPORTED_PARAMETERS
        multipliers: array of multiplicative shocks. Defaults to
            [0.5, 0.75, 1.0, 1.25, 1.5].

    Returns:
        List of SensitivityPoint, one per multiplier.
    """
    if parameter not in SUPPORTED_PARAMETERS:
        raise ValueError(
            f"parameter must be one of {SUPPORTED_PARAMETERS}, got {parameter}"
        )
    if multipliers is None:
        multipliers = np.array([0.5, 0.75, 1.0, 1.25, 1.5])

    points = []
    for m in multipliers:
        cfg = _config_with_param(base_config, parameter, float(m))
        result = run_simulation(cfg)
        points.append(_summarize_result(parameter, float(m), result))
    return points


def tornado(
    base_config: SimulationConfig,
    shock: float = 0.25,
    metric: str = "pml_250yr",
) -> list[dict]:
    """Run tornado analysis: OAT shock for each parameter, ranked by impact.

    Args:
        base_config: baseline simulation configuration
        shock: relative shock magnitude (e.g. 0.25 means +/- 25%)
        metric: which output metric to rank by; one of
            "aal", "pml_100yr", "pml_250yr", "tvar_250yr"

    Returns:
        List of dicts (one per parameter) sorted by absolute impact descending.
        Each dict has keys: parameter, baseline, low_value, high_value,
        low_pct_change, high_pct_change, impact_range.
    """
    if metric not in ("aal", "pml_100yr", "pml_250yr", "tvar_250yr"):
        raise ValueError(f"Unknown metric: {metric}")

    base_result = run_simulation(base_config)
    base_metric_map = {
        "aal": base_result.aal,
        "pml_100yr": base_result.pml(100),
        "pml_250yr": base_result.pml(250),
        "tvar_250yr": base_result.tvar(250),
    }
    base_value = base_metric_map[metric]

    rows = []
    for param in SUPPORTED_PARAMETERS:
        low_cfg = _config_with_param(base_config, param, 1.0 - shock)
        high_cfg = _config_with_param(base_config, param, 1.0 + shock)
        low_result = run_simulation(low_cfg)
        high_result = run_simulation(high_cfg)
        low_pt = _summarize_result(param, 1.0 - shock, low_result)
        high_pt = _summarize_result(param, 1.0 + shock, high_result)

        low_v = getattr(low_pt, metric)
        high_v = getattr(high_pt, metric)

        rows.append(
            {
                "parameter": param,
                "baseline": float(base_value),
                "low_value": float(low_v),
                "high_value": float(high_v),
                "low_pct_change": float((low_v - base_value) / base_value * 100),
                "high_pct_change": float((high_v - base_value) / base_value * 100),
                "impact_range": float(abs(high_v - low_v)),
            }
        )

    rows.sort(key=lambda r: r["impact_range"], reverse=True)
    return rows
