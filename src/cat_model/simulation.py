"""
Monte Carlo simulation engine for the casualty catastrophe model.

The simulation generates synthetic years. For each year:
  1. Draw an event count from the frequency distribution
  2. For each event, draw a ground-up severity
  3. Apply contract terms to get insured losses
  4. Aggregate to annual portfolio loss

Standard cat model outputs (AAL, EP curve, PMLs, TVaR) are then computed
from the resulting annual loss distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import (
    FrequencyParams,
    SeverityParams,
    sample_frequency,
    sample_severity,
)
from .exposure import Portfolio
from .financial import apply_contract_terms_vectorized


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a Monte Carlo simulation run."""

    frequency: FrequencyParams
    severity: SeverityParams
    portfolio: Portfolio
    n_years: int = 100_000
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.n_years < 1000:
            raise ValueError(
                f"n_years should be >= 1000 for stable tail estimates, got {self.n_years}"
            )


@dataclass
class SimulationResult:
    """Results of a Monte Carlo simulation run.

    Attributes:
        annual_losses: array of shape (n_years,) of total insured losses per year
        event_counts: array of shape (n_years,) of event counts per year
        config: the configuration used to generate these results
    """

    annual_losses: np.ndarray
    event_counts: np.ndarray
    config: SimulationConfig

    @property
    def n_years(self) -> int:
        return len(self.annual_losses)

    @property
    def aal(self) -> float:
        """Average Annual Loss — expected annual insured loss."""
        return float(self.annual_losses.mean())

    @property
    def aal_std_error(self) -> float:
        """Standard error of the AAL estimate (Monte Carlo error)."""
        return float(self.annual_losses.std(ddof=1) / np.sqrt(self.n_years))

    def pml(self, return_period: float) -> float:
        """Probable Maximum Loss at a given return period (in years).

        A 100-year PML is the loss exceeded with 1% probability in any given year.
        Equivalently, the 99th percentile of the annual loss distribution.
        """
        if return_period <= 1:
            raise ValueError(f"return_period must be > 1, got {return_period}")
        q = 1.0 - 1.0 / return_period
        return float(np.quantile(self.annual_losses, q))

    def tvar(self, return_period: float) -> float:
        """Tail Value at Risk: expected loss conditional on exceeding the PML.

        TVaR is more robust than PML for capital adequacy because it captures
        the average severity of tail losses, not just the threshold.
        """
        threshold = self.pml(return_period)
        tail = self.annual_losses[self.annual_losses >= threshold]
        if len(tail) == 0:
            return threshold
        return float(tail.mean())

    def ep_curve(
        self,
        return_periods: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Exceedance Probability curve.

        Args:
            return_periods: array of return periods to evaluate. Defaults to
                a standard set: 2, 5, 10, 25, 50, 100, 250, 500, 1000.

        Returns:
            (return_periods, losses) where losses[i] is the PML at return_periods[i].
        """
        if return_periods is None:
            return_periods = np.array(
                [2, 5, 10, 25, 50, 100, 250, 500, 1000], dtype=float
            )
        losses = np.array([self.pml(rp) for rp in return_periods])
        return return_periods, losses


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run a Monte Carlo simulation of annual portfolio losses.

    The simulation is fully vectorized over events for efficiency, but
    iterates over years to handle variable event counts.
    """
    rng = np.random.default_rng(config.seed)

    # Draw event counts for all years at once
    event_counts = sample_frequency(config.frequency, config.n_years, rng)
    total_events = int(event_counts.sum())

    # Draw severities for all events across all years at once, then
    # apply contract terms in one vectorized call
    if total_events == 0:
        annual_losses = np.zeros(config.n_years, dtype=float)
    else:
        ground_up = sample_severity(config.severity, total_events, rng)
        insured_per_event = apply_contract_terms_vectorized(ground_up, config.portfolio)

        # Aggregate event losses back to annual buckets using cumulative counts
        annual_losses = np.zeros(config.n_years, dtype=float)
        event_idx = 0
        for year_idx, count in enumerate(event_counts):
            if count > 0:
                annual_losses[year_idx] = insured_per_event[
                    event_idx : event_idx + count
                ].sum()
                event_idx += int(count)

    return SimulationResult(
        annual_losses=annual_losses,
        event_counts=event_counts,
        config=config,
    )


def bootstrap_metric_ci(
    annual_losses: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for a metric of the loss dist.

    Useful for quantifying Monte Carlo uncertainty in tail metrics like PMLs,
    where the standard error is not easily computed analytically.

    Args:
        annual_losses: simulated annual losses
        metric_fn: function taking annual_losses array and returning a scalar
        n_bootstrap: number of bootstrap resamples
        confidence: confidence level for the interval (e.g. 0.95)
        seed: RNG seed for reproducibility

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(annual_losses)
    point = float(metric_fn(annual_losses))

    boot_estimates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(annual_losses, size=n, replace=True)
        boot_estimates[i] = metric_fn(resample)

    alpha = 1 - confidence
    lo = float(np.quantile(boot_estimates, alpha / 2))
    hi = float(np.quantile(boot_estimates, 1 - alpha / 2))
    return point, lo, hi
