"""
Frequency and severity distributions for casualty catastrophe modeling.

In casualty cat modeling, we model two things separately:
  - Event frequency: how many catastrophic events occur per year
  - Event severity: given an event occurs, how large is the loss

This separation is standard practice (the "frequency-severity" or "collective risk"
model) and lets us reason about each component independently.

We use:
  - Negative Binomial for frequency (allows for overdispersion vs Poisson, which
    is important for casualty where event clustering is common)
  - Lognormal for severity (heavy-tailed, standard for liability losses)

We also support parameter uncertainty: the distribution parameters themselves
are drawn from a prior distribution. This is critical for casualty cat where
parameters are deeply uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class FrequencyParams:
    """Parameters for a Negative Binomial frequency distribution.

    Parameterized by mean and dispersion (variance / mean ratio).
    Dispersion = 1 reduces to Poisson; dispersion > 1 indicates clustering.
    """

    mean: float
    dispersion: float  # variance / mean; must be >= 1

    def __post_init__(self) -> None:
        if self.mean <= 0:
            raise ValueError(f"mean must be positive, got {self.mean}")
        if self.dispersion < 1:
            raise ValueError(
                f"dispersion must be >= 1 (Poisson lower bound), got {self.dispersion}"
            )


@dataclass(frozen=True)
class SeverityParams:
    """Parameters for a Lognormal severity distribution.

    Parameterized by mu and sigma of the underlying normal distribution.
    Mean of the lognormal = exp(mu + sigma^2 / 2).
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    @classmethod
    def from_mean_cv(cls, mean: float, cv: float) -> "SeverityParams":
        """Construct from a target mean and coefficient of variation.

        This is more intuitive for domain experts than mu/sigma directly.
        """
        if mean <= 0:
            raise ValueError(f"mean must be positive, got {mean}")
        if cv <= 0:
            raise ValueError(f"cv must be positive, got {cv}")
        sigma_sq = np.log(1 + cv**2)
        sigma = float(np.sqrt(sigma_sq))
        mu = float(np.log(mean) - sigma_sq / 2)
        return cls(mu=mu, sigma=sigma)

    @property
    def mean(self) -> float:
        return float(np.exp(self.mu + self.sigma**2 / 2))


def sample_frequency(
    params: FrequencyParams,
    n_years: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample event counts for n_years simulated years.

    Returns array of shape (n_years,) with non-negative integer counts.
    """
    if params.dispersion == 1.0:
        return rng.poisson(lam=params.mean, size=n_years)

    # Negative binomial parameterized by (n, p) where mean = n(1-p)/p
    # and variance = n(1-p)/p^2 = mean / p.
    # So dispersion = variance/mean = 1/p, giving p = 1/dispersion.
    p = 1.0 / params.dispersion
    n = params.mean * p / (1 - p)
    return rng.negative_binomial(n=n, p=p, size=n_years)


def sample_severity(
    params: SeverityParams,
    n_events: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample loss amounts for n_events occurred events.

    Returns array of shape (n_events,) with positive loss amounts.
    """
    if n_events == 0:
        return np.array([], dtype=float)
    return rng.lognormal(mean=params.mu, sigma=params.sigma, size=n_events)


def severity_quantile(params: SeverityParams, q: float) -> float:
    """Return the q-th quantile of the severity distribution.

    Useful for sanity checks and for computing analytical reference values.
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return float(stats.lognorm.ppf(q, s=params.sigma, scale=np.exp(params.mu)))
