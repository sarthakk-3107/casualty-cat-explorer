"""
Exposure module: represents the portfolio of insureds at risk.

For a casualty cat model, exposure is typically a set of policies, each with:
  - A coverage limit (max payout per event or aggregate)
  - A deductible / self-insured retention
  - An exposure base (revenue, headcount, etc.) that scales loss potential
  - A class of business or industry segment

For this simplified model, we represent exposure as a set of policy "cells"
each with a relative exposure weight. Losses are then allocated across cells
in proportion to exposure when an event occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Policy:
    """A single insurance policy in the exposure portfolio."""

    policy_id: str
    industry_segment: str
    exposure_weight: float  # relative share of total exposure
    deductible: float  # per-event retention
    limit: float  # per-event policy limit

    def __post_init__(self) -> None:
        if self.exposure_weight < 0:
            raise ValueError(
                f"exposure_weight must be non-negative, got {self.exposure_weight}"
            )
        if self.deductible < 0:
            raise ValueError(f"deductible must be non-negative, got {self.deductible}")
        if self.limit <= 0:
            raise ValueError(f"limit must be positive, got {self.limit}")


@dataclass
class Portfolio:
    """A collection of policies representing the insurer's book of business."""

    policies: list[Policy]

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("Portfolio must contain at least one policy")
        total_weight = sum(p.exposure_weight for p in self.policies)
        if total_weight <= 0:
            raise ValueError("Total exposure weight must be positive")

    @property
    def total_limit(self) -> float:
        """Aggregate policy limit across the portfolio."""
        return sum(p.limit for p in self.policies)

    @property
    def n_policies(self) -> int:
        return len(self.policies)

    def exposure_shares(self) -> np.ndarray:
        """Return normalized exposure shares as a numpy array (sums to 1)."""
        weights = np.array([p.exposure_weight for p in self.policies], dtype=float)
        return weights / weights.sum()

    def deductibles(self) -> np.ndarray:
        return np.array([p.deductible for p in self.policies], dtype=float)

    def limits(self) -> np.ndarray:
        return np.array([p.limit for p in self.policies], dtype=float)


def build_sample_portfolio(
    n_policies: int = 50,
    total_limit: float = 500_000_000.0,
    seed: int | None = 42,
) -> Portfolio:
    """Build a synthetic portfolio for testing and demos.

    Mimics a casualty book of business with varying policy sizes drawn from
    a Pareto-like distribution (a few large insureds, many small ones).
    """
    rng = np.random.default_rng(seed)
    segments = ["chemicals", "pharmaceuticals", "consumer_products", "industrial"]

    # Pareto-distributed exposure weights (heavy-tailed: a few large insureds)
    weights = rng.pareto(a=2.0, size=n_policies) + 1.0
    weights = weights / weights.sum()

    # Each policy gets a limit roughly proportional to its weight, scaled so
    # total limit matches the target.
    limits = weights * total_limit * n_policies / weights.sum() * (1 / n_policies)
    limits = limits * (total_limit / limits.sum())
    # Round limits to nicer numbers
    limits = np.round(limits / 100_000) * 100_000
    limits = np.maximum(limits, 1_000_000)  # floor at $1M

    # Deductibles ~ 1-5% of limit
    ded_pct = rng.uniform(0.01, 0.05, size=n_policies)
    deductibles = limits * ded_pct
    deductibles = np.round(deductibles / 10_000) * 10_000

    policies = [
        Policy(
            policy_id=f"POL-{i:04d}",
            industry_segment=str(rng.choice(segments)),
            exposure_weight=float(weights[i]),
            deductible=float(deductibles[i]),
            limit=float(limits[i]),
        )
        for i in range(n_policies)
    ]
    return Portfolio(policies=policies)
