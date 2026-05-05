"""
Financial module: applies insurance contract terms to convert ground-up
losses into insured losses.

For each event, ground-up loss is allocated across policies in proportion
to exposure share, then per-policy deductibles and limits are applied:

    insured_loss_i = min(max(allocated_loss_i - deductible_i, 0), limit_i)

The portfolio insured loss is the sum across policies.
"""

from __future__ import annotations

import numpy as np

from .exposure import Portfolio


def apply_contract_terms(
    ground_up_loss: float,
    portfolio: Portfolio,
) -> float:
    """Apply per-policy deductibles and limits to a single event's ground-up loss.

    Args:
        ground_up_loss: total economic loss from the event before insurance
        portfolio: the insured portfolio

    Returns:
        Total insured loss across the portfolio for this event.
    """
    if ground_up_loss <= 0:
        return 0.0

    shares = portfolio.exposure_shares()
    deductibles = portfolio.deductibles()
    limits = portfolio.limits()

    allocated = ground_up_loss * shares
    after_ded = np.maximum(allocated - deductibles, 0.0)
    insured = np.minimum(after_ded, limits)
    return float(insured.sum())


def apply_contract_terms_vectorized(
    ground_up_losses: np.ndarray,
    portfolio: Portfolio,
) -> np.ndarray:
    """Vectorized version: apply contract terms to many events at once.

    Args:
        ground_up_losses: array of shape (n_events,) of ground-up losses
        portfolio: the insured portfolio

    Returns:
        Array of shape (n_events,) of portfolio insured losses.
    """
    if len(ground_up_losses) == 0:
        return np.array([], dtype=float)

    shares = portfolio.exposure_shares()  # (n_pol,)
    deductibles = portfolio.deductibles()  # (n_pol,)
    limits = portfolio.limits()  # (n_pol,)

    # Broadcast: (n_events, 1) * (n_pol,) -> (n_events, n_pol)
    allocated = ground_up_losses[:, np.newaxis] * shares[np.newaxis, :]
    after_ded = np.maximum(allocated - deductibles[np.newaxis, :], 0.0)
    insured = np.minimum(after_ded, limits[np.newaxis, :])
    return insured.sum(axis=1)
