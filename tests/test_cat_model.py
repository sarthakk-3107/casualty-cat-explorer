"""Tests for the casualty cat model core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow tests to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cat_model import (
    FrequencyParams,
    Policy,
    Portfolio,
    SeverityParams,
    SimulationConfig,
    apply_contract_terms,
    apply_contract_terms_vectorized,
    bootstrap_metric_ci,
    build_sample_portfolio,
    one_at_a_time,
    run_simulation,
    sample_frequency,
    sample_severity,
    severity_quantile,
    tornado,
)


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------


def test_frequency_poisson_when_dispersion_one():
    rng = np.random.default_rng(0)
    params = FrequencyParams(mean=2.0, dispersion=1.0)
    samples = sample_frequency(params, n_years=50_000, rng=rng)
    assert samples.shape == (50_000,)
    # Sample mean should be close to true mean
    assert abs(samples.mean() - 2.0) < 0.05
    # Variance ~ mean for Poisson
    assert abs(samples.var() - 2.0) < 0.1


def test_frequency_overdispersed_has_more_variance():
    rng = np.random.default_rng(0)
    base = FrequencyParams(mean=2.0, dispersion=1.0)
    over = FrequencyParams(mean=2.0, dispersion=3.0)
    base_samples = sample_frequency(base, n_years=50_000, rng=rng)
    over_samples = sample_frequency(over, n_years=50_000, rng=rng)
    # Same mean, higher variance
    assert abs(base_samples.mean() - over_samples.mean()) < 0.1
    assert over_samples.var() > base_samples.var() * 2


def test_frequency_invalid_params_raise():
    with pytest.raises(ValueError):
        FrequencyParams(mean=-1.0, dispersion=1.0)
    with pytest.raises(ValueError):
        FrequencyParams(mean=1.0, dispersion=0.5)


def test_severity_from_mean_cv_recovers_mean():
    sev = SeverityParams.from_mean_cv(mean=10_000_000, cv=2.0)
    rng = np.random.default_rng(0)
    samples = sample_severity(sev, n_events=200_000, rng=rng)
    # Lognormal sample mean converges slowly; allow 5% tolerance
    assert abs(samples.mean() / 10_000_000 - 1) < 0.05
    assert abs(sev.mean - 10_000_000) < 1


def test_severity_quantile_monotone():
    sev = SeverityParams.from_mean_cv(mean=1e6, cv=2.0)
    q50 = severity_quantile(sev, 0.5)
    q90 = severity_quantile(sev, 0.9)
    q99 = severity_quantile(sev, 0.99)
    assert q50 < q90 < q99


def test_sample_severity_empty():
    rng = np.random.default_rng(0)
    sev = SeverityParams.from_mean_cv(mean=1e6, cv=2.0)
    samples = sample_severity(sev, n_events=0, rng=rng)
    assert samples.shape == (0,)


# ---------------------------------------------------------------------------
# Financial module tests
# ---------------------------------------------------------------------------


def _two_policy_portfolio() -> Portfolio:
    return Portfolio(
        policies=[
            Policy("A", "chemicals", 0.6, deductible=1_000_000, limit=10_000_000),
            Policy("B", "pharma", 0.4, deductible=500_000, limit=5_000_000),
        ]
    )


def test_apply_contract_terms_below_deductibles():
    portfolio = _two_policy_portfolio()
    # Allocate $1M total: A gets $600k (below $1M ded), B gets $400k (below $500k ded)
    insured = apply_contract_terms(1_000_000, portfolio)
    assert insured == 0.0


def test_apply_contract_terms_above_deductibles_below_limits():
    portfolio = _two_policy_portfolio()
    # Allocate $5M total: A gets $3M (-$1M ded = $2M), B gets $2M (-$500k ded = $1.5M)
    insured = apply_contract_terms(5_000_000, portfolio)
    assert insured == pytest.approx(2_000_000 + 1_500_000)


def test_apply_contract_terms_capped_by_limits():
    portfolio = _two_policy_portfolio()
    # Allocate $100M total: A would get $60M but capped at $10M; B would get $40M
    # but capped at $5M
    insured = apply_contract_terms(100_000_000, portfolio)
    assert insured == pytest.approx(10_000_000 + 5_000_000)


def test_vectorized_matches_scalar():
    portfolio = _two_policy_portfolio()
    losses = np.array([0, 1_000_000, 5_000_000, 100_000_000, 500_000])
    vec = apply_contract_terms_vectorized(losses, portfolio)
    scalar = np.array([apply_contract_terms(float(l), portfolio) for l in losses])
    np.testing.assert_allclose(vec, scalar)


def test_apply_contract_terms_zero_loss():
    portfolio = _two_policy_portfolio()
    assert apply_contract_terms(0.0, portfolio) == 0.0


# ---------------------------------------------------------------------------
# Simulation end-to-end tests
# ---------------------------------------------------------------------------


def _baseline_config(n_years: int = 20_000) -> SimulationConfig:
    return SimulationConfig(
        frequency=FrequencyParams(mean=0.15, dispersion=2.0),
        severity=SeverityParams.from_mean_cv(mean=50_000_000, cv=2.5),
        portfolio=build_sample_portfolio(n_policies=50, total_limit=500_000_000),
        n_years=n_years,
        seed=42,
    )


def test_simulation_outputs_have_expected_shape():
    config = _baseline_config()
    result = run_simulation(config)
    assert result.annual_losses.shape == (config.n_years,)
    assert result.event_counts.shape == (config.n_years,)
    assert (result.annual_losses >= 0).all()
    assert (result.event_counts >= 0).all()


def test_simulation_reproducible_with_same_seed():
    cfg = _baseline_config()
    r1 = run_simulation(cfg)
    r2 = run_simulation(cfg)
    np.testing.assert_array_equal(r1.annual_losses, r2.annual_losses)


def test_pml_monotone_in_return_period():
    result = run_simulation(_baseline_config())
    rps = [10, 50, 100, 250, 500]
    pmls = [result.pml(rp) for rp in rps]
    for a, b in zip(pmls, pmls[1:]):
        assert b >= a, f"PML not monotone: {pmls}"


def test_tvar_at_least_pml():
    result = run_simulation(_baseline_config())
    for rp in [50, 100, 250]:
        assert result.tvar(rp) >= result.pml(rp)


def test_aal_within_standard_error_tolerance():
    """AAL should be within a few standard errors of the analytical expectation.

    With independent ground-up severities allocated proportionally and capped,
    the exact AAL depends on contract terms. We instead check that two runs
    with disjoint seeds agree within their combined standard errors.
    """
    cfg1 = _baseline_config()
    cfg2 = SimulationConfig(
        frequency=cfg1.frequency,
        severity=cfg1.severity,
        portfolio=cfg1.portfolio,
        n_years=cfg1.n_years,
        seed=123,
    )
    r1 = run_simulation(cfg1)
    r2 = run_simulation(cfg2)
    combined_se = np.sqrt(r1.aal_std_error**2 + r2.aal_std_error**2)
    assert abs(r1.aal - r2.aal) < 4 * combined_se


def test_higher_frequency_gives_higher_aal():
    base = _baseline_config()
    high_freq = SimulationConfig(
        frequency=FrequencyParams(mean=base.frequency.mean * 2, dispersion=base.frequency.dispersion),
        severity=base.severity,
        portfolio=base.portfolio,
        n_years=base.n_years,
        seed=base.seed,
    )
    base_result = run_simulation(base)
    high_result = run_simulation(high_freq)
    assert high_result.aal > base_result.aal


def test_bootstrap_ci_brackets_point_estimate():
    result = run_simulation(_baseline_config())
    point, lo, hi = bootstrap_metric_ci(
        result.annual_losses,
        metric_fn=lambda x: float(np.quantile(x, 0.99)),
        n_bootstrap=200,
    )
    assert lo <= point <= hi
    assert hi > lo


# ---------------------------------------------------------------------------
# Sensitivity tests
# ---------------------------------------------------------------------------


def test_oat_sensitivity_severity_mean_increasing():
    cfg = _baseline_config(n_years=10_000)
    points = one_at_a_time(
        cfg, "severity_mean", multipliers=np.array([0.5, 1.0, 2.0])
    )
    assert len(points) == 3
    aals = [p.aal for p in points]
    # Higher severity -> higher AAL (allow slight Monte Carlo noise)
    assert aals[0] < aals[1] < aals[2]


def test_tornado_returns_all_parameters():
    cfg = _baseline_config(n_years=10_000)
    rows = tornado(cfg, shock=0.25, metric="pml_250yr")
    params = {r["parameter"] for r in rows}
    assert params == {
        "frequency_mean",
        "frequency_dispersion",
        "severity_mean",
        "severity_cv",
    }
    # Sorted by impact descending
    for a, b in zip(rows, rows[1:]):
        assert a["impact_range"] >= b["impact_range"]


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------


def test_sample_portfolio_constructs_valid():
    pf = build_sample_portfolio(n_policies=20, total_limit=100_000_000)
    assert pf.n_policies == 20
    assert sum(p.exposure_weight for p in pf.policies) > 0
    for p in pf.policies:
        assert p.deductible >= 0
        assert p.limit > 0


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        Policy("X", "chem", -0.1, deductible=0, limit=1_000_000)
    with pytest.raises(ValueError):
        Policy("X", "chem", 0.5, deductible=-1, limit=1_000_000)
    with pytest.raises(ValueError):
        Policy("X", "chem", 0.5, deductible=0, limit=0)
