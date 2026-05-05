"""Tests for the agent tool registry — these don't require an OpenAI API key."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.tools import ModelContext, build_registry, execute_tool_call


@pytest.fixture
def ctx() -> ModelContext:
    # Smaller n_years for test speed
    c = ModelContext(default_n_years=5_000)
    return c


@pytest.fixture
def registry(ctx):
    return build_registry(ctx)


def test_registry_exposes_expected_tools(registry):
    expected = {
        "describe_scenario",
        "compute_key_metrics",
        "compute_pml_with_ci",
        "run_sensitivity",
        "run_tornado",
        "update_assumption",
    }
    assert set(registry.keys()) == expected


def test_each_tool_has_valid_openai_schema(registry):
    for name, tool in registry.items():
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == name
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params


def test_describe_scenario_returns_full_summary(registry):
    out = json.loads(execute_tool_call(registry, "describe_scenario", "{}"))
    assert "frequency" in out
    assert "severity" in out
    assert "portfolio" in out
    assert "simulation" in out


def test_key_metrics_returns_aal_and_ep_curve(registry):
    out = json.loads(execute_tool_call(registry, "compute_key_metrics", "{}"))
    assert out["aal"] > 0
    assert out["aal_standard_error"] > 0
    assert isinstance(out["ep_curve"], list)
    assert len(out["ep_curve"]) > 0
    # PMLs should be non-decreasing in return period
    losses = [pt["loss"] for pt in out["ep_curve"]]
    for a, b in zip(losses, losses[1:]):
        assert b >= a


def test_pml_with_ci_returns_bracket(registry):
    out = json.loads(
        execute_tool_call(registry, "compute_pml_with_ci", '{"return_period": 100}')
    )
    assert out["return_period_years"] == 100
    assert out["ci_lower_95"] <= out["pml"] <= out["ci_upper_95"]
    assert out["ci_width_pct_of_point"] > 0


def test_pml_with_ci_invalid_return_period(registry):
    out = json.loads(
        execute_tool_call(registry, "compute_pml_with_ci", '{"return_period": 1}')
    )
    assert "error" in out


def test_run_sensitivity_severity_mean(registry):
    out = json.loads(
        execute_tool_call(
            registry,
            "run_sensitivity",
            '{"parameter": "severity_mean", "multipliers": [0.5, 1.0, 2.0]}',
        )
    )
    assert out["parameter"] == "severity_mean"
    assert len(out["results"]) == 3
    aals = [r["aal"] for r in out["results"]]
    # AAL should be monotone increasing with severity scale
    assert aals[0] < aals[1] < aals[2]


def test_run_tornado_ranks_drivers(registry):
    out = json.loads(
        execute_tool_call(registry, "run_tornado", '{"shock": 0.25}')
    )
    assert out["metric"] == "pml_250yr"
    rows = out["ranked_drivers"]
    assert len(rows) == 4
    impact = [r["impact_range"] for r in rows]
    for a, b in zip(impact, impact[1:]):
        assert a >= b


def test_update_assumption_persists(registry, ctx):
    original_freq = ctx.frequency.mean
    out = json.loads(
        execute_tool_call(
            registry,
            "update_assumption",
            '{"parameter": "frequency_mean", "new_value": 0.5}',
        )
    )
    assert out["status"] == "updated"
    assert ctx.frequency.mean == 0.5
    assert ctx.frequency.mean != original_freq


def test_update_assumption_invalid_parameter(registry):
    out = json.loads(
        execute_tool_call(
            registry,
            "update_assumption",
            '{"parameter": "fake_param", "new_value": 1.0}',
        )
    )
    assert "error" in out


def test_unknown_tool_returns_error(registry):
    out = json.loads(execute_tool_call(registry, "made_up_tool", "{}"))
    assert "error" in out


def test_invalid_json_returns_error(registry):
    out = json.loads(execute_tool_call(registry, "describe_scenario", "{not json"))
    assert "error" in out


def test_caching_avoids_redundant_simulation(ctx, registry):
    """Calling key_metrics twice with no change should re-use the cached result."""
    json.loads(execute_tool_call(registry, "compute_key_metrics", "{}"))
    cached = ctx._cached_result
    json.loads(execute_tool_call(registry, "compute_key_metrics", "{}"))
    # Same instance means we used the cache
    assert ctx._cached_result is cached


def test_cache_invalidates_on_assumption_change(ctx, registry):
    json.loads(execute_tool_call(registry, "compute_key_metrics", "{}"))
    cached = ctx._cached_result
    execute_tool_call(
        registry, "update_assumption", '{"parameter": "frequency_mean", "new_value": 0.5}'
    )
    json.loads(execute_tool_call(registry, "compute_key_metrics", "{}"))
    assert ctx._cached_result is not cached
