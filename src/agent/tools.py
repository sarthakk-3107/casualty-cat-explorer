"""
Tool registry for the cat model agent.

Each tool wraps a piece of cat model functionality with:
  - A JSON Schema description for OpenAI function calling
  - A Python implementation that takes parsed args and returns JSON-serializable results

This pattern (schema-defined tool registry) mirrors the Model Context Protocol (MCP)
approach: tools are first-class registered entities with declarative schemas, making
them composable, inspectable, and easy to add to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from cat_model import (
    FrequencyParams,
    SeverityParams,
    SimulationConfig,
    bootstrap_metric_ci,
    build_sample_portfolio,
    one_at_a_time,
    run_simulation,
    tornado,
)


@dataclass
class Tool:
    """A registered tool callable by the agent."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    implementation: Callable[..., dict[str, Any]]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# ---------------------------------------------------------------------------
# Shared scenario state (kept on the registry so all tools see the same model)
# ---------------------------------------------------------------------------


class ModelContext:
    """Holds the current model configuration and last simulation result.

    The agent operates on this shared context; tools read from and write to it.
    This makes follow-up questions efficient (no re-simulation when not needed)
    and keeps the agent grounded in a consistent scenario.
    """

    def __init__(self, default_n_years: int = 30_000) -> None:
        # Default scenario: PFAS-style emerging mass tort
        self.frequency = FrequencyParams(mean=0.15, dispersion=2.0)
        self.severity = SeverityParams.from_mean_cv(mean=50_000_000, cv=2.5)
        self.portfolio = build_sample_portfolio(
            n_policies=50, total_limit=500_000_000, seed=42
        )
        self.n_years = default_n_years
        self._cached_result = None
        self._cache_key: tuple | None = None

    def _key(self) -> tuple:
        return (
            self.frequency.mean,
            self.frequency.dispersion,
            self.severity.mu,
            self.severity.sigma,
            self.n_years,
            self.portfolio.n_policies,
        )

    def get_result(self):
        """Return cached simulation result, re-running if config changed."""
        key = self._key()
        if self._cache_key != key or self._cached_result is None:
            cfg = SimulationConfig(
                frequency=self.frequency,
                severity=self.severity,
                portfolio=self.portfolio,
                n_years=self.n_years,
                seed=42,
            )
            self._cached_result = run_simulation(cfg)
            self._cache_key = key
        return self._cached_result


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _describe_scenario(ctx: ModelContext) -> dict[str, Any]:
    """Return a summary of the current scenario configuration."""
    return {
        "frequency": {
            "mean_events_per_year": ctx.frequency.mean,
            "dispersion": ctx.frequency.dispersion,
            "interpretation": (
                f"Roughly 1 event every {1/ctx.frequency.mean:.1f} years on average; "
                f"{'overdispersed (events cluster)' if ctx.frequency.dispersion > 1.05 else 'Poisson-like'}"
            ),
        },
        "severity": {
            "distribution": "lognormal",
            "mean_loss": ctx.severity.mean,
            "median_loss": float(np.exp(ctx.severity.mu)),
            "sigma": ctx.severity.sigma,
        },
        "portfolio": {
            "n_policies": ctx.portfolio.n_policies,
            "total_limit": ctx.portfolio.total_limit,
        },
        "simulation": {"n_years": ctx.n_years},
    }


def _key_metrics(ctx: ModelContext) -> dict[str, Any]:
    """Compute and return key cat model output metrics."""
    result = ctx.get_result()
    return {
        "aal": result.aal,
        "aal_standard_error": result.aal_std_error,
        "ep_curve": [
            {"return_period_years": int(rp), "loss": float(loss)}
            for rp, loss in zip(*result.ep_curve())
        ],
        "tvar_250yr": result.tvar(250),
        "tvar_500yr": result.tvar(500),
        "n_years_simulated": result.n_years,
        "fraction_loss_free_years": float((result.annual_losses == 0).mean()),
    }


def _pml_with_ci(ctx: ModelContext, return_period: float) -> dict[str, Any]:
    """Compute a PML at a given return period with a bootstrap CI."""
    result = ctx.get_result()
    if return_period <= 1:
        return {"error": "return_period must be > 1"}

    q = 1 - 1.0 / return_period
    point, lo, hi = bootstrap_metric_ci(
        result.annual_losses,
        metric_fn=lambda x: float(np.quantile(x, q)),
        n_bootstrap=500,
    )
    return {
        "return_period_years": return_period,
        "pml": point,
        "ci_lower_95": lo,
        "ci_upper_95": hi,
        "ci_width_pct_of_point": float((hi - lo) / point * 100) if point > 0 else None,
        "interpretation": (
            f"With 95% confidence, the {return_period:g}-year PML is between "
            f"${lo:,.0f} and ${hi:,.0f}. Width relative to point estimate "
            f"indicates simulation precision in the tail."
        ),
    }


def _run_sensitivity(
    ctx: ModelContext,
    parameter: str,
    multipliers: list[float] | None = None,
) -> dict[str, Any]:
    """Run one-at-a-time sensitivity analysis for a single parameter."""
    cfg = SimulationConfig(
        frequency=ctx.frequency,
        severity=ctx.severity,
        portfolio=ctx.portfolio,
        # Use fewer years for sensitivity to stay fast; tail metrics
        # will be slightly noisier but the relative pattern is robust.
        n_years=max(10_000, ctx.n_years // 3),
        seed=42,
    )
    mult_array = (
        np.array(multipliers, dtype=float) if multipliers else None
    )
    points = one_at_a_time(cfg, parameter, mult_array)
    return {
        "parameter": parameter,
        "results": [
            {
                "multiplier": p.multiplier,
                "aal": p.aal,
                "pml_100yr": p.pml_100yr,
                "pml_250yr": p.pml_250yr,
                "tvar_250yr": p.tvar_250yr,
            }
            for p in points
        ],
    }


def _run_tornado(ctx: ModelContext, shock: float = 0.25, metric: str = "pml_250yr") -> dict[str, Any]:
    """Rank parameters by impact on a chosen metric."""
    cfg = SimulationConfig(
        frequency=ctx.frequency,
        severity=ctx.severity,
        portfolio=ctx.portfolio,
        n_years=max(10_000, ctx.n_years // 3),
        seed=42,
    )
    rows = tornado(cfg, shock=shock, metric=metric)
    return {
        "metric": metric,
        "shock_size": shock,
        "ranked_drivers": rows,
        "interpretation": (
            f"Parameters ranked by impact on {metric} given a +/-{shock*100:.0f}% shock. "
            "Top entries are the dominant drivers of tail risk."
        ),
    }


def _update_assumption(
    ctx: ModelContext,
    parameter: str,
    new_value: float,
) -> dict[str, Any]:
    """Update a model parameter to a new absolute value."""
    if parameter == "frequency_mean":
        ctx.frequency = FrequencyParams(
            mean=new_value, dispersion=ctx.frequency.dispersion
        )
    elif parameter == "frequency_dispersion":
        ctx.frequency = FrequencyParams(
            mean=ctx.frequency.mean, dispersion=max(1.0, new_value)
        )
    elif parameter == "severity_mean":
        old_cv = float(np.sqrt(np.exp(ctx.severity.sigma**2) - 1))
        ctx.severity = SeverityParams.from_mean_cv(new_value, old_cv)
    elif parameter == "severity_cv":
        old_mean = ctx.severity.mean
        ctx.severity = SeverityParams.from_mean_cv(old_mean, new_value)
    else:
        return {"error": f"Unknown parameter: {parameter}"}
    return {
        "status": "updated",
        "parameter": parameter,
        "new_value": new_value,
        "scenario": _describe_scenario(ctx),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def build_registry(ctx: ModelContext) -> dict[str, Tool]:
    """Build the tool registry, binding each tool to the shared context."""

    return {
        "describe_scenario": Tool(
            name="describe_scenario",
            description=(
                "Return a summary of the current model scenario: frequency, "
                "severity, portfolio, and simulation configuration."
            ),
            parameters_schema={"type": "object", "properties": {}, "required": []},
            implementation=lambda: _describe_scenario(ctx),
        ),
        "compute_key_metrics": Tool(
            name="compute_key_metrics",
            description=(
                "Compute the standard cat model output metrics: AAL, full EP "
                "curve at standard return periods, and TVaR at 250 and 500 years."
            ),
            parameters_schema={"type": "object", "properties": {}, "required": []},
            implementation=lambda: _key_metrics(ctx),
        ),
        "compute_pml_with_ci": Tool(
            name="compute_pml_with_ci",
            description=(
                "Compute a Probable Maximum Loss at a specific return period, "
                "with a bootstrap 95% confidence interval to quantify simulation "
                "uncertainty. Use this when the user asks about a specific return "
                "period or wants to know how reliable a tail estimate is."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "return_period": {
                        "type": "number",
                        "description": (
                            "Return period in years (e.g. 100, 250, 500, 1000). "
                            "Must be > 1."
                        ),
                    }
                },
                "required": ["return_period"],
            },
            implementation=lambda return_period: _pml_with_ci(ctx, return_period),
        ),
        "run_sensitivity": Tool(
            name="run_sensitivity",
            description=(
                "Run one-at-a-time sensitivity analysis for a single parameter. "
                "Returns AAL and tail metrics at a range of multiplicative shocks."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": [
                            "frequency_mean",
                            "frequency_dispersion",
                            "severity_mean",
                            "severity_cv",
                        ],
                        "description": "Which parameter to vary.",
                    },
                    "multipliers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": (
                            "Optional list of multiplicative shocks to apply. "
                            "Defaults to [0.5, 0.75, 1.0, 1.25, 1.5]."
                        ),
                    },
                },
                "required": ["parameter"],
            },
            implementation=lambda parameter, multipliers=None: _run_sensitivity(
                ctx, parameter, multipliers
            ),
        ),
        "run_tornado": Tool(
            name="run_tornado",
            description=(
                "Run tornado analysis: rank all model parameters by impact on a "
                "chosen output metric given a uniform shock. Use this when the "
                "user asks 'what drives the result' or 'which assumption matters most'."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "shock": {
                        "type": "number",
                        "description": "Relative shock magnitude (0.25 = +/-25%). Default 0.25.",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["aal", "pml_100yr", "pml_250yr", "tvar_250yr"],
                        "description": "Output metric to rank by. Default pml_250yr.",
                    },
                },
                "required": [],
            },
            implementation=lambda shock=0.25, metric="pml_250yr": _run_tornado(
                ctx, shock, metric
            ),
        ),
        "update_assumption": Tool(
            name="update_assumption",
            description=(
                "Update a model assumption to a new absolute value. Subsequent "
                "tool calls will use the updated assumption. Use this for "
                "what-if scenarios where the user wants to change a parameter "
                "and then ask follow-up questions."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": [
                            "frequency_mean",
                            "frequency_dispersion",
                            "severity_mean",
                            "severity_cv",
                        ],
                    },
                    "new_value": {"type": "number"},
                },
                "required": ["parameter", "new_value"],
            },
            implementation=lambda parameter, new_value: _update_assumption(
                ctx, parameter, new_value
            ),
        ),
    }


def execute_tool_call(
    registry: dict[str, Tool], name: str, arguments_json: str
) -> str:
    """Execute a tool call and return a JSON string of the result.

    Args:
        registry: tool registry
        name: tool name
        arguments_json: JSON-encoded arguments string from the model

    Returns:
        JSON string of the tool result, suitable for sending back to the model.
    """
    if name not in registry:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON arguments: {e}"})

    try:
        result = registry[name].implementation(**args)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments: {e}"})
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps({"error": f"Tool execution failed: {e}"})

    return json.dumps(result, default=float)
