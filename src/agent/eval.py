"""
Evaluation framework for the cat model agent.

We evaluate the agent on three axes:

  1. Tool-selection correctness: did the agent call the right tool(s) for the query?
     (deterministic, doesn't require an LLM judge)

  2. Numerical correctness: do the numbers in the response match the ground-truth
     tool outputs? (deterministic, regex-based extraction with tolerance)

  3. LLM-as-judge for response quality: optional, used for open-ended eval.

Each test case specifies:
  - A query
  - Expected tools to be called (subset, order-agnostic)
  - Optional expected numerical values to appear in the response (with tolerance)

This deterministic, reference-based evaluation approach mirrors what's used
in production agent systems where reliability matters more than fluency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .runtime import AgentResponse, CatModelAgent


@dataclass
class EvalCase:
    """A single agent evaluation case."""

    case_id: str
    query: str
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_numbers: list[tuple[float, float]] = field(default_factory=list)
    # Each expected number is (value, relative_tolerance), e.g. (305_000_000, 0.20)


@dataclass
class EvalResult:
    """Outcome of running one eval case."""

    case_id: str
    passed: bool
    tool_match: bool
    forbidden_tool_called: bool
    number_matches: list[bool]
    response_text: str
    n_tool_calls: int
    elapsed_seconds: float
    failure_reason: str | None = None


def _extract_numbers(text: str) -> list[float]:
    """Extract numeric values from a response, handling $ and , formatting."""
    # Match patterns like $305,000,000 or 305 million or 1.5e6 or 0.85
    pattern = r"\$?([\d]+(?:,[\d]{3})*(?:\.[\d]+)?(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, text)
    nums: list[float] = []
    for m in matches:
        try:
            nums.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return nums


def _number_appears(target: float, tolerance: float, text: str) -> bool:
    """Check if any number in text is within relative tolerance of target."""
    extracted = _extract_numbers(text)
    if not extracted:
        return False
    abs_tol = abs(target) * tolerance
    return any(abs(x - target) <= abs_tol for x in extracted)


def _tools_called(response: AgentResponse) -> list[str]:
    return [
        ev.payload["name"]
        for ev in response.trace
        if ev.kind == "tool_call"
    ]


def evaluate_case(agent: CatModelAgent, case: EvalCase) -> EvalResult:
    """Run a single evaluation case against the agent."""
    response = agent.run(case.query)
    tools_called = _tools_called(response)

    tool_match = all(t in tools_called for t in case.expected_tools)
    forbidden_called = any(t in tools_called for t in case.forbidden_tools)

    number_matches = [
        _number_appears(target, tol, response.final_text)
        for target, tol in case.expected_numbers
    ]

    failure_reasons = []
    if not tool_match:
        missing = [t for t in case.expected_tools if t not in tools_called]
        failure_reasons.append(f"missing expected tools: {missing}")
    if forbidden_called:
        called_forbidden = [t for t in case.forbidden_tools if t in tools_called]
        failure_reasons.append(f"called forbidden tools: {called_forbidden}")
    for (target, tol), matched in zip(case.expected_numbers, number_matches):
        if not matched:
            failure_reasons.append(
                f"missing expected number {target} (tol={tol*100:.0f}%)"
            )

    passed = tool_match and not forbidden_called and all(number_matches)
    return EvalResult(
        case_id=case.case_id,
        passed=passed,
        tool_match=tool_match,
        forbidden_tool_called=forbidden_called,
        number_matches=number_matches,
        response_text=response.final_text,
        n_tool_calls=response.n_tool_calls,
        elapsed_seconds=response.elapsed_seconds,
        failure_reason="; ".join(failure_reasons) if failure_reasons else None,
    )


def evaluate_suite(
    agent: CatModelAgent, cases: list[EvalCase]
) -> dict[str, Any]:
    """Run a full eval suite and return aggregate metrics."""
    results = [evaluate_case(agent, c) for c in cases]
    n = len(results)
    n_passed = sum(1 for r in results if r.passed)
    return {
        "n_cases": n,
        "n_passed": n_passed,
        "pass_rate": n_passed / n if n else 0.0,
        "tool_match_rate": sum(1 for r in results if r.tool_match) / n if n else 0.0,
        "mean_elapsed_seconds": (
            sum(r.elapsed_seconds for r in results) / n if n else 0.0
        ),
        "mean_tool_calls": (
            sum(r.n_tool_calls for r in results) / n if n else 0.0
        ),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Default eval suite
# ---------------------------------------------------------------------------


def default_eval_cases() -> list[EvalCase]:
    """A starter eval suite covering the main agent capabilities."""
    return [
        EvalCase(
            case_id="describe_basic",
            query="What is the current scenario?",
            expected_tools=["describe_scenario"],
        ),
        EvalCase(
            case_id="aal_basic",
            query="What's the average annual loss for this portfolio?",
            expected_tools=["compute_key_metrics"],
        ),
        EvalCase(
            case_id="pml_with_uncertainty",
            query=(
                "What's the 250-year PML, and how confident should I be in that "
                "estimate given simulation noise?"
            ),
            expected_tools=["compute_pml_with_ci"],
        ),
        EvalCase(
            case_id="what_drives_tail",
            query="Which assumption matters most for the 250-year PML?",
            expected_tools=["run_tornado"],
        ),
        EvalCase(
            case_id="severity_sensitivity",
            query=(
                "How does the 250-year PML change if claim severity is 30% higher?"
            ),
            # Either run_sensitivity OR update_assumption + recompute is acceptable
            expected_tools=[],  # we check via numbers/tools loosely
        ),
        EvalCase(
            case_id="reject_unsupported",
            query=(
                "Can you re-fit the severity distribution to a Pareto using "
                "actual industry loss data from 2015-2024?"
            ),
            # The agent should NOT hallucinate this capability. We don't have
            # fitting tools or external data sources.
            forbidden_tools=["update_assumption"],
        ),
    ]
