"""Run the agent eval suite against a live OpenAI-backed agent.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m scripts.run_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import CatModelAgent, ModelContext
from agent.eval import default_eval_cases, evaluate_suite


def main() -> int:
    try:
        ctx = ModelContext(default_n_years=10_000)
        agent = CatModelAgent(ctx=ctx, model="gpt-4o-mini")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    cases = default_eval_cases()
    print(f"Running {len(cases)} eval cases...\n")

    summary = evaluate_suite(agent, cases)

    for r in summary["results"]:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.case_id}  ({r.n_tool_calls} tool calls, {r.elapsed_seconds:.1f}s)")
        if not r.passed:
            print(f"         reason: {r.failure_reason}")

    print(f"\n{'=' * 60}")
    print(f"  Pass rate:        {summary['pass_rate']*100:.1f}% ({summary['n_passed']}/{summary['n_cases']})")
    print(f"  Tool match rate:  {summary['tool_match_rate']*100:.1f}%")
    print(f"  Mean tool calls:  {summary['mean_tool_calls']:.1f}")
    print(f"  Mean latency:     {summary['mean_elapsed_seconds']:.2f}s")
    print(f"{'=' * 60}")

    return 0 if summary["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
