"""Interactive CLI demo for the casualty cat model agent.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m scripts.demo

Type questions about the model. Type 'trace' to see the trace of the last query,
'reset' to reset the scenario, or 'quit' to exit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import CatModelAgent, ModelContext, trace_summary


BANNER = """
========================================================================
 Casualty Catastrophe Model — Agentic Explorer
========================================================================
Ask questions about the cat model in natural language. Examples:

  * What is the current scenario?
  * What's the 250-year PML and how reliable is that estimate?
  * Which assumption matters most for tail risk?
  * If claim severity is 30% higher, how does the 100-year PML change?
  * Update severity mean to $80 million and recompute key metrics.

Commands:  trace  |  reset  |  quit
========================================================================
"""


def main() -> int:
    ctx = ModelContext(default_n_years=20_000)
    try:
        agent = CatModelAgent(ctx=ctx, model="gpt-4o-mini")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(BANNER)
    last_response = None

    while True:
        try:
            query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            return 0
        if query.lower() == "trace":
            if last_response is None:
                print("(no query yet)")
            else:
                print(trace_summary(last_response.trace))
            continue
        if query.lower() == "reset":
            ctx = ModelContext(default_n_years=20_000)
            agent = CatModelAgent(ctx=ctx, model="gpt-4o-mini")
            print("(scenario reset)")
            continue

        response = agent.run(query)
        last_response = response
        print(f"\nAgent> {response.final_text}")
        print(
            f"\n[{response.n_tool_calls} tool calls, "
            f"{response.n_llm_calls} LLM calls, "
            f"{response.elapsed_seconds:.1f}s]"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
