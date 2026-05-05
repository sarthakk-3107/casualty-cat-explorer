"""
Agent runtime: orchestrates the conversation between an OpenAI model and the
cat model tool registry.

Design choices:
  - We use OpenAI function calling directly rather than a heavyweight agent
    framework. This keeps the core simple, transparent, and easy to debug.
  - We cap the tool-use loop at a configurable number of iterations to prevent
    runaway loops or token blowups.
  - We emit structured trace events so callers can inspect the agent's
    reasoning, tool calls, and timing — useful for evaluation and debugging.
  - We support a "dry run" mode that bypasses the LLM and exercises only the
    tools, which is useful for tests and CI.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .tools import ModelContext, Tool, build_registry, execute_tool_call

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert assistant for a casualty catastrophe model.

You help users understand model behavior, sensitivities, and outputs. The model
simulates losses from emerging/systemic liability risks (mass torts, product
defect cascades, etc.) using Monte Carlo simulation with frequency-severity
distributions.

Tools available:
- describe_scenario: see current model configuration
- compute_key_metrics: compute AAL, EP curve, TVaR
- compute_pml_with_ci: a specific PML with bootstrap confidence interval
- run_sensitivity: vary one parameter and see the effect
- run_tornado: rank parameters by impact on a metric
- update_assumption: change a parameter for what-if analysis

Guidelines:
- When asked about results, call the relevant tool — do NOT make up numbers.
- When asked about uncertainty in a tail estimate, use compute_pml_with_ci.
- When asked "what drives the result", use run_tornado.
- For multi-step questions, call multiple tools in sequence as needed.
- Format dollar amounts with thousands separators (e.g. $305,000,000).
- Keep responses concise and quantitative. Lead with the number, then explain.
- If a user asks for something the tools don't support, say so clearly rather
  than guessing.
"""


@dataclass
class TraceEvent:
    """A single event in the agent's execution trace."""

    timestamp: float
    kind: str  # 'llm_call', 'tool_call', 'tool_result', 'final_response'
    payload: dict[str, Any]


@dataclass
class AgentResponse:
    """Result of running the agent on a user query."""

    final_text: str
    trace: list[TraceEvent] = field(default_factory=list)
    n_tool_calls: int = 0
    n_llm_calls: int = 0
    elapsed_seconds: float = 0.0


class CatModelAgent:
    """Agent that answers questions about the cat model using tool calling."""

    def __init__(
        self,
        ctx: ModelContext | None = None,
        client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        max_iterations: int = 6,
        api_key: str | None = None,
    ) -> None:
        self.ctx = ctx or ModelContext()
        self.registry = build_registry(self.ctx)
        self.model = model
        self.max_iterations = max_iterations

        if client is not None:
            self.client = client
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Pass api_key= or set the env var."
                )
            self.client = OpenAI(api_key=key)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self.registry.values()]

    def run(self, user_query: str) -> AgentResponse:
        """Run the agent on a single user query.

        The agent will call tools as needed (up to max_iterations rounds) and
        return a final natural-language response.
        """
        start = time.time()
        trace: list[TraceEvent] = []
        n_tool_calls = 0
        n_llm_calls = 0

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        for iteration in range(self.max_iterations):
            llm_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.openai_tools,
                tool_choice="auto",
                temperature=0.0,
            )
            n_llm_calls += 1
            trace.append(
                TraceEvent(
                    timestamp=time.time(),
                    kind="llm_call",
                    payload={
                        "iteration": iteration,
                        "elapsed": time.time() - llm_start,
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                    },
                )
            )

            choice = response.choices[0]
            msg = choice.message

            if not msg.tool_calls:
                # Agent has produced a final answer
                final_text = msg.content or ""
                trace.append(
                    TraceEvent(
                        timestamp=time.time(),
                        kind="final_response",
                        payload={"text": final_text},
                    )
                )
                return AgentResponse(
                    final_text=final_text,
                    trace=trace,
                    n_tool_calls=n_tool_calls,
                    n_llm_calls=n_llm_calls,
                    elapsed_seconds=time.time() - start,
                )

            # Otherwise, execute the tool calls and append results
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                trace.append(
                    TraceEvent(
                        timestamp=time.time(),
                        kind="tool_call",
                        payload={
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    )
                )
                tool_start = time.time()
                result_json = execute_tool_call(
                    self.registry, tc.function.name, tc.function.arguments
                )
                n_tool_calls += 1
                trace.append(
                    TraceEvent(
                        timestamp=time.time(),
                        kind="tool_result",
                        payload={
                            "name": tc.function.name,
                            "elapsed": time.time() - tool_start,
                            "result_size_bytes": len(result_json),
                        },
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_json,
                    }
                )

        # Hit max iterations without a final answer; return a best-effort message
        return AgentResponse(
            final_text=(
                "[Agent stopped after reaching the maximum tool-use iterations. "
                "The last partial state has been preserved in the trace.]"
            ),
            trace=trace,
            n_tool_calls=n_tool_calls,
            n_llm_calls=n_llm_calls,
            elapsed_seconds=time.time() - start,
        )


def trace_summary(trace: list[TraceEvent]) -> str:
    """Return a human-readable one-line-per-event summary of a trace."""
    lines = []
    for ev in trace:
        if ev.kind == "llm_call":
            lines.append(
                f"[LLM] iter={ev.payload['iteration']} "
                f"in={ev.payload['input_tokens']} out={ev.payload['output_tokens']} "
                f"({ev.payload['elapsed']*1000:.0f}ms)"
            )
        elif ev.kind == "tool_call":
            args_preview = ev.payload["arguments"][:80]
            lines.append(f"[TOOL] {ev.payload['name']}({args_preview})")
        elif ev.kind == "tool_result":
            lines.append(
                f"[RESULT] {ev.payload['name']} "
                f"({ev.payload['elapsed']*1000:.0f}ms, "
                f"{ev.payload['result_size_bytes']}B)"
            )
        elif ev.kind == "final_response":
            preview = ev.payload["text"][:80].replace("\n", " ")
            lines.append(f"[FINAL] {preview}...")
    return "\n".join(lines)
