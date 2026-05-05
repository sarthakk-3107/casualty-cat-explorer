"""Tests for the agent runtime using a mocked OpenAI client.

These tests exercise the full tool-calling loop — multi-turn dispatch, trace
emission, max_iterations guardrail — without requiring a real API key.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import CatModelAgent, ModelContext


# ---------------------------------------------------------------------------
# Mock OpenAI client objects
# ---------------------------------------------------------------------------


@dataclass
class _MockFunction:
    name: str
    arguments: str


@dataclass
class _MockToolCall:
    id: str
    function: _MockFunction
    type: str = "function"


@dataclass
class _MockMessage:
    content: str | None
    tool_calls: list | None = None


@dataclass
class _MockChoice:
    message: _MockMessage


@dataclass
class _MockUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _MockResponse:
    choices: list[_MockChoice]
    usage: _MockUsage


class MockOpenAIClient:
    """Mock OpenAI client that returns a programmed sequence of responses."""

    def __init__(self, responses: list[_MockResponse]) -> None:
        self.responses = list(responses)
        self.call_log: list[dict] = []
        self.chat = self  # so client.chat.completions.create works

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.call_log.append(kwargs)
        if not self.responses:
            raise RuntimeError("MockOpenAIClient ran out of responses")
        return self.responses.pop(0)


def _final(text: str) -> _MockResponse:
    return _MockResponse(
        choices=[_MockChoice(message=_MockMessage(content=text))],
        usage=_MockUsage(prompt_tokens=10, completion_tokens=5),
    )


def _tool_call(name: str, arguments: dict, content: str | None = None) -> _MockResponse:
    return _MockResponse(
        choices=[
            _MockChoice(
                message=_MockMessage(
                    content=content,
                    tool_calls=[
                        _MockToolCall(
                            id=f"call_{name}",
                            function=_MockFunction(
                                name=name, arguments=json.dumps(arguments)
                            ),
                        )
                    ],
                )
            )
        ],
        usage=_MockUsage(prompt_tokens=20, completion_tokens=10),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> ModelContext:
    return ModelContext(default_n_years=3_000)


def test_agent_returns_immediately_on_no_tool_use(ctx):
    client = MockOpenAIClient([_final("Hello, I cannot help without context.")])
    agent = CatModelAgent(ctx=ctx, client=client, model="test-model")
    response = agent.run("hi")
    assert "Hello" in response.final_text
    assert response.n_tool_calls == 0
    assert response.n_llm_calls == 1


def test_agent_dispatches_single_tool_then_finalizes(ctx):
    client = MockOpenAIClient(
        [
            _tool_call("describe_scenario", {}),
            _final("The scenario uses a Negative Binomial frequency distribution."),
        ]
    )
    agent = CatModelAgent(ctx=ctx, client=client, model="test-model")
    response = agent.run("what's the scenario?")
    assert response.n_tool_calls == 1
    assert response.n_llm_calls == 2
    assert "scenario" in response.final_text
    # Verify trace shape: llm -> tool_call -> tool_result -> llm -> final
    kinds = [ev.kind for ev in response.trace]
    assert kinds == ["llm_call", "tool_call", "tool_result", "llm_call", "final_response"]


def test_agent_dispatches_multiple_tools_in_sequence(ctx):
    client = MockOpenAIClient(
        [
            _tool_call("compute_key_metrics", {}),
            _tool_call("compute_pml_with_ci", {"return_period": 250}),
            _final("AAL is X and the 250-year PML is Y with 95% CI."),
        ]
    )
    agent = CatModelAgent(ctx=ctx, client=client, model="test-model")
    response = agent.run("AAL and 250-year PML?")
    assert response.n_tool_calls == 2
    assert response.n_llm_calls == 3


def test_agent_respects_max_iterations(ctx):
    # Construct a client that always asks for a tool, never finalizes
    looping = [_tool_call("describe_scenario", {}) for _ in range(20)]
    client = MockOpenAIClient(looping)
    agent = CatModelAgent(
        ctx=ctx, client=client, model="test-model", max_iterations=3
    )
    response = agent.run("loop forever")
    # Should hit the cap and bail out gracefully
    assert response.n_llm_calls == 3
    assert "maximum tool-use iterations" in response.final_text


def test_tool_arguments_are_passed_through(ctx):
    client = MockOpenAIClient(
        [
            _tool_call("compute_pml_with_ci", {"return_period": 100}),
            _final("100-year PML is X."),
        ]
    )
    agent = CatModelAgent(ctx=ctx, client=client, model="test-model")
    response = agent.run("100-year PML?")
    # The tool result event should reflect a successful call (non-error)
    tool_results = [ev for ev in response.trace if ev.kind == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].payload["name"] == "compute_pml_with_ci"


def test_invalid_tool_args_returned_as_error_to_model(ctx):
    """If the LLM emits malformed JSON arguments, we send an error tool result
    back rather than crashing the agent loop. The LLM can then recover."""
    client = MockOpenAIClient(
        [
            _MockResponse(
                choices=[
                    _MockChoice(
                        message=_MockMessage(
                            content=None,
                            tool_calls=[
                                _MockToolCall(
                                    id="call_bad",
                                    function=_MockFunction(
                                        name="compute_pml_with_ci",
                                        arguments="{not valid json",
                                    ),
                                )
                            ],
                        )
                    )
                ],
                usage=_MockUsage(prompt_tokens=10, completion_tokens=5),
            ),
            _final("Sorry, encountered an error."),
        ]
    )
    agent = CatModelAgent(ctx=ctx, client=client, model="test-model")
    response = agent.run("...")
    # Did not crash; produced a final response
    assert response.final_text == "Sorry, encountered an error."


def test_openai_tools_schema_well_formed(ctx):
    client = MockOpenAIClient([_final("ok")])
    agent = CatModelAgent(ctx=ctx, client=client, model="test-model")
    schemas = agent.openai_tools
    assert len(schemas) == 6
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert s["function"]["parameters"]["type"] == "object"
