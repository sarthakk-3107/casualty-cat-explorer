"""Agentic interface to the casualty cat model."""

from .runtime import AgentResponse, CatModelAgent, TraceEvent, trace_summary
from .tools import ModelContext, Tool, build_registry, execute_tool_call

__all__ = [
    "AgentResponse",
    "CatModelAgent",
    "ModelContext",
    "Tool",
    "TraceEvent",
    "build_registry",
    "execute_tool_call",
    "trace_summary",
]
