"""LangGraph state schema for the Context Engine."""

from typing import Any, TypedDict

from blog_mas.engine.tracer import ExecutionTrace


class EngineState(TypedDict):
    goal: str
    plan: list | None
    current_step: int
    step_outputs: dict
    trace: ExecutionTrace | None
    final_output: Any | None
    error: str | None
