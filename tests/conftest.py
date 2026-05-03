"""Shared test fixtures and helpers."""

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel


def _to_ai_message(value) -> AIMessage:
    """Convert a Pydantic model or dict to an AIMessage with JSON content."""
    if isinstance(value, BaseModel):
        content = value.model_dump_json()
    elif isinstance(value, dict):
        content = json.dumps(value)
    else:
        content = str(value)
    return AIMessage(content=content)


def make_mock_llm(return_value) -> RunnableLambda:
    """Create a mock LLM (Runnable) that returns an AIMessage with the given value."""
    msg = _to_ai_message(return_value)

    async def _invoke(messages, config=None):
        return msg

    return RunnableLambda(_invoke)


def make_mock_llm_sequence(return_values: list) -> RunnableLambda:
    """Create a mock LLM (Runnable) that returns AIMessages in sequence."""
    it = iter([_to_ai_message(v) for v in return_values])

    async def _invoke(messages, config=None):
        return next(it)

    return RunnableLambda(_invoke)


def make_failing_llm(error: Exception) -> RunnableLambda:
    """Create a mock LLM (Runnable) that raises the given error."""
    async def _invoke(messages, config=None):
        raise error

    return RunnableLambda(_invoke)


def make_config(llm):
    """Create a LangGraph RunnableConfig dict with the given LLM."""
    return {"configurable": {"llm": llm}}
