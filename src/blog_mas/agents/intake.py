"""Intake agent: decomposes raw user text into a structured BlogSpec."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.mcp.models import BlogSpec
from blog_mas.prompts import INTAKE_SYSTEM_PROMPT
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


async def intake_node(state: BlogState, config: RunnableConfig) -> dict:
    """Take raw user input and return a BlogSpec via PydanticOutputParser."""
    raw_input = state.get("raw_input")
    if raw_input is None or not raw_input.strip():
        raise ValueError("[Intake] No raw_input provided")

    logger.info("[Intake] Decomposing your request...")
    spec = await run_agent_chain(
        config=config,
        model_cls=BlogSpec,
        system_prompt=INTAKE_SYSTEM_PROMPT,
        user_message=raw_input,
        agent_name="Intake",
    )
    logger.info(
        '[Intake] Structured spec: topic="%s", audience="%s", ...',
        spec.topic, spec.audience,
    )
    return {"blog_spec": spec}
