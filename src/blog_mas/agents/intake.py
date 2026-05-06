"""Intake agent: decomposes raw user text into a structured BlogSpec."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.mcp.models import BlogSpec, GoalDecomposition
from blog_mas.prompts import INTAKE_SYSTEM_PROMPT
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)

GOAL_DECOMP_SYSTEM_PROMPT = (
    "You decompose a blog writing request into two search queries. "
    '"intent_query" captures WHY the user wants the blog (tone, goal, audience). '
    '"topic_query" captures WHAT the blog is about (subject matter). '
    "Output ONLY valid JSON."
)


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

    intent_query, topic_query = await _decompose_goal(spec, config, raw_input)

    return {"blog_spec": spec, "intent_query": intent_query, "topic_query": topic_query}


async def _decompose_goal(spec: BlogSpec, config, raw_input: str) -> tuple[str, str]:
    for _ in range(2):
        try:
            goal = await run_agent_chain(
                config=config,
                model_cls=GoalDecomposition,
                system_prompt=GOAL_DECOMP_SYSTEM_PROMPT,
                user_message=raw_input,
                agent_name="Intake-GoalDecomp",
            )
            return goal.intent_query, goal.topic_query
        except Exception as exc:
            logger.warning("[Intake] goal decomp attempt failed: %s", exc)

    # Deterministic fallback (F10)
    intent_query = f"{spec.tone} {spec.goal} for {spec.audience}"
    topic_query = spec.topic
    logger.info("[Intake] using deterministic fallback for goal decomposition")
    return intent_query, topic_query
