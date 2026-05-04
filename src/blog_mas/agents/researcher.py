"""Researcher agent: synthesizes knowledge base content into research bullet points."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.knowledge_base import lookup_topic
from blog_mas.mcp.models import ResearchRequest, ResearchSummary
from blog_mas.prompts import RESEARCHER_SYSTEM_PROMPT
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


async def research_node(state: BlogState, config: RunnableConfig) -> dict:
    """Look up a topic in the knowledge base and synthesize it into bullet points."""
    blog_spec = state.get("blog_spec")
    if blog_spec is None:
        raise ValueError("[Researcher] Upstream agent failed — no blog spec in state")

    ResearchRequest(topic=blog_spec.topic, audience=blog_spec.audience, goal=blog_spec.goal)

    logger.info('[Researcher] Looking up topic: "%s"', blog_spec.topic)
    kb_content = lookup_topic(blog_spec.topic)

    if kb_content is not None:
        logger.info("[Researcher] Topic found in knowledge base. Synthesizing...")
        user_message = (
            f"Topic: {blog_spec.topic}\n"
            f"Audience: {blog_spec.audience}\n"
            f"Goal: {blog_spec.goal}\n\n"
            f"Source material:\n{kb_content}"
        )
    else:
        logger.info("[Researcher] Topic not found. Proceeding with limited information.")
        user_message = (
            f"Topic: {blog_spec.topic}\n"
            f"Audience: {blog_spec.audience}\n"
            f"Goal: {blog_spec.goal}\n\n"
            "No information found on this topic in the knowledge base."
        )

    summary = await run_agent_chain(
        config=config,
        model_cls=ResearchSummary,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="Researcher",
    )

    return {"research_summary": summary}
