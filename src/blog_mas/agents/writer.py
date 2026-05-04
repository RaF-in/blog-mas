"""Writer agent: generates or revises a blog post draft from research + spec."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain, sanitize_feedback
from blog_mas.mcp.models import BlogDraft, WriterInput
from blog_mas.prompts import WRITER_REVISION_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


async def write_node(state: BlogState, config: RunnableConfig) -> dict:
    """Generate or revise a blog post draft and return a BlogDraft."""
    blog_spec = state.get("blog_spec")
    if blog_spec is None:
        raise ValueError("[Writer] Upstream agent failed — no blog spec in state")

    research_summary = state.get("research_summary")
    if research_summary is None:
        raise ValueError("[Writer] Upstream agent failed — no research summary in state")

    revision_feedback = state.get("revision_feedback")
    is_revision = revision_feedback is not None

    WriterInput(
        research_summary=research_summary,
        blog_spec=blog_spec,
        revision_feedback=revision_feedback,
    )

    if is_revision:
        clean_feedback = sanitize_feedback(revision_feedback)
        system_prompt = WRITER_REVISION_SYSTEM_PROMPT.format(feedback=clean_feedback)
    else:
        system_prompt = WRITER_SYSTEM_PROMPT

    lines = ["Research summary:"]
    for bp in research_summary.bullet_points:
        lines.append(f"- {bp}")
    lines.append("")
    lines.append("Blog specification:")
    lines.append(f"- Tone: {blog_spec.tone}")
    lines.append(f"- Audience: {blog_spec.audience}")
    lines.append(f"- Goal: {blog_spec.goal}")
    if blog_spec.constraints:
        lines.append(f"- Constraints: {', '.join(blog_spec.constraints)}")

    user_message = "\n".join(lines)

    if is_revision:
        logger.info("[Writer] Revising draft based on feedback...")
    else:
        logger.info("[Writer] Drafting blog post...")

    draft = await run_agent_chain(
        config=config,
        model_cls=BlogDraft,
        system_prompt=system_prompt,
        user_message=user_message,
        agent_name="Writer",
    )

    return {"draft": draft}
