"""Writer agent: generates or revises a blog post draft from research + spec + blueprint."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain, sanitize_feedback
from blog_mas.mcp.models import BlogDraft, WriterInput
from blog_mas.prompts import WRITER_REVISION_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


def _blueprint_scaffold(blueprint) -> str:
    json_str = blueprint.model_dump_json()
    return f"--- SEMANTIC BLUEPRINT (JSON) ---\n{json_str}\n--- END SEMANTIC BLUEPRINT ---"


async def write_node(state: BlogState, config: RunnableConfig) -> dict:
    """Generate or revise a blog post draft and return a BlogDraft."""
    blog_spec = state.get("blog_spec")
    if blog_spec is None:
        raise ValueError("[Writer] Upstream agent failed — no blog spec in state")

    research_summary = state.get("research_summary")
    previous_content = state.get("previous_content")

    if research_summary is None and previous_content is None:
        raise ValueError(
            "[Writer] Upstream agent failed — no research summary in state "
            "(and no previous_content for rewrite mode)."
        )

    blueprint = state.get("blueprint")
    if blueprint is None:
        raise ValueError("[Writer] Upstream agent failed — no blueprint in state")

    revision_feedback = state.get("revision_feedback")
    is_revision = revision_feedback is not None
    is_rewrite = previous_content is not None and research_summary is None

    # Validate via WriterInput only in fresh mode (it requires research_summary)
    if not is_rewrite:
        WriterInput(
            research_summary=research_summary,
            blog_spec=blog_spec,
            revision_feedback=revision_feedback,
        )

    scaffold = _blueprint_scaffold(blueprint)

    if is_revision:
        clean_feedback = sanitize_feedback(revision_feedback)
        system_prompt = WRITER_REVISION_SYSTEM_PROMPT.format(
            blueprint_scaffold=scaffold, feedback=clean_feedback,
        )
    else:
        system_prompt = WRITER_SYSTEM_PROMPT.format(blueprint_scaffold=scaffold)

    if is_rewrite:
        prev_body = previous_content.body if hasattr(previous_content, "body") else str(previous_content)
        lines = [
            "PREVIOUS CONTENT (rewrite this in the new blueprint's style):",
            prev_body,
            "",
            "Blog specification:",
            f"- Tone: {blog_spec.tone}",
            f"- Audience: {blog_spec.audience}",
            f"- Goal: {blog_spec.goal}",
        ]
        if blog_spec.constraints:
            lines.append(f"- Constraints: {', '.join(blog_spec.constraints)}")
        user_message = "\n".join(lines)
    else:
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
    elif is_rewrite:
        logger.info("[Writer] Rewriting previous content in new style...")
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
