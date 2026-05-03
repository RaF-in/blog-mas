"""Writer agent: generates or revises a blog post draft from research + spec."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from blog_mas.mcp.models import BlogDraft
from blog_mas.prompts import WRITER_REVISION_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from blog_mas.state import BlogState


async def write_node(state: BlogState, config: RunnableConfig) -> dict:
    """Generate or revise a blog post draft and return a BlogDraft."""
    llm = config["configurable"]["llm"]
    parser = PydanticOutputParser(pydantic_object=BlogDraft)
    chain = llm | parser

    blog_spec = state["blog_spec"]
    research_summary = state["research_summary"]
    revision_feedback = state.get("revision_feedback")
    is_revision = revision_feedback is not None

    if is_revision:
        system_prompt = WRITER_REVISION_SYSTEM_PROMPT.format(feedback=revision_feedback)
    else:
        system_prompt = WRITER_SYSTEM_PROMPT

    system_prompt += "\n\n" + parser.get_format_instructions()

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
        print("[Writer] Revising draft based on feedback...")
    else:
        print("[Writer] Drafting blog post...")

    draft = await chain.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )

    return {"draft": draft}
