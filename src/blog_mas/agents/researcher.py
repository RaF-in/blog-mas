"""Researcher agent: synthesizes knowledge base content into research bullet points."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig

from blog_mas.knowledge_base import lookup_topic
from blog_mas.mcp.models import ResearchSummary
from blog_mas.prompts import RESEARCHER_SYSTEM_PROMPT
from blog_mas.state import BlogState


async def research_node(state: BlogState, config: RunnableConfig) -> dict:
    """Look up a topic in the knowledge base and synthesize it into bullet points."""
    llm = config["configurable"]["llm"]
    parser = PydanticOutputParser(pydantic_object=ResearchSummary)
    chain = llm | parser
    blog_spec = state["blog_spec"]

    print(f'[Researcher] Looking up topic: "{blog_spec.topic}"')
    kb_content = lookup_topic(blog_spec.topic)

    if kb_content is not None:
        print("[Researcher] Topic found in knowledge base. Synthesizing...")
        user_message = (
            f"Topic: {blog_spec.topic}\n"
            f"Audience: {blog_spec.audience}\n"
            f"Goal: {blog_spec.goal}\n\n"
            f"Source material:\n{kb_content}"
        )
    else:
        print("[Researcher] Topic not found. Proceeding with limited information.")
        user_message = (
            f"Topic: {blog_spec.topic}\n"
            f"Audience: {blog_spec.audience}\n"
            f"Goal: {blog_spec.goal}\n\n"
            "No information found on this topic in the knowledge base."
        )

    system_prompt = RESEARCHER_SYSTEM_PROMPT + "\n\n" + parser.get_format_instructions()
    summary = await chain.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )

    return {"research_summary": summary}
