"""Researcher agent: hybrid retrieval → citation prompt → synthesis."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.mcp.models import ResearchSummary
from blog_mas.prompts import RESEARCHER_SYSTEM_PROMPT
from blog_mas.rag.retrieval import hybrid_search
from blog_mas.rag.vector_store import ScoredPoint
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


async def research_node(state: BlogState, config: RunnableConfig) -> dict:
    blog_spec = state.get("blog_spec")
    if blog_spec is None:
        raise ValueError("[Researcher] Upstream agent failed — no blog spec in state")

    query = state.get("topic_query") or blog_spec.topic
    logger.info('[Researcher] Retrieving chunks for: "%s"', query)

    store = config["configurable"].get("store")
    embedder = config["configurable"].get("embedder")
    reranker = config["configurable"].get("reranker")

    chunks = []
    if store and embedder and reranker:
        chunks = hybrid_search(
            query=query, namespace="knowledge", top_k=3,
            store=store, embedder=embedder, reranker=reranker,
        )
        logger.info("[Researcher] retrieved %d chunks", len(chunks))

    user_message = _build_user_message(chunks, blog_spec)

    summary = await run_agent_chain(
        config=config,
        model_cls=ResearchSummary,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="Researcher",
    )

    # Override source with actual chunk IDs when we have them
    if chunks and summary.source != "none":
        chunk_ids = ",".join(c.id for c in chunks if c.id in user_message)
        if chunk_ids:
            summary = ResearchSummary(
                topic=summary.topic,
                bullet_points=summary.bullet_points,
                source=chunk_ids,
            )

    return {"research_summary": summary}


def _build_user_message(chunks: list[ScoredPoint], blog_spec) -> str:
    header = f"Topic: {blog_spec.topic}\nAudience: {blog_spec.audience}\nGoal: {blog_spec.goal}"

    if not chunks:
        return f"{header}\n\nNo source material was found for this topic."

    citations = []
    for c in chunks:
        text = c.payload.get("raw_text", "")
        citations.append(f"[Source {c.id}]\n{text}")

    return f"{header}\n\nSource material:\n" + "\n\n".join(citations)
