"""Librarian agent: retrieves and validates blueprints from the vector store."""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.rag.blueprints import NEUTRAL_BLUEPRINT, validate_blueprint_payload
from blog_mas.rag.retrieval import hybrid_search
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)


async def librarian_node(state: BlogState, config: RunnableConfig) -> dict:
    intent_query = state.get("intent_query")
    if not intent_query:
        logger.info("[Librarian] no intent_query, using neutral blueprint")
        return _neutral(fallback_reason="no_intent_query")

    store = config["configurable"]["store"]
    embedder = config["configurable"]["embedder"]
    reranker = config["configurable"]["reranker"]

    results = hybrid_search(
        query=intent_query,
        namespace="blueprints",
        top_k=5,
        store=store,
        embedder=embedder,
        reranker=reranker,
    )

    if not results:
        logger.info("[Librarian] empty retrieval, using neutral blueprint")
        return _neutral(fallback_reason="empty_retrieval")

    top = results[0]

    bp_json = top.payload.get("blueprint_json")
    if not bp_json:
        logger.info("[Librarian] missing blueprint_json in payload")
        return _neutral(fallback_reason="missing_payload")

    bp = validate_blueprint_payload(bp_json)
    if bp is None:
        logger.warning("[Librarian] blueprint validation failed — possible security event")
        return _neutral(fallback_reason="schema_violation")

    return {
        "blueprint": bp,
        "blueprint_match_score": top.score,
        "blueprint_alternatives": None,
        "blueprint_fallback_reason": None,
    }


def _neutral(fallback_reason: str) -> dict:
    return {
        "blueprint": NEUTRAL_BLUEPRINT,
        "blueprint_match_score": None,
        "blueprint_alternatives": None,
        "blueprint_fallback_reason": fallback_reason,
    }
