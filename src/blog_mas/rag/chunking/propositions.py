"""Stage 3: agentic proposition extraction from parent chunks.

For every parent ``Chunk``, call the LLM to extract atomic, self-contained
factual propositions.  Each proposition becomes a child ``Chunk`` linked to
its parent via ``parent_id``.  On JSON-parse failure for a given parent,
discard that parent's propositions — the parent itself is preserved
upstream by the ingestion graph.
"""

import hashlib
import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from blog_mas.rag.chunking.types import Chunk

logger = logging.getLogger(__name__)

PROPOSITION_SYSTEM_PROMPT = (
    "You are a precise information extraction assistant. "
    "Given a text passage, extract all factual propositions as a JSON object "
    'with a single key "propositions" mapping to an array of strings. '
    "Each proposition must be atomic, self-contained, and factually grounded "
    "in the source text. Output ONLY valid JSON — no markdown, no commentary."
)


async def extract_propositions(
    parents: list[Chunk], llm
) -> list[Chunk]:
    """Extract child proposition chunks from *parents*.

    Returns a flat list of child ``Chunk`` objects (``chunk_type="proposition"``).
    The *parents* list is not modified.
    """
    children: list[Chunk] = []

    for parent in parents:
        propositions = await _extract_one(parent, llm)
        for i, prop_text in enumerate(propositions):
            children.append(
                Chunk(
                    raw_text=prop_text.strip(),
                    parent_id=parent.content_hash,
                    doc_id=parent.doc_id,
                    headings_path=list(parent.headings_path),
                    content_hash=_child_hash(parent.doc_id, parent.chunk_index, i, prop_text),
                    chunk_type="proposition",
                    chunk_index=parent.chunk_index * 1000 + i,
                )
            )

    return children


async def _extract_one(parent: Chunk, llm) -> list[str]:
    """Call LLM for one parent, parse propositions. Returns [] on any failure."""
    messages = [
        SystemMessage(content=PROPOSITION_SYSTEM_PROMPT),
        HumanMessage(content=parent.raw_text),
    ]

    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        logger.warning("propositions.llm_error parent=%s detail=%s", parent.content_hash[:12], exc)
        return []

    content = response.content if isinstance(response, AIMessage) else str(response)

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("propositions.json_parse_error parent=%s", parent.content_hash[:12])
        return []

    if not isinstance(data, dict) or "propositions" not in data:
        logger.warning("propositions.missing_key parent=%s", parent.content_hash[:12])
        return []

    props = data["propositions"]
    if not isinstance(props, list):
        logger.warning("propositions.non_list parent=%s", parent.content_hash[:12])
        return []

    return props


def _child_hash(doc_id: str, parent_idx: int, child_idx: int, text: str) -> str:
    return hashlib.sha256(
        f"{doc_id}:{parent_idx}:prop:{child_idx}:{text}".encode("utf-8")
    ).hexdigest()
