"""Chapter 7 §D — Hi-Fi Researcher agent: four sequential stages.

The book (§D) defines the upgrade as four stages that must run in order:

  1. Retrieve  — pull chunks with their source metadata.
  2. Sanitize  — run each chunk through the injection gate; skip failures.
  3. Synthesize — prompt the LLM with numbered, citation-aware passages.
  4. Format    — append the Sources list *programmatically* from the
                  cited_passage_ids mapping, not from what the LLM claimed.

Stage 4 is the "belt and suspenders" move from §B: LLMs are stochastic and can
drop or hallucinate citations on a bad day; the code-built Sources list is
deterministic and reliable.

This node is wired into the *engine* path only (via agent_adapters.py).
The linear LangGraph orchestrator keeps the existing research_node unchanged —
so old flows keep working without a single line change (§G backward compat).
"""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.mcp.models import CitedAnswer, SynthesizedAnswer
from blog_mas.prompts import (
    INSUFFICIENT_EVIDENCE_REFUSAL,
    RESEARCHER_HIFI_SYSTEM_PROMPT,
    RESEARCHER_HIFI_USER_TEMPLATE,
)
from blog_mas.rag.retrieval import hybrid_search
from blog_mas.rag.vector_store import ScoredPoint
from blog_mas.security.sanitizer import SanitizationResult, sanitize_chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public node — called by the engine adapter
# ---------------------------------------------------------------------------

async def research_hifi_node(state: dict, config: RunnableConfig) -> dict:
    """Four-stage hi-fi research node.

    State keys consumed:
      blog_spec    — BlogSpec (provides topic, audience, goal)
      topic_query  — optional override query string

    State keys produced:
      cited_answer — CitedAnswer (the hi-fi output contract)
    """
    blog_spec = state.get("blog_spec")
    if blog_spec is None:
        raise ValueError("[ResearcherHiFi] Upstream agent failed — no blog_spec in state")

    query = state.get("topic_query") or blog_spec.topic

    # ── Stage 1: Retrieve ──────────────────────────────────────────────────
    # hybrid_search returns ScoredPoints whose payload contains raw_text,
    # doc_id (the source filename stem), and other ingestion metadata.
    store = config["configurable"].get("store")
    embedder = config["configurable"].get("embedder")
    reranker = config["configurable"].get("reranker")

    raw_chunks: list[ScoredPoint] = []
    if store and embedder and reranker:
        raw_chunks = hybrid_search(
            query=query, namespace="knowledge", top_k=5,
            store=store, embedder=embedder, reranker=reranker,
        )
    logger.info("[ResearcherHiFi] Stage 1 — Retrieved %d chunks", len(raw_chunks))

    # ── Stage 2: Sanitize ─────────────────────────────────────────────────
    # Each chunk is run through the injection gate.  Rejected chunks are
    # counted and logged; they never touch the LLM.  Chapter 7 §E.
    passages: list[dict] = []   # {"id": 1, "text": ..., "doc_id": ...}
    rejected_count = 0

    for chunk in raw_chunks:
        text = chunk.payload.get("raw_text", "")
        result: SanitizationResult = sanitize_chunk(text)

        if result.ok:
            passages.append({
                "id": len(passages) + 1,   # 1-based numbering for LLM prompt
                "text": result.cleaned_text,
                "doc_id": chunk.payload.get("doc_id", "unknown"),
                "chunk_id": chunk.id,
            })
        else:
            rejected_count += 1
            logger.warning(
                "[ResearcherHiFi] Stage 2 — Chunk rejected. "
                "chunk_id=%s pattern=%r",
                chunk.id, result.matched_pattern,
            )

    logger.info(
        "[ResearcherHiFi] Stage 2 — %d passages clean, %d rejected",
        len(passages), rejected_count,
    )

    # ── Defensive fallback ────────────────────────────────────────────────
    # If nothing survived sanitization (or nothing was retrieved at all),
    # return an honest refusal rather than hallucinating.
    # Chapter 7 §G: "An agent that correctly refuses to answer outside its
    # knowledge base is more valuable than one that confidently hallucinates."
    if not passages:
        logger.warning(
            "[ResearcherHiFi] No trustworthy passages available. Returning refusal."
        )
        cited_answer = CitedAnswer(
            answer=INSUFFICIENT_EVIDENCE_REFUSAL,
            sources=[],
            passages_retrieved=len(raw_chunks),
            passages_used=0,
            passages_rejected=rejected_count,
        )
        return {"cited_answer": cited_answer}

    # ── Stage 3: Synthesize ───────────────────────────────────────────────
    # Build the numbered passages block and let the LLM write a cited answer.
    # The system prompt instructs it to use inline [N] markers and return
    # cited_passage_ids — a structured contract that lets us verify its work.
    numbered_passages = "\n\n".join(
        f"[{p['id']}] (source: {p['doc_id']})\n{p['text']}"
        for p in passages
    )
    user_message = RESEARCHER_HIFI_USER_TEMPLATE.format(
        question=query,
        numbered_passages=numbered_passages,
    )

    synthesized: SynthesizedAnswer = await run_agent_chain(
        config=config,
        model_cls=SynthesizedAnswer,
        system_prompt=RESEARCHER_HIFI_SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="ResearcherHiFi",
    )

    logger.info(
        "[ResearcherHiFi] Stage 3 — LLM cited %d passages",
        len(synthesized.cited_passage_ids),
    )

    # ── Stage 4: Format ───────────────────────────────────────────────────
    # Rebuild the Sources list from the passage→doc_id mapping in *our* code,
    # not from whatever the LLM claimed.  This is the programmatic source append
    # from Chapter 7 §D / §B.  sorted() makes the list deterministic across runs.
    passage_map = {p["id"]: p["doc_id"] for p in passages}
    used_sources = sorted({
        passage_map[pid]
        for pid in synthesized.cited_passage_ids
        if pid in passage_map          # guard against LLM hallucinating an ID
    })

    cited_answer = CitedAnswer(
        answer=synthesized.answer,
        sources=used_sources,
        passages_retrieved=len(raw_chunks),
        passages_used=len(synthesized.cited_passage_ids),
        passages_rejected=rejected_count,
    )

    logger.info(
        "[ResearcherHiFi] Stage 4 — Sources: %s",
        cited_answer.sources,
    )

    return {"cited_answer": cited_answer}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cited_answer_to_research_summary(cited_answer):
    """Translate a CitedAnswer into the ResearchSummary shape write_node expects.

    This is the same contract-translation-at-the-boundary pattern used in
    Chapter 6 §F (_summary_result_to_research_summary).  write_node is stable;
    only the adapter layer knows about the new upstream shape.

    The translation preserves the sources so the Writer can weave them into the
    draft body (Chapter 7 §H — "trilingual Writer").
    """
    from blog_mas.mcp.models import CitedAnswer as _CitedAnswer, ResearchSummary

    # Accept both a CitedAnswer model instance and a plain dict (from resolved
    # $$STEP_N_OUTPUT$$ references in the executor trace).
    if isinstance(cited_answer, _CitedAnswer):
        answer_text = cited_answer.answer
        sources = cited_answer.sources
        retrieved = cited_answer.passages_retrieved
        used = cited_answer.passages_used
        rejected = cited_answer.passages_rejected
    elif isinstance(cited_answer, dict):
        answer_text = cited_answer.get("answer", "")
        sources = cited_answer.get("sources", [])
        retrieved = cited_answer.get("passages_retrieved", 0)
        used = cited_answer.get("passages_used", 0)
        rejected = cited_answer.get("passages_rejected", 0)
    else:
        answer_text = str(cited_answer)
        sources = []
        retrieved = used = rejected = 0

    sources_line = (
        f"[Sources: {', '.join(sources)}]"
        if sources
        else "[Sources: none — refusal or no evidence]"
    )
    provenance_note = (
        f"[Retrieved: {retrieved} chunks | Used: {used} | Rejected: {rejected}]"
    )

    return ResearchSummary(
        topic="hi-fi research",
        bullet_points=[answer_text, sources_line, provenance_note],
        source=", ".join(sources) if sources else "none",
    )
