"""Thin adapters: wrap existing LangGraph nodes into (mcp_message) -> mcp_message handlers.

Chapter 5 §B — the adapter layer is the contract translation boundary.
Each make_*_handler closure captures its dependencies at construction time
(dependency injection via closure) and presents a uniform single-argument
interface to the Executor: handler(mcp_message) -> mcp_message.

Chapter 6 §F — the Writer adapter is "bilingual": it accepts both a
ResearchSummary (from the Researcher) and a SummaryResult (from the new
Summarizer) for its `research_summary` slot.  Contract translation happens
here, not inside write_node — keeping the core agent stable.

Chapter 7 §H — the Writer adapter is upgraded to "trilingual": it now also
accepts a CitedAnswer (from the new Hi-Fi Researcher).  The priority order
for translation is:
  cited_answer  (Ch 7 Hi-Fi Researcher — richest, has inline citations + sources)
  → summary_result  (Ch 6 Summarizer)
  → research_summary  (Ch 5 Researcher — plain bullets)
The Writer's core (write_node) is never touched — contract translation
happens only here, at the adapter boundary.

Chapter 7 also swaps the engine Researcher to the hi-fi four-stage node while
keeping the *linear* LangGraph orchestrator on the original research_node.
This is the "engine-only upgrade" the plan specified — old flows continue to
work without modification (§G backward compatibility).
"""

import logging

from blog_mas.agents.intake import intake_node
from blog_mas.agents.librarian import librarian_node
from blog_mas.agents.researcher_hifi import cited_answer_to_research_summary, research_hifi_node
from blog_mas.agents.summarizer import summarize_node
from blog_mas.agents.validator import validate_node
from blog_mas.agents.writer import write_node
from blog_mas.engine.mcp_envelope import create_mcp_message
from blog_mas.engine.registry import AgentRegistry
from blog_mas.mcp.models import CitedAnswer, ResearchSummary, SummaryResult
from blog_mas.rag.blueprints import NEUTRAL_BLUEPRINT

logger = logging.getLogger(__name__)


def _config(llm, store, embedder, reranker):
    return {"configurable": {
        "llm": llm, "store": store, "embedder": embedder, "reranker": reranker,
    }}


def make_intake_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        raw_input = msg["content"]["raw_input"]
        state = {"raw_input": raw_input, "revision_count": 0}
        out = await intake_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Intake", {
            "blog_spec": out["blog_spec"],
            "intent_query": out["intent_query"],
            "topic_query": out["topic_query"],
        })
    return handler


def make_librarian_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        intent_query = msg["content"].get("intent_query")
        state = {"intent_query": intent_query, "revision_count": 0}
        out = await librarian_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Librarian", out["blueprint"])
    return handler


def make_researcher_handler(llm, store, embedder, reranker):
    """Chapter 7 — engine Researcher now routes through the hi-fi four-stage node.

    The linear orchestrator still uses the original research_node (unchanged).
    Only the engine path gets the upgrade: retrieve → sanitize → synthesize →
    format.  Output contract changes from ResearchSummary to CitedAnswer.
    """
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        state = {
            "blog_spec": content["blog_spec"],
            "topic_query": content.get("topic_query"),
            "revision_count": 0,
        }
        out = await research_hifi_node(state, _config(llm, store, embedder, reranker))
        # Emit CitedAnswer as a dict so the Executor stores plain JSON in
        # step_outputs — same pattern as make_summarizer_handler.
        cited_answer: CitedAnswer = out["cited_answer"]
        return create_mcp_message("Researcher", cited_answer.model_dump())
    return handler


def make_writer_handler(llm, store, embedder, reranker):
    """Chapter 7 §H — trilingual Writer adapter.

    Priority order for the research_summary slot:
      1. cited_answer   (Ch 7 Hi-Fi Researcher — richest contract)
      2. summary_result (Ch 6 Summarizer)
      3. research_summary (Ch 5 Researcher — plain bullets, pass-through)

    write_node is never touched.  All contract translation happens here.
    """
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        blueprint = content["blueprint"] or NEUTRAL_BLUEPRINT
        blog_spec = content["blog_spec"]

        research_summary = content.get("research_summary")
        previous_content = content.get("previous_content")

        if research_summary is None:
            # Priority 1 — Chapter 7 CitedAnswer (new hi-fi Researcher output)
            cited_answer = content.get("cited_answer")
            if cited_answer is not None:
                research_summary = cited_answer_to_research_summary(cited_answer)
                sources = (
                    cited_answer.get("sources", [])
                    if isinstance(cited_answer, dict)
                    else cited_answer.sources
                )
                logger.info(
                    "[WriterAdapter] Received CitedAnswer — sources: %s", sources
                )

        if research_summary is None:
            # Priority 2 — Chapter 6 SummaryResult (existing bilingual path)
            summary_result = content.get("summary_result")
            if summary_result is not None:
                research_summary = _summary_result_to_research_summary(summary_result)
                logger.info(
                    "[WriterAdapter] Received SummaryResult from Summarizer — "
                    "synthesised ResearchSummary (%d tokens saved).",
                    summary_result.input_tokens - summary_result.output_tokens
                    if isinstance(summary_result, SummaryResult)
                    else 0,
                )

        # Priority 3 — research_summary already set (plain Ch 5 Researcher output)

        state = {
            "blog_spec": blog_spec,
            "blueprint": blueprint,
            "research_summary": research_summary,
            "previous_content": previous_content,
            "draft": None,
            "revision_feedback": None,
            "revision_count": 0,
        }
        out = await write_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Writer", out["draft"])
    return handler


def _summary_result_to_research_summary(summary_result) -> ResearchSummary:
    """Translate a SummaryResult into the ResearchSummary shape write_node expects.

    This is the "contract translation at the boundary" lesson from Chapter 6 §F.
    The Writer's core (write_node) stays stable; only the adapter knows about
    the new upstream shape.  A priority-list approach handles both dict payloads
    (straight from the trace) and Pydantic model instances (from test stubs).
    """
    # Pydantic model instance (normal runtime path)
    if isinstance(summary_result, SummaryResult):
        summary_text = summary_result.summary
        tokens_note = (
            f"[Summarized from {summary_result.input_tokens} → "
            f"{summary_result.output_tokens} tokens, "
            f"{summary_result.reduction_percent:.1f}% reduction]"
        )
    # dict (from resolved $$STEP_N_OUTPUT$$ references in the trace)
    elif isinstance(summary_result, dict):
        summary_text = summary_result.get("summary", "")
        in_t = summary_result.get("input_tokens", 0)
        out_t = summary_result.get("output_tokens", 0)
        pct = summary_result.get("reduction_percent", 0.0)
        tokens_note = f"[Summarized from {in_t} → {out_t} tokens, {pct:.1f}% reduction]"
    else:
        summary_text = str(summary_result)
        tokens_note = "[Source: Summarizer]"

    return ResearchSummary(
        topic="summarized content",
        bullet_points=[summary_text, tokens_note],
        source="Summarizer",
    )


def make_summarizer_handler(llm, store, embedder, reranker):
    """Adapter for the Chapter 6 Summarizer agent.

    The Summarizer only needs the LLM — no vector store, no embedder.  The
    other arguments are accepted for signature uniformity with all other
    make_*_handler functions, which makes build_default_registry simpler.

    Input content keys:
      text_to_summarize  — (String/Reference) the bulk text to compress
      summary_objective  — (String) the micro-context goal for the summary

    Output:  SummaryResult serialised as dict in the MCP envelope content.
    """
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        text_to_summarize = content.get("text_to_summarize")
        summary_objective = content.get("summary_objective")

        if not text_to_summarize:
            raise ValueError("[SummarizerAdapter] Missing 'text_to_summarize' in content.")
        if not summary_objective:
            raise ValueError("[SummarizerAdapter] Missing 'summary_objective' in content.")

        state = {
            "text_to_summarize": text_to_summarize,
            "summary_objective": summary_objective,
        }
        out = await summarize_node(state, _config(llm, store, embedder, reranker))
        # Return the SummaryResult as a dict so the Executor stores plain JSON
        # in step_outputs — both dict and Pydantic model instances work as
        # upstream inputs for $$STEP_N_OUTPUT$$ resolution.
        summary_result: SummaryResult = out["summary_result"]
        return create_mcp_message("Summarizer", summary_result.model_dump())
    return handler


def make_validator_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        state = {
            "research_summary": content["research_summary"],
            "draft": content["draft"],
            "revision_count": 0,
        }
        out = await validate_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Validator", out["verdict"])
    return handler


def build_default_registry(llm, store, embedder, reranker) -> AgentRegistry:
    """Build the default registry of all six blog-mas agents (Ch 4 + Ch 6)."""
    reg = AgentRegistry()

    reg.register(
        "Intake",
        make_intake_handler(llm, store, embedder, reranker),
        role="Decomposes a free-text user request into a structured BlogSpec plus intent_query (style) and topic_query (subject).",
        inputs={"raw_input": "(String) The user's original blog request."},
        output="A dict with keys 'blog_spec' (BlogSpec), 'intent_query' (string), 'topic_query' (string).",
    )

    reg.register(
        "Librarian",
        make_librarian_handler(llm, store, embedder, reranker),
        role="Retrieves a Semantic Blueprint (style/structure instructions) from the Qdrant blueprint collection via hybrid search.",
        inputs={"intent_query": "(String/Reference) A descriptive phrase of the desired style or format."},
        output="A Blueprint object (Pydantic model) describing the target voice, structure, and constraints.",
    )

    reg.register(
        "Researcher",
        make_researcher_handler(llm, store, embedder, reranker),
        role=(
            "Chapter 7 Hi-Fi Researcher: retrieves chunks, sanitizes for injection, "
            "synthesizes with inline [N] citations, and appends a deterministic Sources list. "
            "Returns a CitedAnswer — not a ResearchSummary. "
            "Pass the output to Writer via the 'cited_answer' key."
        ),
        inputs={
            "topic_query": "(String/Reference) The subject matter to research.",
            "blog_spec": "(Reference) The BlogSpec produced by Intake (provides topic, audience, goal context).",
        },
        output=(
            "A CitedAnswer dict with keys: 'answer' (str with inline [N] cites), "
            "'sources' (list[str] of doc names), 'passages_retrieved' (int), "
            "'passages_used' (int), 'passages_rejected' (int). "
            "Pass to Writer using the 'cited_answer' key."
        ),
    )

    reg.register(
        "Writer",
        make_writer_handler(llm, store, embedder, reranker),
        role=(
            "Generates or rewrites a blog draft. Chapter 7 upgrade: now TRILINGUAL — "
            "accepts three research input shapes (priority order): "
            "(1) cited_answer from the Hi-Fi Researcher, "
            "(2) summary_result from the Summarizer, "
            "(3) research_summary from the plain Researcher. "
            "Also has rewrite mode via previous_content."
        ),
        inputs={
            "blueprint": "(Reference) The Blueprint from a Librarian step (style instructions).",
            "blog_spec": "(Reference) The BlogSpec from Intake.",
            "cited_answer": (
                "(Reference, OPTIONAL) CitedAnswer from a Chapter 7 Researcher step. "
                "Use this for FRESH content when the Researcher ran in hi-fi mode. "
                "Takes priority over research_summary and summary_result."
            ),
            "research_summary": "(Reference, OPTIONAL) ResearchSummary from a Chapter 5 Researcher step. Use this for FRESH content generation.",
            "summary_result": "(Reference, OPTIONAL) SummaryResult from a Summarizer step. Use this when you ran a Summarizer before the Writer.",
            "previous_content": "(Reference, OPTIONAL) A prior BlogDraft from an earlier Writer step. Use this for REWRITING in a new style.",
        },
        output="A BlogDraft object with title, body, and word_count.",
    )

    # Chapter 6 — Summarizer agent registered here.
    # The engine core (engine_graph.py, executor.py, planner.py) is untouched —
    # this single register() call is all it takes to make the agent discoverable
    # by the Planner and callable by the Executor.  This is zero-touch
    # extensibility / the open-closed principle in action.
    reg.register(
        "Summarizer",
        make_summarizer_handler(llm, store, embedder, reranker),
        role=(
            "Reduces a large text to a concise summary based on a specific objective. "
            "Ideal for managing token counts before an expensive generation step. "
            "Use when upstream text is large, off-topic-heavy, or needs extraction "
            "of specific entities before the Writer processes it."
        ),
        inputs={
            "text_to_summarize": (
                "(String/Reference) The long text to compress. "
                "Typically the output of a Researcher step or raw user-supplied text."
            ),
            "summary_objective": (
                "(String) A precise goal for the summary. "
                "Example: 'Extract the key scientific mission facts and instrument names.' "
                "The more specific this is, the more useful the output will be."
            ),
        },
        output=(
            "A dict with keys: 'summary' (str), 'input_tokens' (int), "
            "'output_tokens' (int), 'reduction_percent' (float). "
            "Pass this to the Writer via the 'summary_result' key."
        ),
    )

    reg.register(
        "Validator",
        make_validator_handler(llm, store, embedder, reranker),
        role="Fact-checks a BlogDraft against a ResearchSummary and returns a pass/fail verdict.",
        inputs={
            "draft": "(Reference) The BlogDraft from a Writer step.",
            "research_summary": "(Reference) The ResearchSummary the draft should be grounded in.",
        },
        output="A ValidationVerdict object with verdict ('pass'|'fail') and reason.",
    )

    return reg
