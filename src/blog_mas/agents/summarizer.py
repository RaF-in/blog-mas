"""Summarizer agent: reduces large text to a goal-directed summary.

Chapter 6 introduces this as a *first-class architectural primitive* — not a
utility function bolted on, but a proper agent that lives in the registry and
is chosen dynamically by the Planner.

Key design decisions aligned with the book:

1. Same LangGraph node signature as every other agent (state, config).
   The Executor calls all agents through the same interface.

2. Dependency injection — only `llm` is needed (no vector store, no embedder).
   Dependencies are explicit in the node signature, not global.

3. Guard clauses on both required fields (Chapter 5 §D data contracts).
   Missing either field raises ValueError immediately, not silently.

4. Micro-context engineering (Chapter 6 §E): the power is entirely in
   `summary_objective`. A precise objective transforms this generic agent into
   a precision entity-extractor, keyword-distiller, or decision-summariser.

5. Token metrics on output (Chapter 6 §7): `SummaryResult` carries
   input_tokens, output_tokens, and reduction_percent so the CLI can print
   the business-value summary without doing any extra computation.

6. Structured logging from day one — not retrofitted (Chapter 5 §C).
"""

import logging

from langchain_core.runnables import RunnableConfig

from blog_mas.agent_helpers import run_agent_chain
from blog_mas.mcp.models import SummaryResult, SummaryText
from blog_mas.prompts import SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_USER_PROMPT_TEMPLATE
from blog_mas.tokens import count_tokens

logger = logging.getLogger(__name__)


async def summarize_node(state: dict, config: RunnableConfig) -> dict:
    """Reduce `text_to_summarize` to a goal-directed summary.

    Expected state keys (both required — guard-clause pattern):
      text_to_summarize  — the bulk text to compress
      summary_objective  — precise goal for the summary (micro-context)

    Returns:
      {"summary_result": SummaryResult}
    """
    text_to_summarize = state.get("text_to_summarize")
    summary_objective = state.get("summary_objective")

    # Chapter 5 §D — data contract enforcement.  Both fields are mandatory.
    # Fail fast here rather than produce a confusing downstream error.
    if not text_to_summarize:
        raise ValueError("[Summarizer] Missing required state key: 'text_to_summarize'")
    if not summary_objective:
        raise ValueError("[Summarizer] Missing required state key: 'summary_objective'")

    input_tokens = count_tokens(text_to_summarize)
    logger.info(
        "[Summarizer] Activated — input: %d tokens. Objective: %.80s…",
        input_tokens, summary_objective,
    )

    user_message = SUMMARIZER_USER_PROMPT_TEMPLATE.format(
        summary_objective=summary_objective,
        text_to_summarize=text_to_summarize,
    )

    # The LLM produces only the summary text (SummaryText).
    # We compute the efficiency metrics ourselves — the model can't know token
    # counts, and we want them to be accurate rather than LLM-hallucinated.
    summary_text_obj: SummaryText = await run_agent_chain(
        config=config,
        model_cls=SummaryText,
        system_prompt=SUMMARIZER_SYSTEM_PROMPT,
        user_message=user_message,
        agent_name="Summarizer",
    )

    output_tokens = count_tokens(summary_text_obj.summary)
    reduction_pct = (
        (1 - output_tokens / input_tokens) * 100 if input_tokens > 0 else 0.0
    )

    # Chapter 6 §7 — translate tech to business value.
    logger.info(
        "[Summarizer] Done — output: %d tokens. Reduction: %.1f%% (%d → %d tokens).",
        output_tokens, reduction_pct, input_tokens, output_tokens,
    )

    result = SummaryResult(
        summary=summary_text_obj.summary,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reduction_percent=round(reduction_pct, 1),
    )

    return {"summary_result": result}
