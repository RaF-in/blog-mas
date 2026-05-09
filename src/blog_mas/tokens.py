"""Token counting utilities — the system's fuel gauge.

Chapter 5 introduces count_tokens as the production-readiness signal: an engine
that doesn't know its token consumption will hit context limits without warning
and generate surprise API bills.  This module centralises that measurement so
every agent and the agent_helpers chain wrapper can use the same logic.

Key ideas from the book implemented here:
  • count_tokens  — exact token count via tiktoken (cl100k_base fallback)
  • estimate_cost — translates tokens into dollars for business conversations
  • compute_token_savings — post-run analytics pulled from the ExecutionTrace,
      so the Chapter 6 Summarizer's efficiency shows up in the CLI output
"""

import logging

import tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared encoder — cl100k_base covers GPT-4, GPT-3.5, and most modern models.
# Module-level so it's constructed once and reused across all calls.
# ---------------------------------------------------------------------------
_ENC = tiktoken.get_encoding("cl100k_base")


# Illustrative pricing table ($/1K tokens). Update to current rates as needed.
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4":    {"input": 0.03,  "output": 0.06},
    "gpt-4o":   {"input": 0.005, "output": 0.015},
    "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
}
_DEFAULT_PRICE = {"input": 0.01, "output": 0.03}


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Return the exact token count for *text* using the model's tokenizer.

    Falls back to cl100k_base when the model isn't in tiktoken's registry —
    which is common for locally-hosted or newly-released models.

    This is the "fuel gauge" from Chapter 5 Section C.  Without it you can't
    do cost tracking, context-window safety checks, or auto-summarisation.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Locally-hosted models (e.g. Qwen via LM Studio) won't be in the
        # tiktoken registry — cl100k_base gives a close-enough count.
        enc = _ENC
    return len(enc.encode(text))


def estimate_cost(prompt_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    """Translate token counts into an estimated dollar cost.

    This is the "translate tech to business value" step from Chapter 6 §1:
    56.5% token reduction is nice; "$12K/month saved" is what leadership hears.
    """
    prices = _MODEL_PRICING.get(model, _DEFAULT_PRICE)
    return (prompt_tokens * prices["input"] + output_tokens * prices["output"]) / 1000


def compute_token_savings(trace) -> dict:
    """Analyse an ExecutionTrace and report Summarizer efficiency metrics.

    Walks the trace steps looking for Summarizer agents.  For each one it
    reads the input_tokens / output_tokens fields stored in the step output
    and computes the aggregate reduction.

    Returns a dict with:
      summarizer_steps  — list of per-step dicts with token metrics
      total_input_tokens  — tokens consumed across all Summarizer inputs
      total_output_tokens — tokens produced across all Summarizer outputs
      overall_reduction_pct — weighted average reduction across all steps

    Returns an empty dict when no Summarizer steps are present, so callers
    can use a simple ``if savings:`` guard.
    """
    if trace is None:
        return {}

    summarizer_steps = []
    for step in trace.steps:
        if step.get("agent") != "Summarizer":
            continue
        output = step.get("output", {})
        if not isinstance(output, dict):
            continue
        in_tok = output.get("input_tokens", 0)
        out_tok = output.get("output_tokens", 0)
        reduction = output.get("reduction_percent", 0.0)
        summarizer_steps.append({
            "step": step["step"],
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reduction_percent": reduction,
        })

    if not summarizer_steps:
        return {}

    total_in = sum(s["input_tokens"] for s in summarizer_steps)
    total_out = sum(s["output_tokens"] for s in summarizer_steps)
    overall_reduction = (1 - total_out / total_in) * 100 if total_in > 0 else 0.0

    return {
        "summarizer_steps": summarizer_steps,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "overall_reduction_pct": round(overall_reduction, 1),
    }
