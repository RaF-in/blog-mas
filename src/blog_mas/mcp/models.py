"""Pydantic content models for inter-agent communication."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BlogSpec(BaseModel):
    topic: str = Field(max_length=200)
    audience: str = Field(max_length=100)
    tone: str = Field(max_length=100)
    goal: str = Field(max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("constraints")
    @classmethod
    def _validate_constraint_items(cls, v: list[str]) -> list[str]:
        for item in v:
            if len(item) > 500:
                raise ValueError("Each constraint must be at most 500 characters")
        return v


class ResearchRequest(BaseModel):
    topic: str = Field(max_length=100)
    audience: str = Field(max_length=100)
    goal: str = Field(max_length=100)


class ResearchSummary(BaseModel):
    topic: str
    bullet_points: list[str] = Field(max_length=10)
    source: str

    @field_validator("bullet_points")
    @classmethod
    def _validate_bullet_items(cls, v: list[str]) -> list[str]:
        for item in v:
            if len(item) > 1000:
                raise ValueError("Each bullet point must be at most 1000 characters")
        return v


class WriterInput(BaseModel):
    research_summary: ResearchSummary
    blog_spec: BlogSpec
    revision_feedback: str | None = None


class BlogDraft(BaseModel):
    title: str
    body: str = Field(max_length=10000)
    word_count: int


class ValidationInput(BaseModel):
    research_summary: ResearchSummary
    draft: BlogDraft


class ValidationVerdict(BaseModel):
    verdict: Literal["pass", "fail"] = Field(max_length=50)
    reason: str


class GoalDecomposition(BaseModel):
    intent_query: str
    topic_query: str

    @field_validator("intent_query", "topic_query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Chapter 6 — Summarizer agent models
# ---------------------------------------------------------------------------

class SummarizerInput(BaseModel):
    """Input contract for the Summarizer agent.

    Both fields are mandatory — the agent raises ValueError if either is absent
    (guard-clause pattern from Chapter 5 §D data contracts).
    """
    text_to_summarize: str = Field(max_length=50_000)
    summary_objective: str = Field(
        max_length=500,
        description=(
            "A precise goal for the summary, e.g. "
            "'Extract the names of all involved parties and the key financial figures.'"
        ),
    )


class SummaryText(BaseModel):
    """What the LLM returns: the raw summary string.

    Kept separate from SummaryResult because the LLM can't know token counts —
    those are computed after the fact in summarize_node.
    """
    summary: str = Field(max_length=5_000)


class SummaryResult(BaseModel):
    """Full output contract for the Summarizer agent.

    Includes the business-value metrics from Chapter 6 §7 — input/output token
    counts and reduction percentage — so the CLI can print a one-liner like
    "Token reduction: 56.5% (1240 → 540 tokens)" without extra computation.
    """
    summary: str = Field(max_length=5_000)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reduction_percent: float = Field(ge=0.0, le=100.0)


# ---------------------------------------------------------------------------
# Chapter 7 — High-Fidelity Researcher models
# ---------------------------------------------------------------------------

class SynthesizedAnswer(BaseModel):
    """What the LLM returns from the hi-fi Researcher synthesis step.

    The LLM is given numbered passages [1], [2], [3] … and asked to:
      1. Write the answer using only those passages.
      2. Cite inline with [N] markers.
      3. Return the exact list of passage IDs it cited.

    We trust the answer text (with its [N] markers) but we rebuild the
    Sources: block deterministically from cited_passage_ids — the
    "belt and suspenders" move from Chapter 7 §B.
    """
    answer: str = Field(
        max_length=5_000,
        description=(
            "Prose answer with inline [N] citation markers matching the "
            "numbered passages supplied in the user prompt."
        ),
    )
    cited_passage_ids: list[int] = Field(
        description=(
            "List of passage numbers (1-based) that were cited in the answer. "
            "Used to programmatically rebuild the Sources list."
        ),
    )


class CitedAnswer(BaseModel):
    """Full output contract for the Chapter 7 Hi-Fi Researcher agent.

    Contains the synthesized answer (with inline citations), the deterministic
    source list, and observability counters so downstream agents and the CLI
    can surface data quality information.
    """
    answer: str = Field(
        max_length=5_000,
        description="Prose answer with inline [N] citation markers.",
    )
    sources: list[str] = Field(
        description=(
            "Sorted unique list of source document names (doc_id) that were "
            "actually cited.  Built deterministically from the passage→doc_id "
            "mapping — never from the LLM's claimed sources."
        ),
    )
    passages_retrieved: int = Field(ge=0, description="Total chunks returned by hybrid_search.")
    passages_used: int = Field(ge=0, description="Chunks that survived sanitization and were cited.")
    passages_rejected: int = Field(ge=0, description="Chunks blocked by the sanitizer.")
