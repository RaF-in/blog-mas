"""LangGraph state schema for the multi-agent blog generation pipeline."""

import operator
from typing import Annotated, TypedDict

from blog_mas.mcp.models import (
    BlogDraft,
    BlogSpec,
    GoalDecomposition,
    ResearchSummary,
    ValidationVerdict,
)
from blog_mas.rag.blueprints import Blueprint


class BlogState(TypedDict):
    raw_input: str
    blog_spec: BlogSpec | None
    research_summary: ResearchSummary | None
    draft: BlogDraft | None
    verdict: ValidationVerdict | None
    revision_feedback: str | None
    revision_count: Annotated[int, operator.add]
    intent_query: str | None
    topic_query: str | None
    blueprint: Blueprint | None
    blueprint_match_score: float | None
    blueprint_alternatives: list[str] | None
    blueprint_fallback_reason: str | None
