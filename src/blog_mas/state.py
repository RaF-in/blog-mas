"""LangGraph state schema for the multi-agent blog generation pipeline."""

import operator
from typing import Annotated, TypedDict

from blog_mas.mcp.models import (
    BlogDraft,
    BlogSpec,
    ResearchSummary,
    ValidationVerdict,
)


class BlogState(TypedDict):
    raw_input: str
    blog_spec: BlogSpec | None
    research_summary: ResearchSummary | None
    draft: BlogDraft | None
    verdict: ValidationVerdict | None
    revision_feedback: str | None
    revision_count: Annotated[int, operator.add]
