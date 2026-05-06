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
