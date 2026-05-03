"""Pydantic content models for inter-agent communication."""

from typing import Literal

from pydantic import BaseModel


class BlogSpec(BaseModel):
    topic: str
    audience: str
    tone: str
    goal: str
    constraints: list[str]


class ResearchRequest(BaseModel):
    topic: str
    audience: str
    goal: str


class ResearchSummary(BaseModel):
    topic: str
    bullet_points: list[str]
    source: str


class WriterInput(BaseModel):
    research_summary: ResearchSummary
    blog_spec: BlogSpec
    revision_feedback: str | None = None


class BlogDraft(BaseModel):
    title: str
    body: str
    word_count: int


class ValidationInput(BaseModel):
    research_summary: ResearchSummary
    draft: BlogDraft


class ValidationVerdict(BaseModel):
    verdict: Literal["pass", "fail"]
    reason: str
