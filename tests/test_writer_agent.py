"""Tests for the Writer agent (LangGraph node with LCEL chain)."""

import pytest

from blog_mas.agents.writer import write_node
from blog_mas.mcp.models import (
    BlogDraft,
    BlogSpec,
    ResearchSummary,
)
from tests.conftest import make_config, make_failing_llm, make_mock_llm


def _make_research_summary(
    topic="Mediterranean diet",
    bullet_points=None,
    source="knowledge_base",
):
    return ResearchSummary(
        topic=topic,
        bullet_points=bullet_points or [
            "The Mediterranean diet emphasizes plant-based foods.",
            "Olive oil is the primary fat source.",
            "Studies show reduced cardiovascular risk.",
        ],
        source=source,
    )


def _make_blog_spec(
    topic="Mediterranean diet",
    audience="general readers",
    tone="informative and engaging",
    goal="educate the reader",
    constraints=None,
):
    return BlogSpec(
        topic=topic,
        audience=audience,
        tone=tone,
        goal=goal,
        constraints=constraints or [],
    )


def _make_draft(
    title="The Wonders of the Mediterranean Diet",
    body="The Mediterranean diet is great. Olive oil is key.",
    word_count=11,
):
    return BlogDraft(title=title, body=body, word_count=word_count)


class TestWriterFirstPass:
    @pytest.mark.asyncio
    async def test_generates_blog_draft(self):
        draft = _make_draft()
        llm = make_mock_llm(draft)
        state = {
            "blog_spec": _make_blog_spec(),
            "research_summary": _make_research_summary(),
            "revision_feedback": None,
        }

        result = await write_node(state, config=make_config(llm))
        assert isinstance(result["draft"], BlogDraft)
        assert result["draft"].title != ""
        assert result["draft"].body != ""
        assert result["draft"].word_count > 0


class TestWriterRevisionPass:
    @pytest.mark.asyncio
    async def test_incorporates_feedback_into_revised_draft(self):
        draft = _make_draft()
        llm = make_mock_llm(draft)
        state = {
            "blog_spec": _make_blog_spec(),
            "research_summary": _make_research_summary(),
            "revision_feedback": "Remove unsupported claim about X",
        }

        result = await write_node(state, config=make_config(llm))
        assert isinstance(result["draft"], BlogDraft)


class TestWriterFailure:
    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self):
        llm = make_failing_llm(ConnectionError("down"))
        state = {
            "blog_spec": _make_blog_spec(),
            "research_summary": _make_research_summary(),
            "revision_feedback": None,
        }

        with pytest.raises(ConnectionError, match="down"):
            await write_node(state, config=make_config(llm))
