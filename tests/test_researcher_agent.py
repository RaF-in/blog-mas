"""Tests for the Researcher agent (LangGraph node with LCEL chain)."""

import pytest

from blog_mas.agents.researcher import research_node
from blog_mas.mcp.models import BlogSpec, ResearchSummary
from tests.conftest import make_config, make_failing_llm, make_mock_llm


def _make_blog_spec(topic="Mediterranean diet"):
    return BlogSpec(
        topic=topic,
        audience="general readers",
        tone="informative",
        goal="educate",
        constraints=[],
    )


def _make_research_summary(
    topic="Mediterranean diet",
    bullet_points=None,
    source="knowledge_base",
):
    return ResearchSummary(
        topic=topic,
        bullet_points=bullet_points or [
            "The Mediterranean diet emphasizes fruits, vegetables, and whole grains.",
            "It is associated with reduced cardiovascular disease risk.",
            "Olive oil is the primary source of fat.",
        ],
        source=source,
    )


class TestResearcherAgentHappyPath:
    @pytest.mark.asyncio
    async def test_returns_research_summary_with_bullet_points(self):
        summary = _make_research_summary()
        llm = make_mock_llm(summary)

        result = await research_node(
            {"blog_spec": _make_blog_spec()},
            config=make_config(llm),
        )
        assert isinstance(result["research_summary"], ResearchSummary)
        assert len(result["research_summary"].bullet_points) > 0

    @pytest.mark.asyncio
    async def test_includes_source_indicator_in_output(self):
        summary = _make_research_summary(source="knowledge_base")
        llm = make_mock_llm(summary)

        result = await research_node(
            {"blog_spec": _make_blog_spec()},
            config=make_config(llm),
        )
        assert result["research_summary"].source == "knowledge_base"


class TestResearcherAgentTopicNotFound:
    @pytest.mark.asyncio
    async def test_handles_unknown_topic(self):
        summary = _make_research_summary(
            topic="quantum physics",
            bullet_points=["No information was found on this topic."],
            source="none",
        )
        llm = make_mock_llm(summary)

        result = await research_node(
            {"blog_spec": _make_blog_spec(topic="quantum physics")},
            config=make_config(llm),
        )
        assert isinstance(result["research_summary"], ResearchSummary)
        assert result["research_summary"].source == "none"
        assert "No information" in result["research_summary"].bullet_points[0]


class TestResearcherAgentFailure:
    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self):
        llm = make_failing_llm(ConnectionError("down"))

        with pytest.raises(RuntimeError, match="Researcher failed"):
            await research_node(
                {"blog_spec": _make_blog_spec()},
                config=make_config(llm),
            )

    @pytest.mark.asyncio
    async def test_raises_on_missing_blog_spec(self):
        llm = make_mock_llm(
            ResearchSummary(topic="AI", bullet_points=["p1"], source="kb")
        )

        with pytest.raises(ValueError, match="no blog spec"):
            await research_node(
                {"blog_spec": None},
                config=make_config(llm),
            )
