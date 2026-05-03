"""Tests for the Validator agent (LangGraph node with LCEL chain)."""

import pytest

from blog_mas.agents.validator import validate_node
from blog_mas.mcp.models import (
    BlogDraft,
    ResearchSummary,
    ValidationVerdict,
)
from tests.conftest import make_config, make_failing_llm, make_mock_llm


def _make_research_summary():
    return ResearchSummary(
        topic="Python",
        bullet_points=["Python is great.", "Python is widely used."],
        source="knowledge_base",
    )


def _make_draft(body="Python is great."):
    return BlogDraft(title="All About Python", body=body, word_count=4)


class TestValidatorPass:
    @pytest.mark.asyncio
    async def test_returns_pass_verdict_when_draft_matches_research(self):
        verdict = ValidationVerdict(verdict="pass", reason="All claims are supported.")
        llm = make_mock_llm(verdict)

        result = await validate_node(
            {
                "research_summary": _make_research_summary(),
                "draft": _make_draft(),
            },
            config=make_config(llm),
        )
        assert result["verdict"].verdict == "pass"
        assert "supported" in result["verdict"].reason.lower()

    @pytest.mark.asyncio
    async def test_does_not_set_revision_feedback_on_pass(self):
        verdict = ValidationVerdict(verdict="pass", reason="All good.")
        llm = make_mock_llm(verdict)

        result = await validate_node(
            {
                "research_summary": _make_research_summary(),
                "draft": _make_draft(),
            },
            config=make_config(llm),
        )
        assert "revision_feedback" not in result
        assert "revision_count" not in result


class TestValidatorFail:
    @pytest.mark.asyncio
    async def test_returns_fail_verdict_with_reason(self):
        verdict = ValidationVerdict(
            verdict="fail",
            reason="Draft claims 'Python was invented in 1990' but research does not mention this.",
        )
        llm = make_mock_llm(verdict)

        result = await validate_node(
            {
                "research_summary": _make_research_summary(),
                "draft": _make_draft(body="Python was invented in 1990."),
            },
            config=make_config(llm),
        )
        assert result["verdict"].verdict == "fail"
        assert "1990" in result["verdict"].reason

    @pytest.mark.asyncio
    async def test_sets_revision_feedback_and_count_on_fail(self):
        verdict = ValidationVerdict(
            verdict="fail",
            reason="unsupported claim about X",
        )
        llm = make_mock_llm(verdict)

        result = await validate_node(
            {
                "research_summary": _make_research_summary(),
                "draft": _make_draft(),
            },
            config=make_config(llm),
        )
        assert result["revision_feedback"] == "unsupported claim about X"
        assert result["revision_count"] == 1


class TestValidatorFailure:
    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self):
        llm = make_failing_llm(ConnectionError("down"))

        with pytest.raises(ConnectionError, match="down"):
            await validate_node(
                {
                    "research_summary": _make_research_summary(),
                    "draft": _make_draft(),
                },
                config=make_config(llm),
            )
