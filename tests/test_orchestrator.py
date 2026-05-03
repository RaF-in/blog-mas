"""Tests for the Orchestrator (LangGraph StateGraph with revision loop)."""

import pytest

from blog_mas.mcp.models import (
    BlogDraft,
    BlogSpec,
    ResearchSummary,
    ValidationVerdict,
)
from blog_mas.orchestrator import MAX_REVISIONS, build_graph, should_continue
from tests.conftest import make_failing_llm, make_mock_llm_sequence


def _spec():
    return BlogSpec(
        topic="Mediterranean diet",
        audience="general readers",
        tone="informative",
        goal="educate",
        constraints=[],
    )


def _research_summary():
    return ResearchSummary(
        topic="Mediterranean diet",
        bullet_points=["Rich in omega-3 fatty acids", "Reduces heart disease risk"],
        source="knowledge_base",
    )


def _draft():
    return BlogDraft(title="Mediterranean Diet Guide", body="A great diet.", word_count=3)


_INITIAL_STATE = {
    "raw_input": "Write about Mediterranean diet",
    "blog_spec": None,
    "research_summary": None,
    "draft": None,
    "verdict": None,
    "revision_feedback": None,
    "revision_count": 0,
    "error": None,
}


class TestShouldContinue:
    def test_returns_end_on_pass(self):
        state = {"verdict": ValidationVerdict(verdict="pass", reason="ok")}
        assert should_continue(state) == "end"

    def test_returns_retry_on_fail_under_max(self):
        state = {
            "verdict": ValidationVerdict(verdict="fail", reason="bad"),
            "revision_count": 1,
        }
        assert should_continue(state) == "retry"

    def test_returns_end_at_max_revisions(self):
        state = {
            "verdict": ValidationVerdict(verdict="fail", reason="bad"),
            "revision_count": MAX_REVISIONS,
        }
        assert should_continue(state) == "end"

    def test_returns_end_when_no_verdict(self):
        assert should_continue({}) == "end"


class TestOrchestratorHappyPath:
    @pytest.mark.asyncio
    async def test_runs_complete_pipeline_and_returns_blog_draft(self):
        llm = make_mock_llm_sequence([
            _spec(),                           # intake → BlogSpec
            _research_summary(),               # research → ResearchSummary
            _draft(),                          # write → BlogDraft
            ValidationVerdict(verdict="pass", reason="All claims supported"),
        ])

        graph = build_graph()
        result = await graph.ainvoke(
            _INITIAL_STATE,
            config={"configurable": {"llm": llm}},
        )
        assert result["draft"].title == "Mediterranean Diet Guide"
        assert result["verdict"].verdict == "pass"


class TestRevisionLoop:
    @pytest.mark.asyncio
    async def test_retries_on_validation_fail_then_passes(self):
        revised_draft = BlogDraft(title="Fixed", body="Fixed body.", word_count=2)
        llm = make_mock_llm_sequence([
            _spec(),                                        # intake
            _research_summary(),                            # research
            _draft(),                                       # write (1st)
            ValidationVerdict(verdict="fail", reason="unsupported claim about X"),  # validate (1st)
            revised_draft,                                  # write (2nd)
            ValidationVerdict(verdict="pass", reason="ok"), # validate (2nd)
        ])

        graph = build_graph()
        result = await graph.ainvoke(
            _INITIAL_STATE,
            config={"configurable": {"llm": llm}},
        )

        assert result["verdict"].verdict == "pass"
        assert result["draft"].title == "Fixed"

    @pytest.mark.asyncio
    async def test_stops_after_max_revisions(self):
        llm = make_mock_llm_sequence([
            _spec(),                                        # intake
            _research_summary(),                            # research
            _draft(),                                       # write (1st)
            ValidationVerdict(verdict="fail", reason="bad"),  # validate (1st)
            _draft(),                                       # write (2nd)
            ValidationVerdict(verdict="fail", reason="bad"),  # validate (2nd)
            _draft(),                                       # write (3rd)
            ValidationVerdict(verdict="fail", reason="bad"),  # validate (3rd)
        ])

        graph = build_graph()
        result = await graph.ainvoke(
            _INITIAL_STATE,
            config={"configurable": {"llm": llm}},
        )

        assert result["verdict"].verdict == "fail"
        assert result["revision_count"] == MAX_REVISIONS


class TestAgentFailure:
    @pytest.mark.asyncio
    async def test_exits_gracefully_when_agent_fails(self):
        llm = make_failing_llm(ConnectionError("timeout"))

        graph = build_graph()
        with pytest.raises(ConnectionError, match="timeout"):
            await graph.ainvoke(
                _INITIAL_STATE,
                config={"configurable": {"llm": llm}},
            )
