"""Tests for the Orchestrator — parallel wiring + revision loop."""

import pytest

from blog_mas.mcp.models import (
    BlogDraft,
    BlogSpec,
    GoalDecomposition,
    ResearchSummary,
    ValidationVerdict,
)
from blog_mas.rag.blueprints import NEUTRAL_BLUEPRINT
from blog_mas.orchestrator import (
    MAX_REVISIONS,
    build_graph,
    run_pipeline_async,
    should_continue,
)
from tests.conftest import (
    FakeEmbedder,
    FakeReranker,
    FakeVectorStore,
    make_failing_llm,
    make_mock_llm_sequence,
)


def _spec():
    return BlogSpec(topic="Mediterranean diet", audience="general readers", tone="informative", goal="educate", constraints=[])


def _research_summary():
    return ResearchSummary(topic="Mediterranean diet", bullet_points=["Rich in omega-3"], source="knowledge_base")


def _draft():
    return BlogDraft(title="Mediterranean Diet Guide", body="A great diet.", word_count=3)


def _goal():
    return GoalDecomposition(intent_query="informative educate for general readers", topic_query="Mediterranean diet")


_INITIAL_STATE = {
    "raw_input": "Write about Mediterranean diet",
    "blog_spec": None,
    "research_summary": None,
    "draft": None,
    "verdict": None,
    "revision_feedback": None,
    "revision_count": 0,
    "intent_query": None,
    "topic_query": None,
    "blueprint": NEUTRAL_BLUEPRINT,
    "blueprint_match_score": None,
    "blueprint_alternatives": None,
    "blueprint_fallback_reason": None,
}


def _retrieval_config(llm):
    store = FakeVectorStore(dim=4)
    store.ensure_collection("blueprints", dim=4)
    store.ensure_collection("knowledge", dim=4)
    return {"configurable": {"llm": llm, "store": store, "embedder": FakeEmbedder(dim=4), "reranker": FakeReranker()}}


class TestShouldContinue:
    def test_returns_end_on_pass(self):
        assert should_continue({"verdict": ValidationVerdict(verdict="pass", reason="ok")}) == "end"

    def test_returns_retry_on_fail_under_max(self):
        assert should_continue({"verdict": ValidationVerdict(verdict="fail", reason="bad"), "revision_count": 1}) == "retry"

    def test_returns_end_at_max_revisions(self):
        assert should_continue({"verdict": ValidationVerdict(verdict="fail", reason="bad"), "revision_count": MAX_REVISIONS}) == "end"

    def test_returns_end_when_no_verdict(self):
        assert should_continue({}) == "end"


class TestGraphStructure:
    def test_librarian_node_exists(self):
        graph = build_graph()
        assert "librarian" in graph.get_graph().nodes

    def test_parallel_branches_from_intake(self):
        graph = build_graph()
        g = graph.get_graph()
        edges = [(e.source, e.target) for e in g.edges]
        assert ("intake", "librarian") in edges
        assert ("intake", "research") in edges
        assert ("librarian", "write") in edges
        assert ("research", "write") in edges


class TestOrchestratorHappyPath:
    @pytest.mark.asyncio
    async def test_runs_complete_pipeline(self):
        llm = make_mock_llm_sequence([
            _spec(), _goal(),
            _research_summary(),
            _draft(),
            ValidationVerdict(verdict="pass", reason="All claims supported"),
        ])
        graph = build_graph()
        result = await graph.ainvoke(_INITIAL_STATE, config=_retrieval_config(llm))
        assert result["draft"].title == "Mediterranean Diet Guide"
        assert result["verdict"].verdict == "pass"


class TestRevisionLoop:
    @pytest.mark.asyncio
    async def test_retries_on_fail_then_passes(self):
        revised = BlogDraft(title="Fixed", body="Fixed body.", word_count=2)
        llm = make_mock_llm_sequence([
            _spec(), _goal(),
            _research_summary(),
            _draft(),
            ValidationVerdict(verdict="fail", reason="bad"),
            revised,
            ValidationVerdict(verdict="pass", reason="ok"),
        ])
        graph = build_graph()
        result = await graph.ainvoke(_INITIAL_STATE, config=_retrieval_config(llm))
        assert result["verdict"].verdict == "pass"
        assert result["draft"].title == "Fixed"

    @pytest.mark.asyncio
    async def test_stops_after_max_revisions(self):
        llm = make_mock_llm_sequence([
            _spec(), _goal(),
            _research_summary(),
            _draft(),
            ValidationVerdict(verdict="fail", reason="bad"),
            _draft(),
            ValidationVerdict(verdict="fail", reason="bad"),
            _draft(),
            ValidationVerdict(verdict="fail", reason="bad"),
        ])
        graph = build_graph()
        result = await graph.ainvoke(_INITIAL_STATE, config=_retrieval_config(llm))
        assert result["verdict"].verdict == "fail"
        assert result["revision_count"] == MAX_REVISIONS


class TestAgentFailure:
    @pytest.mark.asyncio
    async def test_exits_gracefully_when_agent_fails(self):
        llm = make_failing_llm(ConnectionError("timeout"))
        graph = build_graph()
        with pytest.raises(RuntimeError, match="Intake failed"):
            await graph.ainvoke(_INITIAL_STATE, config=_retrieval_config(llm))


class TestRunPipelineAsync:
    @pytest.mark.asyncio
    async def test_success_path(self):
        llm = make_mock_llm_sequence([
            _spec(), _goal(), _research_summary(), _draft(),
            ValidationVerdict(verdict="pass", reason="All good"),
        ])
        result = await run_pipeline_async(raw_input="Write about Mediterranean diet", llm=llm)
        assert result["success"] is True
        assert result["draft"].title == "Mediterranean Diet Guide"

    @pytest.mark.asyncio
    async def test_empty_input_returns_error(self):
        result = await run_pipeline_async(raw_input="   ", llm=None)
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_none_input_returns_error(self):
        result = await run_pipeline_async(raw_input=None, blog_spec=None, llm=None)
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_value_error_returns_configuration_error(self):
        result = await run_pipeline_async(raw_input="test", llm=None)
        assert result["success"] is False
        assert result["error"] == "Pipeline configuration error"

    @pytest.mark.asyncio
    async def test_runtime_error_from_retry_exhaustion(self):
        llm = make_failing_llm(ConnectionError("network down"))
        result = await run_pipeline_async(raw_input="Write about Mediterranean diet", llm=llm)
        assert result["success"] is False
        assert "LLM call failed" in result["error"]

    @pytest.mark.asyncio
    async def test_generic_exception_returns_unexpected_error(self):
        from unittest.mock import AsyncMock, patch
        with patch("blog_mas.orchestrator.build_graph") as mock_build:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.side_effect = TypeError("unexpected")
            mock_build.return_value = mock_graph
            result = await run_pipeline_async(raw_input="test", llm=None)
            assert result["success"] is False
            assert result["error"] == "Unexpected error"
