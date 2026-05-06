"""Tests for the Intake agent (LangGraph node with LCEL chain)."""

import pytest

from blog_mas.agents.intake import intake_node
from blog_mas.mcp.models import BlogSpec, GoalDecomposition
from tests.conftest import make_config, make_failing_llm, make_mock_llm, make_mock_llm_sequence


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


class TestIntakeAgentHappyPath:
    @pytest.mark.asyncio
    async def test_decomposes_valid_request_into_blog_spec(self):
        spec = _make_blog_spec(topic="Mediterranean diet", audience="athletes")
        llm = make_mock_llm(spec)

        result = await intake_node(
            {"raw_input": "Write about the Mediterranean diet for athletes"},
            config=make_config(llm),
        )
        assert isinstance(result["blog_spec"], BlogSpec)
        assert "Mediterranean diet" in result["blog_spec"].topic
        assert "athletes" in result["blog_spec"].audience

    @pytest.mark.asyncio
    async def test_applies_defaults_for_unspecified_fields(self):
        spec = _make_blog_spec(topic="AI")
        llm = make_mock_llm(spec)

        result = await intake_node(
            {"raw_input": "Write about AI"},
            config=make_config(llm),
        )
        assert result["blog_spec"].audience == "general readers"
        assert result["blog_spec"].tone == "informative and engaging"
        assert result["blog_spec"].goal == "educate the reader"
        assert result["blog_spec"].constraints == []


class TestIntakeAgentFailure:
    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self):
        llm = make_failing_llm(ConnectionError("timeout"))

        with pytest.raises(RuntimeError, match="Intake failed"):
            await intake_node(
                {"raw_input": "Write about AI"},
                config=make_config(llm),
            )

    @pytest.mark.asyncio
    async def test_raises_on_missing_raw_input(self):
        llm = make_mock_llm(
            BlogSpec(topic="AI", audience="general readers", tone="neutral", goal="educate", constraints=[])
        )

        with pytest.raises(ValueError, match="No raw_input"):
            await intake_node(
                {"raw_input": ""},
                config=make_config(llm),
            )

    @pytest.mark.asyncio
    async def test_raises_on_no_llm(self):
        with pytest.raises(ValueError, match="No LLM configured"):
            await intake_node(
                {"raw_input": "Write about AI"},
                config={"configurable": {}},
            )


class TestGoalDecomposition:
    @pytest.mark.asyncio
    async def test_emits_intent_and_topic_query(self):
        spec = _make_blog_spec()
        goal = GoalDecomposition(intent_query="technical deep dive", topic_query="Mediterranean diet")
        llm = make_mock_llm_sequence([spec, goal])

        result = await intake_node(
            {"raw_input": "Write about the Mediterranean diet"},
            config=make_config(llm),
        )

        assert result["intent_query"] == "technical deep dive"
        assert result["topic_query"] == "Mediterranean diet"

    @pytest.mark.asyncio
    async def test_llm_fails_uses_deterministic_fallback(self):
        spec = _make_blog_spec()
        # First call succeeds (BlogSpec), second and third fail (GoalDecomposition retries)
        llm = make_mock_llm_sequence([spec, ConnectionError("down"), ConnectionError("still down")])

        result = await intake_node(
            {"raw_input": "Write about the Mediterranean diet"},
            config=make_config(llm),
        )

        assert result["intent_query"] == "informative and engaging educate the reader for general readers"
        assert result["topic_query"] == "Mediterranean diet"

    @pytest.mark.asyncio
    async def test_deterministic_fallback_produces_non_empty(self):
        spec = _make_blog_spec()
        llm = make_mock_llm_sequence([spec, ConnectionError("fail"), ConnectionError("fail")])

        result = await intake_node(
            {"raw_input": "Write about AI"},
            config=make_config(llm),
        )

        assert len(result["intent_query"].strip()) > 0
        assert len(result["topic_query"].strip()) > 0
