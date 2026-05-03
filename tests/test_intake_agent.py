"""Tests for the Intake agent (LangGraph node with LCEL chain)."""

import pytest

from blog_mas.agents.intake import intake_node
from blog_mas.mcp.models import BlogSpec
from tests.conftest import make_config, make_failing_llm, make_mock_llm


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

        with pytest.raises(ConnectionError, match="timeout"):
            await intake_node(
                {"raw_input": "Write about AI"},
                config=make_config(llm),
            )
