"""Tests for the Writer agent — blueprint injection + draft generation."""

import json

import pytest

from blog_mas.agents.writer import write_node, _blueprint_scaffold
from blog_mas.mcp.models import BlogDraft, BlogSpec, ResearchSummary
from blog_mas.rag.blueprints import Blueprint, NEUTRAL_BLUEPRINT
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
        ],
        source=source,
    )


def _make_blog_spec(topic="Mediterranean diet", **kw):
    return BlogSpec(
        topic=topic, audience=kw.get("audience", "general readers"),
        tone=kw.get("tone", "informative"), goal=kw.get("goal", "educate"),
        constraints=kw.get("constraints", []),
    )


def _make_draft(title="The Wonders of the Mediterranean Diet", body="Olive oil is key.", word_count=5):
    return BlogDraft(title=title, body=body, word_count=word_count)


def _base_state(**overrides):
    state = {
        "blog_spec": _make_blog_spec(),
        "research_summary": _make_research_summary(),
        "blueprint": NEUTRAL_BLUEPRINT,
        "revision_feedback": None,
    }
    state.update(overrides)
    return state


class TestWriterFirstPass:
    @pytest.mark.asyncio
    async def test_generates_blog_draft(self):
        result = await write_node(_base_state(), config=make_config(make_mock_llm(_make_draft())))
        assert isinstance(result["draft"], BlogDraft)
        assert result["draft"].word_count > 0


class TestBlueprintInjection:
    def test_scaffold_contains_canonical_json(self):
        bp = Blueprint(
            id="bp-test", description="Test bp.", scene_goal="Goal.",
            style_guide="Style.", participants=[], instruction="Write well.",
        )
        scaffold = _blueprint_scaffold(bp)
        assert "--- SEMANTIC BLUEPRINT (JSON) ---" in scaffold
        assert "--- END SEMANTIC BLUEPRINT ---" in scaffold
        # The scaffold must contain the canonical re-serialization, not the raw string
        assert json.loads(scaffold.split("\n", 1)[1].rsplit("\n", 1)[0])["id"] == "bp-test"

    @pytest.mark.asyncio
    async def test_no_blueprint_raises(self):
        with pytest.raises(ValueError, match="no blueprint"):
            await write_node(
                _base_state(blueprint=None),
                config=make_config(make_mock_llm(_make_draft())),
            )


class TestWriterRevisionPass:
    @pytest.mark.asyncio
    async def test_incorporates_feedback_with_blueprint(self):
        result = await write_node(
            _base_state(revision_feedback="Remove unsupported claim about X"),
            config=make_config(make_mock_llm(_make_draft())),
        )
        assert isinstance(result["draft"], BlogDraft)


class TestWriterFailure:
    @pytest.mark.asyncio
    async def test_raises_on_llm_failure(self):
        with pytest.raises(RuntimeError, match="Writer failed"):
            await write_node(_base_state(), config=make_config(make_failing_llm(ConnectionError("down"))))

    @pytest.mark.asyncio
    async def test_raises_on_missing_blog_spec(self):
        with pytest.raises(ValueError, match="no blog spec"):
            await write_node(
                _base_state(blog_spec=None),
                config=make_config(make_mock_llm(_make_draft())),
            )

    @pytest.mark.asyncio
    async def test_raises_on_missing_research_summary(self):
        with pytest.raises(ValueError, match="no research summary"):
            await write_node(
                _base_state(research_summary=None),
                config=make_config(make_mock_llm(_make_draft())),
            )
