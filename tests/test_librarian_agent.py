"""Tests for the Librarian agent — blueprint retrieval + validation."""

import json
import logging

import pytest

from blog_mas.agents.librarian import librarian_node
from blog_mas.rag.blueprints import NEUTRAL_BLUEPRINT
from tests.conftest import FakeEmbedder, FakeReranker, FakeVectorStore


def _valid_bp_json(bp_id="bp-match", description="A matched blueprint."):
    return json.dumps({
        "id": bp_id,
        "description": description,
        "scene_goal": "Test scene goal.",
        "style_guide": "Clear and concise.",
        "participants": [],
        "instruction": "Write a well-structured blog post.",
        "metadata": {"source": "test"},
    })


def _seed_blueprints(store, items):
    """Seed the blueprints namespace. items = list of (id, vector, payload)."""
    store.ensure_collection("blueprints", dim=4)
    for id_, vec, payload in items:
        store.upsert_points("blueprints", [{"id": id_, "vector": vec, "payload": payload}])


def _make_state(intent_query="test query", **overrides):
    state = {"intent_query": intent_query}
    state.update(overrides)
    return state


def _make_config(store, embedder, reranker):
    return {"configurable": {"store": store, "embedder": embedder, "reranker": reranker}}


class TestLibrarianHappyPath:
    @pytest.mark.asyncio
    async def test_valid_blueprint_above_threshold_writes_to_state(self):
        store = FakeVectorStore(dim=4)
        bp_json = _valid_bp_json()
        _seed_blueprints(store, [
            ("bp1", [0.5] * 4, {"blueprint_json": bp_json}),
        ])

        config = _make_config(store, FakeEmbedder(dim=4), FakeReranker())
        result = await librarian_node(_make_state(), config)

        assert result["blueprint"] is not None
        assert result["blueprint"].id == "bp-match"
        assert result["blueprint_match_score"] is not None
        assert result["blueprint_fallback_reason"] is None


class TestLibrarianFallbacks:
    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_neutral(self):
        store = FakeVectorStore(dim=4)
        store.ensure_collection("blueprints", dim=4)

        config = _make_config(store, FakeEmbedder(dim=4), FakeReranker())
        result = await librarian_node(_make_state(), config)

        assert result["blueprint"] == NEUTRAL_BLUEPRINT

    @pytest.mark.asyncio
    async def test_missing_blueprint_json_returns_neutral(self):
        store = FakeVectorStore(dim=4)
        _seed_blueprints(store, [
            ("bp1", [0.5] * 4, {"description": "no json field"}),
        ])

        config = _make_config(store, FakeEmbedder(dim=4), FakeReranker())
        result = await librarian_node(_make_state(), config)

        assert result["blueprint"] == NEUTRAL_BLUEPRINT
        assert result["blueprint_fallback_reason"] == "missing_payload"

    @pytest.mark.asyncio
    async def test_malformed_json_returns_neutral(self):
        store = FakeVectorStore(dim=4)
        _seed_blueprints(store, [
            ("bp1", [0.5] * 4, {"blueprint_json": "{bad json"}),
        ])

        config = _make_config(store, FakeEmbedder(dim=4), FakeReranker())
        result = await librarian_node(_make_state(), config)

        assert result["blueprint"] == NEUTRAL_BLUEPRINT
        assert result["blueprint_fallback_reason"] == "schema_violation"

    @pytest.mark.asyncio
    async def test_injection_marker_returns_neutral(self, caplog):
        injected = _valid_bp_json()
        data = json.loads(injected)
        data["instruction"] = "Write with {{template}}"
        bad_json = json.dumps(data)

        store = FakeVectorStore(dim=4)
        _seed_blueprints(store, [
            ("bp1", [0.5] * 4, {"blueprint_json": bad_json}),
        ])

        config = _make_config(store, FakeEmbedder(dim=4), FakeReranker())
        with caplog.at_level(logging.WARNING, logger="blog_mas.agents.librarian"):
            result = await librarian_node(_make_state(), config)

        assert result["blueprint"] == NEUTRAL_BLUEPRINT
        assert "security" in caplog.text.lower() or "injection" in caplog.text.lower()
