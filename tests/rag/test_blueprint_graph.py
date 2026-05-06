"""Tests for rag/blueprint_graph.py — blueprint ingestion."""

import json

import pytest

from blog_mas.rag.blueprint_graph import run_blueprint_ingestion
from tests.conftest import FakeEmbedder, FakeVectorStore


def _valid_blueprint(bp_id="bp-test", description="A test blueprint."):
    return json.dumps({
        "id": bp_id,
        "description": description,
        "scene_goal": "Test scene goal.",
        "style_guide": "Clear and concise.",
        "participants": [],
        "instruction": "Write a well-structured blog post.",
        "metadata": {"source": "test"},
    })


class TestBlueprintIngestion:
    def test_validates_and_upserts_blueprints(self, tmp_path):
        (tmp_path / "bp1.json").write_text(_valid_blueprint("bp1", "First blueprint."))
        (tmp_path / "bp2.json").write_text(_valid_blueprint("bp2", "Second blueprint."))

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        run_blueprint_ingestion(str(tmp_path), "blueprints", store, embedder)

        points = store._data["blueprints"]
        assert len(points) == 2
        # Each payload must contain the full blueprint_json
        for pt in points:
            assert "blueprint_json" in pt["payload"]

    def test_description_only_embedding(self, tmp_path):
        (tmp_path / "bp.json").write_text(_valid_blueprint("bp1", "Short desc."))

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        run_blueprint_ingestion(str(tmp_path), "blueprints", store, embedder)

        # The vector should come from embedding just the description, not the full JSON.
        # FakeEmbedder uses hash-based vectors, so we can verify the input was the description.
        expected_vec = embedder.embed_batch(["Short desc."])[0]
        points = store._data["blueprints"]
        assert points[0]["vector"] == expected_vec

    def test_blueprint_json_is_original_string(self, tmp_path):
        original = _valid_blueprint()
        (tmp_path / "bp.json").write_text(original)

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        run_blueprint_ingestion(str(tmp_path), "blueprints", store, embedder)

        points = store._data["blueprints"]
        assert points[0]["payload"]["blueprint_json"] == original


class TestValidationHandling:
    def test_skips_invalid_blueprint(self, tmp_path):
        valid = _valid_blueprint("bp-valid", "Valid one.")
        invalid = json.dumps({"bad": "data"})
        (tmp_path / "good.json").write_text(valid)
        (tmp_path / "bad.json").write_text(invalid)

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        run_blueprint_ingestion(str(tmp_path), "blueprints", store, embedder)

        points = store._data["blueprints"]
        assert len(points) == 1
        assert points[0]["payload"]["blueprint_id"] == "bp-valid"


class TestIdempotency:
    def test_rerun_is_noop(self, tmp_path):
        (tmp_path / "bp.json").write_text(_valid_blueprint())

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        run_blueprint_ingestion(str(tmp_path), "blueprints", store, embedder)
        first = len(store._data["blueprints"])

        run_blueprint_ingestion(str(tmp_path), "blueprints", store, embedder)
        second = len(store._data["blueprints"])

        assert first == second
