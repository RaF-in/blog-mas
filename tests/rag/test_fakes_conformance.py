"""Conformance tests: FakeVectorStore, FakeEmbedder, FakeReranker, fixtures."""

import inspect
import json

import pytest

from blog_mas.rag.blueprints import validate_blueprint_payload
from blog_mas.rag.vector_store import QdrantStore


# ── FakeVectorStore conformance ────────────────────────────────────────


class TestFakeVectorStoreConformance:
    def test_upsert_and_dense_search_round_trips(self):
        from tests.conftest import FakeVectorStore

        fvs = FakeVectorStore(dim=4)
        fvs.ensure_collection("test", dim=4)
        fvs.upsert_points("test", [
            {"id": "p1", "vector": [1, 0, 0, 0], "payload": {"x": 1}},
            {"id": "p2", "vector": [0, 1, 0, 0], "payload": {"x": 2}},
        ])
        results = fvs.dense_search("test", [1, 0, 0, 0], top_n=2)
        assert len(results) == 2

    def test_respects_namespace_isolation(self):
        from tests.conftest import FakeVectorStore

        fvs = FakeVectorStore(dim=4)
        fvs.ensure_collection("knowledge", dim=4)
        fvs.ensure_collection("blueprints", dim=4)
        fvs.upsert_points("knowledge", [{"id": "k1", "vector": [1, 0, 0, 0], "payload": {}}])
        fvs.upsert_points("blueprints", [{"id": "b1", "vector": [1, 0, 0, 0], "payload": {}}])

        results = fvs.dense_search("knowledge", [1, 0, 0, 0])
        ids = [r.id for r in results]
        assert "b1" not in ids

    def test_mirrors_qdrant_store_public_methods(self):
        from tests.conftest import FakeVectorStore

        fake_methods = {name for name, _ in inspect.getmembers(FakeVectorStore, predicate=inspect.isfunction)
                        if not name.startswith("_")}
        real_methods = {name for name, _ in inspect.getmembers(QdrantStore, predicate=inspect.isfunction)
                        if not name.startswith("_")}
        # content_hash is a static utility, not an instance method the fake needs
        instance_methods = real_methods - {"content_hash"}
        assert fake_methods >= instance_methods, (
            f"FakeVectorStore missing methods: {instance_methods - fake_methods}"
        )

    def test_configurable_failure_injection(self):
        from tests.conftest import FakeVectorStore

        fvs = FakeVectorStore()
        fvs.fail_on("dense_search")
        with pytest.raises(RuntimeError, match="dense_search"):
            fvs.dense_search("test", [1.0])


# ── FakeEmbedder conformance ───────────────────────────────────────────


class TestFakeEmbedderConformance:
    def test_deterministic_vectors_per_text(self):
        from tests.conftest import FakeEmbedder

        emb = FakeEmbedder(dim=8)
        v1 = emb.embed_batch(["hello"])
        v2 = emb.embed_batch(["hello"])
        assert v1 == v2

    def test_batch_matches_single_item(self):
        from tests.conftest import FakeEmbedder

        emb = FakeEmbedder(dim=8)
        batch = emb.embed_batch(["a", "b", "c"])
        single_a = emb.embed_batch(["a"])[0]
        single_b = emb.embed_batch(["b"])[0]
        single_c = emb.embed_batch(["c"])[0]
        assert batch[0] == single_a
        assert batch[1] == single_b
        assert batch[2] == single_c

    def test_configurable_dim(self):
        from tests.conftest import FakeEmbedder

        emb = FakeEmbedder(dim=16)
        result = emb.embed_batch(["test"])
        assert len(result[0]) == 16


# ── FakeReranker conformance ───────────────────────────────────────────


class TestFakeRerankerConformance:
    def test_deterministic_ordering(self):
        from tests.conftest import FakeReranker

        rr = FakeReranker()
        docs = [{"raw_text": "short"}, {"raw_text": "longer text here"}, {"raw_text": "mid"}]
        r1 = rr.rerank("query", docs)
        r2 = rr.rerank("query", docs)
        assert r1 == r2

    def test_preserves_top_k(self):
        from tests.conftest import FakeReranker

        rr = FakeReranker()
        docs = [{"raw_text": f"doc{i}"} for i in range(5)]
        results = rr.rerank("query", docs, top_k=3)
        assert len(results) == 3

    def test_soft_fails_to_identity_when_degraded(self):
        from tests.conftest import FakeReranker

        rr = FakeReranker()
        rr.degraded = True
        docs = [{"raw_text": f"doc{i}"} for i in range(5)]
        results = rr.rerank("query", docs, top_k=3)
        assert results == docs[:3]


# ── Shared RAG fixtures ────────────────────────────────────────────────


class TestSharedFixtures:
    def test_sample_markdown_doc(self, sample_markdown_doc):
        assert "# Introduction" in sample_markdown_doc
        assert "# Main Topic" in sample_markdown_doc
        assert sample_markdown_doc.count("## ") >= 3

    def test_sample_blueprint_payload_validates(self, sample_blueprint_payload):
        result = validate_blueprint_payload(sample_blueprint_payload)
        assert result is not None
        assert result.id == "bp-test-fixture"

    def test_make_chunk_produces_content_hash_id(self, make_chunk):
        chunk = make_chunk(doc_id="d1", chunk_index=0, raw_text="hello")
        assert "id" in chunk
        assert len(chunk["id"]) == 64  # sha256 hex
        assert chunk["raw_text"] == "hello"
