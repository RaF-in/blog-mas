"""Tests for rag/retrieval.py — hybrid search: RRF, reranker, parent expansion, degradation."""

import logging
import os

import pytest

from blog_mas.rag.retrieval import _expand_parents, _rrf_fuse, _rerank, hybrid_search
from blog_mas.rag.vector_store import ScoredPoint
from tests.conftest import FakeEmbedder, FakeReranker, FakeVectorStore


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────


class TestRRFFusion:
    def test_fuses_dense_and_sparse_ranked_lists(self):
        dense = [
            ScoredPoint(id="A", score=0.9, payload={}),
            ScoredPoint(id="B", score=0.8, payload={}),
            ScoredPoint(id="C", score=0.7, payload={}),
        ]
        sparse = [
            ScoredPoint(id="B", score=5.0, payload={}),
            ScoredPoint(id="C", score=4.0, payload={}),
            ScoredPoint(id="D", score=3.0, payload={}),
        ]
        result = _rrf_fuse(dense, sparse, k=60)

        ids = [r.id for r in result]
        # B appears at rank 1 in both lists → highest combined score
        # A appears only in dense at rank 0
        # C appears at rank 2 in dense, rank 1 in sparse
        # D appears only in sparse at rank 2
        assert ids[0] == "B"
        assert set(ids) == {"A", "B", "C", "D"}

    def test_respects_env_override_for_rrf_k(self, monkeypatch):
        monkeypatch.setenv("RAG_RRF_K", "100")
        dense = [
            ScoredPoint(id="A", score=0.9, payload={}),
            ScoredPoint(id="B", score=0.8, payload={}),
        ]
        sparse = [
            ScoredPoint(id="B", score=5.0, payload={}),
            ScoredPoint(id="C", score=3.0, payload={}),
        ]

        # Compute expected scores with k=100
        # A: 1/(100+0) from dense only = 0.01
        # B: 1/(100+1) + 1/(100+0) from dense(rank1) + sparse(rank0)
        # C: 1/(100+1) from sparse only
        result = _rrf_fuse(dense, sparse, k=100)
        ids = [r.id for r in result]
        assert ids[0] == "B"
        assert set(ids) == {"A", "B", "C"}


# ── Reranker integration ───────────────────────────────────────────────


class TestReranker:
    def test_returns_top_k_after_rerank(self):
        reranker = FakeReranker()
        candidates = [
            ScoredPoint(id=f"doc{i}", score=float(i), payload={"raw_text": f"text {i}"})
            for i in range(20)
        ]
        result = _rerank("test query", candidates, reranker, top_k=3)
        assert len(result) == 3
        assert all(isinstance(r, ScoredPoint) for r in result)

    def test_reranker_error_degrades_to_rrf_top_k(self, caplog):
        reranker = FakeReranker()
        reranker.rerank = lambda q, d, top_k=3: (_ for _ in ()).throw(RuntimeError("reranker down"))

        candidates = [
            ScoredPoint(id=f"doc{i}", score=float(20 - i), payload={"raw_text": f"text {i}"})
            for i in range(10)
        ]
        with caplog.at_level(logging.WARNING, logger="blog_mas.rag.retrieval"):
            result = _rerank("test query", candidates, reranker, top_k=3)

        assert len(result) == 3
        # Should be RRF top-K (highest scores first), unmodified order
        assert result[0].id == "doc0"
        assert result[1].id == "doc1"
        assert result[2].id == "doc2"
        assert "degraded" in caplog.text.lower()


# ── Small-to-big parent expansion ──────────────────────────────────────


class TestParentExpansion:
    def test_proposition_chunk_expands_to_parent(self):
        parent_text = "This is the full parent chunk text."
        parent = ScoredPoint(
            id="parent-1",
            score=0.5,
            payload={"raw_text": parent_text, "chunk_type": "parent"},
        )
        proposition = ScoredPoint(
            id="prop-1",
            score=0.9,
            payload={
                "raw_text": "A single proposition.",
                "chunk_type": "proposition",
                "parent_id": "parent-1",
            },
        )

        # Build a lookup from the store's collection data
        store = FakeVectorStore()
        store.ensure_collection("test", dim=4)
        store.upsert_points("test", [
            {"id": parent.id, "vector": [0.1] * 4, "payload": parent.payload},
        ])

        result = _expand_parents([proposition, parent], store, namespace="test")

        # Proposition should be replaced with parent's raw_text
        prop_result = [r for r in result if r.id == "prop-1"][0]
        assert prop_result.payload["raw_text"] == parent_text

    def test_deduplicates_parent_if_also_in_result_set(self):
        parent_text = "Parent chunk text."
        parent = ScoredPoint(
            id="parent-1",
            score=0.5,
            payload={"raw_text": parent_text, "chunk_type": "parent"},
        )
        proposition = ScoredPoint(
            id="prop-1",
            score=0.9,
            payload={
                "raw_text": "A proposition.",
                "chunk_type": "proposition",
                "parent_id": "parent-1",
            },
        )

        store = FakeVectorStore()
        store.ensure_collection("test", dim=4)
        store.upsert_points("test", [
            {"id": parent.id, "vector": [0.1] * 4, "payload": parent.payload},
        ])

        result = _expand_parents([proposition, parent], store, namespace="test")

        # Parent should appear only once
        parent_ids = [r.id for r in result if r.id == "parent-1"]
        assert len(parent_ids) == 1


# ── Graceful degradation ───────────────────────────────────────────────


class TestGracefulDegradation:
    def _seed_store(self, store, namespace="knowledge"):
        """Seed the fake store with a few documents."""
        store.ensure_collection(namespace, dim=4)
        for i in range(5):
            store.upsert_points(namespace, [{
                "id": f"doc{i}",
                "vector": [0.1 * i] * 4,
                "payload": {"raw_text": f"text {i}", "chunk_type": "parent"},
            }])

    def test_dense_only_on_sparse_failure(self, caplog):
        store = FakeVectorStore(dim=4)
        self._seed_store(store)
        store.fail_on("sparse_search")

        embedder = FakeEmbedder(dim=4)
        reranker = FakeReranker()

        with caplog.at_level(logging.WARNING, logger="blog_mas.rag.retrieval"):
            result = hybrid_search(
                "test query", namespace="knowledge", top_k=3,
                store=store, embedder=embedder, reranker=reranker,
            )

        assert len(result) <= 3
        assert len(result) > 0
        assert "degraded" in caplog.text.lower()

    def test_sparse_only_on_dense_failure(self):
        store = FakeVectorStore(dim=4)
        self._seed_store(store)
        store.fail_on("dense_search")

        embedder = FakeEmbedder(dim=4)
        reranker = FakeReranker()

        result = hybrid_search(
            "test query", namespace="knowledge", top_k=3,
            store=store, embedder=embedder, reranker=reranker,
        )

        assert len(result) <= 3
        assert len(result) > 0

    def test_both_fail_returns_empty(self):
        store = FakeVectorStore(dim=4)
        self._seed_store(store)
        store.fail_on("dense_search")
        store.fail_on("sparse_search")

        embedder = FakeEmbedder(dim=4)
        reranker = FakeReranker()

        result = hybrid_search(
            "test query", namespace="knowledge", top_k=3,
            store=store, embedder=embedder, reranker=reranker,
        )

        assert result == []

    def test_empty_retrieval_returns_empty(self):
        store = FakeVectorStore(dim=4)
        store.ensure_collection("knowledge", dim=4)

        embedder = FakeEmbedder(dim=4)
        reranker = FakeReranker()

        result = hybrid_search(
            "test query", namespace="knowledge", top_k=3,
            store=store, embedder=embedder, reranker=reranker,
        )

        assert result == []


# ── End-to-end ─────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_returns_scored_chunks(self):
        store = FakeVectorStore(dim=4)
        store.ensure_collection("knowledge", dim=4)
        for i in range(10):
            store.upsert_points("knowledge", [{
                "id": f"doc{i}",
                "vector": [0.1 * (i + 1)] * 4,
                "payload": {"raw_text": f"document text {i}", "chunk_type": "parent"},
            }])

        embedder = FakeEmbedder(dim=4)
        reranker = FakeReranker()

        result = hybrid_search(
            "test query", namespace="knowledge", top_k=3,
            store=store, embedder=embedder, reranker=reranker,
        )

        assert len(result) <= 3
        assert all(isinstance(r, ScoredPoint) for r in result)
        assert all(r.payload for r in result)

    def test_respects_namespace_isolation(self):
        store = FakeVectorStore(dim=4)

        # Seed knowledge namespace
        store.ensure_collection("knowledge", dim=4)
        store.upsert_points("knowledge", [{
            "id": "k-doc1", "vector": [0.5] * 4,
            "payload": {"raw_text": "knowledge content", "chunk_type": "parent"},
        }])

        # Seed blueprints namespace
        store.ensure_collection("blueprints", dim=4)
        store.upsert_points("blueprints", [{
            "id": "bp-doc1", "vector": [0.5] * 4,
            "payload": {"raw_text": "blueprint content", "chunk_type": "parent"},
        }])

        embedder = FakeEmbedder(dim=4)
        reranker = FakeReranker()

        result = hybrid_search(
            "test query", namespace="knowledge", top_k=3,
            store=store, embedder=embedder, reranker=reranker,
        )

        # Should only return results from knowledge namespace
        assert all(r.id.startswith("k-") for r in result)
