"""Tests for rag/ingestion_graph.py — knowledge ingestion LangGraph."""

import json
import os
import tempfile

import pytest

from blog_mas.rag.chunking.types import Chunk
from blog_mas.rag.ingestion_graph import run_ingestion
from blog_mas.rag.vector_store import ScoredPoint
from tests.conftest import FakeEmbedder, FakeVectorStore, make_mock_llm


def _mock_propositions_response():
    """JSON response the mock LLM returns for proposition extraction."""
    return {"propositions": ["Fact one.", "Fact two."]}


def _mock_context_response():
    """Plain text response the mock LLM returns for contextualization."""
    return "This chunk discusses testing."


def _make_llm():
    """Mock LLM that alternates between proposition and context responses."""
    from tests.conftest import make_mock_llm_sequence
    # Each chunk needs: 1 proposition call + 1 contextual call
    # With 1 parent + 2 propositions = 3 chunks, so 3 contextual calls
    # Plus 1 proposition extraction call
    return make_mock_llm_sequence([
        _mock_propositions_response(),  # proposition extraction
        _mock_context_response(),       # contextualize parent
        _mock_context_response(),       # contextualize prop 1
        _mock_context_response(),       # contextualize prop 2
    ])


class TestIngestionEndToEnd:
    def test_ingests_markdown_doc_into_store(self, tmp_path):
        md_file = tmp_path / "doc1.md"
        md_file.write_text("# Intro\n\nSome intro text.\n\n## Details\n\nDetail content.\n")

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)
        llm = _make_llm()

        run_ingestion(
            source_dir=str(tmp_path),
            namespace="knowledge",
            store=store,
            embedder=embedder,
            llm=llm,
        )

        points = store._data["knowledge"]
        assert len(points) > 0

        # Check payload fields exist
        for pt in points:
            payload = pt["payload"]
            assert "raw_text" in payload
            assert "doc_id" in payload
            assert "chunk_type" in payload
            assert "content_hash" in payload

    def test_parent_and_proposition_chunks_both_present(self, tmp_path):
        md_file = tmp_path / "doc1.md"
        md_file.write_text("# Topic\n\nEnough text to form a parent chunk with sufficient tokens.\n")

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)
        llm = _make_llm()

        run_ingestion(
            source_dir=str(tmp_path),
            namespace="knowledge",
            store=store,
            embedder=embedder,
            llm=llm,
        )

        points = store._data["knowledge"]
        types = {pt["payload"]["chunk_type"] for pt in points}
        assert "parent" in types
        assert "proposition" in types


class TestIdempotency:
    def test_second_run_produces_no_new_points(self, tmp_path):
        md_file = tmp_path / "doc1.md"
        md_file.write_text("# Topic\n\nSome content here.\n")

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)
        llm = _make_llm()

        run_ingestion(str(tmp_path), "knowledge", store, embedder, llm)
        count_first = len(store._data["knowledge"])

        # Re-create LLM for second run (iterator exhausted)
        llm2 = _make_llm()
        run_ingestion(str(tmp_path), "knowledge", store, embedder, llm2)
        count_second = len(store._data["knowledge"])

        assert count_first == count_second


class TestGracefulDegradation:
    def test_proposition_failure_parent_still_indexed(self, tmp_path):
        md_file = tmp_path / "doc1.md"
        md_file.write_text("# Topic\n\nEnough content for a parent chunk.\n")

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        # LLM returns bad JSON for propositions, valid text for contextual
        from tests.conftest import make_mock_llm_sequence
        llm = make_mock_llm_sequence([
            "not valid json",           # proposition extraction fails
            "Context text here.",       # contextualize parent
        ])

        run_ingestion(str(tmp_path), "knowledge", store, embedder, llm)

        points = store._data["knowledge"]
        # Parent should still be there, no propositions
        types = [pt["payload"]["chunk_type"] for pt in points]
        assert "parent" in types


class TestMultiFile:
    def test_processes_all_md_files(self, tmp_path):
        for name in ["a.md", "b.md", "c.md"]:
            (tmp_path / name).write_text(f"# {name}\n\nContent for {name}.\n")

        store = FakeVectorStore(dim=4)
        embedder = FakeEmbedder(dim=4)

        # Need enough responses: 3 files × (1 prop + contextual calls)
        from tests.conftest import make_mock_llm_sequence
        responses = []
        for _ in range(3):
            responses.append(_mock_propositions_response())  # propositions
            responses.extend([_mock_context_response()] * 3)  # contextualize 3 chunks
        llm = make_mock_llm_sequence(responses)

        run_ingestion(str(tmp_path), "knowledge", store, embedder, llm)

        points = store._data["knowledge"]
        doc_ids = {pt["payload"]["doc_id"] for pt in points}
        assert len(doc_ids) == 3
