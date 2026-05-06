"""Shared test fixtures and helpers."""

import hashlib
import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from blog_mas.rag.blueprints import NEUTRAL_BLUEPRINT, validate_blueprint_payload


def _to_ai_message(value) -> AIMessage:
    """Convert a Pydantic model or dict to an AIMessage with JSON content."""
    if isinstance(value, BaseModel):
        content = value.model_dump_json()
    elif isinstance(value, dict):
        content = json.dumps(value)
    else:
        content = str(value)
    return AIMessage(content=content)


def make_mock_llm(return_value) -> RunnableLambda:
    """Create a mock LLM (Runnable) that returns an AIMessage with the given value."""
    msg = _to_ai_message(return_value)

    async def _invoke(messages, config=None):
        return msg

    return RunnableLambda(_invoke)


def make_mock_llm_sequence(return_values: list) -> RunnableLambda:
    """Create a mock LLM (Runnable) that returns AIMessages in sequence."""
    it = iter([_to_ai_message(v) for v in return_values])

    async def _invoke(messages, config=None):
        return next(it)

    return RunnableLambda(_invoke)


def make_failing_llm(error: Exception) -> RunnableLambda:
    """Create a mock LLM (Runnable) that raises the given error."""
    async def _invoke(messages, config=None):
        raise error

    return RunnableLambda(_invoke)


def make_config(llm):
    """Create a LangGraph RunnableConfig dict with the given LLM."""
    return {"configurable": {"llm": llm}}


# ── RAG test doubles ───────────────────────────────────────────────────


class FakeVectorStore:
    """In-memory test double mirroring QdrantStore's public surface."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._data: dict[str, list[dict]] = {}
        self._metadata: dict[str, dict] = {}
        self._fail_on: dict[str, Exception] = {}

    def ensure_collection(self, name: str, dim: int, sparse: bool = False) -> None:
        self._check_fail("ensure_collection")
        if name not in self._data:
            self._data[name] = []
            self._metadata[name] = {}

    def upsert_points(self, name: str, points: list) -> None:
        self._check_fail("upsert_points")
        if name not in self._data:
            self._data[name] = []
        existing = {p["id"] for p in self._data[name]}
        for pt in points:
            pid = pt["id"] if isinstance(pt, dict) else pt.id
            vec = pt["vector"] if isinstance(pt, dict) else pt.vector
            payload = pt.get("payload", {}) if isinstance(pt, dict) else pt.payload
            if pid in existing:
                self._data[name] = [
                    {"id": pid, "vector": vec, "payload": payload}
                    if p["id"] == pid else p
                    for p in self._data[name]
                ]
            else:
                self._data[name].append(
                    {"id": pid, "vector": vec, "payload": payload}
                )

    def dense_search(
        self, name: str, query_vec: list[float], top_n: int = 20
    ) -> list[dict]:
        from blog_mas.rag.vector_store import ScoredPoint

        self._check_fail("dense_search")
        pts = self._data.get(name, [])
        scored = []
        for p in pts:
            dot = sum(a * b for a, b in zip(query_vec, p["vector"]))
            scored.append(ScoredPoint(id=p["id"], score=dot, payload=p["payload"]))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]

    def sparse_search(
        self, name: str, query_text: str, top_n: int = 20
    ) -> list[dict]:
        from blog_mas.rag.vector_store import ScoredPoint

        self._check_fail("sparse_search")
        pts = self._data.get(name, [])
        return [ScoredPoint(id=p["id"], score=1.0, payload=p["payload"]) for p in pts[:top_n]]

    def delete_collection_with_polling(self, name: str, timeout_s: float = 30) -> None:
        self._check_fail("delete_collection_with_polling")
        self._data.pop(name, None)
        self._metadata.pop(name, None)

    def acquire_lock(self, name: str, hostname: str | None = None, pid: int | None = None) -> None:
        self._check_fail("acquire_lock")
        meta = self._metadata.setdefault(name, {})
        if meta.get("ingestion_lock_held_by"):
            from blog_mas.rag.vector_store import LockHeldError
            raise LockHeldError(f"locked by {meta['ingestion_lock_held_by']}")
        meta["ingestion_lock_held_by"] = f"{hostname or 'fake'}-{pid or 0}"

    def release_lock(self, name: str) -> None:
        self._check_fail("release_lock")
        self._metadata.get(name, {}).pop("ingestion_lock_held_by", None)

    def fail_on(self, method_name: str, exc: Exception | None = None) -> None:
        self._fail_on[method_name] = exc or RuntimeError(f"forced failure on {method_name}")

    def _check_fail(self, method_name: str) -> None:
        if method_name in self._fail_on:
            exc = self._fail_on.pop(method_name)
            raise exc


class FakeEmbedder:
    """Deterministic fake embedding client."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def _hash_vector(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = _SeededRNG(int(h[:8], 16))
        return [rng.random() for _ in range(self._dim)]


class _SeededRNG:
    def __init__(self, seed: int):
        self._state = seed

    def random(self) -> float:
        self._state = (self._state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self._state / 0xFFFFFFFF


class FakeReranker:
    """Deterministic fake reranker."""

    def __init__(self):
        self.degraded = False

    def rerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
        if self.degraded:
            return docs[:top_k]
        scored = []
        for i, doc in enumerate(docs):
            text = doc.get("raw_text", "") if isinstance(doc, dict) else str(doc)
            score = len(text) * 0.01 + (100 - i)
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ── RAG fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_markdown_doc() -> str:
    """A small markdown doc with 2 H1, 3 H2 sections."""
    return (
        "# Introduction\n\nSome intro text.\n\n"
        "## Background\n\nBackground details.\n\n"
        "# Main Topic\n\nMain content.\n\n"
        "## Details\n\nDetailed info.\n\n"
        "## Summary\n\nSummary text.\n"
    )


@pytest.fixture
def sample_blueprint_payload() -> str:
    """A valid Blueprint JSON string."""
    return json.dumps({
        "id": "bp-test-fixture",
        "description": "A test blueprint for fixture use.",
        "scene_goal": "Test scene goal.",
        "style_guide": "Clear and concise.",
        "participants": [{"name": "Editor", "role": "review"}],
        "instruction": "Write a well-structured blog post.",
        "metadata": {"source": "fixture"},
    })


@pytest.fixture
def make_chunk():
    """Factory to construct a chunk dict with a content_hash ID."""
    def _make(doc_id: str = "doc1", chunk_index: int = 0, raw_text: str = "sample") -> dict:
        content_hash = hashlib.sha256(
            f"{doc_id}:{chunk_index}:{raw_text}".encode("utf-8")
        ).hexdigest()
        return {
            "id": content_hash,
            "raw_text": raw_text,
            "doc_id": doc_id,
            "chunk_index": chunk_index,
        }
    return _make
