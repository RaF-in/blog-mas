"""Tests for rag/vector_store.py — upsert, search, lock, deletion polling."""

import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from blog_mas.rag.vector_store import (
    EmbeddingDimDriftError,
    LockHeldError,
    QdrantStore,
    ScoredPoint,
)


# ── Fake qdrant-client double ──────────────────────────────────────────


@dataclass
class _FakeCollection:
    name: str
    dim: int = 384
    metadata: dict = field(default_factory=dict)


class FakeQdrantClient:
    """In-memory fake mimicking qdrant_client.QdrantClient surface."""

    def __init__(self, url=None, api_key=None, **kwargs):
        self._collections: dict[str, _FakeCollection] = {}
        self._points: dict[str, list[dict]] = {}
        self._delete_delay: dict[str, int] = {}

    def get_collections(self):
        # Decrement delete delay counters
        for name in list(self._delete_delay):
            self._delete_delay[name] -= 1
            if self._delete_delay[name] <= 0:
                del self._delete_delay[name]
                self._collections.pop(name, None)
                self._points.pop(name, None)

        Collections = type("Collections", (), {})
        ColInfo = type("ColInfo", (), {})
        cols = []
        for name in self._collections:
            ci = ColInfo()
            ci.name = name
            cols.append(ci)
        resp = Collections()
        resp.collections = cols
        return resp

    def get_collection(self, name):
        if name not in self._collections:
            raise ValueError(f"Collection {name!r} not found")
        c = self._collections[name]

        class Info:
            pass

        info = Info()
        info.config = type("Config", (), {})()
        info.config.params = type("Params", (), {})()

        from qdrant_client.models import VectorParams, Distance

        info.config.params.vectors = VectorParams(size=c.dim, distance=Distance.COSINE)
        info.metadata = dict(c.metadata)
        return info

    def create_collection(self, collection_name, vectors_config, **kwargs):
        dim = vectors_config.size
        self._collections[collection_name] = _FakeCollection(
            name=collection_name, dim=dim
        )
        self._points[collection_name] = []

    def delete_collection(self, collection_name):
        # Mark for delayed deletion — collection lingers for N get_collections calls
        if collection_name in self._delete_delay:
            return
        self._delete_delay[collection_name] = 0

    def upsert(self, collection_name, points):
        if collection_name not in self._points:
            self._points[collection_name] = []
        existing = {p["id"] for p in self._points[collection_name]}
        for pt in points:
            if pt.id in existing:
                self._points[collection_name] = [
                    p if p["id"] != pt.id
                    else {"id": pt.id, "vector": pt.vector, "payload": pt.payload}
                    for p in self._points[collection_name]
                ]
            else:
                self._points[collection_name].append(
                    {"id": pt.id, "vector": pt.vector, "payload": pt.payload}
                )

    def search(self, collection_name, query_vector, limit=10, **kwargs):
        pts = self._points.get(collection_name, [])
        scored = []
        for p in pts:
            score = sum(a * b for a, b in zip(query_vector, p["vector"]))
            scored.append(_make_hit(p, score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def query_points(self, collection_name, query, using=None, limit=10, **kwargs):
        pts = self._points.get(collection_name, [])
        scored = []
        for idx, p in enumerate(pts):
            text = p["payload"].get("raw_text", "")
            score = float(len(text))
            scored.append(_make_hit(p, score))
        scored.sort(key=lambda h: h.score, reverse=True)
        result = type("Result", (), {})()
        result.points = scored[:limit]
        return result

    def update_collection_metadata(self, name, metadata):
        if name in self._collections:
            self._collections[name].metadata.update(
                {k: v for k, v in metadata.items() if v is not None}
            )
            for k, v in metadata.items():
                if v is None:
                    self._collections[name].metadata.pop(k, None)

    def set_delete_delay(self, name, poll_count):
        """Set how many get_collections() calls before the collection actually disappears."""
        self._delete_delay[name] = poll_count


def _make_hit(point, score):
    class Hit:
        pass
    h = Hit()
    h.id = point["id"]
    h.score = score
    h.payload = point["payload"]
    return h


def _make_store(fake_client):
    store = QdrantStore.__new__(QdrantStore)
    store._url = "fake"
    store._client = fake_client
    store._sparse_model = None
    return store


# ── ensure_collection ──────────────────────────────────────────────────


class TestEnsureCollection:
    def test_creates_when_missing(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=384)
        assert "knowledge" in fc._collections
        assert fc._collections["knowledge"].dim == 384

    def test_noop_when_existing_matching_dim(self):
        fc = FakeQdrantClient()
        fc.create_collection("knowledge", vectors_config=_vec_params(384))
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=384)
        assert len(fc._collections) == 1

    def test_raises_on_dim_mismatch(self):
        fc = FakeQdrantClient()
        fc.create_collection("knowledge", vectors_config=_vec_params(768))
        store = _make_store(fc)
        with pytest.raises(EmbeddingDimDriftError):
            store.ensure_collection("knowledge", dim=384)

    def test_fails_fast_on_connection_error(self):
        fc = FakeQdrantClient()
        fc.get_collections = MagicMock(side_effect=ConnectionError("unreachable"))
        store = _make_store(fc)
        with pytest.raises(ConnectionError):
            store.ensure_collection("knowledge", dim=384)


# ── Upsert idempotency ─────────────────────────────────────────────────


class TestUpsertIdempotency:
    def test_same_content_same_id_single_point(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=384)

        doc_id, raw_text = "doc1", "hello world"
        h = QdrantStore.content_hash(doc_id, 0, raw_text)
        vec = [0.1] * 384

        from qdrant_client.models import PointStruct

        pt = PointStruct(id=h, vector=vec, payload={"raw_text": raw_text})
        store.upsert_points("knowledge", [pt])
        store.upsert_points("knowledge", [pt])

        assert len(fc._points["knowledge"]) == 1

    def test_different_chunk_index_different_ids(self):
        h1 = QdrantStore.content_hash("doc1", 0, "same text")
        h2 = QdrantStore.content_hash("doc1", 1, "same text")
        assert h1 != h2


# ── Dense search ───────────────────────────────────────────────────────


class TestDenseSearch:
    def test_returns_top_n_with_scores(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)

        from qdrant_client.models import PointStruct

        for i, (vec, txt) in enumerate([
            ([1.0, 0.0, 0.0, 0.0], "aaa"),
            ([0.0, 1.0, 0.0, 0.0], "bbb"),
            ([0.9, 0.1, 0.0, 0.0], "ccc"),
        ]):
            store.upsert_points("knowledge", [PointStruct(id=f"p{i}", vector=vec, payload={"raw_text": txt})])

        results = store.dense_search("knowledge", [1.0, 0.0, 0.0, 0.0], top_n=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score

    def test_namespace_isolation(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        store.ensure_collection("blueprints", dim=4)

        from qdrant_client.models import PointStruct

        store.upsert_points("knowledge", [PointStruct(id="k1", vector=[1, 0, 0, 0], payload={"t": "k"})])
        store.upsert_points("blueprints", [PointStruct(id="b1", vector=[1, 0, 0, 0], payload={"t": "b"})])

        results = store.dense_search("knowledge", [1, 0, 0, 0])
        assert all(r.payload.get("t") != "b" for r in results)


# ── Sparse search ──────────────────────────────────────────────────────


class TestSparseSearch:
    def test_soft_fails_on_backend_error(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        store._get_sparse_model = MagicMock(side_effect=ImportError("no fastembed"))
        results = store.sparse_search("knowledge", "test query")
        assert results == []


# ── Deletion polling ────────────────────────────────────────────────────


class TestDeletionPolling:
    @patch("blog_mas.rag.vector_store.time.sleep", return_value=None)
    @patch("blog_mas.rag.vector_store.time.monotonic")
    def test_polls_until_absent(self, mock_time, mock_sleep):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        # Collection lingers for 2 get_collections calls after delete
        fc._delete_delay["knowledge"] = 2

        # monotonic is called: deadline calc, then while-check per iteration
        # With delay=2: poll1 (still present), poll2 (still present), poll3 (gone)
        t = iter([0, 1, 2, 3, 4, 5, 6, 7, 8])
        mock_time.side_effect = lambda: next(t)

        store.delete_collection_with_polling("knowledge", timeout_s=10)
        assert "knowledge" not in fc._collections

    @patch("blog_mas.rag.vector_store.time.sleep", return_value=None)
    @patch("blog_mas.rag.vector_store.time.monotonic")
    def test_raises_on_timeout(self, mock_time, mock_sleep):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        # Collection never disappears — delete is a no-op
        original_delete = fc.delete_collection

        def no_op_delete(name):
            pass

        fc.delete_collection = no_op_delete

        # monotonic: 0 (deadline=10), then 5, 11 → 11 > 10 → timeout
        t = iter([0, 5, 11])
        mock_time.side_effect = lambda: next(t)

        with pytest.raises(TimeoutError):
            store.delete_collection_with_polling("knowledge", timeout_s=10)


# ── Collection lock ────────────────────────────────────────────────────


class TestCollectionLock:
    def test_acquire_writes_metadata(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        store.acquire_lock("knowledge", hostname="host1", pid=1234)
        assert fc._collections["knowledge"].metadata["ingestion_lock_held_by"] == "host1-1234"

    def test_second_acquire_fails(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        store.acquire_lock("knowledge", hostname="host1", pid=1234)
        with pytest.raises(LockHeldError):
            store.acquire_lock("knowledge", hostname="host2", pid=5678)

    def test_release_allows_reacquire(self):
        fc = FakeQdrantClient()
        store = _make_store(fc)
        store.ensure_collection("knowledge", dim=4)
        store.acquire_lock("knowledge", hostname="host1", pid=1234)
        store.release_lock("knowledge")
        store.acquire_lock("knowledge", hostname="host2", pid=5678)
        assert "host2-5678" in fc._collections["knowledge"].metadata["ingestion_lock_held_by"]


def _vec_params(dim):
    from qdrant_client.models import Distance, VectorParams
    return VectorParams(size=dim, distance=Distance.COSINE)
