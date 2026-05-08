"""Qdrant store wrapper: collections, upsert, search, lock, async deletion polling."""

import hashlib
import logging
import os
import socket
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.5
_POLL_TIMEOUT_S = 30


class EmbeddingDimDriftError(Exception):
    pass


class LockHeldError(Exception):
    pass


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: dict


class QdrantStore:
    """Single wrapper around qdrant-client used by all RAG modules."""

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient

        self._url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._client = QdrantClient(url=self._url, api_key=api_key)
        self._sparse_model = None

    @staticmethod
    def content_hash(doc_id: str, chunk_index: int, raw_text: str) -> str:
        return hashlib.sha256(
            f"{doc_id}:{chunk_index}:{raw_text}".encode("utf-8")
        ).hexdigest()

    def ensure_collection(
        self, name: str, dim: int, sparse: bool = False
    ) -> None:
        from qdrant_client.models import Distance, PointVectors, VectorParams

        try:
            collections = self._client.get_collections().collections
        except Exception as exc:
            logger.error("qdrant.ensure_collection connection_failed name=%s detail=%s", name, exc)
            raise

        existing = [c.name for c in collections]
        if name in existing:
            info = self._client.get_collection(name)
            existing_dim = (
                info.config.params.vectors.size
                if hasattr(info.config.params, "vectors") and isinstance(info.config.params.vectors, VectorParams)
                else None
            )
            if existing_dim is not None and existing_dim != dim:
                raise EmbeddingDimDriftError(
                    f"Collection {name!r} has dim={existing_dim}, expected dim={dim}. "
                    f"Use --rebuild to recreate."
                )
            return

        vectors_config = VectorParams(size=dim, distance=Distance.COSINE)
        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
        )
        logger.info("qdrant.ensure_collection created name=%s dim=%d", name, dim)

    def upsert_points(self, name: str, points: list) -> None:
        from qdrant_client.models import PointStruct

        qdrant_points = []
        for p in points:
            if isinstance(p, PointStruct):
                qdrant_points.append(p)
            else:
                qdrant_points.append(
                    PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p.get("payload", {}),
                    )
                )
        self._client.upsert(collection_name=name, points=qdrant_points)

    def dense_search(
        self, name: str, query_vec: list[float], top_n: int = 20
    ) -> list[ScoredPoint]:
        hits = self._client.query_points(
            collection_name=name,
            query=query_vec,
            limit=top_n,
        ).points
        return [
            ScoredPoint(id=str(h.id), score=h.score, payload=h.payload)
            for h in hits
        ]

    def sparse_search(
        self, name: str, query_text: str, top_n: int = 20
    ) -> list[ScoredPoint]:
        try:
            model = self._get_sparse_model()
            sparse_vectors = list(model.embed([query_text]))
            sparse_vec = sparse_vectors[0]

            from qdrant_client.models import SparseVector

            sparse_query = SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist(),
            )

            hits = self._client.query_points(
                collection_name=name,
                query=sparse_query,
                using="bm25",
                limit=top_n,
            ).points

            return [
                ScoredPoint(id=str(h.id), score=h.score, payload=h.payload)
                for h in hits
            ]
        except Exception as exc:
            logger.warning("qdrant.sparse_search degraded name=%s detail=%s", name, exc)
            return []

    def delete_collection_with_polling(
        self, name: str, timeout_s: float = _POLL_TIMEOUT_S
    ) -> None:
        self._client.delete_collection(name)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            collections = self._client.get_collections().collections
            if name not in [c.name for c in collections]:
                logger.info("qdrant.delete_collection confirmed name=%s", name)
                return
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(
            f"Collection {name!r} still exists after {timeout_s}s"
        )

    def acquire_lock(self, name: str, hostname: str | None = None, pid: int | None = None) -> None:
        holder = f"{hostname or socket.gethostname()}-{pid or os.getpid()}"
        try:
            existing = self._client.get_collection(name)
            meta = existing.metadata or {}
            if meta.get("ingestion_lock_held_by"):
                raise LockHeldError(
                    f"Collection {name!r} locked by {meta['ingestion_lock_held_by']}"
                )
        except LockHeldError:
            raise
        except Exception:
            pass

        self._set_collection_metadata(name, {"ingestion_lock_held_by": holder})
        logger.info("qdrant.acquire_lock name=%s holder=%s", name, holder)

    def release_lock(self, name: str) -> None:
        self._set_collection_metadata(name, {"ingestion_lock_held_by": None})
        logger.info("qdrant.release_lock name=%s", name)

    def _set_collection_metadata(self, name: str, metadata: dict) -> None:
        self._client.update_collection_metadata(name, metadata)

    def _get_sparse_model(self):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model
