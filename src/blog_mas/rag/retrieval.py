"""Hybrid retrieval: dense + sparse → RRF fusion → rerank → small-to-big parent expansion."""

import logging
import os

from blog_mas.rag.vector_store import ScoredPoint

logger = logging.getLogger(__name__)


def _rrf_fuse(
    dense: list[ScoredPoint],
    sparse: list[ScoredPoint],
    k: int = 60,
) -> list[ScoredPoint]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, pt in enumerate(dense):
        scores[pt.id] = scores.get(pt.id, 0.0) + 1.0 / (k + rank)
        payloads[pt.id] = pt.payload

    for rank, pt in enumerate(sparse):
        scores[pt.id] = scores.get(pt.id, 0.0) + 1.0 / (k + rank)
        payloads[pt.id] = pt.payload

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [ScoredPoint(id=pid, score=sc, payload=payloads[pid]) for pid, sc in ranked]


def _rerank(
    query: str,
    candidates: list[ScoredPoint],
    reranker,
    top_k: int = 3,
) -> list[ScoredPoint]:
    docs = [{"raw_text": pt.payload.get("raw_text", ""), "id": pt.id} for pt in candidates]
    try:
        ranked_docs = reranker.rerank(query, docs, top_k=top_k)
    except Exception as exc:
        logger.warning("retrieval.rerank degraded detail=%s", exc)
        return candidates[:top_k]

    reranked_ids = {d["id"] if isinstance(d, dict) else d.id for d in ranked_docs}
    return [pt for pt in candidates if pt.id in reranked_ids][:top_k]


def _expand_parents(
    candidates: list[ScoredPoint],
    store,
    namespace: str,
) -> list[ScoredPoint]:
    seen_parent_ids: set[str] = set()
    result: list[ScoredPoint] = []

    for pt in candidates:
        parent_id = pt.payload.get("parent_id")
        if pt.payload.get("chunk_type") == "proposition" and parent_id:
            parent_payload = _fetch_parent_payload(store, namespace, parent_id)
            if parent_payload:
                expanded = ScoredPoint(
                    id=pt.id,
                    score=pt.score,
                    payload={**pt.payload, "raw_text": parent_payload.get("raw_text", pt.payload.get("raw_text", ""))},
                )
                result.append(expanded)
            else:
                result.append(pt)
            seen_parent_ids.add(parent_id)
        else:
            result.append(pt)

    return result


def _fetch_parent_payload(store, namespace: str, parent_id: str) -> dict | None:
    """Look up a parent point's payload from the store by searching with a zero vector."""
    # FakeVectorStore stores points by id; we do a dense_search with a dummy vector
    # and filter for our target. For a real QdrantStore we'd use scroll/point lookup.
    dim = getattr(store, "_dim", 384)
    hits = store.dense_search(namespace, [0.0] * dim, top_n=1000)
    for hit in hits:
        if hit.id == parent_id:
            return hit.payload
    return None


def hybrid_search(
    query: str,
    namespace: str,
    top_k: int = 3,
    store=None,
    embedder=None,
    reranker=None,
) -> list[ScoredPoint]:
    dense_top_n = int(os.environ.get("RAG_DENSE_TOP_N", "20"))
    sparse_top_n = int(os.environ.get("RAG_SPARSE_TOP_N", "20"))
    rrf_k = int(os.environ.get("RAG_RRF_K", "60"))

    # Dense search
    try:
        query_vec = embedder.embed_batch([query])[0]
        dense_results = store.dense_search(namespace, query_vec, top_n=dense_top_n)
    except Exception as exc:
        logger.warning("retrieval.dense_search degraded detail=%s", exc)
        dense_results = []

    # Sparse search
    try:
        sparse_results = store.sparse_search(namespace, query, top_n=sparse_top_n)
    except Exception as exc:
        logger.warning("retrieval.sparse_search degraded detail=%s", exc)
        sparse_results = []

    if not dense_results and not sparse_results:
        return []

    # RRF fusion
    fused = _rrf_fuse(dense_results, sparse_results, k=rrf_k)

    # Rerank
    reranked = _rerank(query, fused, reranker, top_k=top_k)

    # Small-to-big parent expansion
    return _expand_parents(reranked, store, namespace)
