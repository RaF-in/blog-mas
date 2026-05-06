"""Recall@k eval harness — measures retrieval quality against labeled queries."""

import logging
from pathlib import Path

import yaml

from blog_mas.rag.retrieval import hybrid_search

logger = logging.getLogger(__name__)

QUERIES_PATH = Path(__file__).resolve().parents[3] / "tests" / "eval" / "queries.yaml"


def load_queries(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else QUERIES_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def compute_recall(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    if not expected_ids:
        return 0.0
    return len(top_k & set(expected_ids)) / len(expected_ids)


def run_recall_eval(queries_path: str | None = None, store=None, embedder=None, reranker=None):
    """Run recall evaluation and print results."""
    if store is None:
        from blog_mas.rag.vector_store import QdrantStore
        store = QdrantStore()
    if embedder is None:
        from blog_mas.rag.embedding import EmbeddingClient
        embedder = EmbeddingClient()
    if reranker is None:
        from tests.conftest import FakeReranker
        reranker = FakeReranker()

    queries = load_queries(queries_path)
    results = {"1": [], "3": [], "10": []}

    for q in queries:
        hits = hybrid_search(
            query=q["query"], namespace="knowledge", top_k=10,
            store=store, embedder=embedder, reranker=reranker,
        )
        retrieved = [h.id for h in hits]
        for k in [1, 3, 10]:
            results[str(k)].append(compute_recall(retrieved, q["expected_doc_ids"], k))

    print("\n--- Recall@k Results ---")
    for k in [1, 3, 10]:
        vals = results[str(k)]
        mean = sum(vals) / len(vals) if vals else 0.0
        print(f"  recall@{k}: {mean:.2f} ({len(vals)} queries)")
    print()
