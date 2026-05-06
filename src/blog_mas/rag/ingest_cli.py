"""CLI subcommand handlers for ingestion."""

import argparse
import asyncio
import logging

from blog_mas.rag.blueprint_graph import run_blueprint_ingestion
from blog_mas.rag.ingestion_graph import run_ingestion
from blog_mas.rag.embedding import EmbeddingClient
from blog_mas.rag.vector_store import QdrantStore
from tests.conftest import FakeEmbedder, FakeReranker, FakeVectorStore

logger = logging.getLogger(__name__)


def cmd_ingest(args):
    store = QdrantStore()
    embedder = EmbeddingClient()
    llm = _make_llm()

    if args.rebuild:
        logger.info("Rebuilding knowledge collection...")
        try:
            store.delete_collection_with_polling("knowledge")
        except Exception:
            pass

    run_ingestion(
        source_dir=args.path or "data/knowledge",
        namespace="knowledge",
        store=store,
        embedder=embedder,
        llm=llm,
    )
    print("Knowledge ingestion complete.")


def cmd_ingest_blueprints(args):
    store = QdrantStore()
    embedder = EmbeddingClient()

    if args.rebuild:
        logger.info("Rebuilding blueprints collection...")
        try:
            store.delete_collection_with_polling("blueprints")
        except Exception:
            pass

    run_blueprint_ingestion(
        source_dir="data/blueprints",
        namespace="blueprints",
        store=store,
        embedder=embedder,
    )
    print("Blueprint ingestion complete.")


def cmd_eval(args):
    from blog_mas.eval.recall import run_recall_eval
    run_recall_eval(queries_path=getattr(args, "queries", None))


def _make_llm():
    from blog_mas.llm import create_llm
    return create_llm()
