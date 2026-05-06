"""Knowledge ingestion: load .md files → chunk pipeline → embed → upsert."""

import asyncio
import logging
import os
from pathlib import Path

from blog_mas.rag.chunking.contextual import contextualize_chunks
from blog_mas.rag.chunking.propositions import extract_propositions
from blog_mas.rag.chunking.recursive import recursive_split
from blog_mas.rag.chunking.structural import split_by_headers
from blog_mas.rag.chunking.types import IngestionDoc
from blog_mas.rag.vector_store import ScoredPoint

logger = logging.getLogger(__name__)


def run_ingestion(
    source_dir: str,
    namespace: str,
    store,
    embedder,
    llm,
) -> None:
    """Load .md files from source_dir, chunk, embed, and upsert into store."""
    asyncio.run(_run_async(source_dir, namespace, store, embedder, llm))


async def _run_async(
    source_dir: str,
    namespace: str,
    store,
    embedder,
    llm,
) -> None:
    dim = getattr(embedder, "_dim", 384)
    store.ensure_collection(namespace, dim=dim)

    md_files = sorted(Path(source_dir).glob("*.md"))
    if not md_files:
        return

    for md_file in md_files:
        doc_id = md_file.stem
        raw_text = md_file.read_text()
        doc = IngestionDoc(doc_id=doc_id, raw_text=raw_text)

        # Stage 1: structural split
        sections = split_by_headers(raw_text)

        # Stage 2: recursive split into parent chunks
        parents = recursive_split(sections, doc_id=doc_id)

        # Stage 3: extract propositions (LLM)
        propositions = await extract_propositions(parents, llm)

        # Combine all chunks
        all_chunks = parents + propositions

        # Stage 4: contextualize (LLM)
        all_chunks = await contextualize_chunks(all_chunks, raw_text, llm)

        # Stage 5: embed
        texts = [c.contextualized_text or c.raw_text for c in all_chunks]
        vectors = embedder.embed_batch(texts)

        # Stage 6: upsert with content-hash IDs
        points = []
        for chunk, vec in zip(all_chunks, vectors):
            points.append({
                "id": chunk.content_hash,
                "vector": vec,
                "payload": {
                    "raw_text": chunk.raw_text,
                    "contextualized_text": chunk.contextualized_text,
                    "parent_id": chunk.parent_id,
                    "doc_id": chunk.doc_id,
                    "headings_path": chunk.headings_path,
                    "chunk_type": chunk.chunk_type,
                    "content_hash": chunk.content_hash,
                },
            })

        store.upsert_points(namespace, points)
        logger.info("ingestion.upserted namespace=%s doc=%s points=%d", namespace, doc_id, len(points))
