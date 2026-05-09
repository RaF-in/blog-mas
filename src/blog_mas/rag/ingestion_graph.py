"""Knowledge ingestion: load .md files → chunk pipeline → embed → upsert.

Chapter 7 defense-in-depth: the sanitizer runs as a second ring of defense
at ingest time, before chunks ever reach the vector store.  This catches
poisoned source documents before they can be retrieved by a legitimate query
and used in a prompt-injection attack (Chapter 7 §B).

Chunks that fail sanitization are skipped and logged — they are never
embedded or upserted.  The ingestion log will show which documents triggered
rejections so a data engineer can review and remediate the source files.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

from blog_mas.rag.chunking.contextual import contextualize_chunks
from blog_mas.rag.chunking.propositions import extract_propositions
from blog_mas.rag.chunking.recursive import recursive_split
from blog_mas.rag.chunking.structural import split_by_headers
from blog_mas.rag.chunking.types import IngestionDoc
from blog_mas.rag.vector_store import ScoredPoint
from blog_mas.security.sanitizer import sanitize_chunk

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

        # Stage 4b: sanitize — Chapter 7 §B defense-in-depth (second ring).
        # Chunks that contain injection patterns are dropped before embedding.
        # This prevents poisoned source documents from ever reaching the vector
        # store, where a later legitimate query could retrieve them.
        clean_chunks = []
        ingest_rejected = 0
        for chunk in all_chunks:
            result = sanitize_chunk(chunk.raw_text)
            if result.ok:
                clean_chunks.append(chunk)
            else:
                ingest_rejected += 1
                logger.warning(
                    "ingestion.sanitizer_rejected doc=%s chunk_hash=%s pattern=%r",
                    doc_id, chunk.content_hash[:12], result.matched_pattern,
                )

        if ingest_rejected:
            logger.warning(
                "ingestion.rejected_summary doc=%s rejected=%d kept=%d",
                doc_id, ingest_rejected, len(clean_chunks),
            )

        all_chunks = clean_chunks

        # Stage 5: embed
        texts = [c.contextualized_text or c.raw_text for c in all_chunks]
        vectors = embedder.embed_batch(texts)

        # Stage 6: upsert with content-hash IDs
        points = []
        for chunk, vec in zip(all_chunks, vectors):
            points.append({
                "id": str(uuid.UUID(chunk.content_hash[:32])),
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
        logger.info(
            "ingestion.upserted namespace=%s doc=%s points=%d rejected=%d",
            namespace, doc_id, len(points), ingest_rejected,
        )
