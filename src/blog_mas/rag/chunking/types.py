"""Shared dataclasses for the chunking pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Section:
    """A section of a document delimited by markdown headers."""
    text: str
    headings_path: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A chunk produced by the pipeline, ready for embedding and upsert."""
    raw_text: str
    contextualized_text: str | None = None
    parent_id: str | None = None
    doc_id: str = ""
    headings_path: list[str] = field(default_factory=list)
    content_hash: str = ""
    chunk_type: Literal["parent", "proposition"] = "parent"
    chunk_index: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IngestionDoc:
    """A document to be ingested."""
    doc_id: str
    raw_text: str
