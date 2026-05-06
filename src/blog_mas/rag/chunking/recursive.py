"""Stage 2: token-budgeted recursive splitting with overlap.

For sections exceeding ``target_chunk_tokens``, recursively split using
the separator hierarchy ``["\\n\\n", "\\n", ". ", " "]`` with a
configurable overlap.  Token counts use ``tiktoken cl100k_base``.
Each output chunk gets a deterministic content-hash ID:
``sha256(doc_id:chunk_index:raw_text)``.
"""

import hashlib
import logging

import tiktoken

from blog_mas.rag.chunking.types import Chunk, Section

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

SEPARATORS = ["\n\n", "\n", ". ", " "]
DEFAULT_TARGET = 400
DEFAULT_OVERLAP = 50


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def content_hash(doc_id: str, chunk_index: int, raw_text: str) -> str:
    return hashlib.sha256(
        f"{doc_id}:{chunk_index}:{raw_text}".encode("utf-8")
    ).hexdigest()


def recursive_split(
    sections: list[Section],
    doc_id: str,
    target_chunk_tokens: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    start_index: int = 0,
) -> list[Chunk]:
    """Split sections into parent chunks respecting token budgets."""
    chunks: list[Chunk] = []
    idx = start_index

    for section in sections:
        tokens = _token_count(section.text)
        if tokens <= target_chunk_tokens:
            raw = section.text.strip()
            chunks.append(_make_chunk(raw, doc_id, section.headings_path, idx))
            idx += 1
        else:
            parts = _recursive_split_text(
                section.text, target_chunk_tokens, overlap
            )
            for part in parts:
                raw = part.strip()
                if raw:
                    chunks.append(_make_chunk(raw, doc_id, section.headings_path, idx))
                    idx += 1

    return chunks


def _make_chunk(
    raw_text: str, doc_id: str, headings_path: list[str], chunk_index: int
) -> Chunk:
    return Chunk(
        raw_text=raw_text,
        doc_id=doc_id,
        headings_path=list(headings_path),
        content_hash=content_hash(doc_id, chunk_index, raw_text),
        chunk_type="parent",
        chunk_index=chunk_index,
    )


def _recursive_split_text(
    text: str, target: int, overlap: int, sep_index: int = 0
) -> list[str]:
    """Split text using separator hierarchy until each part ≤ target tokens."""
    tokens = _token_count(text)
    if tokens <= target:
        return [text]

    for i in range(sep_index, len(SEPARATORS)):
        sep = SEPARATORS[i]
        if sep in text:
            parts = text.split(sep)
            merged = _merge_parts(parts, sep, target, overlap)
            # Re-split any still-over-budget chunks with next separator
            final: list[str] = []
            for r in merged:
                if _token_count(r) > target:
                    final.extend(_recursive_split_text(r, target, overlap, i + 1))
                else:
                    final.append(r)
            return final

    # No separator works — split by character positions
    return _split_by_chars(text, target, overlap)


def _merge_parts(
    parts: list[str], sep: str, target: int, overlap: int
) -> list[str]:
    """Merge small parts back together until they approach target size."""
    result: list[str] = []
    current = parts[0]

    for part in parts[1:]:
        candidate = current + sep + part
        if _token_count(candidate) <= target:
            current = candidate
        else:
            result.append(current)
            if overlap > 0 and result:
                overlap_text = _last_n_tokens(current, overlap)
                current = overlap_text + sep + part if overlap_text else part
            else:
                current = part
    result.append(current)
    return result


def _last_n_tokens(text: str, n: int) -> str:
    """Return the last *n* tokens of *text*."""
    tokens = _enc.encode(text)
    if len(tokens) <= n:
        return text
    return _enc.decode(tokens[-n:])


def _split_by_chars(text: str, target: int, overlap: int) -> list[str]:
    """Fallback: split text at character boundaries to hit target token count."""
    result: list[str] = []
    chars_per_token = max(1, len(text) // max(1, _token_count(text)))
    chunk_chars = target * chars_per_token

    i = 0
    while i < len(text):
        end = min(i + chunk_chars, len(text))
        result.append(text[i:end])
        if overlap > 0:
            overlap_chars = overlap * chars_per_token
            i = end - overlap_chars if end < len(text) else end
        else:
            i = end
    return result
