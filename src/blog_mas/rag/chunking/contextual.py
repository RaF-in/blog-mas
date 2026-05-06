"""Stage 4: Anthropic-style Contextual Retrieval.

For every ``Chunk``, call the LLM with the chunk plus its surrounding
doc-window context to produce 50–100 tokens of "situating context".
Sets ``chunk.contextualized_text = f"{context}\\n\\n{chunk.raw_text}"``.

**Load-bearing invariant:** ``raw_text`` is never mutated.  Only
``contextualized_text`` is set.  ``raw_text`` is what reaches the
Writer's LLM; ``contextualized_text`` is what gets embedded.
"""

import logging

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from blog_mas.rag.chunking.types import Chunk

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

MAX_DOC_WINDOW_TOKENS = 8000

CONTEXTUAL_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides brief situating context "
    "for a text chunk within its source document. "
    "Given the document context and a specific chunk, write 1-2 sentences "
    "(50-100 tokens) describing where this chunk sits in the document and "
    "what implicit references it contains. "
    "Output ONLY the situating context text — no labels, no markdown, no commentary."
)


async def contextualize_chunks(
    chunks: list[Chunk], doc_text: str, llm
) -> list[Chunk]:
    """Add situating context to each chunk based on the surrounding document.

    Mutates ``chunk.contextualized_text`` in place and returns the same list.
    On LLM failure for any chunk, falls back to ``contextualized_text = raw_text``.
    """
    windows = _build_windows(doc_text)

    for chunk in chunks:
        window = _find_window(chunk, doc_text, windows)
        context = await _get_context(chunk.raw_text, window, llm)
        chunk.contextualized_text = f"{context}\n\n{chunk.raw_text}"

    return chunks


async def _get_context(chunk_text: str, window: str, llm) -> str:
    """Call LLM for one chunk. Falls back to empty string on error."""
    messages = [
        SystemMessage(content=CONTEXTUAL_SYSTEM_PROMPT),
        HumanMessage(content=f"<document>\n{window}\n</document>\n\n<chunk>\n{chunk_text}\n</chunk>"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response, AIMessage) else str(response)
        return content.strip()
    except Exception as exc:
        logger.warning(
            "contextual.llm_error chunk=%s.. detail=%s",
            chunk_text[:20], exc,
        )
        return ""


def _build_windows(doc_text: str) -> list[tuple[int, int]]:
    """Split doc into token-bounded windows. Returns (char_start, char_end) pairs."""
    tokens = _enc.encode(doc_text)
    if len(tokens) <= MAX_DOC_WINDOW_TOKENS:
        return [(0, len(doc_text))]

    windows: list[tuple[int, int]] = []
    chars_per_token = max(1, len(doc_text) / max(1, len(tokens)))
    window_chars = int(MAX_DOC_WINDOW_TOKENS * chars_per_token)

    i = 0
    while i < len(doc_text):
        end = min(i + window_chars, len(doc_text))
        windows.append((i, end))
        if end >= len(doc_text):
            break
        i = end
    return windows


def _find_window(
    chunk: Chunk, doc_text: str, windows: list[tuple[int, int]]
) -> str:
    """Find the window containing the chunk's midpoint."""
    if len(windows) == 1:
        return doc_text

    offset = doc_text.find(chunk.raw_text)
    if offset < 0:
        # Fallback: use full doc
        return doc_text

    midpoint = offset + len(chunk.raw_text) // 2
    for start, end in windows:
        if start <= midpoint < end:
            return doc_text[start:end]

    # Fallback to last window
    start, end = windows[-1]
    return doc_text[start:end]
