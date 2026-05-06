"""Tests for rag/chunking/contextual.py — Stage 4 contextual retrieval."""

import pytest

from blog_mas.rag.chunking.contextual import (
    CONTEXTUAL_SYSTEM_PROMPT,
    contextualize_chunks,
    _build_windows,
    _find_window,
)
from blog_mas.rag.chunking.types import Chunk
from tests.conftest import make_mock_llm, make_failing_llm


def _chunk(text: str = "sample chunk text", chunk_type: str = "parent") -> Chunk:
    return Chunk(
        raw_text=text,
        doc_id="doc1",
        content_hash="hash123",
        chunk_type=chunk_type,
        chunk_index=0,
    )


def _doc(n_words: int = 200) -> str:
    return ("word " * n_words).strip()


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_prepends_situating_context(self):
        context_text = "This chunk is from section X discussing Y."
        llm = make_mock_llm(context_text)
        chunk = _chunk("The actual content.")
        doc = "Some doc text. " + "The actual content. " + "More doc text."
        result = await contextualize_chunks([chunk], doc, llm)
        assert result[0].contextualized_text.startswith(context_text)
        assert "The actual content." in result[0].contextualized_text

    @pytest.mark.asyncio
    async def test_raw_text_untouched(self):
        llm = make_mock_llm("Some context.")
        chunk = _chunk("Original text.")
        doc = "Original text."
        original = chunk.raw_text
        await contextualize_chunks([chunk], doc, llm)
        assert chunk.raw_text == original


class TestWindowingPreflight:
    def test_short_doc_uses_full_window(self):
        doc = _doc(200)
        windows = _build_windows(doc)
        assert len(windows) == 1
        assert windows[0] == (0, len(doc))

    def test_long_doc_splits_into_windows(self):
        doc = _doc(10000)  # ~12,500 tokens, well over 8000
        windows = _build_windows(doc)
        assert len(windows) > 1

    def test_chunk_straddling_boundary_uses_midpoint(self):
        doc = "A" * 10000 + "CHUNK_CONTENT" + "B" * 10000
        chunk = _chunk("CHUNK_CONTENT")
        windows = _build_windows(doc)
        window = _find_window(chunk, doc, windows)
        assert "CHUNK_CONTENT" in window


class TestLLMFailures:
    @pytest.mark.asyncio
    async def test_falls_back_on_llm_error(self):
        llm = make_failing_llm(RuntimeError("LLM down"))
        chunk = _chunk("some text")
        doc = "some text"
        result = await contextualize_chunks([chunk], doc, llm)
        # Falls back to empty context + raw_text
        assert result[0].contextualized_text == "\n\nsome text"


class TestOutputIntegrity:
    @pytest.mark.asyncio
    async def test_works_on_parent_and_proposition_chunks(self):
        llm = make_mock_llm("Context.")
        parent = _chunk("parent text", "parent")
        prop = _chunk("proposition text", "proposition")
        doc = "parent text proposition text"
        result = await contextualize_chunks([parent, prop], doc, llm)
        assert result[0].contextualized_text is not None
        assert result[1].contextualized_text is not None
        assert result[0].chunk_type == "parent"
        assert result[1].chunk_type == "proposition"
