"""Tests for rag/chunking/recursive.py — Stage 2 recursive split."""

import hashlib

import tiktoken

from blog_mas.rag.chunking.recursive import (
    content_hash,
    recursive_split,
    _token_count,
)
from blog_mas.rag.chunking.types import Section

_enc = tiktoken.get_encoding("cl100k_base")


def _section(text: str) -> Section:
    return Section(text=text, headings_path=["Test"])


def _long_section(n_tokens: int = 800) -> Section:
    """Build a section with approximately *n_tokens* tokens."""
    words = []
    total = 0
    idx = 0
    while total < n_tokens:
        w = f"word{idx}"
        words.append(w)
        total = len(_enc.encode(" ".join(words)))
        idx += 1
    return Section(text=" ".join(words), headings_path=["Test"])


class TestNoOpShortSection:
    def test_section_under_budget_emits_single_chunk(self):
        sec = _section("Short text here.")
        chunks = recursive_split([sec], "doc1")
        assert len(chunks) == 1
        assert chunks[0].raw_text == "Short text here."


class TestSplittingOverBudget:
    def test_splits_on_double_newline_first(self):
        text = "word " * 200 + "\n\n" + "word " * 200
        sec = Section(text=text, headings_path=["A"])
        chunks = recursive_split([sec], "doc1", target_chunk_tokens=100)
        assert len(chunks) >= 2
        assert all(_token_count(c.raw_text) <= 120 for c in chunks)

    def test_resulting_chunks_under_budget(self):
        sec = _long_section(800)
        chunks = recursive_split([sec], "doc1", target_chunk_tokens=400)
        for c in chunks:
            assert _token_count(c.raw_text) <= 440


class TestTokenCounting:
    def test_uses_cl100k_base(self):
        text = "Hello, world!"
        expected = len(_enc.encode(text))
        assert _token_count(text) == expected


class TestOverlap:
    def test_overlap_produces_shared_tokens(self):
        text = ("paragraph " * 50 + "\n\n") * 4
        sec = Section(text=text, headings_path=[])
        chunks = recursive_split([sec], "doc1", target_chunk_tokens=50, overlap=10)
        if len(chunks) >= 2:
            tail_tokens = set(_enc.encode(chunks[0].raw_text)[-10:])
            head_tokens = set(_enc.encode(chunks[1].raw_text)[:10:])
            assert tail_tokens & head_tokens

    def test_zero_overlap_no_shared_tokens(self):
        text = ("sentence. " * 200)
        sec = Section(text=text, headings_path=[])
        chunks = recursive_split([sec], "doc1", target_chunk_tokens=50, overlap=0)
        assert len(chunks) >= 2


class TestContentHashIds:
    def test_deterministic_same_input(self):
        h1 = content_hash("doc1", 0, "hello")
        h2 = content_hash("doc1", 0, "hello")
        assert h1 == h2

    def test_different_chunk_index_different_hash(self):
        h1 = content_hash("doc1", 0, "same text")
        h2 = content_hash("doc1", 1, "same text")
        assert h1 != h2

    def test_uses_sha256_format(self):
        h = content_hash("doc1", 0, "test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestChunkMetadata:
    def test_headings_path_inherited(self):
        sec = Section(text="some text", headings_path=["A", "B"])
        chunks = recursive_split([sec], "doc1")
        assert chunks[0].headings_path == ["A", "B"]

    def test_chunk_type_is_parent(self):
        sec = _section("text")
        chunks = recursive_split([sec], "doc1")
        assert all(c.chunk_type == "parent" for c in chunks)

    def test_chunk_index_increments_across_sections(self):
        s1 = _section("first section text.")
        s2 = _section("second section text.")
        chunks = recursive_split([s1, s2], "doc1")
        indices = [c.chunk_index for c in chunks]
        assert indices == sorted(set(indices))
        assert indices[0] == 0
