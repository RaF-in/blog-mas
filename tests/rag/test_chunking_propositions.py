"""Tests for rag/chunking/propositions.py — Stage 3 proposition extraction."""

import json

import pytest

from blog_mas.rag.chunking.propositions import (
    PROPOSITION_SYSTEM_PROMPT,
    extract_propositions,
)
from blog_mas.rag.chunking.types import Chunk
from tests.conftest import make_mock_llm, make_failing_llm


def _parent(text: str = "The sky is blue. Water is wet.") -> Chunk:
    return Chunk(
        raw_text=text,
        doc_id="doc1",
        headings_path=["Test"],
        content_hash="abc123",
        chunk_type="parent",
        chunk_index=0,
    )


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_extracts_propositions(self):
        llm = make_mock_llm({"propositions": ["The sky is blue.", "Water is wet."]})
        parents = [_parent()]
        children = await extract_propositions(parents, llm)
        assert len(children) == 2
        assert all(c.chunk_type == "proposition" for c in children)

    @pytest.mark.asyncio
    async def test_child_links_to_parent(self):
        llm = make_mock_llm({"propositions": ["Fact one."]})
        parents = [_parent()]
        children = await extract_propositions(parents, llm)
        assert children[0].parent_id == "abc123"

    @pytest.mark.asyncio
    async def test_child_has_own_content_hash(self):
        llm = make_mock_llm({"propositions": ["Fact one."]})
        children = await extract_propositions([_parent()], llm)
        assert children[0].content_hash != "abc123"
        assert len(children[0].content_hash) == 64


class TestAppliedToAllParents:
    @pytest.mark.asyncio
    async def test_calls_llm_per_parent(self):
        from tests.conftest import make_mock_llm_sequence

        llm = make_mock_llm_sequence([
            {"propositions": ["P1"]},
            {"propositions": ["P2"]},
        ])
        parents = [_parent("text1"), _parent("text2")]
        parents[1].chunk_index = 1
        parents[1].content_hash = "def456"
        children = await extract_propositions(parents, llm)
        assert len(children) == 2


class TestJsonParseFailures:
    @pytest.mark.asyncio
    async def test_discards_on_malformed_json(self):
        llm = make_mock_llm("not json at all")
        children = await extract_propositions([_parent()], llm)
        assert children == []

    @pytest.mark.asyncio
    async def test_discards_on_missing_propositions_key(self):
        llm = make_mock_llm({"facts": ["something"]})
        children = await extract_propositions([_parent()], llm)
        assert children == []

    @pytest.mark.asyncio
    async def test_discards_on_non_list_propositions(self):
        llm = make_mock_llm({"propositions": "not a list"})
        children = await extract_propositions([_parent()], llm)
        assert children == []


class TestEmptyPropositions:
    @pytest.mark.asyncio
    async def test_empty_list_returns_no_children(self):
        llm = make_mock_llm({"propositions": []})
        children = await extract_propositions([_parent()], llm)
        assert children == []


class TestPromptConstruction:
    @pytest.mark.asyncio
    async def test_prompt_requires_strict_json(self):
        assert "ONLY valid JSON" in PROPOSITION_SYSTEM_PROMPT
