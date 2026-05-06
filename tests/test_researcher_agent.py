"""Tests for the Researcher agent — hybrid retrieval + citation synthesis."""

import pytest

from blog_mas.agents.researcher import research_node, _build_user_message
from blog_mas.mcp.models import BlogSpec, ResearchSummary
from blog_mas.prompts import RESEARCHER_SYSTEM_PROMPT
from tests.conftest import FakeEmbedder, FakeReranker, FakeVectorStore, make_config, make_failing_llm, make_mock_llm


def _make_blog_spec(topic="Mediterranean diet"):
    return BlogSpec(topic=topic, audience="general readers", tone="informative", goal="educate", constraints=[])


def _make_summary(topic="Mediterranean diet", bullets=None, source="chunk1,chunk2"):
    return ResearchSummary(
        topic=topic,
        bullet_points=bullets or ["Fact one.", "Fact two."],
        source=source,
    )


def _seed_knowledge(store, chunks):
    store.ensure_collection("knowledge", dim=4)
    for cid, vec, payload in chunks:
        store.upsert_points("knowledge", [{"id": cid, "vector": vec, "payload": payload}])


def _retrieval_config(store, summary=None):
    return {
        "configurable": {
            "llm": make_mock_llm(summary or _make_summary()),
            "store": store,
            "embedder": FakeEmbedder(dim=4),
            "reranker": FakeReranker(),
        },
    }


class TestHybridRetrieval:
    @pytest.mark.asyncio
    async def test_retrieves_and_synthesizes_with_citations(self):
        store = FakeVectorStore(dim=4)
        _seed_knowledge(store, [
            ("chunk1", [0.5] * 4, {"raw_text": "Olive oil is healthy."}),
            ("chunk2", [0.6] * 4, {"raw_text": "Fish reduces heart risk."}),
        ])

        result = await research_node(
            {"blog_spec": _make_blog_spec(), "topic_query": "Mediterranean diet"},
            config=_retrieval_config(store),
        )

        assert isinstance(result["research_summary"], ResearchSummary)
        assert "chunk1" in result["research_summary"].source

    @pytest.mark.asyncio
    async def test_reads_topic_query_falls_back_to_blog_spec_topic(self):
        store = FakeVectorStore(dim=4)
        store.ensure_collection("knowledge", dim=4)

        result = await research_node(
            {"blog_spec": _make_blog_spec("AI safety"), "topic_query": None},
            config=_retrieval_config(store, _make_summary(source="none")),
        )
        assert isinstance(result["research_summary"], ResearchSummary)


class TestGracefulHandling:
    @pytest.mark.asyncio
    async def test_empty_retrieval_returns_empty_bullets(self):
        store = FakeVectorStore(dim=4)
        store.ensure_collection("knowledge", dim=4)

        result = await research_node(
            {"blog_spec": _make_blog_spec(), "topic_query": "obscure topic"},
            config=_retrieval_config(store, _make_summary(bullets=["No info."], source="none")),
        )

        assert result["research_summary"].source == "none"

    @pytest.mark.asyncio
    async def test_raises_on_missing_blog_spec(self):
        with pytest.raises(ValueError, match="no blog spec"):
            await research_node(
                {"blog_spec": None},
                config=make_config(make_mock_llm(_make_summary())),
            )


class TestPromptConstruction:
    def test_user_message_contains_citation_blocks(self):
        from blog_mas.rag.vector_store import ScoredPoint
        chunks = [
            ScoredPoint(id="c1", score=0.5, payload={"raw_text": "Text one."}),
            ScoredPoint(id="c2", score=0.4, payload={"raw_text": "Text two."}),
        ]
        spec = _make_blog_spec()
        msg = _build_user_message(chunks, spec)

        assert "[Source c1]" in msg
        assert "[Source c2]" in msg
        assert "Audience:" in msg
        assert "Goal:" in msg

    def test_user_message_empty_chunks(self):
        spec = _make_blog_spec()
        msg = _build_user_message([], spec)

        assert "No source material" in msg

    def test_system_prompt_includes_citation_rules(self):
        assert "[Source" in RESEARCHER_SYSTEM_PROMPT
