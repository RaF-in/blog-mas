"""Tests for the simulated knowledge base with topic lookup."""

from blog_mas.knowledge_base import lookup_topic, _KNOWLEDGE_BASE


class TestTopicLookup:
    def test_returns_content_for_exact_topic_match(self):
        result = lookup_topic("Mediterranean diet")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_content_for_case_insensitive_match_lowercase(self):
        result = lookup_topic("mediterranean diet")
        assert result is not None
        assert result == lookup_topic("Mediterranean diet")

    def test_returns_content_for_case_insensitive_match_uppercase(self):
        result = lookup_topic("MEDITERRANEAN DIET")
        assert result is not None
        assert result == lookup_topic("Mediterranean diet")

    def test_returns_content_for_partial_match_diet(self):
        result = lookup_topic("diet")
        assert result is not None
        assert result == lookup_topic("Mediterranean diet")

    def test_returns_content_for_partial_match_climate(self):
        result = lookup_topic("climate")
        assert result is not None
        assert result == lookup_topic("Climate change")

    def test_returns_none_for_unknown_topic(self):
        result = lookup_topic("quantum physics")
        assert result is None


class TestContentRequirements:
    def test_contains_exactly_five_topics(self):
        assert len(_KNOWLEDGE_BASE) == 5

    def test_each_topic_has_substantial_content(self):
        for topic, content in _KNOWLEDGE_BASE.items():
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            assert len(paragraphs) >= 3, (
                f"Topic '{topic}' has only {len(paragraphs)} paragraphs, expected at least 3"
            )
