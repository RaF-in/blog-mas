"""Tests for rag/chunking/structural.py — Stage 1 header-based split."""

import pytest

from blog_mas.rag.chunking.structural import split_by_headers
from blog_mas.rag.chunking.types import Section


def _body(n_words: int = 60) -> str:
    """Generate a body paragraph with enough words to exceed the 50-token merge threshold."""
    return "word " * n_words


class TestHeaderSplit:
    def test_splits_h1_h2_h3_into_sections(self):
        md = (
            f"# Alpha\n\n{_body()}\n\n"
            f"## Beta\n\n{_body()}\n\n"
            f"### Gamma\n\n{_body()}\n"
        )
        sections = split_by_headers(md)
        assert len(sections) == 3
        assert sections[0].headings_path == ["Alpha"]
        assert sections[1].headings_path == ["Alpha", "Beta"]
        assert sections[2].headings_path == ["Alpha", "Beta", "Gamma"]

    def test_preserves_heading_hierarchy(self):
        md = f"# A\n\n{_body()}\n\n## B\n\n{_body()}\n"
        sections = split_by_headers(md)
        assert sections[0].headings_path == ["A"]
        assert sections[1].headings_path == ["A", "B"]

    def test_no_headings_returns_single_section(self):
        md = f"{_body(80)}"
        sections = split_by_headers(md)
        assert len(sections) == 1
        assert sections[0].headings_path == []

    def test_only_h1_no_h2_h3(self):
        md = f"# First\n\n{_body()}\n\n# Second\n\n{_body()}\n"
        sections = split_by_headers(md)
        assert len(sections) == 2
        assert sections[0].headings_path == ["First"]
        assert sections[1].headings_path == ["Second"]

    def test_out_of_order_h3_under_h1(self):
        md = f"# Top\n\n{_body()}\n\n### Deep\n\n{_body()}\n"
        sections = split_by_headers(md)
        assert len(sections) == 2
        assert sections[1].headings_path == ["Top", "Deep"]

    def test_hash_in_code_block_not_treated_as_heading(self):
        md = f"# Real\n\n```\n# Fake heading\n```\n\n{_body()}\n"
        sections = split_by_headers(md)
        assert len(sections) == 1
        assert "# Fake heading" in sections[0].text


class TestMinSectionMerge:
    def test_short_section_merges_forward(self):
        short_body = "x" * 10
        md = f"# Short\n\n{short_body}\n\n# Long\n\n{_body()}\n"
        sections = split_by_headers(md, min_section_tokens=50)
        assert len(sections) == 1
        assert short_body in sections[0].text
        assert sections[0].headings_path == ["Long"]

    def test_last_short_section_merges_backward(self):
        md = f"# Long\n\n{_body()}\n\n# Short\n\n{'x' * 10}\n"
        sections = split_by_headers(md, min_section_tokens=50)
        assert len(sections) == 1

    def test_min_section_tokens_configurable(self):
        md = "# A\n\nshort.\n\n# B\n\nalso short.\n"
        sections = split_by_headers(md, min_section_tokens=1)
        assert len(sections) == 2

    def test_all_empty_input_returns_empty(self):
        assert split_by_headers("") == []
        assert split_by_headers("   \n  \n") == []


class TestDegenerateInputs:
    def test_empty_markdown_returns_empty(self):
        assert split_by_headers("") == []

    def test_only_headers_no_body(self):
        md = "# A\n\n## B\n\n### C\n"
        sections = split_by_headers(md)
        assert all(isinstance(s, Section) for s in sections)
