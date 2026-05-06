"""Tests for the recall@k eval harness."""

import pytest

from blog_mas.eval.recall import compute_recall, load_queries


class TestRecallComputation:
    def test_perfect_recall_at_k(self):
        assert compute_recall(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_partial_recall(self):
        assert compute_recall(["a", "b", "c"], ["a", "d"], k=3) == 0.5

    def test_zero_recall(self):
        assert compute_recall(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_recall_at_k_1(self):
        assert compute_recall(["a", "b"], ["a"], k=1) == 1.0
        assert compute_recall(["b", "a"], ["a"], k=1) == 0.0

    def test_empty_expected_returns_zero(self):
        assert compute_recall(["a", "b"], [], k=3) == 0.0


class TestQueryLoading:
    def test_loads_queries_yaml(self):
        queries = load_queries()
        assert len(queries) >= 5
        assert all("query" in q and "expected_doc_ids" in q for q in queries)
