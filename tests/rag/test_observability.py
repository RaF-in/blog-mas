"""Tests for RAG observability: structlog config + LangSmith tracer."""

import os
from unittest.mock import patch

import pytest


class TestStructlogConfig:
    def test_get_logger_returns_bound_logger(self):
        from blog_mas.rag.observability import get_logger

        logger = get_logger(stage="chunking")
        assert logger is not None

    def test_get_logger_binds_fields(self):
        from blog_mas.rag.observability import get_logger

        logger = get_logger(stage="retrieval", query="test query")
        assert "stage" in logger._context
        assert logger._context["stage"] == "retrieval"
        assert logger._context["query"] == "test query"


class TestLangSmithTracer:
    def test_no_op_when_api_key_unset(self):
        from blog_mas.rag.observability import get_tracer

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LANGCHAIN_API_KEY", None)
            tracer = get_tracer()
            assert tracer is not None

    def test_returns_tracer_when_key_set(self):
        from blog_mas.rag.observability import get_tracer

        with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "test-key", "LANGCHAIN_TRACING_V2": "true"}):
            tracer = get_tracer()
            assert tracer is not None
