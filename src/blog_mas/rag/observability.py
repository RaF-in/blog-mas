"""RAG observability: structlog for structured logs, LangSmith tracer with no-op fallback."""

import os

import structlog


def get_logger(**kwargs) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with the given key-value pairs."""
    return structlog.get_logger().bind(**kwargs)


def get_tracer():
    """Return a LangSmith tracer, or a no-op when LANGCHAIN_API_KEY is unset."""
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return _NoOpTracer()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    from langsmith import Client
    return Client()


class _NoOpTracer:
    """No-op tracer used when LangSmith is not configured."""

    def trace(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self
