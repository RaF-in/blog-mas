"""Chapter 10 §1.4 — Structured JSON logging with trace_id context binding.

Ch10's requirement for production logs:
  "Production logs must be machine-readable … the engine should emit structured
   logs, typically in JSON format … Each entry should include critical metadata
   such as trace_id, which connects the log line to the originating request."

This module configures structlog (already a project dependency) to emit
JSON in production and human-readable console output in development.
Every log line carries:

  timestamp, level, logger, trace_id, request_id, service, event

The ``trace_id`` field is the linchpin: it correlates every log line from
every component (API gateway → FastAPI → Celery worker) back to a single
ExecutionTrace.  This is what the ELK/Splunk/Datadog stack needs to reconstruct
a request's full lifecycle from distributed log lines.

Usage:

  # At application startup (call once):
  from blog_mas.logging_json import setup_logging
  setup_logging()                         # reads LOG_FORMAT / LOG_LEVEL from env

  # In any module — bind context fields for the current request/task:
  import structlog
  log = structlog.get_logger(__name__)
  bound = log.bind(trace_id="abc-123", service="api")
  bound.info("goal received", goal_length=len(goal))

  # The existing logging_config.setup_logging() still works — it is preserved
  # as a shim that calls this module, so no existing imports break.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging(
    level: str | int | None = None,
    log_format: str | None = None,
) -> None:
    """Configure structlog for structured JSON or pretty console output.

    Args:
        level:      Logging level.  Defaults to LOG_LEVEL env var → "INFO".
        log_format: "json" or "console".  Defaults to LOG_FORMAT env var → "console".

    The function is idempotent — safe to call multiple times (subsequent calls
    are a no-op because structlog is already configured).
    """
    resolved_level_str: str = (
        str(level)
        if level is not None
        else os.environ.get("LOG_LEVEL", "INFO")
    )
    resolved_format: str = (
        log_format
        if log_format is not None
        else os.environ.get("LOG_FORMAT", "console")
    ).lower()

    # Map string → int for stdlib logging.
    numeric_level = getattr(logging, resolved_level_str.upper(), logging.INFO)

    # ── Shared processors — run for every log call regardless of renderer ──
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,     # picks up bound trace_id etc.
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if resolved_format == "json":
        # ── Production: machine-readable JSON ─────────────────────────────
        # Compatible with ELK, Splunk, CloudWatch Logs, Datadog.
        # Fluentd/Fluent Bit can forward these lines without transformation.
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # ── Development: colourised console output ──────────────────────
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Keep stdlib logging in sync (langchain, celery, uvicorn all use it).
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )
    # Suppress noisy libraries in production.
    if resolved_format == "json":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("celery").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Context helpers — bind/clear trace_id for the duration of a request or task
# ---------------------------------------------------------------------------

def bind_trace_context(
    trace_id: str,
    request_id: str | None = None,
    service: str = "unknown",
) -> None:
    """Bind trace_id (and optionally request_id) to the structlog context.

    Call this at the start of every request handler and Celery task so that
    every subsequent log.info/warning/error within that scope automatically
    includes the identifiers without explicit passing.

    Structlog's contextvars integration makes this thread/async-safe.
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        trace_id=trace_id,
        service=service,
        **({"request_id": request_id} if request_id else {}),
    )


def clear_trace_context() -> None:
    """Clear all bound context vars (call at the end of a request/task)."""
    structlog.contextvars.clear_contextvars()
