"""Ch10 §1.4 — Prometheus metrics for the Context Engine service.

Ch10's monitoring requirements:
  System metrics:    CPU utilisation, memory, network I/O  (node-exporter)
  Application metrics: request latency, error rates, throughput, task queue length
  AI-specific metrics: total token consumption (cost tracking),
                       LLM call latency, vector DB query latency

This module owns the AI-specific and application-level metrics.  System metrics
are collected by the Prometheus node-exporter sidecar (declared in K8s manifests).

All metrics are registered once at import time.  The FastAPI app exposes them
via the /metrics endpoint using prometheus_client's built-in ASGI handler.

Usage:
  # Count a completed task:
  from blog_mas.service.metrics import TASKS_COMPLETED
  TASKS_COMPLETED.labels(status="ok").inc()

  # Observe LLM call latency:
  with LLM_CALL_LATENCY.labels(model="gpt-4o").time():
      response = await call_llm(prompt)
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram, Summary

# ---------------------------------------------------------------------------
# Application metrics
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total HTTP requests received by the API service.",
    labelnames=["route", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "End-to-end HTTP request latency (seconds).",
    labelnames=["route", "method"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ---------------------------------------------------------------------------
# Task / queue metrics — Ch10 §1.3 (task queue length as autoscale signal)
# ---------------------------------------------------------------------------

TASKS_SUBMITTED = Counter(
    "tasks_submitted_total",
    "Total goals submitted to the Celery task queue.",
)

TASKS_COMPLETED = Counter(
    "tasks_completed_total",
    "Total tasks completed by Celery workers.",
    labelnames=["status"],  # ok | blocked_pre | redacted_post | engine_error
)

TASK_DURATION = Histogram(
    "task_duration_seconds",
    "Wall-clock time from task enqueue to result store write (seconds).",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

TASK_QUEUE_LENGTH = Gauge(
    "task_queue_length",
    "Current number of pending tasks in the Celery queue (scraped from Redis).",
)

# ---------------------------------------------------------------------------
# AI-specific metrics — Ch10 §1.4
# Ch10: "Total token consumption (for cost tracking), latency of external LLM
#         calls (e.g. OpenAI API response time), and vector database query
#         performance (e.g. Pinecone latency)."
# ---------------------------------------------------------------------------

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Cumulative LLM tokens consumed across all tasks.",
    labelnames=["type", "model"],  # type: prompt | completion
)

LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Estimated cumulative LLM cost in USD (computed from token counts + pricing table).",
    labelnames=["model"],
)

LLM_CALL_LATENCY = Histogram(
    "llm_call_latency_seconds",
    "Latency of individual LLM API calls (seconds).",
    labelnames=["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

VECTOR_QUERY_LATENCY = Histogram(
    "vector_query_latency_seconds",
    "Latency of Qdrant vector similarity queries (seconds).",
    labelnames=["collection"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

ENGINE_DURATION = Histogram(
    "engine_duration_seconds",
    "Total wall-clock time for run_context_engine() to complete (seconds).",
    buckets=[1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)

# ---------------------------------------------------------------------------
# Security / moderation metrics — Ch10 §2.4
# ---------------------------------------------------------------------------

MODERATION_BLOCKS_TOTAL = Counter(
    "moderation_blocks_total",
    "Total requests blocked or redacted by the moderation perimeter.",
    labelnames=["stage", "category"],  # stage: api_pre_flight | pre_flight | post_flight
)

SANITIZER_REJECTIONS_TOTAL = Counter(
    "sanitizer_rejections_total",
    "Total chunks rejected by the sanitiser at ingest or retrieval time.",
    labelnames=["stage"],  # ingest | retrieval
)

# ---------------------------------------------------------------------------
# Context-reduction metrics — Ch10 §2.1
# ---------------------------------------------------------------------------

SUMMARIZER_TRIGGERS_TOTAL = Counter(
    "summarizer_triggers_total",
    "Times the proactive context-reduction policy auto-invoked the Summarizer agent.",
)

SUMMARIZER_TOKEN_SAVINGS = Summary(
    "summarizer_token_savings_percent",
    "Percentage of tokens saved when the Summarizer fired (100 * (1 - out/in)).",
)

# ---------------------------------------------------------------------------
# Helper: record LLM token usage from the existing tokens.py helpers
# ---------------------------------------------------------------------------

def record_llm_usage(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    estimated_cost_usd: float = 0.0,
) -> None:
    """Update all LLM-specific counters after a completed call.

    Called from the Celery task after run_context_engine returns its trace.
    The trace's step outputs carry token counts logged by agent_helpers.
    """
    LLM_TOKENS_TOTAL.labels(type="prompt", model=model).inc(prompt_tokens)
    LLM_TOKENS_TOTAL.labels(type="completion", model=model).inc(completion_tokens)
    if estimated_cost_usd > 0:
        LLM_COST_USD_TOTAL.labels(model=model).inc(estimated_cost_usd)


@contextmanager
def measure_llm_call(model: str):
    """Context manager that records LLM call latency in the histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        LLM_CALL_LATENCY.labels(model=model).observe(time.perf_counter() - start)


@contextmanager
def measure_vector_query(collection: str):
    """Context manager that records vector query latency in the histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        VECTOR_QUERY_LATENCY.labels(collection=collection).observe(
            time.perf_counter() - start
        )


def update_queue_length(length: int) -> None:
    """Update the task_queue_length gauge.

    Called from the /metrics scrape handler or a background thread that reads
    the Celery inspect API / Redis LLEN on the task queue key.
    """
    TASK_QUEUE_LENGTH.set(length)
