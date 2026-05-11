"""Ch10 §1.4 — Distributed tracing with OpenTelemetry → OTLP → Jaeger.

Ch10:
  "In an asynchronous deployment, a single request may pass through multiple
   services (the API gateway, FastAPI server, message broker, and Celery
   worker) before completing.  Distributed tracing tools such as Jaeger or
   Zipkin make these flows visible end to end.  By linking each component's
   spans under a shared trace ID, tracing exposes bottlenecks and highlights
   where latency is introduced across the distributed workflow."

Architecture:
  FastAPI request → OpenTelemetry SDK → OTLP exporter → OTel Collector
  → Jaeger backend (see deploy/docker-compose.yml)

  The span hierarchy for a single blog-mas request looks like:
    POST /api/v1/execute         (API span)
    └── pre_flight_moderation    (child span)
    └── celery.apply_async       (child span — async boundary)
        └── execute_goal task    (Worker span — linked via trace context propagation)
            └── run_with_policy  (child span)
                └── engine_invocation
                    └── Planner
                    └── Executor Step 1: Researcher
                    └── Executor Step 2: Librarian
                    └── Executor Step 3: Writer
                └── post_flight_moderation

Usage:
  # Call once at startup:
  from blog_mas.service.tracing import setup_tracing
  setup_tracing()

  # Instrument a block manually:
  from blog_mas.service.tracing import get_tracer
  tracer = get_tracer()
  with tracer.start_as_current_span("my_operation") as span:
      span.set_attribute("goal_length", len(goal))
      ...
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional import — tracing is optional at runtime
# ---------------------------------------------------------------------------
# In local development OTEL_ENABLED=false (the default in .env.example), so
# the OTLP exporter is not imported and no connection is attempted.
# In production OTEL_ENABLED=true and opentelemetry packages must be installed
# (listed in the [prod] optional dependency group in pyproject.toml).


def setup_tracing() -> None:
    """Configure the OpenTelemetry SDK and OTLP exporter.

    Reads configuration from the Settings object (which reads from env vars).
    Safe to call unconditionally — when OTEL_ENABLED=false this is a no-op
    and no OTLP connection is attempted.
    """
    from blog_mas.config import get_settings
    settings = get_settings()

    if not settings.observability.otel_enabled:
        logger.info(
            "[tracing] OTEL_ENABLED=false — distributed tracing is disabled. "
            "Set OTEL_ENABLED=true and OTEL_EXPORTER_OTLP_ENDPOINT to activate."
        )
        return

    try:
        from opentelemetry import trace  # type: ignore[import]
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore[import]
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # type: ignore[import]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import]
    except ImportError as exc:
        logger.warning(
            "[tracing] OpenTelemetry packages not installed (%s). "
            "Add opentelemetry-api, opentelemetry-sdk, "
            "opentelemetry-exporter-otlp, opentelemetry-instrumentation-fastapi, "
            "opentelemetry-instrumentation-httpx to the [prod] dependency group.",
            exc,
        )
        return

    resource = Resource.create({
        "service.name": settings.observability.otel_service_name,
        "service.version": "0.1.0",
        "deployment.environment": settings.env,
    })

    exporter = OTLPSpanExporter(
        endpoint=settings.observability.otel_exporter_otlp_endpoint,
        insecure=True,  # TLS terminated at the collector in production.
    )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI and httpx (covers LLM API calls).
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    logger.info(
        "[tracing] OpenTelemetry configured → %s (service=%s)",
        settings.observability.otel_exporter_otlp_endpoint,
        settings.observability.otel_service_name,
    )


def get_tracer(name: str = "blog_mas"):
    """Return an OpenTelemetry tracer.  Returns a no-op tracer if OTEL is disabled."""
    try:
        from opentelemetry import trace  # type: ignore[import]
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


# ---------------------------------------------------------------------------
# No-op tracer — used when opentelemetry is not installed
# ---------------------------------------------------------------------------

class _NoOpSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def set_attribute(self, key: str, value) -> None:
        pass

    def set_status(self, *_) -> None:
        pass

    def record_exception(self, exc) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **_):
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Helper: inject trace context into Celery task headers (W3C TraceContext)
# ---------------------------------------------------------------------------

def get_current_span_context() -> dict[str, str]:
    """Extract the current W3C TraceContext headers for Celery header injection.

    Celery workers read these from the task's headers and continue the same
    distributed trace, so the API span and Worker span appear as one trace
    in Jaeger — the "end-to-end" visibility Ch10 describes.
    """
    headers: dict[str, str] = {}
    try:
        from opentelemetry import propagate  # type: ignore[import]
        propagate.inject(headers)
    except (ImportError, Exception):
        pass
    return headers


def extract_span_context_from_headers(headers: dict[str, str]) -> None:
    """On the worker side: extract the W3C context and make it current.

    Called at the start of a Celery task so its spans appear as children
    of the originating API request span in Jaeger.
    """
    try:
        from opentelemetry import propagate, context  # type: ignore[import]
        ctx = propagate.extract(headers)
        token = context.attach(ctx)
        return token  # type: ignore[return-value]
    except (ImportError, Exception):
        return None
