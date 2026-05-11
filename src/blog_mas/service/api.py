"""Ch10 §1.2 — FastAPI orchestration layer: the Context Engine as a service.

Ch10:
  "This is the control deck, a high-performance, asynchronous layer that
   validates the request structure.  The API Service does not execute the
   engine itself.  Instead, it dispatches the goal to the Task Queue
   (Celery/RabbitMQ).  This decoupling is vital as it ensures the API
   remains responsive, immediately acknowledging the request even if the
   engine is under heavy load."

Endpoints:
  POST /api/v1/execute           — submit a goal → 202 Accepted + trace_id
  GET  /api/v1/status/{trace_id} — poll for result
  GET  /api/v1/trace/{trace_id}  — full ExecutionTrace (compliance / audit)
  POST /api/v1/webhook/register  — register a callback URL for a trace_id
  GET  /metrics                  — Prometheus exposition
  GET  /healthz                  — liveness probe (K8s)
  GET  /readyz                   — readiness probe (K8s)

Run locally:
  uvicorn blog_mas.service.api:app --host 0.0.0.0 --port 8000 --reload

Or via the CLI:
  uv run blog-mas serve
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from blog_mas.config import get_settings
from blog_mas.logging_json import bind_trace_context, clear_trace_context, setup_logging
from blog_mas.security.moderation import helper_moderate_content
from blog_mas.service.auth import require_api_key
from blog_mas.service.metrics import (
    MODERATION_BLOCKS_TOTAL,
    REQUESTS_TOTAL,
    REQUEST_LATENCY,
    TASKS_SUBMITTED,
    update_queue_length,
)
from blog_mas.service.result_store import get_result_store
from blog_mas.service.schemas import (
    ErrorResponse,
    GoalAcceptedResponse,
    GoalRequest,
    StatusResponse,
    TraceResponse,
    WebhookPayload,
    WebhookRegistration,
)
from blog_mas.service.tracing import get_current_span_context, setup_tracing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state — shared across requests without re-initialisation
# ---------------------------------------------------------------------------

_app_state: dict[str, Any] = {}

# Webhook registry: {trace_id → callback_url} (in-memory, not replicated).
# For multi-worker production, move this to Redis.
_webhook_registry: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources once at startup.

    Ch10 §1.2: "Assume clients (OpenAI, Pinecone) are initialized at startup"
    — FastAPI's lifespan is the clean place to do this.
    """
    settings = get_settings()

    # Structured logging must be set up before anything else logs.
    setup_logging(
        level=settings.observability.log_level,
        log_format=settings.observability.log_format,
    )

    logger.info(
        "[api] Starting Context Engine Service (env=%s, model=%s)",
        settings.env,
        settings.llm.generation_model,
    )

    # Distributed tracing (no-op when OTEL_ENABLED=false).
    setup_tracing()

    # Result store — Redis in prod, filesystem fallback for dev.
    _app_state["result_store"] = get_result_store()
    logger.info(
        "[api] Result store: %s", _app_state["result_store"].name
    )

    # Lazy-import heavy dependencies so the module loads fast.
    from blog_mas.engine.agent_adapters import build_default_registry
    from blog_mas.llm import create_llm
    from blog_mas.rag.embedding import EmbeddingClient
    from blog_mas.rag.vector_store import QdrantStore

    llm = create_llm()
    store = QdrantStore()
    embedder = EmbeddingClient()
    registry = build_default_registry(llm, store, embedder, reranker=None)

    _app_state["llm"] = llm
    _app_state["store"] = store
    _app_state["embedder"] = embedder
    _app_state["registry"] = registry

    logger.info("[api] Context Engine Service ready.")

    yield  # ← application runs here

    logger.info("[api] Shutting down Context Engine Service.")
    _app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Context Engine Service",
    description=(
        "Production API for the glass-box Context Engine.  "
        "Ch10: transforms the prototype into a scalable, enterprise-grade service."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production via the API gateway
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request-ID binding + latency tracking
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    """Bind a request_id to structlog context and record latency per route."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    route = request.url.path
    method = request.method

    bind_trace_context(trace_id="", request_id=request_id, service="api")

    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(route=route, method=method).observe(duration)
        REQUESTS_TOTAL.labels(
            route=route, method=method, status_code=str(response.status_code)
        ).inc()
        return response
    except Exception as exc:
        duration = time.perf_counter() - start
        REQUEST_LATENCY.labels(route=route, method=method).observe(duration)
        REQUESTS_TOTAL.labels(route=route, method=method, status_code="500").inc()
        raise
    finally:
        clear_trace_context()


# ---------------------------------------------------------------------------
# Helper: enqueue a task via Celery (lazy import to keep startup fast)
# ---------------------------------------------------------------------------

def _enqueue_task(
    trace_id: str,
    goal: str,
    template_name: str,
    namespace: str,
    require_audit_trace: bool,
    moderation_active: bool,
    configuration_overrides: dict | None,
) -> None:
    """Dispatch the goal to a Celery worker.

    Ch10 §1.3:
      "The API validates the request and pushes the task onto a message broker
       or task queue."

    The task headers carry the current OTEL trace context so the worker span
    appears as a child of the API span in Jaeger.
    """
    from blog_mas.workers.tasks import execute_goal  # lazy import

    span_context = get_current_span_context()

    execute_goal.apply_async(
        kwargs={
            "trace_id": trace_id,
            "goal": goal,
            "template_name": template_name,
            "namespace": namespace,
            "require_audit_trace": require_audit_trace,
            "moderation_active": moderation_active,
            "configuration_overrides": configuration_overrides or {},
        },
        task_id=trace_id,  # Celery task ID = trace_id for correlation
        headers={"otel_context": json.dumps(span_context)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/execute",
    response_model=GoalAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a goal for async execution",
    description=(
        "Validates the request, runs API-edge pre-flight moderation, "
        "then enqueues the goal for a Celery worker.  "
        "Returns 202 Accepted immediately with a trace_id for polling."
    ),
    dependencies=[Depends(require_api_key)],
)
async def execute_goal_endpoint(request_body: GoalRequest) -> GoalAcceptedResponse:
    """Ch10 §1.2 + §1.3 + §2.4 combined endpoint.

    Three responsibilities:
    1.  Validate the request (Pydantic does this automatically).
    2.  API-edge pre-flight moderation (Ch10 §2.4): screen the goal BEFORE
        it consumes any queue or worker resources.
    3.  Enqueue the task (Ch10 §1.3): dispatch to Celery, return 202.
    """
    settings = get_settings()

    # ── 1. Assign a trace_id ────────────────────────────────────────────────
    trace_id = uuid.uuid4().hex
    bind_trace_context(trace_id=trace_id, service="api")

    logger.info(
        "[api] Goal received",
        goal_preview=request_body.goal[:80],
        template=request_body.template_name,
        trace_id=trace_id,
    )

    # ── 2. API-edge pre-flight moderation (Ch10 §2.4) ───────────────────────
    # Ch10: "By checking user goals BEFORE they are sent to the task queue,
    # the system prevents malicious or inappropriate requests from ever
    # consuming computational resources, acting as a first line of defense."
    if settings.security.api_preflight_moderation and request_body.moderation_active:
        moderation_report = helper_moderate_content(request_body.goal)
        if moderation_report.flagged:
            category = moderation_report.top_category
            logger.warning(
                "[api] API-edge pre-flight BLOCK",
                trace_id=trace_id,
                category=category,
                source=moderation_report.source,
            )
            MODERATION_BLOCKS_TOTAL.labels(
                stage="api_pre_flight", category=category
            ).inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Request blocked by content moderation policy "
                    f"(category: {category}).  "
                    "If you believe this is a false positive, contact your "
                    f"administrator with reference: {trace_id}"
                ),
            )

    # ── 3. Register as pending in the result store ──────────────────────────
    result_store = _app_state.get("result_store") or get_result_store()
    result_store.create_pending(trace_id)

    # ── 4. Enqueue the task (Celery) ────────────────────────────────────────
    try:
        _enqueue_task(
            trace_id=trace_id,
            goal=request_body.goal,
            template_name=request_body.template_name,
            namespace=request_body.namespace,
            require_audit_trace=request_body.require_audit_trace,
            moderation_active=request_body.moderation_active,
            configuration_overrides=request_body.configuration_overrides,
        )
        TASKS_SUBMITTED.inc()
        logger.info("[api] Task enqueued", trace_id=trace_id)
    except Exception as exc:
        logger.exception("[api] Failed to enqueue task", trace_id=trace_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to enqueue task: {exc}",
        )

    # ── 5. Return 202 Accepted immediately ─────────────────────────────────
    # Ch10 §1.3 step 3: "The API immediately returns an HTTP status 202
    # Accepted response to the client, including trace_id."
    return GoalAcceptedResponse(
        status="accepted",
        trace_id=trace_id,
        message=(
            f"Goal accepted and queued for execution.  "
            f"Poll GET /api/v1/status/{trace_id} for results."
        ),
    )


@app.get(
    "/api/v1/status/{trace_id}",
    response_model=StatusResponse,
    summary="Poll execution status",
    description=(
        "Returns the current status of a submitted goal.  "
        "Poll until status is one of: ok | blocked_pre | redacted_post | engine_error."
    ),
    dependencies=[Depends(require_api_key)],
)
async def get_status(trace_id: str) -> StatusResponse:
    """Ch10 §1.3 step 6 — client polls this endpoint for results."""
    result_store = _app_state.get("result_store") or get_result_store()
    entry = result_store.get(trace_id)

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"trace_id '{trace_id}' not found.  "
                   "It may have expired or was never submitted.",
        )

    return StatusResponse(
        trace_id=trace_id,
        status=entry["status"],
        final_output=entry.get("final_output"),
        metadata=entry.get("metadata", {}),
    )


@app.get(
    "/api/v1/trace/{trace_id}",
    response_model=TraceResponse,
    summary="Retrieve full ExecutionTrace (audit)",
    description=(
        "Returns the complete ExecutionTrace for a completed execution.  "
        "Ch10 §2.2: 'a compliance officer or user can retrieve the trace and "
        "see the exact source documents and page numbers used, satisfying "
        "regulatory requirements for explainability.'"
    ),
    dependencies=[Depends(require_api_key)],
)
async def get_trace(trace_id: str) -> TraceResponse:
    """Ch10 §2.2 — immutable audit log retrieval for compliance officers."""
    result_store = _app_state.get("result_store") or get_result_store()
    entry = result_store.get(trace_id)

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"trace_id '{trace_id}' not found.",
        )

    trace_dict: dict = entry.get("trace") or {}

    return TraceResponse(
        trace_id=trace_id,
        goal=trace_dict.get("goal", ""),
        plan=trace_dict.get("plan"),
        steps=trace_dict.get("steps", []),
        status=entry["status"],
        final_output=entry.get("final_output"),
        started_at=trace_dict.get("started_at"),
        duration_seconds=trace_dict.get("duration_seconds"),
        metadata=entry.get("metadata", {}),
    )


@app.post(
    "/api/v1/webhook/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a webhook for task completion",
    description=(
        "Register a callback URL.  When the task with the given trace_id "
        "completes, the service POSTs a WebhookPayload to the callback URL.  "
        "Ch10 §1.3: 'receives the result via a webhook' as an alternative to polling."
    ),
    dependencies=[Depends(require_api_key)],
)
async def register_webhook(payload: WebhookRegistration) -> dict:
    """Ch10 §1.3 — webhook registration (alternative to polling)."""
    _webhook_registry[payload.trace_id] = payload.callback_url
    logger.info(
        "[api] Webhook registered",
        trace_id=payload.trace_id,
        callback_url=payload.callback_url[:60],
    )
    return {"registered": True, "trace_id": payload.trace_id}


async def _fire_webhook(trace_id: str, status_value: str) -> None:
    """POST the WebhookPayload to the registered callback URL.

    Called from the Celery task after it saves the result, so the API process
    itself is not blocked.  Uses httpx (already a project dependency).
    """
    callback_url = _webhook_registry.pop(trace_id, None)
    if not callback_url:
        return

    import hmac
    import hashlib
    import httpx

    settings = get_settings()
    payload = WebhookPayload(
        trace_id=trace_id,
        status=status_value,
        timestamp=datetime.now(timezone.utc).isoformat(),
        result_url=f"/api/v1/status/{trace_id}",
    )
    body = payload.model_dump_json()

    # HMAC signature for the receiver to verify authenticity.
    secret = (settings.security.webhook_secret or "").encode()
    signature = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                callback_url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "x-blog-mas-signature": signature,
                },
            )
        logger.info(
            "[api] Webhook delivered", trace_id=trace_id, callback_url=callback_url
        )
    except Exception as exc:
        logger.warning(
            "[api] Webhook delivery failed",
            trace_id=trace_id,
            callback_url=callback_url,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Observability endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Ch10 §1.4 — Prometheus metrics exposition endpoint.

    Prometheus scrapes this URL at the interval defined in prometheus.yml.
    Grafana reads from Prometheus to render the dashboards Ch10 describes.
    """
    # Update the queue-length gauge just before each scrape.
    result_store = _app_state.get("result_store")
    if result_store and hasattr(result_store, "queue_length"):
        try:
            update_queue_length(result_store.queue_length())
        except Exception:
            pass

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/healthz", include_in_schema=False)
async def liveness() -> dict:
    """K8s liveness probe — returns 200 as long as the process is alive.

    Ch10 §1.6: K8s Deployments reference this path in their liveness probe spec.
    """
    return {"status": "alive"}


@app.get("/readyz", include_in_schema=False)
async def readiness() -> dict:
    """K8s readiness probe — returns 200 only when all dependencies are reachable.

    Ch10 §1.6: K8s waits for this to return 200 before sending traffic
    to a newly-started pod (or after a rolling update).
    """
    checks: dict[str, str] = {}

    # Check result store (Redis).
    result_store = _app_state.get("result_store")
    if result_store:
        try:
            if hasattr(result_store, "_client"):
                result_store._client.ping()
            checks["result_store"] = "ok"
        except Exception as exc:
            checks["result_store"] = f"error: {exc}"
    else:
        checks["result_store"] = "not_initialized"

    # Check that the registry is loaded.
    checks["registry"] = "ok" if _app_state.get("registry") else "not_initialized"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "checks": checks},
        )

    return {"status": "ready", "checks": checks}
