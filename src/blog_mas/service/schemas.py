"""Ch10 §1.2 — Pydantic request/response schemas for the FastAPI service.

Mirrors the book's GoalRequest / ExecutionResponse shapes and extends them
with the async lifecycle fields required for the 202 Accepted / polling pattern.

The book:
  class GoalRequest(BaseModel):
      goal: str
      configuration_overrides: Optional[Dict[str, Any]] = None
      require_audit_trace: bool = False

  class ExecutionResponse(BaseModel):
      status: str
      trace_id: str
      final_output: Optional[str] = None
      metadata: Dict[str, Any]

We keep the same names and extend them for the async lifecycle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound — what clients POST to /api/v1/execute
# ---------------------------------------------------------------------------

class GoalRequest(BaseModel):
    """The contract a client sends to submit a goal to the Context Engine.

    Ch10 book shape preserved exactly; additional fields are additive.
    """

    goal: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The high-level goal string the engine will plan and execute.",
        examples=["Write a cited blog post about the Juno Jupiter mission."],
    )
    template_name: str = Field(
        default="high_fidelity_rag",
        description=(
            "Which ControlDeck template to use: "
            "'high_fidelity_rag' | 'context_reduction' | 'grounded_reasoning'"
        ),
    )
    namespace: str = Field(
        default="default",
        description="Qdrant namespace for sanitiser routing (Ch8 §I).",
    )
    configuration_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Optional overrides merged into EngineConfig at dispatch time.",
    )
    require_audit_trace: bool = Field(
        default=False,
        description=(
            "If True, the full ExecutionTrace JSON is saved to the trace store "
            "and retrievable via GET /api/v1/trace/{trace_id}.  "
            "Ch10 §2.2: the audit trail for compliance officers."
        ),
    )
    moderation_active: bool = Field(
        default=True,
        description="Toggle the moderation perimeter for this request.",
    )


# ---------------------------------------------------------------------------
# Outbound — immediate 202 response after enqueue
# ---------------------------------------------------------------------------

class GoalAcceptedResponse(BaseModel):
    """The 202 Accepted response returned immediately after enqueueing.

    Ch10 §1.3 lifecycle step 3:
    "The API immediately returns an HTTP status 202 Accepted response to the
    client, including trace_id."
    """

    status: str = Field(
        default="accepted",
        description="Always 'accepted' for a 202 response.",
    )
    trace_id: str = Field(
        description="Unique identifier for this execution. Use to poll /status or /trace.",
    )
    message: str = Field(
        default="Goal accepted and queued for execution.",
    )


# ---------------------------------------------------------------------------
# Outbound — polling response from GET /api/v1/status/{trace_id}
# ---------------------------------------------------------------------------

class StatusResponse(BaseModel):
    """Result of polling the status endpoint.

    Ch10 §1.3 lifecycle step 6:
    "The client uses trace_id to poll a status endpoint or receives the result
    via a webhook."
    """

    trace_id: str
    status: str = Field(
        description=(
            "One of: pending | running | ok | blocked_pre | redacted_post | "
            "engine_error | not_found"
        ),
    )
    final_output: str | None = Field(
        default=None,
        description="The engine's final output text, if status='ok'.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Execution metadata: engine_used, duration_seconds, "
            "pre_flight_flagged, post_flight_flagged, token_savings."
        ),
    )


# ---------------------------------------------------------------------------
# Outbound — full trace from GET /api/v1/trace/{trace_id}
# ---------------------------------------------------------------------------

class TraceResponse(BaseModel):
    """Full ExecutionTrace for compliance / auditability.

    Ch10 §2.2: "For any given output, a compliance officer or user can retrieve
    the trace and see the exact source documents and page numbers used."
    """

    trace_id: str
    goal: str
    plan: Any | None = None
    steps: list[dict[str, Any]] = Field(default_factory=list)
    status: str
    final_output: Any | None = None
    started_at: str | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Outbound — webhook payload
# ---------------------------------------------------------------------------

class WebhookPayload(BaseModel):
    """Payload POSTed to a registered webhook URL when a task completes.

    Ch10 §1.3: "receives the result via a webhook" (alternative to polling).
    """

    event: str = "execution.completed"
    trace_id: str
    status: str
    timestamp: str
    result_url: str = Field(
        description="URL the receiver can call to fetch the full result.",
    )


# ---------------------------------------------------------------------------
# Inbound — webhook registration
# ---------------------------------------------------------------------------

class WebhookRegistration(BaseModel):
    """Register a callback URL for a specific trace_id."""

    trace_id: str
    callback_url: str = Field(
        description="HTTPS URL that will receive a POST when the task completes.",
    )


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Uniform error shape for all 4xx/5xx responses."""

    error: str
    detail: str | None = None
    trace_id: str | None = None
