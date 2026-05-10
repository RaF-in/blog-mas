"""Chapter 8 §G — The meta-controller pattern.

The chapter's biggest architectural contribution: instead of cramming
policy logic into the engine, you wrap the engine in a meta-controller
that handles input parsing, policy enforcement, control-deck assembly,
and business actions.  The engine becomes a pure reasoning service.

  meta-controller  =  the operating system
  engine           =  the CPU

Everything deterministic, everything business-specific, everything
organisational lives here.  Nothing in this file modifies an agent, a
planner, an executor, or a graph node.  It only *surrounds* them — the
"perimeter" pattern from Section C.

What ``run_with_policy`` does, in order:
    1. assemble a control deck (already done by caller — we receive one);
    2. pre-flight moderation on ``deck.goal`` (fail-closed);
    3. invoke the existing ``run_context_engine`` unchanged;
    4. post-flight moderation on the engine's textual output;
    5. emit a structured audit log entry per checkpoint;
    6. return a ``MetaResult`` with status, output, audit refs, latency.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from blog_mas.control_decks import ControlDeck
from blog_mas.engine.context_engine import run_context_engine
from blog_mas.observability.latency import LatencyBudget
from blog_mas.security.moderation import (
    ModerationReport,
    helper_moderate_content,
)

logger = logging.getLogger(__name__)

# Default audit sink — JSONL appended to ``data/audit/audit.jsonl``.
_DEFAULT_AUDIT_PATH = Path("data/audit/audit.jsonl")


@dataclass
class ModerationPolicy:
    """Per-deployment moderation configuration.

    The book correctly notes that a single ``moderation_active`` boolean is
    too coarse for production.  Real systems vary thresholds by use case,
    user role, and document namespace.  This dataclass is a small step
    toward that: separable pre/post toggles and a redaction template.
    """

    pre_flight: bool = True
    post_flight: bool = True
    redaction_template: str = (
        "[Redacted by {category} policy. Reference {ref_id} — "
        "contact compliance to appeal.]"
    )
    audit_logger: Callable[[dict[str, Any]], None] | None = None


@dataclass
class MetaResult:
    """What the meta-controller hands back to the caller."""

    status: str  # "ok" | "blocked_pre" | "redacted_post" | "engine_error"
    ref_id: str
    output: Any = None
    pre_report: ModerationReport | None = None
    post_report: ModerationReport | None = None
    latency: LatencyBudget = field(default_factory=LatencyBudget)
    trace: Any = None


# ───────────────────────── Audit sink ─────────────────────────────────────


def default_audit_logger(entry: dict[str, Any]) -> None:
    """Append one JSONL line to ``data/audit/audit.jsonl``.

    Audit trails outlive the notebook session — print statements do not.
    """
    _DEFAULT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _DEFAULT_AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _build_audit_entry(
    *,
    stage: str,
    ref_id: str,
    deck: ControlDeck,
    report: ModerationReport | None,
    text_excerpt: str,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ref_id": ref_id,
        "stage": stage,
        "template": deck.template_name,
        "namespace": deck.namespace,
        "moderation_source": getattr(report, "source", None),
        "flagged": getattr(report, "flagged", None),
        "top_category": (
            report.top_category if report and report.flagged else None
        ),
        "scores": getattr(report, "scores", {}),
        "text_excerpt": text_excerpt[:200],
    }


# ───────────────────────── Main entry point ──────────────────────────────


async def run_with_policy(
    deck: ControlDeck,
    *,
    llm,
    registry,
    policy: ModerationPolicy | None = None,
    save_trace_dir: str | None = None,
) -> MetaResult:
    """Run a control deck through the engine with a moderation perimeter.

    This is the ``execute_and_display`` from Ch8 §3 Code Block 2 reshaped
    for blog-mas: structured policy object, audit logger, latency budget,
    and a typed ``MetaResult`` instead of printing.
    """
    policy = policy or ModerationPolicy(audit_logger=default_audit_logger)
    ref_id = uuid.uuid4().hex[:8]
    budget = LatencyBudget()

    # ── 1. Pre-flight moderation ──────────────────────────────────────────
    pre_report: ModerationReport | None = None
    if deck.moderation_active and policy.pre_flight:
        with budget.measure("pre_flight_moderation", expected_s=0.3):
            pre_report = helper_moderate_content(deck.goal)

        if policy.audit_logger:
            policy.audit_logger(
                _build_audit_entry(
                    stage="pre_flight",
                    ref_id=ref_id,
                    deck=deck,
                    report=pre_report,
                    text_excerpt=deck.goal,
                )
            )

        if pre_report.flagged:
            logger.warning(
                "[MetaController] Pre-flight BLOCK ref=%s top=%s",
                ref_id,
                pre_report.top_category,
            )
            redaction = policy.redaction_template.format(
                category=pre_report.top_category, ref_id=ref_id
            )
            return MetaResult(
                status="blocked_pre",
                ref_id=ref_id,
                output=redaction,
                pre_report=pre_report,
                latency=budget,
            )

    # ── 2. Engine invocation ──────────────────────────────────────────────
    # The engine itself is unchanged.  This is the wrapper architecture:
    # nothing inside ``run_context_engine`` knows the meta-controller exists.
    try:
        with budget.measure("engine_invocation", expected_s=8.0):
            final_output, trace = await run_context_engine(
                goal=deck.goal,
                registry=registry,
                llm=llm,
                save_trace_dir=save_trace_dir,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[MetaController] engine error ref=%s", ref_id)
        return MetaResult(
            status="engine_error",
            ref_id=ref_id,
            output=str(exc),
            pre_report=pre_report,
            latency=budget,
        )

    # ── 3. Post-flight moderation ─────────────────────────────────────────
    post_report: ModerationReport | None = None
    output_text = _coerce_to_text(final_output)
    if deck.moderation_active and policy.post_flight and output_text:
        with budget.measure("post_flight_moderation", expected_s=0.3):
            post_report = helper_moderate_content(output_text)

        if policy.audit_logger:
            policy.audit_logger(
                _build_audit_entry(
                    stage="post_flight",
                    ref_id=ref_id,
                    deck=deck,
                    report=post_report,
                    text_excerpt=output_text,
                )
            )

        if post_report.flagged:
            logger.warning(
                "[MetaController] Post-flight REDACT ref=%s top=%s",
                ref_id,
                post_report.top_category,
            )
            redaction = policy.redaction_template.format(
                category=post_report.top_category, ref_id=ref_id
            )
            return MetaResult(
                status="redacted_post",
                ref_id=ref_id,
                output=redaction,
                pre_report=pre_report,
                post_report=post_report,
                latency=budget,
                trace=trace,
            )

    return MetaResult(
        status="ok",
        ref_id=ref_id,
        output=final_output,
        pre_report=pre_report,
        post_report=post_report,
        latency=budget,
        trace=trace,
    )


def _coerce_to_text(final_output: Any) -> str:
    """Engine output may be a Draft pydantic model or a raw string."""
    if final_output is None:
        return ""
    if isinstance(final_output, str):
        return final_output
    title = getattr(final_output, "title", None)
    body = getattr(final_output, "body", None)
    if title and body:
        return f"{title}\n\n{body}"
    return str(final_output)
