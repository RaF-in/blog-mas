"""Ch10 §1.3 + §2.1 — Celery task: execute_goal.

This module is the "engine room" of the production architecture.  When a
worker pulls a task from the queue, it runs the full glass-box Context Engine
here — the same Planner → Executor → Agents pipeline the CLI runs, but
wrapped in production concerns:

  1.  Trace-ID context binding (every log line includes trace_id)
  2.  OTEL span propagation (links this span to the API span in Jaeger)
  3.  Proactive context-reduction policy (Ch10 §2.1)
       → if goal tokens > SUMMARIZER_TRIGGER_TOKENS, summarise first
  4.  run_with_policy (Ch8 — moderation perimeter, unchanged)
  5.  Result persistence (Ch10 §1.3 — keyed by trace_id in result store)
  6.  Prometheus counters + histograms (Ch10 §1.4)
  7.  Webhook delivery (Ch10 §1.3 — optional, fires if registered)

Nothing inside the engine itself changes — this file only *surrounds* it.
The meta-controller architecture extends all the way to the task layer.
"""

from __future__ import annotations

import asyncio
import logging
import time

from celery import Task

from blog_mas.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialise heavy singletons once per worker process (not per task)
# ---------------------------------------------------------------------------
# Celery workers are long-lived processes.  Creating the LLM client, Qdrant
# store, and registry on every task call would be extremely wasteful.
# Instead we use a module-level dictionary that is populated on first use.

_worker_state: dict = {}


def _get_worker_resources():
    """Lazily build LLM / store / embedder / registry on first task call."""
    if _worker_state:
        return _worker_state

    from blog_mas.engine.agent_adapters import build_default_registry
    from blog_mas.llm import create_llm
    from blog_mas.rag.embedding import EmbeddingClient
    from blog_mas.rag.vector_store import QdrantStore

    logger.info("[worker] Initialising engine resources (once per process)...")
    llm = create_llm()
    store = QdrantStore()
    embedder = EmbeddingClient()
    registry = build_default_registry(llm, store, embedder, reranker=None)

    _worker_state["llm"] = llm
    _worker_state["store"] = store
    _worker_state["embedder"] = embedder
    _worker_state["registry"] = registry
    logger.info("[worker] Engine resources ready.")

    return _worker_state


# ---------------------------------------------------------------------------
# §2.1 — Proactive context-reduction policy
# ---------------------------------------------------------------------------

async def _maybe_prepend_summarizer(goal: str, llm, registry) -> str:
    """Ch10 §2.1 — auto-summarise when the goal exceeds the token budget.

    Ch10:
      "The count_tokens utility from Chapter 6, which measures prompt size
       before an API call, can be integrated into the Executor or a
       pre-processing step.  When an input exceeds a predefined budget, the
       system can be configured to automatically invoke the Summarizer agent
       first."

    This function implements that policy at the task layer (the outermost
    deterministic wrapper), keeping the engine core untouched.

    Returns:
      The original goal if under the threshold, or the Summarizer's output
      appended to a shortened goal string if over it.
    """
    from blog_mas.config import get_settings
    from blog_mas.service.metrics import (
        SUMMARIZER_TOKEN_SAVINGS,
        SUMMARIZER_TRIGGERS_TOTAL,
    )
    from blog_mas.tokens import count_tokens

    settings = get_settings()
    trigger_threshold = settings.llm.summarizer_trigger_tokens
    token_count = count_tokens(goal)

    if token_count <= trigger_threshold:
        logger.debug(
            "[worker] Context reduction policy: no trigger (tokens=%d threshold=%d)",
            token_count, trigger_threshold,
        )
        return goal  # under budget — pass through unchanged

    logger.info(
        "[worker] Context reduction policy TRIGGERED (tokens=%d > threshold=%d) — "
        "auto-invoking Summarizer before primary reasoning agents.",
        token_count, trigger_threshold,
    )
    SUMMARIZER_TRIGGERS_TOTAL.inc()

    # Build a minimal summarise-only deck and run it through the engine.
    # We deliberately NOT use run_with_policy here — the outer task call
    # will apply the moderation perimeter.  This is a pure summarisation pass.
    from blog_mas.control_decks import ControlDeck

    summarize_goal = (
        f"Summarise the following text, retaining only the key facts and "
        f"entities needed to answer questions about it.  "
        f"Be concise — the summary will become the context for a follow-up "
        f"reasoning task.\n\n--- TEXT ---\n{goal}"
    )

    summarize_deck = ControlDeck(
        template_name="context_reduction",
        goal=summarize_goal,
        moderation_active=False,   # already moderated at API edge
    )

    from blog_mas.engine.context_engine import run_context_engine

    try:
        summary_output, summary_trace = await run_context_engine(
            goal=summarize_deck.goal,
            registry=registry,
            llm=llm,
        )

        if summary_output:
            # Coerce to string (output might be a Draft pydantic model).
            summary_text: str = (
                summary_output
                if isinstance(summary_output, str)
                else getattr(summary_output, "body", str(summary_output))
            )
            summary_tokens = count_tokens(summary_text)
            reduction_pct = (1 - summary_tokens / token_count) * 100

            logger.info(
                "[worker] Summariser reduced tokens: %d → %d (%.1f%% savings)",
                token_count, summary_tokens, reduction_pct,
            )
            SUMMARIZER_TOKEN_SAVINGS.observe(reduction_pct)

            return summary_text

    except Exception as exc:
        # Summarisation failure is non-fatal — fall through to the full goal.
        logger.warning(
            "[worker] Summariser failed (%s) — proceeding with original goal.", exc
        )

    return goal


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------

@celery_app.task(
    bind=True,
    name="blog_mas.workers.tasks.execute_goal",
    max_retries=3,
    default_retry_delay=30,   # seconds between retries (exponential backoff applied)
    acks_late=True,
    reject_on_worker_lost=True,
)
def execute_goal(
    self: Task,
    *,
    trace_id: str,
    goal: str,
    template_name: str = "high_fidelity_rag",
    namespace: str = "default",
    require_audit_trace: bool = False,
    moderation_active: bool = True,
    configuration_overrides: dict | None = None,
) -> dict:
    """Execute a goal end-to-end inside a Celery worker.

    Ch10 §1.3 lifecycle steps 4-5:
      4. "Worker processes pull tasks from the queue and execute the Context
          Engine logic."
      5. "Upon completion, the worker persists the final output and the
          detailed execution trace into the Result Store."

    Returns a summary dict (Celery stores this as the task result in Redis
    independently of our own result store).
    """
    from blog_mas.config import get_settings
    from blog_mas.control_decks import ControlDeck, EngineConfig
    from blog_mas.logging_json import bind_trace_context
    from blog_mas.meta_controller import ModerationPolicy, default_audit_logger, run_with_policy
    from blog_mas.service.metrics import (
        ENGINE_DURATION,
        MODERATION_BLOCKS_TOTAL,
        TASKS_COMPLETED,
        TASK_DURATION,
        record_llm_usage,
    )
    from blog_mas.service.result_store import get_result_store
    from blog_mas.service.tracing import extract_span_context_from_headers, get_tracer
    from blog_mas.tokens import compute_token_savings, estimate_cost

    # ── 1. Bind trace context to every log line in this task ────────────────
    bind_trace_context(trace_id=trace_id, service="worker")

    # ── 2. Continue the distributed trace from the API request ──────────────
    # The API embedded the OTEL context in the task headers.
    otel_headers_raw = (self.request.headers or {}).get("otel_context", "{}")
    try:
        import json
        otel_headers = json.loads(otel_headers_raw)
    except Exception:
        otel_headers = {}
    extract_span_context_from_headers(otel_headers)

    tracer = get_tracer()
    result_store = get_result_store()
    settings = get_settings()

    task_start = time.perf_counter()

    with tracer.start_as_current_span("execute_goal_task") as span:
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("template", template_name)
        span.set_attribute("goal_length", len(goal))

        # Mark the task as running in the result store.
        result_store.mark_running(trace_id)
        logger.info(
            "[worker] Task started",
            trace_id=trace_id,
            template=template_name,
            goal_preview=goal[:80],
        )

        try:
            # ── 3. Proactive context-reduction policy (Ch10 §2.1) ─────────
            # Run in an event loop created for this sync Celery task.
            effective_goal = asyncio.run(
                _maybe_prepend_summarizer(
                    goal,
                    llm=_get_worker_resources()["llm"],
                    registry=_get_worker_resources()["registry"],
                )
            )

            # ── 4. Build the control deck ──────────────────────────────────
            engine_config = EngineConfig(**(configuration_overrides or {}))
            deck = ControlDeck(
                template_name=template_name,  # type: ignore[arg-type]
                goal=effective_goal,
                config=engine_config,
                namespace=namespace,
                moderation_active=moderation_active,
            )

            # ── 5. run_with_policy: pre-flight → engine → post-flight ──────
            # The meta-controller wraps the engine unchanged.  All moderation,
            # audit logging, and latency measurement happens here.
            policy = ModerationPolicy(audit_logger=default_audit_logger)
            resources = _get_worker_resources()

            with tracer.start_as_current_span("run_with_policy"):
                meta_result = asyncio.run(
                    run_with_policy(
                        deck,
                        llm=resources["llm"],
                        registry=resources["registry"],
                        policy=policy,
                        save_trace_dir=(
                            settings.observability.trace_store_dir
                            if require_audit_trace else None
                        ),
                    )
                )

            # ── 6. Coerce final output to string ───────────────────────────
            output_str: str | None = None
            if meta_result.output is not None:
                if isinstance(meta_result.output, str):
                    output_str = meta_result.output
                elif hasattr(meta_result.output, "title") and hasattr(meta_result.output, "body"):
                    output_str = f"{meta_result.output.title}\n\n{meta_result.output.body}"
                else:
                    output_str = str(meta_result.output)

            # ── 7. Update Prometheus counters ──────────────────────────────
            TASKS_COMPLETED.labels(status=meta_result.status).inc()

            if meta_result.status in ("blocked_pre", "redacted_post"):
                category = "unknown"
                report = meta_result.pre_report or meta_result.post_report
                if report:
                    category = report.top_category
                stage = (
                    "pre_flight" if meta_result.status == "blocked_pre"
                    else "post_flight"
                )
                MODERATION_BLOCKS_TOTAL.labels(stage=stage, category=category).inc()

            # Token usage from the trace.
            if meta_result.trace:
                token_savings = compute_token_savings(meta_result.trace)
                if token_savings:
                    model = settings.llm.generation_model
                    total_in = token_savings.get("total_input_tokens", 0)
                    total_out = token_savings.get("total_output_tokens", 0)
                    cost = estimate_cost(total_in, total_out, model)
                    record_llm_usage(total_in, total_out, model, cost)

            # ── 8. Build metadata for the result store ─────────────────────
            trace_dict = (
                meta_result.trace.to_dict()
                if meta_result.trace and hasattr(meta_result.trace, "to_dict")
                else {}
            )

            metadata = {
                "engine_used": "GLASS_BOX",
                "template": template_name,
                "duration_seconds": meta_result.latency.total_s,
                "latency_report": meta_result.latency.report(),
                "pre_flight_flagged": (
                    meta_result.pre_report.flagged if meta_result.pre_report else False
                ),
                "post_flight_flagged": (
                    meta_result.post_report.flagged if meta_result.post_report else False
                ),
                "ref_id": meta_result.ref_id,
            }
            if meta_result.trace:
                metadata["engine_trace_id"] = getattr(meta_result.trace, "trace_id", "")

            # ── 9. Persist to result store ─────────────────────────────────
            # Ch10 §1.3: "the worker persists the final output and the
            # detailed execution trace into the Result Store, where the
            # client can retrieve it."
            result_store.save(
                trace_id,
                status=meta_result.status,
                final_output=output_str,
                trace_dict=trace_dict,
                metadata=metadata,
            )

            task_duration = time.perf_counter() - task_start
            TASK_DURATION.observe(task_duration)
            ENGINE_DURATION.observe(
                meta_result.latency.total_s if meta_result.latency else task_duration
            )

            logger.info(
                "[worker] Task completed",
                trace_id=trace_id,
                status=meta_result.status,
                duration_s=round(task_duration, 2),
            )

            return {
                "trace_id": trace_id,
                "status": meta_result.status,
                "duration_s": task_duration,
            }

        except Exception as exc:
            task_duration = time.perf_counter() - task_start
            logger.exception(
                "[worker] Task failed",
                trace_id=trace_id,
                error=str(exc),
                duration_s=round(task_duration, 2),
            )

            # Persist the failure so the polling endpoint shows "engine_error"
            # rather than leaving the client in "running" forever.
            try:
                result_store.save(
                    trace_id,
                    status="engine_error",
                    final_output=f"Execution failed: {exc}",
                    metadata={"error": str(exc), "duration_s": task_duration},
                )
            except Exception:
                pass  # storage failure on top of execution failure — log and move on

            TASKS_COMPLETED.labels(status="engine_error").inc()

            # Retry with exponential backoff (Celery applies jitter automatically).
            try:
                raise self.retry(
                    exc=exc,
                    countdown=30 * (2 ** self.request.retries),  # 30s, 60s, 120s
                )
            except self.MaxRetriesExceededError:
                logger.error(
                    "[worker] Max retries exceeded — task dead-lettered.",
                    trace_id=trace_id,
                )
                return {
                    "trace_id": trace_id,
                    "status": "engine_error",
                    "error": str(exc),
                }
