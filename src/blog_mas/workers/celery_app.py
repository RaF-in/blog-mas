"""Ch10 §1.3 — Celery application configuration.

Ch10:
  "Celery: The most mature distributed task queue system for Python.  It
   integrates well with various message brokers."

  "RabbitMQ or Kafka for robust message passing."
  (This implementation uses Redis as the broker for simplicity — swap
   CELERY_BROKER_URL to amqp://... for RabbitMQ in a larger deployment.)

Worker startup command (matches Dockerfile.worker CMD):
  celery -A blog_mas.workers.celery_app worker --loglevel=INFO --concurrency=4

The Celery app object is imported by tasks.py to decorate the task functions,
and by the K8s worker Deployment's command to start the consumer processes.
"""

from __future__ import annotations

import logging
import os

from celery import Celery

from blog_mas.config import get_settings
from blog_mas.logging_json import setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Read config — settings must be valid before the worker starts.
# ---------------------------------------------------------------------------

settings = get_settings()

# ---------------------------------------------------------------------------
# Celery app construction
# ---------------------------------------------------------------------------

celery_app = Celery(
    "blog_mas",
    broker=settings.queue.celery_broker_url,
    backend=settings.queue.result_backend_url,
    include=["blog_mas.workers.tasks"],
)

# ---------------------------------------------------------------------------
# Celery configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Task serialisation: JSON keeps the payload human-readable in Redis,
    # which simplifies debugging in the Celery task queue.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Ch10 §1.3 reliability settings.
    # acks_late=True: the worker ACKs only AFTER it has finished (or failed).
    # If the worker crashes mid-task, the broker re-queues the task.
    task_acks_late=True,
    # If the worker process is killed while a task is running, do NOT silently
    # drop it — send it back to the queue.
    task_reject_on_worker_lost=True,

    # Result TTL — mirrors result_store TTL so Celery's own backend (Redis)
    # doesn't keep stale keys forever.
    result_expires=settings.queue.result_ttl_seconds,

    # Worker concurrency — how many parallel tasks per worker pod.
    worker_concurrency=settings.queue.worker_concurrency,

    # Retry policy defaults (tasks can override per-task).
    task_max_retries=settings.queue.max_task_retries,

    # Soft/hard time limits (seconds) — prevents runaway tasks from blocking workers.
    # Soft limit: a SoftTimeLimitExceeded exception is raised in the task.
    # Hard limit: the worker process is killed and restarted.
    task_soft_time_limit=300,   # 5 min soft — task can catch and clean up
    task_time_limit=600,        # 10 min hard — killed unconditionally

    # Routing — all blog-mas tasks go to the "celery" queue by default.
    # In a larger deployment, create separate queues per task type (e.g.
    # "fast" for quick lookups, "slow" for full engine runs).
    task_default_queue="celery",

    # Worker logging — integrate with structlog.
    worker_hijack_root_logger=False,  # keep structlog in control
    worker_log_color=False,           # JSON logs don't need colour codes
)


# ---------------------------------------------------------------------------
# Worker startup signal — set up logging when the worker process starts
# ---------------------------------------------------------------------------

from celery.signals import worker_process_init  # noqa: E402 (after app creation)


@worker_process_init.connect
def configure_worker_logging(**kwargs) -> None:
    """Configure structured logging in each worker process.

    Each Celery worker is a separate OS process.  Logging must be configured
    per-process — the lifespan hook in api.py only runs in the API process.
    """
    setup_logging(
        level=settings.observability.log_level,
        log_format=settings.observability.log_format,
    )
    logger.info(
        "[worker] Process started (broker=%s, concurrency=%d)",
        settings.queue.celery_broker_url.split("@")[-1],  # mask credentials
        settings.queue.worker_concurrency,
    )
