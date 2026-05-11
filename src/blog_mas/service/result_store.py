"""Ch10 §1.3 — Result store: trace_id-keyed persistent result retrieval.

Ch10 §1.3 lifecycle steps 5-6:
  5. "The worker stores the execution trace and the final output in a persistent
     store (e.g. Redis Cache or a database), keyed by trace_id."
  6. "The client uses trace_id to poll a status endpoint or receives the result
     via a webhook."

Ch10 §2.2 (auditability):
  "This log, persisted in the Results tracer outputs, serves as an immutable
   record of the engine's reasoning process."

Two implementations behind a common interface:
  RedisResultStore    — primary (production), uses redis-py, TTL-aware.
  FilesystemResultStore — fallback (dev/test), writes JSON files to disk.

Factory function ``get_result_store()`` picks the right backend from config.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------

_STATUS_PENDING = "pending"
_STATUS_RUNNING = "running"


def _make_entry(
    trace_id: str,
    status: str,
    *,
    final_output: str | None = None,
    trace_dict: dict | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "status": status,
        "final_output": final_output,
        "trace": trace_dict or {},
        "metadata": metadata or {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ResultStore(Protocol):
    """Anything that can store and retrieve execution results by trace_id."""

    def create_pending(self, trace_id: str) -> None:
        """Register a trace_id as pending (enqueued, not yet executing)."""
        ...

    def mark_running(self, trace_id: str) -> None:
        """Update status to 'running' when a worker picks up the task."""
        ...

    def save(
        self,
        trace_id: str,
        status: str,
        *,
        final_output: str | None = None,
        trace_dict: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Persist the completed result (overwrites any existing entry)."""
        ...

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Retrieve a result by trace_id, or None if not found."""
        ...


# ---------------------------------------------------------------------------
# Redis implementation
# ---------------------------------------------------------------------------

class RedisResultStore:
    """Primary result store backed by Redis.

    Keys:  result:{trace_id}
    TTL:   configurable (default 24h) — prevents unbounded key growth.

    Why Redis?  Ch10 §1.3 names "Redis Cache" as the canonical example.
    The same Redis instance is also the Celery broker, keeping the
    deployment simple for small-to-medium workloads.
    """

    name = "redis"

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int = 86400,
    ) -> None:
        import redis as redis_lib  # type: ignore[import]

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis_lib.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds

    def _key(self, trace_id: str) -> str:
        return f"result:{trace_id}"

    def create_pending(self, trace_id: str) -> None:
        entry = _make_entry(trace_id, _STATUS_PENDING)
        self._client.setex(
            self._key(trace_id),
            self._ttl,
            json.dumps(entry, default=str),
        )
        logger.debug("[result_store/redis] created pending trace_id=%s", trace_id)

    def mark_running(self, trace_id: str) -> None:
        existing = self.get(trace_id)
        if existing:
            existing["status"] = _STATUS_RUNNING
            existing["updated_at"] = time.time()
            self._client.setex(
                self._key(trace_id),
                self._ttl,
                json.dumps(existing, default=str),
            )

    def save(
        self,
        trace_id: str,
        status: str,
        *,
        final_output: str | None = None,
        trace_dict: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        existing = self.get(trace_id) or {}
        entry = _make_entry(
            trace_id, status,
            final_output=final_output,
            trace_dict=trace_dict,
            metadata=metadata,
        )
        entry["created_at"] = existing.get("created_at", entry["created_at"])
        self._client.setex(
            self._key(trace_id),
            self._ttl,
            json.dumps(entry, default=str),
        )
        logger.info(
            "[result_store/redis] saved trace_id=%s status=%s", trace_id, status
        )

    def get(self, trace_id: str) -> dict[str, Any] | None:
        raw = self._client.get(self._key(trace_id))
        if raw is None:
            return None
        return json.loads(raw)

    def queue_length(self, queue_name: str = "celery") -> int:
        """Read the Celery queue depth from Redis for the HPA custom metric."""
        return self._client.llen(queue_name)


# ---------------------------------------------------------------------------
# Filesystem implementation (dev / test)
# ---------------------------------------------------------------------------

class FilesystemResultStore:
    """Fallback result store backed by JSON files on disk.

    Useful in local dev without Redis and for integration tests.
    Files are written to trace_store_dir (default: data/traces/).

    Note: not suitable for multi-worker production deployments — workers on
    different pods cannot share the local filesystem.
    """

    name = "filesystem"

    def __init__(self, store_dir: str | None = None) -> None:
        self._base = Path(
            store_dir or os.environ.get("TRACE_STORE_DIR", "data/traces")
        )
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, trace_id: str) -> Path:
        return self._base / f"{trace_id}.json"

    def create_pending(self, trace_id: str) -> None:
        entry = _make_entry(trace_id, _STATUS_PENDING)
        self._path(trace_id).write_text(json.dumps(entry, indent=2, default=str))

    def mark_running(self, trace_id: str) -> None:
        existing = self.get(trace_id)
        if existing:
            existing["status"] = _STATUS_RUNNING
            existing["updated_at"] = time.time()
            self._path(trace_id).write_text(json.dumps(existing, indent=2, default=str))

    def save(
        self,
        trace_id: str,
        status: str,
        *,
        final_output: str | None = None,
        trace_dict: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        existing = self.get(trace_id) or {}
        entry = _make_entry(
            trace_id, status,
            final_output=final_output,
            trace_dict=trace_dict,
            metadata=metadata,
        )
        entry["created_at"] = existing.get("created_at", entry["created_at"])
        self._path(trace_id).write_text(json.dumps(entry, indent=2, default=str))
        logger.info(
            "[result_store/fs] saved trace_id=%s status=%s", trace_id, status
        )

    def get(self, trace_id: str) -> dict[str, Any] | None:
        p = self._path(trace_id)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def queue_length(self, queue_name: str = "celery") -> int:
        return 0  # filesystem store has no queue visibility


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_result_store() -> RedisResultStore | FilesystemResultStore:
    """Return the correct result store based on configuration.

    Resolution:
      • REDIS_URL is set AND redis-py is importable → RedisResultStore
      • Otherwise                                    → FilesystemResultStore

    This lets local development work without Redis while production
    automatically uses Redis when the env var is set.
    """
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis as redis_lib  # type: ignore[import]
            redis_lib.Redis.from_url(redis_url).ping()
            from blog_mas.config import get_settings
            settings = get_settings()
            return RedisResultStore(
                redis_url=redis_url,
                ttl_seconds=settings.queue.result_ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "[result_store] Redis not reachable (%s) — falling back to filesystem store.",
                exc,
            )

    return FilesystemResultStore()
