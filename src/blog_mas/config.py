"""Chapter 10 §1.1 — Environment configuration (Twelve-Factor App).

The chapter's core principle: "the same code base can run anywhere, with its
behaviour defined entirely by environment variables."  A production system must
never have API keys or hostnames hardcoded.  This module is the single source
of truth for every tuneable parameter in blog-mas.

Resolution order:
  1. Environment variables (highest priority — what K8s injects)
  2. .env file on disk (local-dev convenience, never committed)
  3. Defaults declared in the Settings sub-models

Usage anywhere in the app:
  from blog_mas.config import get_settings
  settings = get_settings()          # cached singleton
  settings.llm.generation_model     # "gpt-4o" etc.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# ---------------------------------------------------------------------------
# python-dotenv integration — load .env before the Settings class reads env
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env into os.environ when python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv is optional; K8s injects env vars directly


_load_dotenv()


# ---------------------------------------------------------------------------
# Sub-model helpers — plain classes so no pydantic-settings dep required
# ---------------------------------------------------------------------------

class _SubModel:
    """Minimal base: reads from os.environ with typed defaults."""

    def _env(self, key: str, default: str | None = None) -> str | None:
        return os.environ.get(key, default)

    def _env_required(self, key: str) -> str:
        val = os.environ.get(key)
        if not val:
            raise ValueError(
                f"Required environment variable '{key}' is missing. "
                "Set it in your shell, .env file, or K8s Secret/ConfigMap."
            )
        return val

    def _env_int(self, key: str, default: int) -> int:
        return int(os.environ.get(key, str(default)))

    def _env_bool(self, key: str, default: bool) -> bool:
        raw = os.environ.get(key)
        if raw is None:
            return default
        return raw.lower() in ("1", "true", "yes", "on")

    def _env_float(self, key: str, default: float) -> float:
        return float(os.environ.get(key, str(default)))


# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

class LLMSettings(_SubModel):
    """All parameters the engine passes to the language model.

    The book's example:  GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gpt-4o")
    We extend it with token-budget controls for the proactive summariser policy.
    """

    def __init__(self) -> None:
        # Primary LLM key — at least one of these must be set.
        self.openai_api_key: str | None = self._env("OPENAI_API_KEY")
        self.hf_token: str | None = self._env("HF_TOKEN")

        self.generation_model: str = self._env("GENERATION_MODEL", "gpt-4o") or "gpt-4o"
        self.embedding_model: str = (
            self._env("EMBEDDING_MODEL", "text-embedding-3-small") or "text-embedding-3-small"
        )

        # Ch10 §2.1 — proactive context-reduction policy.
        # When an input exceeds this threshold, the Summarizer fires automatically.
        self.summarizer_trigger_tokens: int = self._env_int(
            "SUMMARIZER_TRIGGER_TOKENS", 4000
        )
        # Hard ceiling — inputs beyond this trigger a hard error, not a summary.
        self.max_input_tokens: int = self._env_int("MAX_INPUT_TOKENS", 32000)

    def validate(self) -> None:
        if not self.openai_api_key and not self.hf_token:
            raise ValueError(
                "At least one of OPENAI_API_KEY or HF_TOKEN must be set. "
                "The engine needs an LLM to function."
            )


# ---------------------------------------------------------------------------
# RAG / vector-store settings
# ---------------------------------------------------------------------------

class RAGSettings(_SubModel):
    def __init__(self) -> None:
        self.qdrant_url: str = (
            self._env("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333"
        )
        self.qdrant_collection: str = (
            self._env("QDRANT_COLLECTION", "blog-mas-knowledge") or "blog-mas-knowledge"
        )
        self.blueprint_collection: str = (
            self._env("BLUEPRINT_COLLECTION", "blog-mas-blueprints") or "blog-mas-blueprints"
        )
        self.retrieval_top_k: int = self._env_int("RETRIEVAL_TOP_K", 5)
        # Ch10 §2.3 — toggle ingest-time sanitisation (can disable for trusted corpora).
        self.ingest_sanitization_enabled: bool = self._env_bool(
            "INGEST_SANITIZATION_ENABLED", True
        )


# ---------------------------------------------------------------------------
# Queue / worker settings — Ch10 §1.3 (Celery + Redis)
# ---------------------------------------------------------------------------

class QueueSettings(_SubModel):
    def __init__(self) -> None:
        self.redis_url: str = (
            self._env("REDIS_URL", "redis://localhost:6379/0") or "redis://localhost:6379/0"
        )
        self.celery_broker_url: str = (
            self._env("CELERY_BROKER_URL", "redis://localhost:6379/0")
            or "redis://localhost:6379/0"
        )
        self.result_backend_url: str = (
            self._env("RESULT_BACKEND_URL", "redis://localhost:6379/1")
            or "redis://localhost:6379/1"
        )
        # How long (seconds) to keep completed results in the store.
        self.result_ttl_seconds: int = self._env_int("RESULT_TTL_SECONDS", 86400)
        # Celery worker concurrency.
        self.worker_concurrency: int = self._env_int("WORKER_CONCURRENCY", 4)
        # Max task retries before dead-lettering.
        self.max_task_retries: int = self._env_int("MAX_TASK_RETRIES", 3)


# ---------------------------------------------------------------------------
# Observability settings — Ch10 §1.4 (logs + metrics + tracing)
# ---------------------------------------------------------------------------

class ObservabilitySettings(_SubModel):
    def __init__(self) -> None:
        self.log_level: str = self._env("LOG_LEVEL", "INFO") or "INFO"
        # "json" for production, "console" for local dev.
        self.log_format: str = self._env("LOG_FORMAT", "console") or "console"

        # Prometheus — /metrics endpoint is always exposed; this controls port.
        self.metrics_port: int = self._env_int("METRICS_PORT", 8000)

        # OpenTelemetry → OTLP → Jaeger.
        self.otel_exporter_otlp_endpoint: str = (
            self._env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            or "http://localhost:4317"
        )
        self.otel_service_name: str = (
            self._env("OTEL_SERVICE_NAME", "blog-mas") or "blog-mas"
        )
        # Set to "false" in local dev to avoid OTLP connection errors.
        self.otel_enabled: bool = self._env_bool("OTEL_ENABLED", False)

        # Trace persistence (Ch10 §2.2 — immutable audit log).
        self.trace_store_dir: str = (
            self._env("TRACE_STORE_DIR", "data/traces") or "data/traces"
        )


# ---------------------------------------------------------------------------
# Security settings — Ch10 §2.3 / §2.4
# ---------------------------------------------------------------------------

class SecuritySettings(_SubModel):
    def __init__(self) -> None:
        # API key for the /api/v1/* endpoints (Ch10 §1.2 gateway concerns).
        self.api_key: str | None = self._env("API_KEY")
        # When False the API-edge pre-flight moderation is skipped (test mode only).
        self.api_preflight_moderation: bool = self._env_bool(
            "API_PREFLIGHT_MODERATION", True
        )
        # Webhook HMAC secret for outbound result notifications.
        self.webhook_secret: str | None = self._env("WEBHOOK_SECRET")


# ---------------------------------------------------------------------------
# Top-level Settings — assembled once, cached forever
# ---------------------------------------------------------------------------

class Settings:
    """Root settings object.  Access via get_settings() — do not instantiate directly."""

    def __init__(self) -> None:
        self.env: str = os.environ.get("ENV", "development")
        self.debug: bool = self.env == "development"

        self.llm = LLMSettings()
        self.rag = RAGSettings()
        self.queue = QueueSettings()
        self.observability = ObservabilitySettings()
        self.security = SecuritySettings()

    def validate(self) -> None:
        """Fail fast at startup with a clear error message listing every missing var.

        Ch10 §1.1: "the same code base can run anywhere" — but only if every
        instance has a complete awareness of its environment.
        """
        errors: list[str] = []
        try:
            self.llm.validate()
        except ValueError as exc:
            errors.append(str(exc))

        if errors:
            raise ValueError(
                "blog-mas startup failed — missing required configuration:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    The lru_cache means Settings() is constructed exactly once per process,
    which mirrors the book's 'read dynamically at runtime' pattern while
    avoiding repeated os.environ lookups on every request.
    """
    settings = Settings()
    # In production, validate eagerly so a misconfigured pod fails its readiness
    # probe instead of discovering a missing key mid-request.
    if settings.is_production():
        settings.validate()
    return settings
