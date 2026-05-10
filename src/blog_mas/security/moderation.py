"""Chapter 8 §C-D — Moderation gatekeeper (the "perimeter fence").

The moderation layer is a wrapper around the engine, not part of it.  It runs
twice per request: pre-flight on user input, post-flight on engine output.
Two design rules from the chapter govern this module:

1. **Fail-closed** — on any provider error or timeout, return ``flagged=True``.
   Safety code defaults to blocking, never to allowing.
2. **Provider-swappable** — the engine should not care whether moderation
   came from OpenAI, Llama Guard, or a heuristic stub.  We expose a small
   provider interface so future providers slot in without engine edits.

Two providers are shipped:
- ``OpenAIModerationProvider``  — real API (free tier, no token cost).
- ``HeuristicModerationProvider`` — keyword-based stub used when no API key
  is configured.  Categories mirror OpenAI's so callers see one shape.

Improvements over Ch8 §3 Code Block 1 (the book's helper):
- Returns a structured ``ModerationReport`` dataclass instead of a dict.
- LRU-cached on a SHA-256 hash of the text (retries don't re-bill).
- Request timeout (5s) — the book's version can hang indefinitely.
- Provider provenance is recorded in ``source`` so audit logs distinguish
  "blocked by OpenAI" from "blocked by error fallback".
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)

# OpenAI's 11 harm categories — kept here so stub + real provider agree on shape.
MODERATION_CATEGORIES = (
    "hate",
    "hate/threatening",
    "harassment",
    "harassment/threatening",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
)


@dataclass
class ModerationReport:
    """Result of one moderation call. ``flagged=True`` means "do not pass"."""

    flagged: bool
    categories: dict[str, bool] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    source: str = "unknown"

    @property
    def top_category(self) -> str:
        """Highest-scoring category — used to pick a redaction reason."""
        if not self.scores:
            return "policy"
        return max(self.scores, key=self.scores.get)


class ModerationProvider(Protocol):
    """Anything that can score a string against the harm categories."""

    name: str

    def moderate(self, text: str) -> ModerationReport: ...


# ───────────────────────── OpenAI provider ────────────────────────────────


class OpenAIModerationProvider:
    """Calls OpenAI's ``/v1/moderations`` endpoint. Free, fast (~100-300ms)."""

    name = "openai"

    def __init__(self, client=None, model: str = "omni-moderation-latest"):
        # Lazy import keeps the dependency optional at import time.
        from openai import OpenAI

        self._client = client or OpenAI()
        self._model = model

    def moderate(self, text: str) -> ModerationReport:
        try:
            # 5s timeout: book's version can hang the entire engine.
            response = self._client.moderations.create(
                model=self._model,
                input=text,
                timeout=5.0,
            )
            r = response.results[0]
            return ModerationReport(
                flagged=bool(r.flagged),
                categories=_pydantic_to_dict(r.categories),
                scores=_pydantic_to_dict(r.category_scores),
                source=self.name,
            )
        except Exception as exc:  # noqa: BLE001 — fail-closed by design
            logger.error("[Moderation/openai] error: %s", exc)
            return ModerationReport(
                flagged=True,
                error=str(exc),
                source=f"{self.name}:error",
            )


def _pydantic_to_dict(obj) -> dict:
    """OpenAI SDK returns Pydantic objects; coerce to plain dict for logging."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


# ───────────────────────── Heuristic stub ─────────────────────────────────

# Tiny keyword map: enough to demonstrate categorical scoring without an API
# key.  Real deployments must use OpenAI or Llama Guard.
_HEURISTIC_KEYWORDS: dict[str, list[str]] = {
    "violence": [r"\b(kill|murder|stab|shoot|bomb)\b"],
    "hate": [r"\b(slur1|slur2)\b"],  # placeholder — keep example clean
    "harassment": [r"\b(idiot|moron|stupid)\b"],
    "self-harm": [r"\bsuicide\b", r"\bself[- ]harm\b"],
    "sexual": [r"\bporn\b", r"\bnsfw\b"],
}


class HeuristicModerationProvider:
    """Regex-based stub used when no OpenAI key is configured.

    Demonstrates the *interface* the chapter describes (categorical scores +
    fail-closed) without depending on a paid API.
    """

    name = "heuristic"

    def moderate(self, text: str) -> ModerationReport:
        scores = {c: 0.0 for c in MODERATION_CATEGORIES}
        categories = {c: False for c in MODERATION_CATEGORIES}
        text_lc = text.lower()

        for category, patterns in _HEURISTIC_KEYWORDS.items():
            for pat in patterns:
                if re.search(pat, text_lc):
                    scores[category] = 0.95
                    categories[category] = True

        flagged = any(categories.values())
        return ModerationReport(
            flagged=flagged,
            categories=categories,
            scores=scores,
            source=self.name,
        )


# ───────────────────────── Module-level facade ────────────────────────────

_default_provider: ModerationProvider | None = None


def get_default_provider() -> ModerationProvider:
    """Pick a provider based on environment.

    Resolution order:
    1. Cached singleton (memoised so repeat calls don't re-init the client).
    2. ``OpenAIModerationProvider`` if ``OPENAI_API_KEY`` is set.
    3. ``HeuristicModerationProvider`` fallback (no key needed).
    """
    global _default_provider
    if _default_provider is not None:
        return _default_provider

    if os.environ.get("OPENAI_API_KEY"):
        try:
            _default_provider = OpenAIModerationProvider()
            logger.info("[Moderation] Using OpenAIModerationProvider.")
            return _default_provider
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[Moderation] OpenAI init failed (%s); using heuristic stub.",
                exc,
            )

    _default_provider = HeuristicModerationProvider()
    logger.info("[Moderation] Using HeuristicModerationProvider (stub).")
    return _default_provider


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1024)
def _cached_call(text_hash: str, text: str, provider_name: str) -> ModerationReport:
    """LRU-cached inner call.

    Cache key is (hash, provider) — so swapping providers invalidates cleanly,
    and identical texts within a session don't re-hit the API.  ``text_hash``
    is in the key purely so the cache decorator is keyed on a short string.
    """
    provider = get_default_provider()
    return provider.moderate(text)


def helper_moderate_content(
    text: str,
    provider: ModerationProvider | None = None,
) -> ModerationReport:
    """Public entry point — Chapter 8 §D's gatekeeper.

    Mirrors the book's signature but returns a typed report and uses the
    cached provider singleton.  Pass ``provider`` only when you want to
    override the default (e.g., tests).
    """
    if provider is not None:
        return provider.moderate(text)
    return _cached_call(_hash(text), text, get_default_provider().name)
