"""Chapter 7 §E — Prompt-injection sanitizer.

The sanitizer is the first ring of the defense-in-depth strategy the chapter
describes.  It sits between retrieved text and the LLM: every chunk must pass
this gate before it can influence synthesis.

Design decisions that differ from the book's version (and why):
- Returns SanitizationResult instead of raising, so callers can count
  rejections and degrade gracefully rather than dying on the first bad chunk.
- Patterns are pre-compiled once at module load — compiling on every call is
  wasteful (O(chunks × patterns) regex compilations per request).
- Dropped "act as" from the book's list — it generates huge false-positive
  rates on legitimate text ("act as a catalyst", "act as a deterrent").
- Added Unicode bidi-override characters (U+202E / U+202D) — a real attack
  vector that the book's list misses.
- Added <script> HTML injection pattern.
- Tightened word-boundary anchors so "ignore" doesn't fire on "ignorance".
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pre-compiled patterns — Chapter 7 §E "defense-in-depth: first ring"
_INJECTION_PATTERNS: list[re.Pattern] = [
    # Classic instruction-override phrases
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+(instructions?|commands?)\b", re.I),
    # Mode-switching ("you are now in DAN mode")
    re.compile(r"\byou\s+are\s+now\s+in\s+\w[\w\s]*mode\b", re.I),
    # Prompt-leakage probes
    re.compile(r"\bprint\s+your\s+(system\s+)?instructions?\b", re.I),
    re.compile(r"\breveal\s+your\s+(system\s+)?prompt\b", re.I),
    # Shell / package-manager injection
    re.compile(r"\b(sudo|apt-get|yum|pip\s+install|rm\s+-rf)\b", re.I),
    # HTML script injection
    re.compile(r"<\s*script[^>]*>", re.I),
    # Unicode bidi-override characters (text-direction manipulation)
    re.compile(r"[‮‭‏]"),
    # Jailbreak persona framing
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.I),
    re.compile(r"\byour\s+(real\s+)?instructions?\s+are\b", re.I),
]


@dataclass(frozen=True)
class SanitizationResult:
    """Return value from sanitize_chunk.

    ok=True  → text is clean; cleaned_text holds the original.
    ok=False → text was rejected; cleaned_text is empty; reason and
               matched_pattern explain what triggered the block.
    """
    ok: bool
    cleaned_text: str
    reason: str | None = None
    matched_pattern: str | None = None


def sanitize_chunk(text: str) -> SanitizationResult:
    """Check a single retrieved or ingested text chunk for injection patterns.

    Chapter 7 §E: "The sanitizer rejects suspicious input rather than trying to
    fix it.  Fixing is a losing game; rejection is survivable."

    Returns a SanitizationResult — never raises — so callers control what
    happens to rejected chunks (skip, quarantine, escalate).
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "[Sanitizer] Rejected chunk. Pattern=%r snippet=%r",
                pattern.pattern,
                text[:120],
            )
            return SanitizationResult(
                ok=False,
                cleaned_text="",
                reason="injection_pattern_detected",
                matched_pattern=pattern.pattern,
            )

    logger.debug("[Sanitizer] Chunk passed. length=%d", len(text))
    return SanitizationResult(ok=True, cleaned_text=text)
