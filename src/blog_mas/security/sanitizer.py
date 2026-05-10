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
from dataclasses import dataclass, field
from enum import Enum

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


# ─────────────────────────────────────────────────────────────────────────
# Chapter 8 §I — namespace-aware, severity-tiered sanitiser
# ─────────────────────────────────────────────────────────────────────────
#
# The book's lesson: when legal asks you to add a phrase to your regex list,
# the wrong move is to stuff it into the same flat list that holds your
# prompt-injection patterns.  Doing so conflates two threat models:
#
#   * "ignore previous instructions"  — a *technical* prompt-injection.
#     Hard-block, always, in every namespace.
#   * "ignore any legal advice"       — a *business policy* concern.
#     Only meaningful in the ``emails`` namespace; should NEVER fire on
#     legitimate testimony where the phrase is being quoted as evidence.
#
# Two new types make this explicit:
#   - ``Severity``      — BLOCK / QUARANTINE / FLAG  (not all hits are equal)
#   - ``SanitizerRule`` — pattern + severity + reason + optional namespace set
#
# ``sanitize_chunk_with_policy`` is additive: existing ``sanitize_chunk``
# callers (chapter 7 retrieval path) are untouched.


class Severity(Enum):
    """How seriously to treat a sanitiser hit."""

    BLOCK = "block"            # hard-stop; chunk never reaches the LLM
    QUARANTINE = "quarantine"  # set aside for human review
    FLAG = "flag"              # allow but annotate for downstream handling


@dataclass(frozen=True)
class SanitizerRule:
    """One classifier with namespace scoping.

    ``applies_to_namespaces=None`` means "applies everywhere" (e.g., the
    universal prompt-injection patterns).  A frozenset means "only fires
    when the chunk is being processed under one of these namespaces".
    """

    pattern: re.Pattern
    severity: Severity
    reason: str
    applies_to_namespaces: frozenset[str] | None = None


# Universal technical threats — apply to every namespace.
_TECHNICAL_RULES: tuple[SanitizerRule, ...] = (
    SanitizerRule(
        re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+(instructions?|commands?)\b", re.I),
        Severity.BLOCK,
        "prompt_injection",
    ),
    SanitizerRule(
        re.compile(r"\b(sudo|apt-get|yum|pip\s+install|rm\s+-rf)\b", re.I),
        Severity.BLOCK,
        "command_injection",
    ),
    SanitizerRule(
        re.compile(r"<\s*script[^>]*>", re.I),
        Severity.BLOCK,
        "html_injection",
    ),
)

# Policy rules — namespace-scoped.  This is the chapter's data-segmentation
# fix: the ``emails`` namespace cares about the phrase; the ``testimony``
# namespace deliberately does not (because the phrase is legitimate
# evidence there, not an instruction).
_POLICY_RULES: tuple[SanitizerRule, ...] = (
    SanitizerRule(
        re.compile(r"ignore\s+any\s+legal\s+advice", re.I),
        Severity.FLAG,
        "policy_concern_legal_advice",
        applies_to_namespaces=frozenset({"emails"}),
    ),
)


@dataclass(frozen=True)
class PolicySanitizationResult:
    """Result from the namespace-aware sanitiser.

    ``allowed`` is the only thing callers must check; ``violations`` and
    ``severity`` are for audit logs and tuning.
    """

    allowed: bool
    violations: tuple[str, ...] = field(default_factory=tuple)
    severity: Severity | None = None


def sanitize_chunk_with_policy(
    text: str,
    namespace: str,
) -> PolicySanitizationResult:
    """Severity-tiered, namespace-aware sanitiser (Chapter 8 §I).

    Walks technical rules first (always-on), then namespace-scoped policy
    rules.  A BLOCK hit short-circuits with ``allowed=False``; FLAGs and
    QUARANTINEs accumulate but still allow the chunk through (annotated).
    """
    violations: list[str] = []
    worst: Severity | None = None

    for rule in (*_TECHNICAL_RULES, *_POLICY_RULES):
        if (
            rule.applies_to_namespaces is not None
            and namespace not in rule.applies_to_namespaces
        ):
            continue
        if not rule.pattern.search(text):
            continue

        violations.append(rule.reason)
        worst = rule.severity if worst is None else _max_severity(worst, rule.severity)

        if rule.severity is Severity.BLOCK:
            logger.warning(
                "[Sanitizer/policy] BLOCK ns=%s reason=%s pattern=%r",
                namespace, rule.reason, rule.pattern.pattern,
            )
            return PolicySanitizationResult(
                allowed=False,
                violations=tuple(violations),
                severity=Severity.BLOCK,
            )

    if violations:
        logger.info(
            "[Sanitizer/policy] %s ns=%s reasons=%s",
            worst.name if worst else "FLAG", namespace, violations,
        )

    return PolicySanitizationResult(
        allowed=True, violations=tuple(violations), severity=worst,
    )


_SEVERITY_ORDER = {Severity.FLAG: 0, Severity.QUARANTINE: 1, Severity.BLOCK: 2}


def _max_severity(a: Severity, b: Severity) -> Severity:
    return a if _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b] else b
