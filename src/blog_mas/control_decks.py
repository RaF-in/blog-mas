"""Chapter 8 §H — Generic control deck templates.

Section H's small but important move: replace the domain-specific test
goals from chapter 7 ("NASA Juno mission", "Service Agreement v1") with
*templates*.  The same engine should serve a science-blog use case today
and a law-firm use case tomorrow with no code changes — only a different
deck instance.

The deck is the API between the deterministic meta-controller (where
policy lives) and the non-deterministic engine (where reasoning lives).
Keeping it Pydantic-typed gives us:
- validation at the boundary (typos in ``namespace_knowledge`` no longer
  produce silent runtime errors at retrieval time);
- documented, self-describing payloads in audit logs and traces;
- an explicit ``moderation_active`` field that defaults to ``True`` —
  the book calls the original ``False`` default a "footgun".
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """Run-time engine settings — namespaces, model IDs, vector index."""

    index_name: str = Field(
        default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "blog-mas-knowledge")
    )
    generation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    namespace_context: str = "ContextLibrary"
    namespace_knowledge: str = "KnowledgeStore"


TemplateName = Literal[
    "high_fidelity_rag",
    "context_reduction",
    "grounded_reasoning",
]


class ControlDeck(BaseModel):
    """The contract between meta-controller and engine.

    ``goal`` is the only free-form field; everything else is enumerated or
    typed so the meta-controller can route, log, and audit deterministically.
    """

    template_name: TemplateName
    goal: str
    config: EngineConfig = Field(default_factory=EngineConfig)
    moderation_active: bool = True  # Ch8 §H: safer default than the book.
    namespace: str = "default"      # used by the namespace-aware sanitiser.


# ───────────────────────── Template factories ────────────────────────────


def template_high_fidelity_rag(question: str, *, require_citations: bool = True) -> ControlDeck:
    """Template 1 — cited-research over a knowledge base.

    Maps to the chapter-7 hi-fi Researcher path in this codebase.
    """
    suffix = " Please cite your sources." if require_citations else ""
    return ControlDeck(
        template_name="high_fidelity_rag",
        goal=f"{question}{suffix}",
    )


def template_context_reduction(text: str, write_goal: str) -> ControlDeck:
    """Template 2 — summarise-then-write for any large source document.

    Maps to the chapter-6 Summarizer → Writer path in this codebase.
    """
    goal = (
        f"First, summarize the following text to extract only the key facts. "
        f"Then {write_goal}\n\n--- TEXT TO USE ---\n{text}"
    )
    return ControlDeck(template_name="context_reduction", goal=goal)


def template_grounded_reasoning(question: str) -> ControlDeck:
    """Template 3 — verify the engine refuses out-of-scope queries honestly.

    The book's example: "monkey flies a rocket" should produce a refusal,
    not dreamy nonsense.  An honest "I don't know" is a feature.
    """
    return ControlDeck(
        template_name="grounded_reasoning",
        goal=(
            f"{question}\n\n"
            "If the knowledge base does not contain enough grounded evidence "
            "to answer, refuse explicitly rather than speculating."
        ),
    )
