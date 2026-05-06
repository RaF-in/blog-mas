"""Stage 1: split a Markdown document on H1/H2/H3 headers.

Preserves the heading-ancestor path per section and merges sections
shorter than ``min_section_tokens`` (default 50) forward into the next
section (last section merges backward).
"""

import logging
import re

from blog_mas.rag.chunking.types import Section

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _count_tokens_approx(text: str) -> int:
    """Rough token count: ~4 chars per token (good enough for merge decisions)."""
    return max(1, len(text) // 4)


def split_by_headers(
    md_text: str, min_section_tokens: int = 50
) -> list[Section]:
    """Split *md_text* into sections on H1/H2/H3 boundaries.

    Fenced code blocks (``` ... ```) are respected — ``#`` inside them
    is not treated as a heading.
    """
    if not md_text or not md_text.strip():
        return []

    lines = md_text.split("\n")
    raw_sections = _scan_lines(lines)

    if not raw_sections:
        return []

    sections = [Section(text=t, headings_path=h) for t, h in raw_sections]
    return _merge_short_sections(sections, min_section_tokens)


def _scan_lines(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Walk lines, track heading ancestry, return ``(text, headings_path)`` pairs."""
    sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()

        # Track fenced code blocks
        if stripped.startswith("```"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        if in_fence:
            current_lines.append(line)
            continue

        m = _HEADING_RE.match(line)
        if m:
            # Flush current section
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((body, list(current_path)))

            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            # Truncate path to parent level and append this heading
            current_path = current_path[: level - 1] + [title]
        else:
            current_lines.append(line)

    # Flush final section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((body, list(current_path)))

    return sections


def _merge_short_sections(
    sections: list[Section], min_tokens: int
) -> list[Section]:
    """Merge sections below *min_tokens* forward into the next section.

    Last under-threshold section merges backward (no forward target).
    """
    if not sections:
        return []

    merged: list[Section] = []
    i = 0
    while i < len(sections):
        sec = sections[i]
        if _count_tokens_approx(sec.text) < min_tokens and i + 1 < len(sections):
            # Merge forward: prepend this section's text to the next one
            sections[i + 1].text = sec.text + "\n\n" + sections[i + 1].text
            i += 1
        else:
            merged.append(sec)
            i += 1

    # If the last section is still short, merge it backward
    if len(merged) > 1 and _count_tokens_approx(merged[-1].text) < min_tokens:
        last = merged.pop()
        merged[-1].text = merged[-1].text + "\n\n" + last.text

    return merged
