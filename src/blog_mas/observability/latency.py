"""Chapter 8 §B — Latency is a feature, not a bug.

The book's argument: a multi-agent system pays a structural latency tax
because work is a chain of sequential, dependent network calls.  Pretending
that's a bug leads to bad UX (users see a frozen screen).  Treating it as
an engineered budget — itemised, measured, surfaced — is what the chapter
calls "production hygiene".

This module gives you a tiny ``LatencyBudget`` you can drop around any call
to record its duration, then ``report()`` to print the itemised breakdown.
It's deliberately small: the value is in the *practice*, not the framework.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyEntry:
    label: str
    duration_s: float
    expected_s: float | None = None  # SLA floor, if declared


@dataclass
class LatencyBudget:
    """Itemised network-call timer.  See chapter 8 §B."""

    entries: list[LatencyEntry] = field(default_factory=list)

    @contextmanager
    def measure(self, label: str, expected_s: float | None = None):
        """Context manager that records ``time.perf_counter`` deltas."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.entries.append(
                LatencyEntry(label=label, duration_s=elapsed, expected_s=expected_s)
            )

    @property
    def total_s(self) -> float:
        return sum(e.duration_s for e in self.entries)

    def report(self) -> str:
        """Pretty-printed itemised report (one line per call + total)."""
        if not self.entries:
            return "(no latency entries recorded)"

        lines = ["--- LATENCY BUDGET ---"]
        width = max(len(e.label) for e in self.entries)
        for e in self.entries:
            actual_ms = e.duration_s * 1000
            if e.expected_s is not None:
                expected_ms = e.expected_s * 1000
                delta = actual_ms - expected_ms
                marker = "OK " if delta <= 0 else "OVR"
                lines.append(
                    f"  [{marker}] {e.label:<{width}} "
                    f"{actual_ms:>7.0f} ms  (budget {expected_ms:>5.0f} ms)"
                )
            else:
                lines.append(
                    f"  [   ] {e.label:<{width}} {actual_ms:>7.0f} ms"
                )
        lines.append(f"  {'TOTAL':<{width + 6}} {self.total_s * 1000:>7.0f} ms")
        lines.append("----------------------")
        return "\n".join(lines)
