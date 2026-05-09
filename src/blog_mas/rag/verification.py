"""Chapter 7 §F — Ingestion verification probe.

The book's point: "Thinking time exceeds development time" — a five-line
verification cell is what turns ingestion from "I hope it worked" into "I
proved it worked."

verify_ingestion() takes a list of (query, expected_source_substring) probes
and asserts that each query's top-k results include at least one chunk from
the expected source document.  Run it after every re-ingestion as a smoke
test.

Improved over the book's single-query version:
  - Multi-probe: checks a whole suite in one call.
  - Asserts loudly on failure (suitable for CI) rather than just printing.
  - Returns a structured report so callers can choose how to surface it.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    query: str
    expected_source: str
    found_sources: list[str]
    passed: bool


@dataclass
class VerificationReport:
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(p.passed for p in self.probes)

    @property
    def failures(self) -> list[ProbeResult]:
        return [p for p in self.probes if not p.passed]

    def summary(self) -> str:
        total = len(self.probes)
        ok = sum(1 for p in self.probes if p.passed)
        lines = [f"Verification: {ok}/{total} probes passed"]
        for f in self.failures:
            lines.append(
                f"  FAIL  query={f.query!r}  "
                f"expected_source_contains={f.expected_source!r}  "
                f"found={f.found_sources}"
            )
        return "\n".join(lines)


def verify_ingestion(
    store,
    embedder,
    namespace: str,
    probes: list[tuple[str, str]],
    top_k: int = 3,
    assert_on_failure: bool = True,
) -> VerificationReport:
    """Run a suite of retrieval probes and verify source provenance.

    Args:
        store:     Vector store instance (must support dense_search).
        embedder:  Embedding client (must support embed_batch).
        namespace: Qdrant collection namespace to query.
        probes:    List of (query_text, expected_source_substring) pairs.
                   A probe passes when at least one of the top-k results has
                   a doc_id payload containing expected_source_substring.
        top_k:     Number of results to retrieve per probe.
        assert_on_failure: If True, raises AssertionError on any failure.
                   Set False to get a report without stopping execution.

    Returns:
        VerificationReport with per-probe results.
    """
    report = VerificationReport()

    for query, expected in probes:
        try:
            query_vec = embedder.embed_batch([query])[0]
            hits = store.dense_search(namespace, query_vec, top_n=top_k)
        except Exception as exc:
            logger.error("verification.search_failed query=%r error=%s", query, exc)
            report.probes.append(ProbeResult(
                query=query,
                expected_source=expected,
                found_sources=[],
                passed=False,
            ))
            continue

        found_sources = [h.payload.get("doc_id", "") for h in hits]
        passed = any(expected.lower() in src.lower() for src in found_sources)

        result = ProbeResult(
            query=query,
            expected_source=expected,
            found_sources=found_sources,
            passed=passed,
        )
        report.probes.append(result)

        if passed:
            logger.info("verification.probe_passed query=%r source=%r", query, expected)
        else:
            logger.warning(
                "verification.probe_failed query=%r expected=%r found=%r",
                query, expected, found_sources,
            )

    logger.info(report.summary())

    if assert_on_failure and not report.passed:
        raise AssertionError(
            "Ingestion verification failed:\n" + report.summary()
        )

    return report
