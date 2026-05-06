# Architecture: Chapter 3 — Dual-RAG Multi-Agent Blog System

> **Date:** 2026-05-06
> **Slug:** chapter3-dual-rag
> **Mode:** B (architecture distilled from a pre-existing PLAN; no separate REQ)
> **Requirements source:** [`specs/plans/PLAN-chapter3-dual-rag.md`](../plans/PLAN-chapter3-dual-rag.md) — authoritative for behaviors, edge cases, decisions, and scope.
> **Estimated tasks:** 30–40 across 11 groups (see Task Slicing).

## Architecture Summary

Extend the existing `blog_mas` LangGraph runtime from a hardcoded Python-dict knowledge base to a **Dual RAG** architecture composed of two independent LangGraph applications:

- **Phase 1 (ingestion, separate graph(s)):** chunk Markdown sources via four ordered stages (structural → recursive → agentic propositions → contextual), embed via HF Inference, and upsert to **Qdrant** with content-hash IDs. A second, simpler ingestion graph handles **blueprint** documents (description-only embedding).
- **Phase 2 (runtime, extended graph):** the existing `intake → researcher → writer → validator` graph gains a new **Context Librarian** node and is restructured so that **Librarian** (procedural RAG over `blueprints` namespace) and **Researcher** (factual RAG over `knowledge` namespace, hybrid retrieval + RRF + cross-encoder rerank + small-to-big expansion) run in **parallel via `asyncio.gather(..., return_exceptions=True)`** before the **Writer** consumes both. The existing validator + revision loop is preserved unchanged.

**Central organizing principle:** the *what* (factual chunks) and the *how* (Semantic Blueprint) are split at the data layer into two Qdrant collections and only rejoin in the Writer's prompt assembly. Every blueprint passes a Pydantic schema + injection-marker scan before being injected into an LLM prompt — the vector store is treated as untrusted input.

## Inferred Requirements

The full requirement set lives in the linked PLAN. The architecture below is anchored on these load-bearing requirements:

- **R-Dual-RAG.** Every blog generation retrieves both factual chunks and one Semantic Blueprint before the Writer runs.
- **R-Goal-Decomp.** Intake emits `intent_query` (style/tone) and `topic_query` (factual subject) on state.
- **R-Librarian.** New Librarian node performs semantic search over blueprints, validates against `Blueprint` Pydantic, and falls back to a neutral default on any failure.
- **R-Researcher-Upgrade.** Researcher replaces `lookup_topic()` with hybrid Qdrant retrieval; LLM synthesis includes per-bullet `[Source <id>]` citations; system prompt is anti-hallucination.
- **R-Writer-Upgrade.** Writer accepts `(facts, blueprint)` and injects the *validated, canonically re-serialized* blueprint into its system prompt.
- **R-Parallel.** Librarian + Researcher run concurrently; one branch failing does not nuke the other.
- **R-Validator-Preserved.** Existing validator + max-3 revision loop survives unchanged.
- **R-Two-Graphs.** Ingestion and runtime are independent LangGraph applications with separate entrypoints.
- **R-Hybrid-Chunking.** All four chunking stages applied in order, propositions extracted on every parent.
- **R-Hybrid-Retrieval.** Dense + BM25 sparse, fused via RRF, then cross-encoder reranked, with small-to-big parent expansion when a child wins.
- **R-Idempotent.** Content-hash chunk IDs make re-running ingestion on unchanged sources a no-op upsert.
- **R-Rebuild.** `--rebuild` flag drops + recreates the collection with async-completion polling.
- **R-Corpus.** 5 dict topics migrated verbatim to `data/knowledge/*.md`; 6 seeded blueprints in `data/blueprints/*.json`.
- **R-CLI.** `blog-mas ingest [--rebuild] [--path]`, `blog-mas ingest-blueprints [--rebuild]`, `blog-mas eval`, `blog-mas` (runtime, unchanged invocation).
- **R-Eval.** pytest-integrated `recall@k` harness under `tests/eval/` with labeled `queries.yaml`.
- **R-Observability.** LangSmith tracing + structlog structured local logs across both graphs; degrades silently when API keys absent.
- **R-Graceful.** Every retrieval failure mode resolves to a neutral default rather than a crash.
- **R-NF-Reliability.** Every external call (HF Inference embeddings + LLM, Qdrant) wrapped in tenacity retries with exponential backoff + jitter.
- **R-NF-Latency.** Parallel Librarian + Researcher should keep total runtime ≤ 1.5× the single-retrieval baseline.
- **R-NF-Security.** Blueprints retrieved from the vector store must pass Pydantic schema validation + injection-marker scan before LLM injection.
- **R-NF-Cost.** Contextualization and proposition LLM calls reuse a single batched HF endpoint client; embeddings batched ≤100 per call with adaptive halving on size errors.
- **R-NF-BackCompat.** Existing `blog_mas` test suite continues to pass after Chapter 3 changes (with dict-based KB swapped for a `FakeVectorStore` test double).

## Tech Choices

| Concern | Choice | Source |
|---|---|---|
| Vector DB | Qdrant (self-hosted via Docker, two collections: `knowledge`, `blueprints`) | PLAN Decision 1 |
| Embeddings | HF Inference API (e.g. `BAAI/bge-small-en-v1.5`); batched ≤100; tenacity retries | PLAN Decision 2 |
| Chunking-stage LLMs | Same HF Inference LLM as runtime agents | PLAN Decision 5 |
| Sparse retrieval | BM25 via Qdrant sparse vector support (`fastembed`) | PLAN Component 3 |
| Fusion | Reciprocal Rank Fusion (RRF), `k=60` default | PLAN Decision 19 |
| Reranker | `BAAI/bge-reranker-base` via HF Inference | PLAN Decision 18 |
| Tokenization (chunking) | `tiktoken` `cl100k_base` for chunk-size accounting | PLAN Component 2 / Stage 2 |
| Orchestration | LangGraph for both ingestion and runtime | PLAN Decision 12 |
| Retries | tenacity, `wait_random_exponential(min=1, max=60)`, `stop_after_attempt(6)` | PLAN Component 2 / Stage 5 |
| Tracing | LangSmith, no-op when `LANGCHAIN_API_KEY` unset | PLAN Decision 9 |
| Local logs | structlog, structured | PLAN NF-Observability |
| Eval | hand-rolled `recall@k` pytest harness | PLAN Decision 8 |
| New deps in `pyproject.toml` | `qdrant-client`, `fastembed`, `tenacity`, `structlog`, `langsmith`, `tiktoken`, `pyyaml`, `langchain-text-splitters` (or stdlib equivalent for recursive splitter) | derived from above |

## Module Boundaries

```
src/blog_mas/
├── rag/                              # NEW: shared RAG primitives + ingestion graphs
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── structural.py             # Stage 1 — markdown headers
│   │   ├── recursive.py              # Stage 2 — token-budgeted split
│   │   ├── propositions.py           # Stage 3 — agentic proposition extraction (LLM)
│   │   └── contextual.py             # Stage 4 — Anthropic-style situating context (LLM)
│   ├── embedding.py                  # HF Inference batched embeddings + tenacity + adaptive halving
│   ├── vector_store.py               # Qdrant client wrapper, namespace mgmt, upsert, dense+sparse search, lock, --rebuild polling
│   ├── retrieval.py                  # RRF fusion + reranker call + small-to-big parent expansion
│   ├── ingestion_graph.py            # LangGraph for knowledge ingestion (load → fan-out → 4 stages → embed → upsert → verify)
│   ├── blueprint_graph.py            # LangGraph for blueprint ingestion (load → validate → embed descriptions → upsert)
│   ├── blueprints.py                 # Blueprint Pydantic schema + injection-scan + neutral default
│   ├── observability.py              # LangSmith + structlog wiring (shared; no-op when keys absent)
│   └── ingest_cli.py                 # CLI subcommand glue: ingest, ingest-blueprints, eval
├── agents/
│   ├── __init__.py
│   ├── librarian.py                  # NEW node
│   ├── intake.py                     # MODIFIED: emits intent_query + topic_query
│   ├── researcher.py                 # MODIFIED (rewritten): hybrid retrieval + citation-aware synthesis
│   ├── writer.py                     # MODIFIED: consumes validated blueprint + research
│   └── validator.py                  # UNCHANGED
├── state.py                          # MODIFIED: adds intent_query, topic_query, blueprint, blueprint_match_score, blueprint_alternatives, blueprint_fallback_reason
├── orchestrator.py                   # MODIFIED: librarian node + parallel asyncio.gather branch
├── prompts.py                        # MODIFIED: extend RESEARCHER_SYSTEM_PROMPT (citation rules); add WRITER blueprint scaffold; INTAKE prompt unchanged (decomposition uses GoalDecomposition Pydantic)
├── mcp/models.py                     # MODIFIED: add GoalDecomposition; existing models unchanged
├── knowledge_base.py                 # DEPRECATED then DELETED — content migrated to data/knowledge/*.md; old test rewired to FakeVectorStore
├── cli.py                            # MODIFIED: argparse subcommands (default = runtime; ingest, ingest-blueprints, eval)
├── llm.py                            # UNCHANGED (existing chat LLM factory; new RAG modules use their own HF Inference client where embeddings/reranker are needed)
├── logging_config.py                 # UNCHANGED for stdlib logger; structlog configured separately in rag/observability.py
├── retry.py                          # UNCHANGED — agent retry handler stays; rag uses tenacity directly
└── agent_helpers.py                  # UNCHANGED

data/                                 # NEW
├── knowledge/                        # 5 *.md files migrated verbatim from knowledge_base.py
└── blueprints/                       # 6 *.json files (technical-deep-dive, executive-summary, casual-explainer, tutorial-stepwise, news-brief, opinion-essay)

tests/
├── conftest.py                       # MODIFIED: add FakeVectorStore + RAG-related fixtures
├── test_researcher_agent.py          # MODIFIED: use FakeVectorStore instead of dict KB
├── test_knowledge_base.py            # DELETED — KB module removed
├── test_intake_agent.py              # MODIFIED: assert intent_query + topic_query on output
├── test_writer_agent.py              # MODIFIED: assert blueprint injection into system prompt
├── test_orchestrator.py              # MODIFIED: assert librarian node + parallel branch
├── test_cli.py                       # MODIFIED: assert subcommand wiring
├── test_validator_agent.py           # UNCHANGED (regression guard)
├── test_mcp_protocol.py              # UNCHANGED (regression guard) or MODIFIED if GoalDecomposition added
├── test_llm.py                       # UNCHANGED (regression guard)
├── test_retry_handler.py             # UNCHANGED (regression guard)
├── rag/                              # NEW unit tests for chunking, embedding, vector_store, retrieval, blueprints
│   ├── test_blueprints.py
│   ├── test_chunking_structural.py
│   ├── test_chunking_recursive.py
│   ├── test_chunking_propositions.py
│   ├── test_chunking_contextual.py
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   ├── test_ingestion_graph.py
│   ├── test_blueprint_graph.py
│   └── test_observability.py
└── eval/                             # NEW recall@k harness
    ├── __init__.py
    ├── queries.yaml
    └── test_recall.py
```

## Data Models / Contracts

### `BlogState` (extended)

```
raw_input: str
blog_spec: BlogSpec | None
intent_query: str | None              # NEW — derived from BlogSpec by intake
topic_query: str | None               # NEW — derived from BlogSpec by intake
blueprint: Blueprint | None           # NEW — written by librarian (validated or neutral default)
blueprint_match_score: float | None   # NEW
blueprint_alternatives: list[str] | None   # NEW — top-3 candidate IDs
blueprint_fallback_reason: str | None # NEW — None on success
research_summary: ResearchSummary | None
draft: BlogDraft | None
verdict: ValidationVerdict | None
revision_feedback: str | None
revision_count: Annotated[int, operator.add]
```

### `Blueprint` (Pydantic, security-hardened)

Bounded `id`, `description`, `scene_goal`, `style_guide`, `participants: list[Participant]` (bounded count + per-item length), `instruction` (injection-marker scan), `metadata: dict[str, str|int|bool] | None`. Total serialized ≤ 8 KB. `instruction` rejects: `{{`, `}}`, `<script`, `</script`, `<|`, `|>`. All strings stripped.

### `GoalDecomposition` (Pydantic, new)

```
intent_query: str  (non-empty after strip)
topic_query: str   (non-empty after strip)
```

### Qdrant payload schema (knowledge namespace)

`raw_text`, `contextualized_text`, `parent_id` (for propositions), `doc_id`, `headings_path`, `content_hash` (= point ID), `chunk_type` ∈ {`parent`, `proposition`}, `chunk_index`, `created_at`. Token count of `raw_text`: 50 ≤ N ≤ 600.

### Qdrant payload schema (blueprints namespace)

`blueprint_json` (full JSON string for retrieval-time validation), `id`, `description`, `created_at`. **Only `description` is embedded** (PLAN Decision invariant).

### CLI surface

```
blog-mas                                                 # runtime, unchanged invocation
blog-mas ingest [--rebuild] [--path data/knowledge/]
blog-mas ingest-blueprints [--rebuild]
blog-mas eval [--queries tests/eval/queries.yaml]
```

## Patterns & Conventions

- **Existing pattern: agent node signature.** All runtime nodes follow `async def <name>_node(state: BlogState, config: RunnableConfig) -> dict` and read the LLM via `config["configurable"]["llm"]`. New `librarian_node` follows this. Pattern reference: `src/blog_mas/agents/intake.py`.
- **Existing pattern: structured-output via `run_agent_chain`.** `agent_helpers.run_agent_chain(config, model_cls, system_prompt, user_message, agent_name)` is the canonical path for LLM calls expecting Pydantic output. Goal-decomposition LLM call in intake reuses it with `model_cls=GoalDecomposition`.
- **Existing pattern: retries.** Runtime agents use `blog_mas.retry.retry_handler` (custom). RAG modules use `tenacity` decorators directly because they need `wait_random_exponential` + jitter + `retry_if_exception_type` semantics that `retry_handler` doesn't expose. Both coexist; do not refactor `retry.py` as part of this work.
- **Existing pattern: Pydantic-validated inter-agent contracts** (`mcp/models.py`). New `Blueprint` and `GoalDecomposition` join this set; no behavior change to existing models.
- **New convention: structured logs.** All RAG modules emit structlog records keyed by `stage`, `doc_id`/`query`, `latency_ms`, `chunk_count`, `score`. The runtime agents continue to use stdlib logging (no behavior change there).
- **New convention: namespaces over collections.** The chapter teaches Pinecone namespaces; we use Qdrant *collections* (`knowledge`, `blueprints`). Researcher only ever queries `knowledge`; Librarian only ever queries `blueprints`. Cross-namespace queries are architecturally impossible — no shared retrieval helper allows passing a namespace argument freely; each caller selects via a typed entrypoint.
- **New convention: untrusted vector-store payloads.** Anything read from Qdrant payload that ends up in an LLM prompt MUST pass through Pydantic + an injection-scan first. No exceptions, even for "internal" fields.
- **`raw_text` (not `contextualized_text`) reaches the Writer's LLM.** Contextualization is retrieval scaffolding; showing it to the generator distorts output.
- **Description-only embedding for blueprints.** The full JSON is stored in payload; only `description` is embedded.
- **Idempotency invariant.** Qdrant point IDs are `sha256(doc_id, chunk_index, raw_text)`. Re-running ingestion on unchanged sources must produce a no-op upsert.

## Architecture Decisions Log

(Mirrors PLAN Decisions 1–20; load-bearing entries reproduced here for tasks to reference.)

| # | Decision | Why |
|---|---|---|
| 1 | Qdrant as vector DB; two collections instead of namespaces | Self-hosted free dev; collection-level lock easy; production-grade |
| 2 | HF Inference for embeddings, propositions, contextualization, reranker | Single-provider stack; existing `HF_TOKEN`; no new credentials |
| 3 | Hybrid chunking, all four stages required for v1 | Quality compounds through every downstream stage |
| 4 | Hybrid retrieval (dense + BM25 + RRF + reranker), day one | Production target requires it |
| 5 | Goal decomposition lives inside extended `intake` (one combined node) | Minimal graph topology; one extra LLM call; co-located with BlogSpec |
| 6 | Idempotent content-hash IDs + `--rebuild` flag | Cheap reruns by default; escape hatch for breaking changes |
| 7 | Blueprint Pydantic validation + injection-marker scan at retrieval time | Vector store is untrusted boundary; ingest-time validation insufficient |
| 8 | Validator + revision loop preserved (chapter drops it) | Existing system has it; chapter notes "you'd add it back for production" |
| 9 | Apply proposition extraction to ALL parent chunks | Simpler v1; selectivity can be added later behind same interface |
| 10 | Parallel Librarian + Researcher via `asyncio.gather(return_exceptions=True)` | Latency win + partial-failure tolerance |
| 11 | RRF for sparse+dense fusion | Parameter-light, robust, well-understood |
| 12 | Top-K = 3 (knowledge), top-K = 5 (blueprints), threshold 0.7 (librarian) | Matches chapter; small enough to fit Writer's context budget |
| 13 | Migrate existing 5 dict topics to `data/knowledge/*.md` verbatim | Preserves backward compat; gives eval harness a known-good baseline |
| 14 | 6 seeded blueprints with dummy data | Covers blog-domain breadth without bloating v1 corpus |
| 15 | LangGraph for ingestion (with checkpointer) | Per-doc state machine, resume-from-failure, parity tracing in LangSmith |
| 16 | `recall@k` hand-rolled eval harness | Lightweight; pytest-integrated; LangSmith dataset compatible |
| 17 | `src/blog_mas/rag/` module layout | Cohesive with existing package; clear domain boundary |
| 18 | LangSmith + structlog observability | LangSmith integrates natively with LangGraph already in use |
| 19 | `knowledge_base.py` is deleted, not shimmed | No production caller after Researcher rewrite; tests rewired to FakeVectorStore |
| 20 | Two ingestion graphs (knowledge, blueprint) instead of one with branching | Simpler state schemas; independent CLIs; tracing readability |

## Change Footprint

**New files / modules**

| Path | Purpose | Pattern reference |
|---|---|---|
| `src/blog_mas/rag/__init__.py` | Package init | — |
| `src/blog_mas/rag/blueprints.py` | `Blueprint` Pydantic schema + `validate_blueprint_payload(json_str) -> Blueprint \| None` + `NEUTRAL_BLUEPRINT` constant + injection-marker scan | `mcp/models.py` for Pydantic style |
| `src/blog_mas/rag/embedding.py` | `EmbeddingClient.embed_batch(texts) -> list[list[float]]` with tenacity + adaptive halving + newline normalization | `llm.py` for HF token wiring |
| `src/blog_mas/rag/vector_store.py` | `QdrantStore` wrapper: `ensure_collection`, `upsert_points`, `dense_search`, `sparse_search`, `delete_collection_with_polling`, `acquire_lock` | new |
| `src/blog_mas/rag/retrieval.py` | `hybrid_search(query, namespace, top_k, **opts) -> list[Chunk]` (RRF + reranker + small-to-big parent expansion) | new |
| `src/blog_mas/rag/chunking/__init__.py` | Package init | — |
| `src/blog_mas/rag/chunking/structural.py` | `split_by_headers(md_text) -> list[Section]` (H1/H2/H3 + min-section merge) | new |
| `src/blog_mas/rag/chunking/recursive.py` | `recursive_split(section, target_tokens, overlap) -> list[ParentChunk]` | new |
| `src/blog_mas/rag/chunking/propositions.py` | `extract_propositions(parent_chunk, llm) -> list[ChildChunk]` (LLM, JSON-strict, discard-on-parse-failure) | new |
| `src/blog_mas/rag/chunking/contextual.py` | `contextualize(chunk, doc_window, llm) -> str` (LLM, 50–100 tokens) + windowing pre-flight | new |
| `src/blog_mas/rag/ingestion_graph.py` | LangGraph for knowledge ingestion (load → fan-out → 4 stages → embed → upsert → verify) | `orchestrator.py` for LangGraph style |
| `src/blog_mas/rag/blueprint_graph.py` | LangGraph for blueprint ingestion (load → validate → embed descriptions → upsert) | `orchestrator.py` |
| `src/blog_mas/rag/observability.py` | structlog config + LangSmith no-op-fallback wiring | `logging_config.py` |
| `src/blog_mas/rag/ingest_cli.py` | argparse subcommands `ingest`, `ingest-blueprints`, `eval`; calls into ingestion + blueprint graphs | `cli.py` |
| `src/blog_mas/agents/librarian.py` | New runtime node: hybrid_search blueprints → validate → state | `agents/researcher.py` for node shape |
| `data/knowledge/mediterranean-diet.md` | Verbatim migration | — |
| `data/knowledge/artificial-intelligence.md` | Verbatim migration | — |
| `data/knowledge/climate-change.md` | Verbatim migration | — |
| `data/knowledge/space-exploration.md` | Verbatim migration | — |
| `data/knowledge/mental-health.md` | Verbatim migration | — |
| `data/blueprints/technical-deep-dive.json` | Seeded with dummy data | — |
| `data/blueprints/executive-summary.json` | Seeded with dummy data | — |
| `data/blueprints/casual-explainer.json` | Seeded with dummy data | — |
| `data/blueprints/tutorial-stepwise.json` | Seeded with dummy data | — |
| `data/blueprints/news-brief.json` | Seeded with dummy data | — |
| `data/blueprints/opinion-essay.json` | Seeded with dummy data | — |
| `tests/rag/__init__.py` | — | — |
| `tests/rag/test_blueprints.py` | Schema, injection scan, neutral default | `tests/test_mcp_protocol.py` |
| `tests/rag/test_chunking_structural.py` | Header split + min-section merge | new |
| `tests/rag/test_chunking_recursive.py` | Token-budgeted recursive split | new |
| `tests/rag/test_chunking_propositions.py` | Proposition extraction + JSON-failure path | `tests/test_intake_agent.py` for LLM mocking |
| `tests/rag/test_chunking_contextual.py` | Contextualization + windowing | `tests/test_intake_agent.py` |
| `tests/rag/test_embedding.py` | Batched embeddings, retries, adaptive halving, newline normalization | `tests/test_retry_handler.py` |
| `tests/rag/test_vector_store.py` | Upsert, search, lock, async-deletion polling (with FakeQdrant) | new |
| `tests/rag/test_retrieval.py` | RRF, reranker, small-to-big | new |
| `tests/rag/test_ingestion_graph.py` | End-to-end with fakes; resume-from-failure via checkpointer | `tests/test_orchestrator.py` |
| `tests/rag/test_blueprint_graph.py` | End-to-end with fakes | `tests/test_orchestrator.py` |
| `tests/rag/test_observability.py` | structlog binding + LangSmith no-op when key absent | new |
| `tests/eval/__init__.py` | — | — |
| `tests/eval/queries.yaml` | Labeled query → expected_doc_ids records | — |
| `tests/eval/test_recall.py` | recall@k harness; pytest-skip if Qdrant unreachable | new |

**Modified files / modules**

| Path | What changes here |
|---|---|
| `src/blog_mas/state.py` | Add `intent_query`, `topic_query`, `blueprint`, `blueprint_match_score`, `blueprint_alternatives`, `blueprint_fallback_reason` to `BlogState`; preserve existing fields and reducer semantics |
| `src/blog_mas/mcp/models.py` | Add `GoalDecomposition` model; existing models unchanged |
| `src/blog_mas/agents/intake.py` | After producing `BlogSpec`, run a second `run_agent_chain` for `GoalDecomposition`; on failure, retry once then deterministic fallback; return `intent_query`/`topic_query` on state |
| `src/blog_mas/agents/researcher.py` | Replace `lookup_topic()` call with `hybrid_search(topic_query, namespace="knowledge", top_k=3)`; build prompt with `[Source <id>]` blocks; populate `ResearchSummary.source` as comma-joined chunk IDs |
| `src/blog_mas/agents/writer.py` | Read `blueprint` from state; canonicalize via `Blueprint.model_dump_json()`; inject into a fixed scaffold prompt; revision path unchanged |
| `src/blog_mas/orchestrator.py` | Add `librarian` node; replace single edge `intake → research` with a parallel branch (Librarian + Researcher in `asyncio.gather`); both join into `write` |
| `src/blog_mas/prompts.py` | Extend `RESEARCHER_SYSTEM_PROMPT` with citation rules; add `WRITER_BLUEPRINT_SCAFFOLD`; existing prompts preserved |
| `src/blog_mas/cli.py` | Convert single-purpose CLI to argparse subcommands; default (no subcommand) preserves runtime invocation; `ingest`/`ingest-blueprints`/`eval` delegate to `rag/ingest_cli.py` |
| `tests/conftest.py` | Add `FakeVectorStore`, `FakeEmbedder`, `FakeReranker`, `make_mock_blueprint`, fixtures for chunked sample doc |
| `tests/test_researcher_agent.py` | Replace dict-KB stubs with `FakeVectorStore`; assert citations in prompt + `ResearchSummary.source` |
| `tests/test_intake_agent.py` | Add assertions for `intent_query`/`topic_query` on output state |
| `tests/test_writer_agent.py` | Add assertion that injected blueprint matches canonical JSON; assert prompt scaffold |
| `tests/test_orchestrator.py` | Assert `librarian` node exists; assert parallel branch composition; assert revision loop preserved |
| `tests/test_cli.py` | Assert subcommand routing for `ingest`, `ingest-blueprints`, `eval`, default-runtime |
| `pyproject.toml` | Add deps: `qdrant-client`, `fastembed`, `tenacity`, `structlog`, `langsmith`, `tiktoken`, `pyyaml`, `langchain-text-splitters` (verify if needed) |

**Deleted / replaced**

| Path | Why |
|---|---|
| `src/blog_mas/knowledge_base.py` | No production caller after Researcher rewrite; data migrated to `data/knowledge/*.md`; tests rewired to `FakeVectorStore` (PLAN Decision 19) |
| `tests/test_knowledge_base.py` | Module under test is deleted |

**Touched but not changed (regression-guard hotspots)**

| Path | Why we're touching it |
|---|---|
| `src/blog_mas/agents/validator.py` | Validator logic unchanged; revision loop must still trigger on `verdict=fail` and increment `revision_count` after Writer rewrite |
| `src/blog_mas/agent_helpers.py` | `run_agent_chain` reused by intake decomposition + librarian; signature must not drift |
| `src/blog_mas/retry.py` | Existing retry handler reused by all runtime agents; new RAG modules use tenacity instead, but `retry.py` itself is untouched |
| `src/blog_mas/llm.py` | Chat LLM factory unchanged; RAG modules read `HF_TOKEN` directly |
| `src/blog_mas/logging_config.py` | Stdlib logger config preserved; structlog configured separately |
| `src/blog_mas/mcp/protocol.py` | MCP envelope unchanged |
| `tests/test_validator_agent.py` | Validator behavior must remain identical |
| `tests/test_llm.py` | Chat LLM factory must remain identical |
| `tests/test_retry_handler.py` | Retry handler must remain identical |
| `tests/test_mcp_protocol.py` | Protocol envelope must remain identical (unless `GoalDecomposition` addition motivates a small update) |

## Areas of Impact (risk-rated)

| Area | Files | Risk | Why |
|---|---|---|---|
| Researcher rewrite (dict → hybrid retrieval) | `agents/researcher.py`, `prompts.py` | **H** | Total rewrite of an agent that downstream Writer + Validator depend on; must preserve `ResearchSummary` shape |
| Blueprint security boundary | `rag/blueprints.py`, `agents/librarian.py`, `agents/writer.py` | **H** | Prompt-injection vector if any path bypasses validation |
| Vector store / Qdrant integration | `rag/vector_store.py`, `rag/embedding.py` | **H** | New external dep; async deletion semantics; lock; embedding-dim drift |
| Hybrid retrieval correctness | `rag/retrieval.py` | **H** | RRF + reranker + small-to-big together drive end-to-end quality; bugs are silent |
| Runtime parallel orchestration | `orchestrator.py` | **M** | `asyncio.gather(return_exceptions=True)` semantics; one branch failing must not nuke the other |
| Intake extension | `agents/intake.py`, `mcp/models.py` | **M** | Existing intake test must still pass; new fields are additive |
| Writer rewrite | `agents/writer.py`, `prompts.py` | **M** | Dynamic prompt assembly; revision path must remain unchanged |
| State schema extension | `state.py` | **M** | `Annotated`/`operator.add` reducer must continue to work for `revision_count`; new fields default to None |
| Chunking pipeline (LLM-driven) | `rag/chunking/propositions.py`, `rag/chunking/contextual.py` | **M** | LLM-driven; quality affects retrieval; JSON-parse failures must not lose parents |
| CLI extension | `cli.py`, `rag/ingest_cli.py` | **L–M** | argparse subcommand introduction; default (no subcommand) must remain runtime |
| Test rewiring | `tests/conftest.py`, `tests/test_researcher_agent.py`, deletion of `test_knowledge_base.py` | **M** | Backward compat — full existing suite must remain green |
| Validator revision loop | `agents/validator.py` (touched-not-changed) + `orchestrator.py` (modified) | **M** | Loop semantics must survive parallel-branch refactor |
| Eval harness | `tests/eval/*` | **L** | New surface; pytest-skip on missing infra |
| Observability wiring | `rag/observability.py` | **L** | Must degrade silently when keys absent |
| Data migration | `data/knowledge/*.md`, `data/blueprints/*.json` | **L** | Static content; verbatim migration |

## Risk & Stress-Test Scenarios

### Forward (runtime / ingestion failures the design must handle)

| # | Scenario | How the design handles it |
|---|---|---|
| F1 | Qdrant unreachable on ingest CLI startup | Fail-fast with structured error from `QdrantStore.ensure_collection` |
| F2 | Qdrant unreachable on runtime startup | Runtime proceeds: Researcher returns empty bullets, Librarian returns neutral default, Writer instructed to handle "no data" via blueprint |
| F3 | HF Inference rate limit (429) on embedding | tenacity exponential backoff with jitter, 6 attempts |
| F4 | HF Inference batch too large (413/400) | Adaptive halving (100 → 50 → 25 → ...) |
| F5 | Empty retrieval (no chunks above score threshold) | Researcher returns `bullet_points=[]`, `source="none"`; pipeline continues |
| F6 | Librarian top score < 0.7 | Neutral default blueprint; `blueprint_fallback_reason="low_score"` |
| F7 | Blueprint payload missing/malformed JSON | Neutral default; structured error log |
| F8 | Blueprint fails Pydantic validation | Neutral default; log violating field |
| F9 | Blueprint contains injection markers | Neutral default; log as `security_event` |
| F10 | Goal decomposition LLM returns invalid JSON | Retry once with stricter prompt; on second failure, deterministic fallback derivation |
| F11 | Concurrent ingestion runs against same collection | Distributed lock via collection metadata key; second run fails-fast |
| F12 | Partial ingestion failure mid-run | LangGraph checkpointer resumes from last completed stage on next invocation |
| F13 | Reranker backend errors | Soft-fail; return RRF top-K unmodified; log degraded mode |
| F14 | Dense backend errors but sparse succeeds | Proceed with sparse-only; log degraded retrieval |
| F15 | Both dense + sparse error | Researcher returns empty; Writer proceeds with blueprint-only |
| F16 | Contextualization LLM exceeds context window | Pre-flight: split doc into 8k-token windows; chunk contextualized against its window |
| F17 | Proposition extraction returns malformed JSON | Discard propositions for that parent; index parent solo; log warning |
| F18 | Blueprint corpus retrieval returns 0 results (empty namespace) | Neutral default; log warning advising `ingest-blueprints` |
| F19 | Parallel Librarian + Researcher: one fails | `asyncio.gather(return_exceptions=True)`; Writer proceeds with available context |
| F20 | LangSmith API unreachable | Tracing degrades silently to local logs only |
| F21 | Embedding-dim drift on collection | Refuse upsert; require explicit `--rebuild` |
| F22 | Doc removed from `data/knowledge/` between ingest runs | Documented limitation: incremental mode leaves orphans; `--rebuild` is the supported path |
| F23 | Validator's revision loop triggers post-Chapter-3 changes | Existing flow preserved; blueprint + facts already in state; Writer re-runs with same context + revision feedback |

### Backward (regression risks per touched-but-not-changed area)

| # | Touched area | What could regress | How we'd know / mitigation |
|---|---|---|---|
| B1 | `agents/validator.py` | Verdict-fail path no longer triggers retry after orchestrator refactor | Existing `test_validator_agent.py` + a new orchestrator test that drives a fail→retry→pass sequence end-to-end |
| B2 | `agent_helpers.run_agent_chain` | Signature drift breaks all agents | Existing agent tests serve as the contract; if they still pass, signature is intact |
| B3 | `retry.py` retry handler | Retry semantics change accidentally | `test_retry_handler.py` is unchanged; must remain green |
| B4 | `llm.py` ChatHuggingFace factory | Token resolution / endpoint config changes | `test_llm.py` regression-guards |
| B5 | `mcp/models.py` existing models | Adding `GoalDecomposition` accidentally mutates `BlogSpec`/`ResearchSummary`/`BlogDraft`/`ValidationVerdict` | `test_mcp_protocol.py` regression-guards |
| B6 | `BlogState` reducer for `revision_count` | New `Annotated` fields break the existing `operator.add` reducer | Orchestrator end-to-end test that runs through 2+ revisions |
| B7 | `cli.py` default invocation | Subcommand introduction breaks zero-arg `blog-mas` runtime | `test_cli.py` asserts default-no-subcommand still routes to runtime |
| B8 | `ResearchSummary` shape consumed by Writer + Validator | Researcher rewrite changes the field shape (e.g. drops `source` or renames `bullet_points`) | Pydantic schema is the wire contract; existing Writer/Validator tests serve as integration guards |
| B9 | `BlogDraft` shape consumed by Validator | Writer rewrite changes the draft shape | Existing Validator tests + draft Pydantic schema |
| B10 | Existing intake behavior on minimal input | Goal decomposition step crashes when raw_input is sparse | Existing `test_intake_agent.py` cases stay green; deterministic fallback covers the failure path |

## Out of Scope

(Mirrors PLAN's "Out of Scope (deferred / future)" — load-bearing for tasks to NOT implement.)

- **Manifest-diff incremental delete** of orphaned chunks. Path: `--rebuild` is the supported deletion mechanism.
- **Pluggable chunking-strategy registry** beyond the locked-in default.
- **RAGAS-based generation evals** (faithfulness, answer-relevance).
- **Reranker model swap to Cohere or proprietary**.
- **Multi-tenant namespace isolation** beyond `knowledge`/`blueprints`.
- **Streaming generation in Writer**.
- **Web UI for blueprint authoring**.
- **A/B testing of blueprints**.
- **Caching layer for embeddings/retrievals at runtime**.
- **Refactor of `retry.py`** to merge with tenacity.
- **structlog migration of runtime agents** — runtime agents continue to use stdlib logging; structlog is scoped to `rag/`.

## Task Slicing

Tasks are generated and reviewed **one group at a time**, in dependency order. The ARCH below carries an empty `# Tasks` section that fills up group by group.

| Group | Scope | ~Tasks |
|---|---|---|
| G1 | Foundations: Blueprint schema + injection scan; embedding client; Qdrant store wrapper; FakeVectorStore + RAG fixtures | 4–5 |
| G2 | Chunking: structural / recursive / propositions / contextual stages | 4 |
| G3 | Retrieval: dense+sparse hybrid + RRF + reranker + small-to-big | 1–2 |
| G4 | Ingestion graphs: knowledge ingestion graph; blueprint ingestion graph; idempotency + rebuild + lock | 3–4 |
| G5 | Runtime state & intake: extend `BlogState`; goal decomposition in intake | 2 |
| G6 | Runtime agents: Librarian (new); Researcher rewrite; Writer rewrite | 3 |
| G7 | Orchestrator: parallel `asyncio.gather` + librarian wiring | 1–2 |
| G8 | Data corpus: migrate 5 dict topics → md; seed 6 blueprints | 2 |
| G9 | CLI & ops: `ingest`, `ingest-blueprints`, `eval` subcommands | 2–3 |
| G10 | Eval & test rewiring: `recall@k` harness + queries.yaml; rewire researcher tests; delete `test_knowledge_base.py` | 3 |
| G11 | Observability: LangSmith + structlog wiring | 1 |

## Open Questions

(Carried forward from PLAN — flag in tasks they touch.)

- **Q1.** HF Inference availability/latency for `BAAI/bge-reranker-base`. Default: start with HF Inference; if issues during implementation, add local `sentence-transformers` fallback as a follow-up task. **Affects G3.**
- **Q2.** Knowledge corpus markdown header synthesis. Default: migrate verbatim (no synthetic headers); revisit if `recall@k` shows weak topical separation. **Affects G8.**

# Tasks

_Filled group by group via the generate-tasks skill. Currently populated: G1–G2 (T1–T8). Pending: G3–G11 (T9–T24)._

## Task T1: Blueprint Pydantic Schema + Injection Scan + Neutral Default

> **Status:** not started
> **Effort:** s
> **Priority:** high
> **Depends on:** None
> **Satisfies REQs:** R-NF-Security; R-Librarian (validation portion); R-Graceful (neutral default)
> **Footprint slice:** New: `src/blog_mas/rag/blueprints.py`, `src/blog_mas/rag/__init__.py`, `tests/rag/__init__.py`, `tests/rag/test_blueprints.py`
> **High-risk areas touched:** Blueprint security boundary (H)

### Description

Implement the `Blueprint` Pydantic model that all retrieved blueprints MUST validate against before being injected into any LLM prompt. The vector store is treated as untrusted input — this module is the security boundary. Also export `NEUTRAL_BLUEPRINT` (a known-good fallback used when retrieval fails or validation rejects a payload) and `validate_blueprint_payload(json_str)` (the single entrypoint that callers use).

### Test Plan

#### Test File(s)
- `tests/rag/test_blueprints.py`

#### Test Scenarios

##### `Blueprint schema validation`

- **accepts a well-formed blueprint JSON** — GIVEN a JSON string with all required fields within bounds, WHEN `validate_blueprint_payload(json_str)` runs, THEN it returns a `Blueprint` instance with fields populated _(verifies R-NF-Security happy path)_
- **rejects blueprint exceeding 8 KB serialized** — GIVEN a JSON string whose serialized form is >8 KB, WHEN validated, THEN returns `None` and logs `librarian.fallback reason=schema_violation` _(verifies PLAN edge case 26)_
- **rejects each per-field length-bound violation** — parametrized over `description`, `scene_goal`, `style_guide`, `instruction` exceeding bounds; THEN returns `None` _(verifies R-NF-Security)_
- **strips whitespace on string fields** — GIVEN a payload with leading/trailing whitespace in `description`, WHEN validated, THEN the returned model has stripped values _(verifies PLAN Component 1 validation rules)_
- **bounds participants list count and per-item length** — GIVEN a payload with too many participants OR an oversized participant string, THEN returns `None`
- **rejects metadata with non-primitive values** — GIVEN `metadata` containing nested dicts/lists, THEN returns `None`

##### `Injection-marker scan`

- **rejects `{{` in instruction** — GIVEN a payload with `{{` in `instruction`, WHEN validated, THEN returns `None` and emits a structured `security_event` log _(verifies F9)_
- **rejects `}}`, `<script`, `</script`, `<|`, `|>` in instruction** — parametrized; same outcome as above _(verifies F9)_
- **does NOT reject those markers in non-instruction fields** — GIVEN a payload with `<script` in `description`, THEN validated successfully (only `instruction` is scanned per PLAN Component 1)
- **scan happens after stripping** — GIVEN injection markers padded with whitespace, the scan still detects them

##### `JSON parse failures`

- **returns None on malformed JSON** — GIVEN a non-JSON string, THEN returns `None` and logs `librarian.fallback reason=json_parse_error` _(verifies F7)_
- **returns None on missing required fields** — GIVEN a JSON missing `instruction`, THEN returns `None` and logs `librarian.fallback reason=schema_violation field=instruction` _(verifies F8)_

##### `Neutral default constant`

- **NEUTRAL_BLUEPRINT validates against the schema** — GIVEN the exported `NEUTRAL_BLUEPRINT`, WHEN passed through the same validator, THEN it returns a valid `Blueprint` _(verifies R-Graceful)_
- **NEUTRAL_BLUEPRINT has the documented shape** — assert exact `id="blueprint_neutral_default"` and the canonical fields from PLAN Component 4

### Implementation Notes

- **Module:** `rag/blueprints.py` (new)
- **Pattern reference:** `src/blog_mas/mcp/models.py` for Pydantic v2 model + `field_validator` style; see existing `BlogSpec` and `ResearchSummary` for length-bound field validators.
- **Key decisions:** Decision 7 — validation at retrieval time. `validate_blueprint_payload` MUST be called every time a blueprint is read from Qdrant, not just at ingest. Document this contract in the module docstring.
- **Libraries:** `pydantic>=2.13.3` (already a dep). No new deps for this task.
- **High-risk callouts:** Blueprint security boundary (H) — any field that can flow into the Writer's prompt must pass the injection scan; the test plan covers `instruction` directly. Only `instruction` is scanned per PLAN; the test plan asserts that markers in `description` are intentionally allowed (they don't reach the prompt scaffold by design — the Writer only consumes the canonicalized JSON of the validated model, which Pydantic-encodes safely).

### Scope Boundaries

- Do NOT add Qdrant retrieval logic — that lives in T3 / G3.
- Do NOT call any LLM — pure schema + scan.
- Do NOT integrate with Librarian — that's G6.
- Do NOT introduce structlog — stdlib `logging.getLogger(__name__)` only (structlog is G11).
- Only implement: `Blueprint` model, `Participant` model, `validate_blueprint_payload(json_str: str) -> Blueprint | None`, `NEUTRAL_BLUEPRINT: Blueprint` constant, internal injection-marker scan helper.

### Files Expected

_Anchored on ARCH's Change Footprint._

**New files:** _(from ARCH "New files / modules")_
- `src/blog_mas/rag/__init__.py` — empty package init
- `src/blog_mas/rag/blueprints.py` — `Blueprint`, `Participant`, `NEUTRAL_BLUEPRINT`, `validate_blueprint_payload`, scan helper
- `tests/rag/__init__.py` — empty package init
- `tests/rag/test_blueprints.py`

**Modified files:** _(none)_

**Must NOT modify:** _(out of scope per ARCH and per task scope)_
- `src/blog_mas/mcp/models.py` — `GoalDecomposition` addition is G5
- `src/blog_mas/agents/**` — agent integration is G6
- `tests/conftest.py` — fixtures land in T4

---

## Task T2: EmbeddingClient with Retries + Adaptive Halving

> **Status:** not started
> **Effort:** m
> **Priority:** high
> **Depends on:** None
> **Satisfies REQs:** R-NF-Reliability; R-NF-Cost; F3; F4
> **Footprint slice:** New: `src/blog_mas/rag/embedding.py`, `tests/rag/test_embedding.py`. Modified: `pyproject.toml` (add `tenacity`)
> **High-risk areas touched:** Vector store / Qdrant integration (H — embedding component)

### Description

Implement an `EmbeddingClient` that wraps HF Inference's text-embedding endpoint. It batches inputs (≤100 per call), normalizes newlines before send, retries transient failures with tenacity (`wait_random_exponential(min=1, max=60)`, `stop_after_attempt(6)`), and adaptively halves the batch on 413/400 size errors down to single-item batches. This is the single entrypoint every other module uses to produce embeddings — chunking, retrieval, and ingestion graphs all depend on it.

### Test Plan

#### Test File(s)
- `tests/rag/test_embedding.py`

#### Test Scenarios

##### `Happy path`

- **embeds a batch of N texts to N vectors** — GIVEN a list of 5 strings and a fake HF endpoint returning vectors of dim 384, WHEN `embed_batch(texts)` is called, THEN returns 5 vectors of dim 384 _(verifies R-NF-Reliability normal path)_
- **embeds an empty batch as empty list** — GIVEN `embed_batch([])`, THEN returns `[]` without making an HTTP call

##### `Newline normalization`

- **replaces `\n` with space before embedding** — GIVEN a text containing `\n`, WHEN embedded, THEN the request body sent to HF contains `' '` instead of `\n` _(verifies PLAN "common mistakes": newline normalization invariant)_

##### `Resilience — 429 rate limit`

- **retries on 429 with exponential backoff** — GIVEN the fake endpoint returns 429 twice then succeeds, WHEN `embed_batch` is called, THEN it retries with backoff and returns the vectors on the third attempt _(verifies F3)_
- **gives up after 6 attempts and raises** — GIVEN the fake endpoint always returns 429, WHEN `embed_batch` is called, THEN after 6 attempts a tenacity-derived exception is raised
- **backoff is capped at `wait_random_exponential(min=1, max=60)`** — assert via injected wait-time inspector that no single sleep exceeds 60s

##### `Adaptive halving — 413/400`

- **halves batch on 413** — GIVEN a batch of 100 and the endpoint rejects with 413 once, WHEN `embed_batch` retries, THEN the next request contains 50 texts and the call succeeds combining results _(verifies F4)_
- **continues halving down to single-item batches** — chain 413→413→413→success; assert sequence 100→50→25→12 (using the implementation's halving rule)
- **fails when even a 1-item batch is rejected** — GIVEN 413 even at batch size 1, THEN raises a clear error

##### `Token resolution`

- **reads `HF_TOKEN` from env when not passed** — patches `HF_TOKEN`; assert client uses it
- **constructor token overrides env** — assert constructor-provided token wins

### Implementation Notes

- **Module:** `rag/embedding.py` (new)
- **Pattern reference:** `src/blog_mas/llm.py` for HF token resolution from `HF_TOKEN` env via `python-dotenv`. **Do not** reuse `ChatHuggingFace` — embeddings need a different endpoint surface (`huggingface_hub.InferenceClient.feature_extraction`).
- **Key decisions:**
  - Decision 2 (HF Inference for all model calls).
  - tenacity here, NOT `blog_mas.retry.retry_handler` — the latter doesn't expose `retry_if_exception_type` or `wait_random_exponential` semantics. Both retry layers coexist (PLAN Pattern note).
- **Libraries:** `tenacity` (NEW dep, add to `pyproject.toml`); `huggingface_hub.InferenceClient` (transitive dep from `langchain-huggingface`; verify availability and pin explicitly if needed).
- **Adaptive halving rule:** on `HTTPError` with status 413/400 containing a size-related message, split the batch in half and embed each half separately, recursively. Re-raise other errors.
- **High-risk callouts:** Vector store integration (H) — embedding-dim drift detection lives in T3 (`QdrantStore.ensure_collection`); this task only emits whatever vector dim the HF model returns. The dim is fixed by the chosen model.

### Scope Boundaries

- Do NOT call Qdrant — embedding only.
- Do NOT implement caching — explicitly Out of Scope (PLAN).
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `EmbeddingClient(model: str | None, token: str | None)`, `embed_batch(texts: list[str]) -> list[list[float]]`, internal halving helper, exception types as needed.

### Files Expected

**New files:**
- `src/blog_mas/rag/embedding.py`
- `tests/rag/test_embedding.py`

**Modified files:**
- `pyproject.toml` — add `tenacity` to `[project] dependencies`

**Must NOT modify:**
- `src/blog_mas/llm.py` — chat LLM factory unchanged
- `src/blog_mas/retry.py` — existing agent retry handler unchanged

---

## Task T3: QdrantStore Wrapper

> **Status:** not started
> **Effort:** m
> **Priority:** high
> **Depends on:** None
> **Satisfies REQs:** R-Idempotent; R-Rebuild; R-NF-Reliability (Qdrant calls); F1; F11; F21
> **Footprint slice:** New: `src/blog_mas/rag/vector_store.py`, `tests/rag/test_vector_store.py`. Modified: `pyproject.toml` (add `qdrant-client`, `fastembed`)
> **High-risk areas touched:** Vector store / Qdrant integration (H)

### Description

Implement `QdrantStore`, the single wrapper around `qdrant-client` that all other modules use. It owns: `ensure_collection` with embedding-dim drift detection; idempotent upsert via content-hash IDs; dense + sparse (BM25) search; async-deletion-with-polling (required because Qdrant deletion is async — recreate-immediately-after-delete races and silently drops data); and a collection-level lock (`ingestion_lock_held_by`) for concurrent-ingestion protection.

### Test Plan

#### Test File(s)
- `tests/rag/test_vector_store.py`

**Mocking strategy:** define an inline `FakeQdrantClient` test double in this module that implements only the qdrant-client surface we call (`get_collections`, `recreate_collection`, `delete_collection`, `upsert`, `search`, `query_points`, etc.). This is at a different abstraction level than the `FakeVectorStore` from T4 (which mirrors `QdrantStore`'s public interface, not qdrant-client's).

#### Test Scenarios

##### `ensure_collection`

- **creates collection when missing** — GIVEN the fake client reports no collection named `knowledge`, WHEN `ensure_collection("knowledge", dim=384)` is called, THEN client.create_collection is called once with vector dim 384
- **no-op when existing collection has matching dim** — GIVEN existing `knowledge` with dim 384, WHEN `ensure_collection("knowledge", dim=384)` is called, THEN no create/recreate calls are made
- **raises `EmbeddingDimDriftError` on dim mismatch** — GIVEN existing `knowledge` with dim 768 and call with dim=384, THEN raises and refuses upsert _(verifies F21)_
- **fails-fast with structured error when Qdrant unreachable** — GIVEN client raises `ConnectionError`, THEN `ensure_collection` propagates with a structured log _(verifies F1)_

##### `Upsert idempotency`

- **same content → same content_hash ID → single point** — GIVEN two upsert calls with identical `(doc_id, chunk_index, raw_text)`, WHEN both run, THEN the resulting point count is 1 (Qdrant overwrites by ID) _(verifies R-Idempotent)_
- **different chunk_index produces different IDs** — even with identical raw_text, two distinct `chunk_index` values produce two points

##### `Dense search`

- **returns top-N with scores** — GIVEN seeded points and a query vector, WHEN `dense_search(name, vec, top_n=20)` is called, THEN returns up to 20 results sorted by score desc, each carrying the full payload
- **respects namespace isolation** — `dense_search(name="knowledge", ...)` never returns points from `blueprints` (PLAN Pattern: namespaces over collections)

##### `Sparse search (BM25)`

- **returns top-N by sparse relevance** — GIVEN seeded points with raw_text covering a query term, WHEN `sparse_search(name, query, top_n)` is called, THEN highest-BM25 results return first
- **soft-fails to empty list on backend error** — GIVEN sparse backend raises, THEN method returns `[]` and logs degraded mode _(backs F14; caller decides graceful degradation in G3)_

##### `Async deletion polling`

- **`delete_collection_with_polling` waits until collection is gone** — GIVEN the fake client returns from `delete_collection` immediately but reports the collection still present for 2 polls, WHEN `delete_collection_with_polling` is called, THEN it polls until absence is confirmed before returning _(verifies the chapter's "async deletion" gotcha; backs `--rebuild` correctness in G9)_
- **raises after polling timeout** — GIVEN the collection never disappears, THEN raises a clear timeout error

##### `Collection lock`

- **`acquire_lock` writes the metadata key** — GIVEN a fresh collection, WHEN `acquire_lock(name, hostname, pid)` is called, THEN a metadata key `ingestion_lock_held_by={hostname}-{pid}` is written
- **second acquire on held lock fails-fast with `LockHeldError`** — GIVEN a held lock, WHEN a second process attempts `acquire_lock`, THEN it raises `LockHeldError` carrying the holder identifier _(verifies F11)_
- **`release_lock` removes the key** — round-trip: acquire → release → acquire-by-other succeeds

### Implementation Notes

- **Module:** `rag/vector_store.py` (new)
- **Pattern reference:** No existing pattern in this codebase. Match the rest of `rag/` for module shape and stdlib logging.
- **Key decisions:**
  - Decision 1 (Qdrant + two collections, `knowledge` and `blueprints`).
  - Decision 6 (content-hash IDs + `--rebuild`).
  - Decision 11 (BM25 sparse via `fastembed`).
- **Libraries:** `qdrant-client` (NEW), `fastembed` (NEW — for BM25 sparse vectors), `tenacity` (added by T2).
- **Configuration:** read Qdrant URL from `QDRANT_URL` env; fall back to `http://localhost:6333`. Lock holder format: `{hostname}-{pid}`.
- **Sparse vectors:** use Qdrant's named sparse vector support; embed sparse vectors via `fastembed.SparseTextEmbedding` (loaded lazily on first sparse call to keep import time low).
- **Polling defaults:** `delete_collection_with_polling` polls every 0.5s up to 30s timeout (constants exported for test overrides).
- **High-risk callouts:**
  - Vector store integration (H): `delete_collection_with_polling` is required because Qdrant's deletion is async; racing recreate against in-flight deletion silently drops data. Test plan asserts polling.
  - Embedding-dim drift (F21): `ensure_collection` MUST raise rather than upsert with mismatched dim — this is the only defense against silent corruption when an operator swaps embedding models without `--rebuild`.

### Scope Boundaries

- Do NOT implement RRF, reranking, or small-to-big — those live in `rag/retrieval.py` (G3).
- Do NOT implement the LangGraph ingestion graphs — those are G4.
- Do NOT implement the actual `--rebuild` CLI flag — that's G9; this task provides the building block (`delete_collection_with_polling`).
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `QdrantStore(url, api_key=None)`, `ensure_collection(name, dim, sparse=False)`, `upsert_points(name, points)`, `dense_search(name, query_vec, top_n)`, `sparse_search(name, query_text, top_n)`, `delete_collection_with_polling(name, timeout_s=30)`, `acquire_lock(name, hostname, pid)`, `release_lock(name)`, `EmbeddingDimDriftError`, `LockHeldError`.

### Files Expected

**New files:**
- `src/blog_mas/rag/vector_store.py`
- `tests/rag/test_vector_store.py`

**Modified files:**
- `pyproject.toml` — add `qdrant-client` and `fastembed`

**Must NOT modify:**
- `src/blog_mas/llm.py`, `src/blog_mas/retry.py` — touched-but-not-changed
- `src/blog_mas/agents/**`, `src/blog_mas/orchestrator.py`, `src/blog_mas/state.py` — runtime path is later groups
- `src/blog_mas/rag/blueprints.py`, `src/blog_mas/rag/embedding.py` — owned by T1/T2

---

## Task T4: FakeVectorStore + FakeEmbedder + FakeReranker + RAG Fixtures

> **Status:** not started
> **Effort:** s
> **Priority:** high
> **Depends on:** T1 (for `sample_blueprint_payload` fixture content), T2 (interface conformance for `FakeEmbedder`), T3 (interface conformance for `FakeVectorStore`)
> **Satisfies REQs:** R-NF-BackCompat (gates G10 rewiring of researcher tests); supports all G2–G10 test infrastructure
> **Footprint slice:** Modified: `tests/conftest.py`. New: `tests/rag/test_fakes_conformance.py`
> **High-risk areas touched:** Test rewiring (M)

### Description

Implement the cross-group test doubles: `FakeVectorStore` (mirrors `QdrantStore`'s public surface, in-memory per namespace), `FakeEmbedder` (deterministic fake embeddings), `FakeReranker` (deterministic fake reranker — used starting G3). Plus shared pytest fixtures (`sample_markdown_doc`, `sample_blueprint_payload`, `make_chunk`) consumed by G2–G10 test files. A small conformance test file ensures the fakes track the real interfaces over time.

### Test Plan

#### Test File(s)
- `tests/rag/test_fakes_conformance.py`

The fakes themselves get exercised by their downstream consumers across G2–G10. This file's job is to catch interface drift early.

#### Test Scenarios

##### `FakeVectorStore conformance`

- **upsert + dense_search round-trips a payload** — GIVEN a `FakeVectorStore` seeded with a few points, WHEN `dense_search(name, vec, top_n=2)` is called, THEN returns 2 points with their full payloads
- **respects namespace isolation** — points upserted into `knowledge` are not returned from a `blueprints` query
- **mirrors `QdrantStore` public method names** — assert via `inspect.signature` that `FakeVectorStore` exposes `ensure_collection`, `upsert_points`, `dense_search`, `sparse_search`, `delete_collection_with_polling`, `acquire_lock`, `release_lock` (drift detector)
- **supports configurable failure injection** — GIVEN `fake.fail_on("dense_search")`, THEN the next `dense_search` raises (used by G6/G7 tests for parallel-failure scenarios)

##### `FakeEmbedder conformance`

- **deterministic vectors per text** — same text in → same vector out
- **batch matches single-item-at-a-time** — `embed_batch([a, b, c])` returns the same vectors as embedding each individually
- **configurable dim** — fixture parameter sets the dim; default 384

##### `FakeReranker conformance`

- **deterministic order from a fixed scoring function** — same `(query, docs)` → same reranked order
- **preserves caller's top-K request** — GIVEN top_k=3 and 5 docs, returns 3
- **soft-fails to identity ordering when configured** — `fake.degraded=True` returns input order unchanged _(used by F13 retrieval test in G3)_

##### `Shared RAG fixtures`

- **`sample_markdown_doc` fixture** — yields a small markdown doc with 2 H1, 3 H2 sections, used by G2 chunking tests
- **`sample_blueprint_payload` fixture** — yields a valid `Blueprint` JSON string usable by T1's validator and by G6 librarian tests (sanity-asserted to validate via T1's `validate_blueprint_payload`)
- **`make_chunk` factory** — helper to construct a chunk dict with a content_hash ID, used by G3 retrieval tests

### Implementation Notes

- **Module:** Code lives in `tests/conftest.py` (since fixtures are pytest-discovered there); the fakes themselves are exported from `tests/conftest.py` so other test files can `from tests.conftest import FakeVectorStore`.
- **Pattern reference:** `tests/conftest.py` already exports `make_mock_llm`, `make_mock_llm_sequence`, `make_failing_llm`, `make_config` — match this style. Use plain classes / module-level functions; no pytest plugins.
- **Key decisions:**
  - Use duck typing — do NOT introduce a `Protocol` class. Keeps friction low and matches the existing code's style.
  - Conformance test asserts via `inspect.signature` that public method names line up between `FakeVectorStore` and `QdrantStore`. Drift triggers a clear test failure.
- **Failure injection:** `FakeVectorStore.fail_on(method_name, exc=...)` — used by G6/G7 tests for parallel-failure scenarios; not used in G1 itself.
- **High-risk callouts:** Test rewiring (M) — divergence between `FakeVectorStore` and `QdrantStore` is the #1 failure mode for downstream test reliability. The conformance test catches name drift; signature drift requires manual diff during code review.

### Scope Boundaries

- Do NOT modify any existing test file — the rewiring of `test_researcher_agent.py` and deletion of `test_knowledge_base.py` is G10's job.
- Do NOT implement RAG production logic in the fakes — they are test doubles only.
- Do NOT add the `FakeQdrantClient` (qdrant-client level fake) here — that's an inline helper inside `tests/rag/test_vector_store.py` (T3) at a different abstraction level.
- Only implement: `FakeVectorStore`, `FakeEmbedder`, `FakeReranker`, and the three pytest fixtures listed above.

### Files Expected

**New files:**
- `tests/rag/test_fakes_conformance.py`

**Modified files:**
- `tests/conftest.py` — add `FakeVectorStore`, `FakeEmbedder`, `FakeReranker`, `sample_markdown_doc`, `sample_blueprint_payload`, `make_chunk`

**Must NOT modify:**
- `tests/test_*.py` (existing tests untouched in G1; rewiring is G10)
- `src/blog_mas/**` (no production code in T4)

---

### TDD Sequence (G1)

T1, T2, and T3 are independent and may be implemented in parallel by separate sessions/agents. T4 should follow T1+T2+T3 because its conformance tests reference their interfaces.

If implemented serially: **T1 → T2 → T3 → T4**.

---

## Task T5: Structural Split (Stage 1, no LLM)

> **Status:** not started
> **Effort:** s
> **Priority:** high
> **Depends on:** None
> **Satisfies REQs:** R-Hybrid-Chunking (Stage 1); PLAN edge case 25 (no-headers doc)
> **Footprint slice:** New: `src/blog_mas/rag/chunking/__init__.py`, `src/blog_mas/rag/chunking/types.py`, `src/blog_mas/rag/chunking/structural.py`, `tests/rag/test_chunking_structural.py`
> **High-risk areas touched:** Chunking pipeline (M)

### Description

Implement Stage 1 of the chunking pipeline: split a Markdown document on H1/H2/H3 headers, preserve the heading-ancestor path per section, and merge sections shorter than `min_section_tokens` (default 50) forward into the next section. Also introduces the shared chunking dataclasses (`Section`, `Chunk`, `IngestionDoc`) used by T6/T7/T8 and by the ingestion graph in G4.

### Test Plan

#### Test File(s)
- `tests/rag/test_chunking_structural.py`

#### Test Scenarios

##### `Header-based split`

- **splits H1/H2/H3 into sections** — GIVEN a markdown doc with three heading levels, WHEN `split_by_headers(md_text)` runs, THEN returns an ordered `Section` list with correct `headings_path` _(verifies PLAN Component 2 / Stage 1)_
- **preserves heading hierarchy in headings_path** — GIVEN `# A` then `## B`, THEN the section under B has `headings_path == ["A", "B"]`
- **markdown with no headings returns single section** — whole doc as one section, empty `headings_path` _(verifies PLAN edge case 25)_
- **only H1, no H2/H3** — works correctly
- **out-of-order header levels (H3 directly under H1)** — splits, `headings_path` reflects the actual ancestor chain
- **`#` inside fenced code blocks is NOT treated as a heading** — code-block content stays attached to the surrounding section

##### `Min-section merge`

- **section shorter than `min_section_tokens` (default 50) merges forward** — under-threshold section's content flows into the next section _(verifies PLAN Component 2 / Stage 1)_
- **last under-threshold section merges backward** — when no forward target exists
- **`min_section_tokens` is configurable** — override via parameter
- **all-empty input returns empty list**

##### `Empty / degenerate inputs`

- **empty markdown returns empty list**
- **markdown with only headers and no body** — sections returned have empty body and merge under min-section rule

### Implementation Notes

- **Module:** `rag/chunking/structural.py` (new); shared types in `rag/chunking/types.py` (new)
- **Types module:** `Section(text: str, headings_path: list[str])`, `Chunk(raw_text: str, contextualized_text: str | None, parent_id: str | None, doc_id: str, headings_path: list[str], content_hash: str, chunk_type: Literal["parent", "proposition"], chunk_index: int, created_at: datetime)`, `IngestionDoc(doc_id: str, raw_text: str)`. Use `dataclasses.dataclass(frozen=False)` — these are mutable through the pipeline (T8 sets `contextualized_text`).
- **Pattern reference:** No existing pattern in this codebase. Match the `mcp/models.py` style for module shape and stdlib `logging.getLogger(__name__)`.
- **Code-block awareness:** Use a small state-machine over lines (track "inside fenced block" via ``` toggling) rather than a regex-only approach. Avoids false-positive heading detection.
- **Tokenization:** This stage uses tiktoken (added by T6 to deps) only for the min-section-merge size check. If T6 hasn't been merged yet, T5's branch should add `tiktoken` to `pyproject.toml`; if T6 lands first, this is a no-op.
- **High-risk callouts:** Chunking pipeline (M) — Stage 1 quality affects every downstream stage. Bugs here are silent (sections wrongly merged or split). Test plan covers the canonical heading patterns plus the code-block edge case.

### Scope Boundaries

- Do NOT implement recursive splitting — that's T6.
- Do NOT call any LLM — pure function.
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `split_by_headers(md_text: str, min_section_tokens: int = 50) -> list[Section]`, `Section`/`Chunk`/`IngestionDoc` dataclasses, internal code-block-aware line scanner.

### Files Expected

**New files:**
- `src/blog_mas/rag/chunking/__init__.py` — empty package init
- `src/blog_mas/rag/chunking/types.py` — `Section`, `Chunk`, `IngestionDoc` dataclasses
- `src/blog_mas/rag/chunking/structural.py` — `split_by_headers`
- `tests/rag/test_chunking_structural.py`

**Modified files:** _(none — `tiktoken` is added by T6 unless T5 ships first)_

**Must NOT modify:**
- `src/blog_mas/rag/blueprints.py`, `src/blog_mas/rag/embedding.py`, `src/blog_mas/rag/vector_store.py` — owned by G1
- `src/blog_mas/agents/**`, `src/blog_mas/orchestrator.py`, `src/blog_mas/state.py` — runtime path is later groups

---

## Task T6: Recursive Split (Stage 2, no LLM)

> **Status:** not started
> **Effort:** s
> **Priority:** high
> **Depends on:** T5 (consumes `Section`; produces `Chunk` defined in `types.py`)
> **Satisfies REQs:** R-Hybrid-Chunking (Stage 2); R-Idempotent (content-hash IDs)
> **Footprint slice:** New: `src/blog_mas/rag/chunking/recursive.py`, `tests/rag/test_chunking_recursive.py`. Modified: `pyproject.toml` (add `tiktoken`)
> **High-risk areas touched:** Chunking pipeline (M)

### Description

Implement Stage 2 of the chunking pipeline: for any `Section` whose token count exceeds `target_chunk_tokens` (default 400), recursively split using the separator hierarchy `["\n\n", "\n", ". ", " "]` with a 50-token overlap. Token counts are measured with `tiktoken cl100k_base` (PLAN Component 2 / Stage 2). Output is the set of **parent chunks** that flow into Stages 3 (propositions) and 4 (contextualization). Each chunk receives a content-hash ID `sha256(doc_id, chunk_index, raw_text)` — this is the load-bearing invariant T3's idempotent upsert relies on.

### Test Plan

#### Test File(s)
- `tests/rag/test_chunking_recursive.py`

#### Test Scenarios

##### `No-op when section under budget`

- **section ≤ `target_chunk_tokens` (default 400) emits as a single ParentChunk** — output content matches input section text

##### `Splitting when section exceeds budget`

- **splits on `\n\n` first** — paragraph boundaries are tried before lower-precedence separators
- **falls back to `\n` → `. ` → ` ` in order** — separator hierarchy used in PLAN-specified order _(verifies PLAN Component 2 / Stage 2)_
- **resulting chunks all ≤ `target_chunk_tokens`** — every output under budget
- **chunks fall within 50 ≤ N ≤ 600 token sanity bound** _(verifies PLAN Component 2 validation rules)_

##### `Token counting (tiktoken cl100k_base)`

- **token counts use tiktoken cl100k_base** — assert via a known string whose cl100k token count is fixed _(PLAN Component 2 / Stage 2 reference)_

##### `Overlap`

- **adjacent chunks share `overlap` tokens (default 50)** — last 50 tokens of chunk N appear at the start of chunk N+1
- **overlap=0 produces non-overlapping chunks** — config override works

##### `Content-hash IDs (R-Idempotent)`

- **`content_hash = sha256(doc_id, chunk_index, raw_text)`** — deterministic; same inputs → same ID
- **different `chunk_index` produces different hash** even when raw_text is identical
- **same content → same hash across runs** _(verifies R-Idempotent invariant relied on by T3 idempotent upsert)_

##### `Chunk metadata`

- **`headings_path` inherited from input section**
- **`chunk_type="parent"` set on every output** — proposition extraction is a separate stage
- **`chunk_index` increments globally across the doc** — across all sections

### Implementation Notes

- **Module:** `rag/chunking/recursive.py` (new)
- **Pattern reference:** No existing pattern in this codebase. The canonical algorithm is langchain's `RecursiveCharacterTextSplitter.from_tiktoken_encoder`. Either reuse it (langchain is already a project dep) OR roll-your-own — both acceptable as long as token counts go through `tiktoken cl100k_base` and the separator hierarchy and overlap match PLAN Stage 2.
- **Key decisions:**
  - PLAN Component 2 / Stage 2: separators in order `["\n\n", "\n", ". ", " "]`, target 400, overlap 50.
  - Content-hash ID format `sha256(f"{doc_id}|{chunk_index}|{raw_text}".encode())` — document the canonical join format in a module docstring; T3's idempotent upsert depends on this exact format.
- **Libraries:** `tiktoken` (NEW dep); optionally `langchain-text-splitters` (already a transitive dep via langchain).
- **High-risk callouts:** Chunking pipeline (M) — Stage 2 quality drives chunk size distribution; oversized chunks fail the embedding model's window, undersized chunks lose context. The 50/600 sanity bound test catches both directions.

### Scope Boundaries

- Do NOT call any LLM — pure function.
- Do NOT implement proposition extraction or contextualization — those are T7/T8.
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `recursive_split(sections: list[Section], doc_id: str, target_chunk_tokens: int = 400, overlap: int = 50, start_index: int = 0) -> list[Chunk]`, internal token-counting helper, internal content-hash helper.

### Files Expected

**New files:**
- `src/blog_mas/rag/chunking/recursive.py`
- `tests/rag/test_chunking_recursive.py`

**Modified files:**
- `pyproject.toml` — add `tiktoken` to `[project] dependencies`

**Must NOT modify:**
- `src/blog_mas/rag/chunking/structural.py`, `src/blog_mas/rag/chunking/types.py` — owned by T5
- `src/blog_mas/rag/blueprints.py`, `src/blog_mas/rag/embedding.py`, `src/blog_mas/rag/vector_store.py` — owned by G1

---

## Task T7: Agentic Proposition Extraction (Stage 3, LLM)

> **Status:** not started
> **Effort:** s
> **Priority:** high
> **Depends on:** T6 (consumes parent `Chunk`s; produces child `Chunk`s linked via `parent_id`)
> **Satisfies REQs:** R-Hybrid-Chunking (Stage 3); F17 (JSON parse failure → discard, keep parent)
> **Footprint slice:** New: `src/blog_mas/rag/chunking/propositions.py`, `tests/rag/test_chunking_propositions.py`
> **High-risk areas touched:** Chunking pipeline (M — LLM-driven stages)

### Description

Implement Stage 3 of the chunking pipeline: for every parent `Chunk`, call the LLM to extract atomic, self-contained factual propositions and emit each as a child `Chunk` linked to its parent via `parent_id`. PLAN Decision 9 mandates this runs on every parent (no fact-density gating in v1). On JSON-parse failure for a given parent, discard that parent's propositions and continue — the parent itself is preserved upstream by the ingestion graph (G4); this function does not modify the parent list.

### Test Plan

#### Test File(s)
- `tests/rag/test_chunking_propositions.py`

**Mocking strategy:** the existing `make_mock_llm` / `make_mock_llm_sequence` / `make_failing_llm` helpers from `tests/conftest.py`. The function takes the LLM as a parameter (these are not LangGraph nodes — they're called from the ingestion graph in G4). LLM responses are constructed as `AIMessage`s carrying JSON or non-JSON content.

#### Test Scenarios

##### `Happy path`

- **extracts propositions from a parent chunk** — GIVEN mock LLM returns `{"propositions": ["P1", "P2"]}`, WHEN `extract_propositions([parent], llm)` runs, THEN returns 2 `Chunk` objects with `chunk_type="proposition"`
- **each child links to parent via `parent_id`** — `child.parent_id == parent.content_hash` (parent's ID)
- **each child has its own content_hash based on its own raw_text** — assert hash differs from parent's

##### `Applied to all parents (Decision 9)`

- **runs once per parent in the input list** — assert LLM call count equals `len(parents)`

##### `JSON parse failures (F17)`

- **discards propositions on malformed JSON; preserves parent** — GIVEN LLM returns non-JSON, WHEN extract runs, THEN returns `[]` for that parent and logs a warning. Parent itself is not deleted by this function _(verifies F17)_
- **discards on missing `propositions` key** — same handling
- **discards on non-list `propositions` value** — same handling

##### `Empty propositions`

- **`{"propositions": []}` returns empty list** — no error, no warning

##### `Prompt construction`

- **prompt includes the parent's `raw_text`** — assert via captured LLM input
- **system prompt requires strict JSON output** — assert prompt contains the strict-JSON instruction from PLAN Component 2 / Stage 3 verbatim

### Implementation Notes

- **Module:** `rag/chunking/propositions.py` (new)
- **Pattern reference:** `src/blog_mas/agent_helpers.py` shows the canonical structured-LLM-call pattern. **Do not** call `run_agent_chain` here — it reads the LLM from `RunnableConfig`, but T7 takes the LLM as a direct parameter (these are not LangGraph nodes). Direct invocation: `await llm.ainvoke([SystemMessage(...), HumanMessage(...)])`, then `json.loads` the response content.
- **Key decisions:**
  - Decision 9 — run on ALL parents.
  - Prompt copy is verbatim from PLAN Component 2 / Stage 3, declared as a module constant `PROPOSITION_SYSTEM_PROMPT` in `propositions.py` (NOT in `src/blog_mas/prompts.py` — `prompts.py` is reserved for runtime agent prompts).
  - Manual `json.loads` + shape check, NOT `PydanticOutputParser`. We need explicit, granular error handling for the three failure modes (non-JSON, missing key, non-list value); Pydantic would mask these into a single `ValidationError`.
- **Libraries:** `langchain_core.messages.{SystemMessage, HumanMessage}` (already a dep via langchain).
- **High-risk callouts:** Chunking pipeline (M) — JSON parse failures must not corrupt the parent list. Test plan covers all three failure modes; the function returns `[]` rather than raising so the ingestion graph can continue.

### Scope Boundaries

- Do NOT modify the input `parents` list — `extract_propositions` returns a new list of children only.
- Do NOT add proposition prompts to `src/blog_mas/prompts.py`.
- Do NOT add tenacity retries — the LLM passed in is expected to handle its own retries (or not, depending on caller). G4's ingestion graph wires the retry-aware LLM client.
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `extract_propositions(parents: list[Chunk], llm) -> list[Chunk]`, `PROPOSITION_SYSTEM_PROMPT` constant, internal content-hash helper for children (or import from `recursive.py` if T6 exports one).

### Files Expected

**New files:**
- `src/blog_mas/rag/chunking/propositions.py`
- `tests/rag/test_chunking_propositions.py`

**Modified files:** _(none)_

**Must NOT modify:**
- `src/blog_mas/prompts.py` — runtime agent prompts only
- `src/blog_mas/rag/chunking/types.py`, `src/blog_mas/rag/chunking/structural.py`, `src/blog_mas/rag/chunking/recursive.py` — owned by T5/T6

---

## Task T8: Contextual Retrieval (Stage 4, LLM, Anthropic-style)

> **Status:** not started
> **Effort:** m
> **Priority:** high
> **Depends on:** T6 (consumes parent `Chunk`s); independent of T7 in API surface (operates on any `Chunk`, parent or child)
> **Satisfies REQs:** R-Hybrid-Chunking (Stage 4); F16 (windowing pre-flight for docs > 8k tokens); load-bearing invariant: `raw_text` reaches Writer's LLM, `contextualized_text` is what gets embedded
> **Footprint slice:** New: `src/blog_mas/rag/chunking/contextual.py`, `tests/rag/test_chunking_contextual.py`
> **High-risk areas touched:** Chunking pipeline (M — biggest measured retrieval-quality lever per PLAN)

### Description

Implement Stage 4 of the chunking pipeline (Anthropic-style Contextual Retrieval): for every `Chunk` (parent and child), call the LLM with the chunk plus its surrounding doc-window context to produce 50–100 tokens of "situating context" — a short natural-language description of where the chunk sits in the doc and what implicit references resolve to. Set `chunk.contextualized_text = f"{situating_context}\n\n{chunk.raw_text}"`. Pre-flight windowing: if a doc exceeds 8k tokens, split into 8k-token windows; chunks contextualize against their containing window only (chunk-with-midpoint-in-window rule).

### Test Plan

#### Test File(s)
- `tests/rag/test_chunking_contextual.py`

**Mocking:** existing LLM mocks from `tests/conftest.py`; `tiktoken` (added by T6) used for window-size measurement.

#### Test Scenarios

##### `Happy path`

- **prepends situating context to chunk** — GIVEN mock LLM returns `"This chunk is from section X discussing Y"`, WHEN `contextualize_chunks(chunks, doc_text, llm)` runs, THEN sets `chunk.contextualized_text == f"{situating_context}\n\n{chunk.raw_text}"` _(verifies PLAN Component 2 / Stage 4)_
- **`raw_text` is untouched** — only `contextualized_text` is set; `chunk.raw_text` unchanged _(verifies the load-bearing invariant: `raw_text` reaches Writer's LLM, `contextualized_text` is what's embedded)_

##### `Windowing pre-flight (F16)`

- **doc ≤ 8k tokens uses full doc as context window** — assert prompt sent to LLM includes the entire doc
- **doc > 8k tokens splits into 8k-token windows** — chunks contextualized against their containing window, not the full doc _(verifies F16)_
- **chunk straddling a window boundary uses the window containing its midpoint** — deterministic placement rule (chunk's midpoint character offset selects the window)

##### `LLM failures`

- **on LLM error after retries, falls back to `contextualized_text = raw_text` + warning log** — graceful degradation; no chunk loss

##### `Output integrity`

- **operates on parent and proposition chunks alike** — GIVEN a mixed list, WHEN run, THEN every chunk has `contextualized_text` populated regardless of `chunk_type`

### Implementation Notes

- **Module:** `rag/chunking/contextual.py` (new)
- **Pattern reference:** `propositions.py` (T7) for direct LLM invocation pattern; same direct-call rationale (not a LangGraph node).
- **Key decisions:**
  - PLAN Component 2 / Stage 4: 50–100 tokens situating context, prepended to raw_text via `\n\n`.
  - Load-bearing invariant (PLAN Patterns + Key Constraints): `raw_text` is what reaches the Writer; `contextualized_text` is what gets embedded. Document this in the module docstring and never mutate `raw_text`.
  - Windowing rule: chunk's midpoint character offset selects the window. Deterministic + reproducible.
  - Window size: 8k tokens, configurable via `MAX_DOC_WINDOW_TOKENS = 8000` constant.
  - Prompt copy declared as `CONTEXTUAL_SYSTEM_PROMPT` module constant in `contextual.py` (NOT in `src/blog_mas/prompts.py`).
- **Libraries:** `tiktoken` (added by T6); `langchain_core.messages`.
- **Chunk-to-window mapping:** the input `Chunk` doesn't currently carry a character offset into the source doc. T8 needs this — propose: T8 computes the offset by searching for `chunk.raw_text` in `doc_text` (first occurrence). Alternative: extend `Chunk` to carry a `source_offset: int` set by T6. **Decision flagged for the developer at the bottom of this file.**
- **High-risk callouts:** Chunking pipeline (M) — per PLAN, this is the biggest measured retrieval-quality lever. Bugs here are silent retrieval-quality regressions detected only by the eval harness (G10). The fall-back-to-raw_text on LLM failure is the safety valve.

### Scope Boundaries

- Do NOT mutate `chunk.raw_text` under any circumstances.
- Do NOT add contextualization prompts to `src/blog_mas/prompts.py`.
- Do NOT add tenacity retries here — the LLM passed in is expected to handle its own retries.
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `contextualize_chunks(chunks: list[Chunk], doc_text: str, llm) -> list[Chunk]`, `CONTEXTUAL_SYSTEM_PROMPT` constant, internal windowing helper, internal chunk-to-window mapping helper.

### Files Expected

**New files:**
- `src/blog_mas/rag/chunking/contextual.py`
- `tests/rag/test_chunking_contextual.py`

**Modified files:** _(none — `tiktoken` is added by T6)_

**Must NOT modify:**
- `src/blog_mas/prompts.py` — runtime agent prompts only
- `src/blog_mas/rag/chunking/types.py`, `src/blog_mas/rag/chunking/structural.py`, `src/blog_mas/rag/chunking/recursive.py`, `src/blog_mas/rag/chunking/propositions.py` — owned by T5/T6/T7

### Open question for T8

**Chunk-to-window mapping needs the chunk's character offset in the source doc.** Two options:

- **A.** T8 computes the offset by searching `doc_text` for `chunk.raw_text` (first occurrence). Risk: ambiguous for repeated text fragments. Mitigation: use `chunk_index` + cumulative offset accounting.
- **B.** Extend `Chunk` (T5's types.py) with a `source_offset: int` field; T6's recursive splitter populates it. Cleaner and unambiguous, but expands T5/T6 scope.

**Default for now:** option A in T8 with a note. If recall@k regressions surface in G10, revisit by adding `source_offset` to `Chunk` (small refactor — additive).

---

### TDD Sequence (G2)

T5 must come first (defines shared `Section` and `Chunk` dataclasses).
T6 follows T5 (consumes `Section`, produces parent `Chunk`s).
T7 and T8 may run in parallel after T6 (both consume `Chunk`s; T7 produces children, T8 mutates `contextualized_text` — no conflict).

If implemented serially: **T5 → T6 → T7 → T8**.

---

## Task T9: Hybrid Retrieval — RRF + Reranker + Small-to-Big

> **Status:** done
> **Effort:** m
> **Priority:** high
> **Depends on:** T2 (EmbeddingClient), T3 (QdrantStore), T4 (FakeEmbedder, FakeReranker, FakeVectorStore)
> **Satisfies REQs:** R-Hybrid-Retrieval; R-Graceful; F13; F14; F15
> **Footprint slice:** New: `src/blog_mas/rag/retrieval.py`, `tests/rag/test_retrieval.py`
> **High-risk areas touched:** Hybrid retrieval correctness (H)

### Description

Implement `hybrid_search()`: the single function that all runtime callers (Librarian, Researcher) use to retrieve chunks. It chains dense search → sparse search → Reciprocal Rank Fusion → cross-encoder reranking → small-to-big parent expansion. Every error path degrades gracefully rather than crashing.

### Test Plan

#### Test File(s)
- `tests/rag/test_retrieval.py`

#### Test Scenarios

##### `Reciprocal Rank Fusion`

- **fuses dense + sparse ranked lists** — GIVEN dense returns `[A, B, C]` and sparse returns `[B, C, D]` with scores, WHEN `_rrf_fuse` runs with k=60, THEN the unified ranking is ordered by combined RRF score
- **respects `RAG_RRF_K` env override** — GIVEN env sets k=100, THEN fusion uses that value

##### `Reranker integration`

- **returns top-K after rerank** — GIVEN 20 fused candidates and `top_k=3`, WHEN reranker runs, THEN exactly 3 results returned
- **reranker error degrades to RRF top-K** — GIVEN FakeReranker raises, THEN returns RRF top-K unmodified, logs degraded mode _(verifies F13)_

##### `Small-to-big parent expansion`

- **proposition chunk expands to parent** — GIVEN a winning chunk has `chunk_type="proposition"`, WHEN expansion runs, THEN the parent's `raw_text` is substituted in the result
- **deduplicates parent if also in result set** — GIVEN both a proposition and its parent appear, THEN the parent appears only once

##### `Graceful degradation`

- **dense-only on sparse failure** — GIVEN sparse_search returns [], THEN proceeds with dense results only, logs degraded _(verifies F14)_
- **sparse-only on dense failure** — GIVEN dense_search returns [], THEN proceeds with sparse results only
- **both fail → returns []** — GIVEN both return [], THEN `hybrid_search` returns `[]` _(verifies F15)_
- **empty retrieval returns []** — no points in collection → `[]`

##### `End-to-end`

- **full pipeline returns scored chunks** — GIVEN FakeVectorStore seeded with chunks, FakeEmbedder, FakeReranker, WHEN `hybrid_search(query, namespace="knowledge", top_k=3)` runs, THEN returns up to 3 `ScoredPoint`-like results with payloads
- **respects namespace isolation** — queries `knowledge` collection only; blueprints query `blueprints` collection

### Implementation Notes

- **Module:** `rag/retrieval.py` (new)
- **Pattern reference:** No existing pattern. `QdrantStore` provides `dense_search` and `sparse_search`; `EmbeddingClient` provides `embed_batch`. This module orchestrates them.
- **Key decisions:**
  - Decision 4 (hybrid retrieval day one).
  - Decision 11 (RRF for fusion, k=60 default).
  - Decision 18 (BAAI/bge-reranker-base via HF Inference).
  - Decision 12 (top_k=3 knowledge, top_k=5 blueprints, threshold 0.7).
- **Libraries:** `huggingface_hub.InferenceClient` for reranker; everything else via T2/T3.
- **Config from env:** `RAG_DENSE_TOP_N=20`, `RAG_SPARSE_TOP_N=20`, `RAG_FUSION_TOP_N=20`, `RAG_RERANK_TOP_K=3`, `RAG_BLUEPRINT_TOP_K=5`, `RAG_RRF_K=60`, `RAG_LIBRARIAN_SCORE_THRESHOLD=0.7`.
- **High-risk callouts:** Hybrid retrieval correctness (H) — RRF + reranker + small-to-big together drive end-to-end quality. Bugs are silent. Test plan covers each stage in isolation plus end-to-end.

### Scope Boundaries

- Do NOT implement agent logic (Librarian/Researcher) — those are G6.
- Do NOT implement ingestion — that's G4.
- Do NOT introduce structlog — stdlib `logging` only.
- Only implement: `hybrid_search(query, namespace, top_k, store, embedder, reranker)`, internal `_rrf_fuse`, internal `_expand_parents`, reranker client helper.

### Files Expected

**New files:**
- `src/blog_mas/rag/retrieval.py`
- `tests/rag/test_retrieval.py`

**Modified files:** _(none)_

**Must NOT modify:**
- `src/blog_mas/rag/vector_store.py`, `src/blog_mas/rag/embedding.py` — owned by G1
- `src/blog_mas/agents/**` — G6

---

## Task T10: Knowledge Ingestion Graph

> **Status:** done
> **Effort:** m
> **Priority:** high
> **Depends on:** T5–T8 (chunking pipeline), T2 (EmbeddingClient), T3 (QdrantStore)
> **Satisfies REQs:** R-Two-Graphs; R-Hybrid-Chunking (orchestration); R-Idempotent; R-NF-Cost
> **Footprint slice:** New: `src/blog_mas/rag/ingestion_graph.py`, `tests/rag/test_ingestion_graph.py`
> **High-risk areas touched:** Vector store / Qdrant integration (H — ingestion side)

### Description

Implement the knowledge ingestion LangGraph: loads `.md` files from a directory, fans out per document through the 4-stage chunking pipeline (structural → recursive → propositions → contextualization), embeds all chunks via `EmbeddingClient`, and upserts to Qdrant with content-hash IDs. Re-running on unchanged sources is a no-op.

### Test Plan

#### Test File(s)
- `tests/rag/test_ingestion_graph.py`

#### Test Scenarios

##### `End-to-end with fakes`

- **ingests a markdown doc into FakeVectorStore** — GIVEN a temp dir with one `.md` file, WHEN ingestion graph runs with FakeVectorStore/FakeEmbedder/mock LLMs, THEN points are upserted with correct payloads (`raw_text`, `contextualized_text`, `parent_id`, `doc_id`, `headings_path`, `chunk_type`, `content_hash`)
- **parent and proposition chunks both present** — output includes both `chunk_type="parent"` and `chunk_type="proposition"` points

##### `Idempotency`

- **second run on same data produces no new points** — GIVEN first run created N points, WHEN run again on same files, THEN point count is still N _(verifies R-Idempotent)_

##### `Graceful degradation`

- **proposition extraction failure: parent still indexed** — GIVEN mock LLM returns bad JSON for one chunk's propositions, WHEN graph runs, THEN that parent is still upserted, only its propositions are missing _(verifies F17)_

##### `Multi-file`

- **processes all .md files in directory** — GIVEN 3 files, THEN each produces its own set of points with distinct `doc_id`

### Implementation Notes

- **Module:** `rag/ingestion_graph.py` (new)
- **Pattern reference:** `orchestrator.py` for LangGraph `StateGraph` style. This is a separate graph (not part of the runtime graph).
- **Key decisions:**
  - Decision 12 (LangGraph for ingestion with checkpointer).
  - Decision 15 (resume-from-failure via checkpointer — wire but don't deeply test for v1).
  - Decision 9 (propositions on ALL parents).
- **Graph topology (simplified for v1):**
  ```
  load_docs → chunk_all → embed_all → upsert_all
  ```
  Not a full fan-out/fan-in for v1 — keep it sequential per document to stay simple. The checkpointer still works.
- **Libraries:** `langgraph`, chunking modules from T5–T8.
- **High-risk callouts:** Vector store integration (H) — ingestion is where content-hash IDs are first generated and upserted. Must match T3's expected format exactly.

### Scope Boundaries

- Do NOT implement the `--rebuild` flag or CLI wiring — that's G9.
- Do NOT implement blueprint ingestion — that's T11.
- Do NOT implement LangSmith tracing — that's G11.
- Do NOT add structlog — stdlib `logging` only.
- Only implement: `build_ingestion_graph(store, embedder, llm)` function, state schema for ingestion, node functions for each stage.

### Files Expected

**New files:**
- `src/blog_mas/rag/ingestion_graph.py`
- `tests/rag/test_ingestion_graph.py`

**Modified files:** _(none)_

**Must NOT modify:**
- `src/blog_mas/rag/chunking/**` — owned by G2
- `src/blog_mas/orchestrator.py` — G7

---

## Task T11: Blueprint Ingestion Graph

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T1 (Blueprint schema), T2 (EmbeddingClient), T3 (QdrantStore)
> **Satisfies REQs:** R-Two-Graphs; R-Corpus (blueprints); description-only embedding invariant
> **Footprint slice:** New: `src/blog_mas/rag/blueprint_graph.py`, `tests/rag/test_blueprint_graph.py`

### Description

Implement the blueprint ingestion LangGraph: loads `.json` blueprint files, validates each against the `Blueprint` Pydantic schema, embeds only the `description` field (not the full JSON), and upserts to Qdrant with the full `blueprint_json` in the payload. Simpler than the knowledge graph — no chunking pipeline.

### Test Plan

#### Test File(s)
- `tests/rag/test_blueprint_graph.py`

#### Test Scenarios

##### `End-to-end with fakes`

- **validates and upserts blueprints** — GIVEN a temp dir with valid `.json` blueprint files, WHEN graph runs, THEN each is upserted with `description`-only embedding and full `blueprint_json` in payload _(verifies description-only embedding invariant)_
- **blueprint_json payload is the original JSON string** — for retrieval-time re-validation

##### `Validation handling`

- **skips invalid blueprint with warning** — GIVEN one file fails `validate_blueprint_payload`, WHEN graph runs, THEN that file is skipped and others are still ingested

##### `Idempotency`

- **re-run on same files is no-op** — point count unchanged after second run

### Implementation Notes

- **Module:** `rag/blueprint_graph.py` (new)
- **Pattern reference:** `orchestrator.py` for LangGraph style; simpler graph than T10.
- **Key decisions:**
  - Decision 14 (6 seeded blueprints).
  - Description-only embedding (PLAN invariant).
  - Decision 20 (separate graph from knowledge ingestion).
- **Graph topology:**
  ```
  load_blueprints → validate_all → embed_descriptions → upsert_all
  ```
- **Libraries:** `langgraph`, T1's `validate_blueprint_payload`, T2's `EmbeddingClient`.

### Scope Boundaries

- Do NOT implement `--rebuild` or CLI wiring — G9.
- Do NOT implement knowledge ingestion — T10.
- Only implement: `build_blueprint_graph(store, embedder)` function, state schema, node functions.

### Files Expected

**New files:**
- `src/blog_mas/rag/blueprint_graph.py`
- `tests/rag/test_blueprint_graph.py`

**Modified files:** _(none)_

**Must NOT modify:**
- `src/blog_mas/rag/blueprints.py` — owned by T1
- `src/blog_mas/rag/ingestion_graph.py` — owned by T10

---

## Task T12: Extend BlogState + Add GoalDecomposition Model

> **Status:** done
> **Effort:** xs
> **Priority:** high
> **Depends on:** T1 (Blueprint model referenced by state)
> **Satisfies REQs:** R-Goal-Decomp; R-Librarian (state fields); R-Dual-RAG (state contract)
> **Footprint slice:** Modified: `src/blog_mas/state.py`, `src/blog_mas/mcp/models.py`

### Description

Add new fields to `BlogState` for Chapter 3's dual-RAG flow and add the `GoalDecomposition` Pydantic model to `mcp/models.py`. This is a small, purely additive task — no existing fields or reducers change.

### Test Plan

#### Test File(s)
- Tested implicitly by T13 (intake), T14 (librarian), T17 (orchestrator). No dedicated test file — this task is a data contract, not behavior.

#### Test Scenarios

_(Verified by downstream tasks.)_

- **BlogState accepts new fields with None defaults** — constructing a state dict with only existing fields still works
- **GoalDecomposition model validates non-empty queries** — empty string after `.strip()` is rejected

### Implementation Notes

- **State fields to add to `BlogState`:**
  ```
  intent_query: str | None
  topic_query: str | None
  blueprint: Blueprint | None
  blueprint_match_score: float | None
  blueprint_alternatives: list[str] | None
  blueprint_fallback_reason: str | None
  ```
- **New model in `mcp/models.py`:**
  ```
  class GoalDecomposition(BaseModel):
      intent_query: str  (non-empty after strip, field_validator)
      topic_query: str   (non-empty after strip, field_validator)
  ```
- **Import:** `BlogState` needs to import `Blueprint` from `rag.blueprints`.
- **Reducer:** `revision_count`'s `Annotated[int, operator.add]` must remain unchanged.

### Scope Boundaries

- Do NOT modify any agent code — G5/G6 handles that.
- Do NOT change existing `BlogState` fields or reducers.
- Only implement: 6 new `BlogState` fields, 1 new `GoalDecomposition` model.

### Files Expected

**Modified files:**
- `src/blog_mas/state.py` — add 6 fields
- `src/blog_mas/mcp/models.py` — add `GoalDecomposition`

**Must NOT modify:**
- `src/blog_mas/agents/**` — G5/G6
- `src/blog_mas/orchestrator.py` — G7

---

## Task T13: Goal Decomposition in Intake

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T12 (GoalDecomposition model, extended BlogState)
> **Satisfies REQs:** R-Goal-Decomp; F10 (deterministic fallback)
> **Footprint slice:** Modified: `src/blog_mas/agents/intake.py`, `tests/test_intake_agent.py`

### Description

Extend the `intake_node` to produce `intent_query` and `topic_query` after creating the `BlogSpec`. After the existing `run_agent_chain` call for `BlogSpec`, make a second call for `GoalDecomposition`. On failure, retry once with a stricter prompt; on second failure, derive deterministically from BlogSpec fields.

### Test Plan

#### Test File(s)
- `tests/test_intake_agent.py` (extend existing)

#### Test Scenarios

##### `Goal decomposition happy path`

- **emits intent_query and topic_query on state** — GIVEN a valid raw_input and mock LLM returning `GoalDecomposition(intent_query="technical deep dive", topic_query="Mediterranean diet")`, WHEN intake runs, THEN result dict contains both fields populated

##### `Deterministic fallback (F10)`

- **LLM fails twice → deterministic derivation** — GIVEN a sequence LLM that succeeds for BlogSpec then fails for GoalDecomposition twice, WHEN intake runs, THEN `intent_query = f"{tone} {goal} for {audience}"` and `topic_query = topic`
- **deterministic fallback produces non-empty strings** — verify the derived strings are non-empty

##### `Regression`

- **existing BlogSpec behavior preserved** — GIVEN the existing test case (no goal decomposition assertions), WHEN it runs, THEN `blog_spec` is still returned correctly

### Implementation Notes

- **Module:** `agents/intake.py` (modify)
- **Pattern reference:** `agent_helpers.run_agent_chain` — same pattern as the existing BlogSpec call. Use `model_cls=GoalDecomposition`.
- **Key decisions:**
  - Decision 6 (goal decomposition inside intake, not a separate node).
  - Fallback: `intent_query = f"{spec.tone} {spec.goal} for {spec.audience}"`, `topic_query = spec.topic`.
- **Retry strategy:** on first `GoalDecomposition` failure, retry once with the same call. On second failure, use deterministic fallback. Wrap in try/except — do not let this crash the intake.
- **Return dict:** `{"blog_spec": spec, "intent_query": ..., "topic_query": ...}`.

### Scope Boundaries

- Do NOT modify `prompts.py` — the GoalDecomposition prompt can be a module constant in `intake.py`.
- Do NOT modify `state.py` — that's T12.
- Only implement: second `run_agent_chain` call in `intake_node`, retry + fallback logic, `GOAL_DECOMP_SYSTEM_PROMPT` constant.

### Files Expected

**Modified files:**
- `src/blog_mas/agents/intake.py` — add goal decomposition step
- `tests/test_intake_agent.py` — add goal decomposition test cases

**Must NOT modify:**
- `src/blog_mas/state.py` — T12 owns the schema
- `src/blog_mas/prompts.py` — runtime prompts only
- `src/blog_mas/orchestrator.py` — G7

---

## Task T14: Librarian Agent (NEW)

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T9 (hybrid_search), T12 (BlogState blueprint fields)
> **Satisfies REQs:** R-Librarian; R-Graceful; R-NF-Security; F6; F7; F8; F9
> **Footprint slice:** New: `src/blog_mas/agents/librarian.py`, `tests/test_librarian_agent.py`
> **High-risk areas touched:** Blueprint security boundary (H)

### Description

Implement the Context Librarian agent: a new LangGraph node that reads `intent_query` from state, performs hybrid search over the `blueprints` namespace, validates the top result against the `Blueprint` Pydantic schema, and writes the validated blueprint (or neutral fallback) to state. Every failure path resolves to the neutral default.

### Test Plan

#### Test File(s)
- `tests/test_librarian_agent.py`

#### Test Scenarios

##### `Happy path`

- **valid blueprint above threshold → writes to state** — GIVEN FakeVectorStore seeded with a valid blueprint (score > 0.7), WHEN librarian runs, THEN `blueprint` is set, `blueprint_match_score` > 0.7, `blueprint_fallback_reason` is None

##### `Fallback paths`

- **low score → neutral default** — GIVEN top match score < 0.7, WHEN librarian runs, THEN `blueprint` is NEUTRAL_BLUEPRINT, `blueprint_fallback_reason == "low_score"` _(verifies F6)_
- **missing blueprint_json in payload → neutral** — GIVEN payload lacks `blueprint_json`, THEN neutral default with `fallback_reason="missing_payload"` _(verifies F7)_
- **Pydantic validation failure → neutral** — GIVEN malformed JSON in `blueprint_json`, THEN neutral default with `fallback_reason="schema_violation"` _(verifies F8)_
- **injection marker → neutral + security log** — GIVEN `blueprint_json` contains `{{` in instruction, THEN neutral default, logged as security event _(verifies F9)_
- **empty retrieval → neutral** — GIVEN no points in blueprints namespace, THEN neutral default

### Implementation Notes

- **Module:** `agents/librarian.py` (new)
- **Pattern reference:** `agents/intake.py` for node signature `async def librarian_node(state: BlogState, config: RunnableConfig) -> dict`. Read LLM via `config["configurable"]["llm"]` — but Librarian doesn't call an LLM directly; it calls `hybrid_search` and `validate_blueprint_payload`.
- **Key decisions:**
  - Decision 7 (validation at retrieval time).
  - Decision 10 (parallel with Researcher via asyncio.gather — wiring in G7).
  - Decision 12 (top_k=5 for blueprints, threshold=0.7).
- **Flow:** read `intent_query` → `hybrid_search(query=intent_query, namespace="blueprints", top_k=5)` → check score → parse `blueprint_json` → `validate_blueprint_payload` → write to state or neutral fallback.
- **High-risk callouts:** Blueprint security boundary (H) — MUST call `validate_blueprint_payload` on every retrieved blueprint. Never inject raw payload into state.

### Scope Boundaries

- Do NOT implement hybrid search — that's T9. Call it as a dependency.
- Do NOT implement orchestrator wiring — that's G7.
- Do NOT implement FakeVectorStore seeding in this file — use existing fixtures.
- Only implement: `librarian_node`, reading `hybrid_search` and `validate_blueprint_payload` as imports.

### Files Expected

**New files:**
- `src/blog_mas/agents/librarian.py`
- `tests/test_librarian_agent.py`

**Modified files:** _(none)_

**Must NOT modify:**
- `src/blog_mas/orchestrator.py` — G7
- `src/blog_mas/rag/retrieval.py` — T9
- `src/blog_mas/rag/blueprints.py` — T1

---

## Task T15: Researcher Rewrite (Dict → Hybrid Retrieval)

> **Status:** done
> **Effort:** m
> **Priority:** high
> **Depends on:** T9 (hybrid_search), T12 (BlogState topic_query field)
> **Satisfies REQs:** R-Researcher-Upgrade; R-Graceful; F5; F17
> **Footprint slice:** Modified: `src/blog_mas/agents/researcher.py`, `src/blog_mas/prompts.py`, `tests/test_researcher_agent.py`
> **High-risk areas touched:** Researcher rewrite (H)

### Description

Replace the `lookup_topic()` dict-based knowledge retrieval with `hybrid_search()` over the `knowledge` namespace. Build the user prompt with `[Source <id>]` citation blocks. The `ResearchSummary.source` field becomes a comma-joined list of cited chunk IDs. Anti-hallucination instructions are added to the system prompt.

### Test Plan

#### Test File(s)
- `tests/test_researcher_agent.py` (rewrite existing)

#### Test Scenarios

##### `Hybrid retrieval integration`

- **retrieves chunks and synthesizes with citations** — GIVEN FakeVectorStore seeded with chunks, WHEN researcher runs with `topic_query`, THEN the user message sent to LLM contains `[Source <chunk_id>]` blocks, and `research_summary.source` contains cited chunk IDs
- **reads topic_query from state, falls back to blog_spec.topic** — GIVEN `topic_query` is set, uses it; GIVEN only `blog_spec.topic`, uses that

##### `Graceful handling`

- **empty retrieval → bullet_points=[], source="none"** — GIVEN FakeVectorStore returns no results, WHEN researcher runs, THEN returns empty bullets with source="none" _(verifies F5)_
- **no blog_spec → ValueError** — preserved from existing behavior

##### `Prompt construction`

- **user message includes audience and goal** — verify prompt contains `Audience:` and `Goal:` lines
- **system prompt includes anti-hallucination + citation rules** — verify extended `RESEARCHER_SYSTEM_PROMPT` contains citation instruction

### Implementation Notes

- **Module:** `agents/researcher.py` (rewrite), `prompts.py` (extend RESEARCHER_SYSTEM_PROMPT)
- **Pattern reference:** Keep the existing `research_node` signature. The key change is replacing `lookup_topic()` with `hybrid_search()`.
- **Key decisions:**
  - Decision 4 (hybrid retrieval).
  - Decision 12 (top_k=3 for knowledge).
  - `source` field format: comma-joined chunk IDs (e.g. `"abc123,def456"`).
- **How to get `hybrid_search` into the node:** inject via `config["configurable"]` (e.g. `"retrieval_fn"`), OR import and call directly with a `QdrantStore` / `FakeVectorStore` passed in. For testability, prefer config injection — the orchestrator (G7) wires the real store.
- **High-risk callouts:** Researcher rewrite (H) — `ResearchSummary` shape must not change; Writer + Validator depend on it.

### Scope Boundaries

- Do NOT modify Writer or Validator — G6 handles Writer separately.
- Do NOT modify orchestrator — G7.
- Do NOT import from `knowledge_base.py` — that module is deleted in G10.
- Only implement: replace `lookup_topic` with `hybrid_search`, build citation prompt, extend `RESEARCHER_SYSTEM_PROMPT`.

### Files Expected

**Modified files:**
- `src/blog_mas/agents/researcher.py` — replace dict lookup with hybrid retrieval
- `src/blog_mas/prompts.py` — extend `RESEARCHER_SYSTEM_PROMPT` with citation rules
- `tests/test_researcher_agent.py` — rewrite tests to use FakeVectorStore

**Must NOT modify:**
- `src/blog_mas/agents/writer.py` — T16
- `src/blog_mas/agents/validator.py` — unchanged
- `src/blog_mas/orchestrator.py` — G7
- `src/blog_mas/knowledge_base.py` — deletion is T23

---

## Task T16: Writer Rewrite (Blueprint Injection)

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T12 (BlogState blueprint field)
> **Satisfies REQs:** R-Writer-Upgrade; R-NF-Security (canonical re-serialization)
> **Footprint slice:** Modified: `src/blog_mas/agents/writer.py`, `src/blog_mas/prompts.py`, `tests/test_writer_agent.py`

### Description

Upgrade the Writer to consume `blueprint` from state and inject the validated, canonically re-serialized blueprint JSON into its system prompt via a fixed scaffold. The revision path is unchanged.

### Test Plan

#### Test File(s)
- `tests/test_writer_agent.py` (extend)

#### Test Scenarios

##### `Blueprint injection`

- **injects canonical blueprint JSON into system prompt** — GIVEN a `Blueprint` in state, WHEN writer runs, THEN the system prompt contains `--- SEMANTIC BLUEPRINT (JSON) ---` followed by `Blueprint.model_dump_json()` output
- **uses validated blueprint, not raw payload** — the injected JSON is the canonical re-serialization of the Pydantic model

##### `Revision path`

- **revision still works with blueprint** — GIVEN `revision_feedback` is set and blueprint is in state, WHEN writer runs, THEN uses `WRITER_REVISION_SYSTEM_PROMPT` with feedback + blueprint scaffold is still present

##### `Error handling`

- **no blueprint in state → ValueError** — GIVEN blueprint is None, WHEN writer runs, THEN raises ValueError
- **no research_summary → ValueError** — preserved from existing behavior

### Implementation Notes

- **Module:** `agents/writer.py` (modify), `prompts.py` (add `WRITER_BLUEPRINT_SCAFFOLD`)
- **Pattern reference:** Keep the existing `write_node` signature and structure. Add blueprint reading and prompt assembly before the LLM call.
- **Key decisions:**
  - Canonical re-serialization: `Blueprint.model_dump_json()` of the validated model, never the raw string from Qdrant.
  - Scaffold format:
    ```
    --- SEMANTIC BLUEPRINT (JSON) ---
    {canonical_json}
    --- END SEMANTIC BLUEPRINT ---
    ```
  - Revision path: prepend blueprint scaffold to the revision system prompt as well.
- **High-risk callouts:** The blueprint must come from the already-validated `Blueprint` Pydantic object in state (set by Librarian), never re-read from Qdrant.

### Scope Boundaries

- Do NOT modify Validator — unchanged.
- Do NOT modify the revision loop — that's orchestrator (G7).
- Only implement: read `blueprint` from state, build scaffold prompt, integrate into both initial and revision paths.

### Files Expected

**Modified files:**
- `src/blog_mas/agents/writer.py` — read blueprint, inject scaffold
- `src/blog_mas/prompts.py` — add `WRITER_BLUEPRINT_SCAFFOLD`
- `tests/test_writer_agent.py` — add blueprint injection tests

**Must NOT modify:**
- `src/blog_mas/agents/validator.py` — unchanged
- `src/blog_mas/rag/blueprints.py` — T1

---

## Task T17: Orchestrator Parallel Wiring

> **Status:** done
> **Effort:** m
> **Priority:** high
> **Depends on:** T13 (intake with goal decomp), T14 (librarian), T15 (researcher rewrite), T16 (writer rewrite)
> **Satisfies REQs:** R-Parallel; R-Validator-Preserved; F19
> **Footprint slice:** Modified: `src/blog_mas/orchestrator.py`, `tests/test_orchestrator.py`
> **High-risk areas touched:** Runtime parallel orchestration (M)

### Description

Restructure the orchestrator graph: add the `librarian` node, replace the linear `intake → research → write` path with a parallel branch where Librarian and Researcher run concurrently via `asyncio.gather(..., return_exceptions=True)`, both feeding into Writer. The existing validator + revision loop must survive unchanged.

### Test Plan

#### Test File(s)
- `tests/test_orchestrator.py` (extend)

#### Test Scenarios

##### `Graph structure`

- **librarian node exists in compiled graph** — assert `"librarian"` is a node name
- **parallel branch: intake → [librarian, research] → write** — verify graph edges show both paths from intake to write
- **validator revision loop preserved** — GIVEN a fail verdict with `revision_count < MAX_REVISIONS`, WHEN `should_continue` runs, THEN returns `"retry"` _(regression guard for B1)_

##### `Parallel execution`

- **both librarian and research run before write** — GIVEN mock LLMs, WHEN graph runs, THEN write receives both `blueprint` and `research_summary` from state
- **one branch fails: other proceeds** — GIVEN librarian raises, WHEN graph runs with `return_exceptions=True`, THEN `research_summary` is still populated; Writer proceeds with neutral blueprint _(verifies F19)_

##### `Initial state`

- **initial state includes new fields with None defaults** — `intent_query`, `topic_query`, `blueprint`, `blueprint_match_score`, `blueprint_alternatives`, `blueprint_fallback_reason` all default to None

### Implementation Notes

- **Module:** `orchestrator.py` (modify)
- **Pattern reference:** The current graph is linear. The new topology:
  ```
  intake → [librarian ∥ research] → write → validate → END/retry
  ```
  LangGraph doesn't natively support `asyncio.gather` inside the graph — the parallelism is achieved by having both `librarian` and `research` as nodes that both receive state from `intake` and both feed into `write`. LangGraph's StateGraph will run them concurrently if they're independent fan-out branches.
- **Implementation approach:**
  1. Add `"librarian"` node with `librarian_node`.
  2. Remove `intake → research` edge.
  3. Add `intake → librarian` and `intake → research` edges.
  4. Add `librarian → write` and `research → write` edges.
  5. Keep `write → validate → END/retry` unchanged.
- **Wire dependencies:** The `retrieval_fn`, `store`, and `embedder` need to be available in config for the librarian and researcher nodes. Add them to `run_pipeline_async`'s config dict.
- **High-risk callouts:** Runtime parallel orchestration (M) — one branch failing must not crash the other. LangGraph handles this if both branches feed into a join node.

### Scope Boundaries

- Do NOT modify individual agent implementations — T13–T16 handle those.
- Do NOT modify `should_continue` or `MAX_REVISIONS` — revision loop is preserved.
- Only implement: graph topology changes, config wiring for retrieval dependencies.

### Files Expected

**Modified files:**
- `src/blog_mas/orchestrator.py` — add librarian node, parallel edges, config wiring
- `tests/test_orchestrator.py` — add parallel branch and regression tests

**Must NOT modify:**
- `src/blog_mas/agents/**` — T13–T16
- `src/blog_mas/state.py` — T12
- `src/blog_mas/agents/validator.py` — unchanged

---

## Task T18: Knowledge Data Migration

> **Status:** done
> **Effort:** xs
> **Priority:** medium
> **Depends on:** None
> **Satisfies REQs:** R-Corpus (knowledge); Decision 13

### Description

Migrate the 5 existing knowledge topics from the Python dict in `knowledge_base.py` to individual Markdown files in `data/knowledge/`. Verbatim — no header synthesis, no editing.

### Test Plan

#### Test File(s)
- No dedicated test file. Verified by T10 (ingestion graph) and T22 (recall@k) consuming the files.

#### Test Scenarios

_(Verified by downstream tasks that read these files.)_

- **all 5 files exist in `data/knowledge/`** — `mediterranean-diet.md`, `artificial-intelligence.md`, `climate-change.md`, `space-exploration.md`, `mental-health.md`
- **content matches the dict values verbatim** — manual spot-check against `knowledge_base.py`

### Implementation Notes

- **Source:** `src/blog_mas/knowledge_base.py` — read the `KNOWLEDGE_BASE` dict values.
- **Target:** `data/knowledge/*.md` — one file per topic, content is the dict value verbatim.
- **File naming:** slug-case of the topic key (e.g. `"Mediterranean Diet"` → `mediterranean-diet.md`).

### Scope Boundaries

- Do NOT modify `knowledge_base.py` — deletion is T23.
- Do NOT add synthetic headers — Decision 13 (verbatim migration).
- Only implement: create 5 `.md` files.

### Files Expected

**New files:**
- `data/knowledge/mediterranean-diet.md`
- `data/knowledge/artificial-intelligence.md`
- `data/knowledge/climate-change.md`
- `data/knowledge/space-exploration.md`
- `data/knowledge/mental-health.md`

---

## Task T19: Blueprint Seeding

> **Status:** done
> **Effort:** xs
> **Priority:** medium
> **Depends on:** T1 (Blueprint schema for validation)
> **Satisfies REQs:** R-Corpus (blueprints); Decision 14

### Description

Create 6 blueprint JSON files in `data/blueprints/` with dummy data: `technical-deep-dive`, `executive-summary`, `casual-explainer`, `tutorial-stepwise`, `news-brief`, `opinion-essay`. Each must validate against the `Blueprint` Pydantic schema.

### Test Plan

#### Test File(s)
- `tests/test_blueprint_files.py` (light)

#### Test Scenarios

##### `File validation`

- **all 6 blueprint files exist and are valid JSON** — iterate `data/blueprints/*.json`, parse each
- **each file validates against Blueprint schema** — `validate_blueprint_payload(content)` returns a non-None `Blueprint` for each

### Implementation Notes

- **Target:** `data/blueprints/*.json` — one file per blueprint type.
- **Content:** dummy but realistic. Each file has all required Blueprint fields within bounds.
- **Example structure:**
  ```json
  {
    "id": "technical-deep-dive",
    "description": "An in-depth technical analysis with code examples and detailed explanations.",
    "scene_goal": "Provide comprehensive technical understanding.",
    "style_guide": "Formal, precise, code-heavy, assumes technical audience.",
    "participants": [],
    "instruction": "Write a detailed technical analysis with code examples, diagrams, and step-by-step explanations."
  }
  ```

### Scope Boundaries

- Do NOT ingest these files — that's T11.
- Only implement: create 6 `.json` files.

### Files Expected

**New files:**
- `data/blueprints/technical-deep-dive.json`
- `data/blueprints/executive-summary.json`
- `data/blueprints/casual-explainer.json`
- `data/blueprints/tutorial-stepwise.json`
- `data/blueprints/news-brief.json`
- `data/blueprints/opinion-essay.json`
- `tests/test_blueprint_files.py`

---

## Task T20: CLI Subcommands

> **Status:** done
> **Effort:** s
> **Priority:** medium
> **Depends on:** T10 (knowledge ingestion graph), T11 (blueprint ingestion graph)
> **Satisfies REQs:** R-CLI
> **Footprint slice:** Modified: `src/blog_mas/cli.py`. New: `src/blog_mas/rag/ingest_cli.py`, `tests/test_cli.py` (extend)

### Description

Convert the single-purpose CLI to argparse subcommands: `blog-mas` (runtime, unchanged), `blog-mas ingest [--rebuild] [--path]`, `blog-mas ingest-blueprints [--rebuild]`, `blog-mas eval [--queries]`. The `--rebuild` flag triggers collection drop + recreate with polling.

### Test Plan

#### Test File(s)
- `tests/test_cli.py` (extend)

#### Test Scenarios

##### `Subcommand routing`

- **default (no subcommand) runs runtime** — `main()` with no args starts the interactive loop
- **`ingest` calls knowledge ingestion** — GIVEN `["ingest", "--path", "data/knowledge/"]`, WHEN parsed, THEN the ingestion graph is invoked
- **`ingest --rebuild` drops and recreates collection** — verify `delete_collection_with_polling` is called before ingestion
- **`ingest-blueprints` calls blueprint ingestion** — GIVEN `["ingest-blueprints"]`, WHEN parsed, THEN the blueprint ingestion graph is invoked
- **`eval` calls recall harness** — GIVEN `["eval"]`, WHEN parsed, THEN the eval harness is invoked

### Implementation Notes

- **Module:** `cli.py` (modify), `rag/ingest_cli.py` (new — houses the subcommand handler functions)
- **Pattern reference:** Python `argparse` subparsers.
- **Key decisions:**
  - Decision 6 (`--rebuild` flag with async-deletion polling).
  - Default invocation (no subcommand) must remain the interactive runtime loop.
- **`--rebuild` flow:** `store.delete_collection_with_polling(name)` → `store.ensure_collection(name, dim)` → run ingestion graph.

### Scope Boundaries

- Do NOT implement the eval harness itself — T22.
- Do NOT modify agent or orchestrator code.
- Only implement: argparse setup, subcommand handlers, `--rebuild` wiring.

### Files Expected

**New files:**
- `src/blog_mas/rag/ingest_cli.py`

**Modified files:**
- `src/blog_mas/cli.py` — add argparse subcommands
- `tests/test_cli.py` — add subcommand routing tests

**Must NOT modify:**
- `src/blog_mas/orchestrator.py` — G7
- `src/blog_mas/agents/**` — G6

---

## Task T21: Eval Subcommand Wiring

> **Status:** done
> **Effort:** xs
> **Priority:** medium
> **Depends on:** T20 (CLI framework), T22 (recall@k harness)
> **Satisfies REQs:** R-Eval (CLI surface); R-CLI (`blog-mas eval`)

### Description

Wire the `eval` CLI subcommand to invoke the `recall@k` test harness from T22. Small task — just CLI glue that calls the eval function with the query file path.

### Test Plan

#### Test File(s)
- Covered by T20's CLI tests and T22's harness tests.

#### Test Scenarios

_(Covered by T20 and T22.)_

### Implementation Notes

- **Module:** `rag/ingest_cli.py` (add eval handler)
- **Flow:** parse `--queries` path → load YAML → call `run_recall_eval(queries, store, embedder)` → print results.
- **pytest-skip if Qdrant unreachable** — the eval handler should catch connection errors and print a helpful message.

### Scope Boundaries

- Do NOT implement the recall@k logic — T22.
- Only implement: `eval` subcommand handler that wires to T22's function.

### Files Expected

**Modified files:**
- `src/blog_mas/rag/ingest_cli.py` — add eval handler

---

## Task T22: Recall@k Eval Harness

> **Status:** done
> **Effort:** s
> **Priority:** medium
> **Depends on:** T9 (hybrid_search), T10 (ingestion — needs indexed data)
> **Satisfies REQs:** R-Eval; Decision 16
> **Footprint slice:** New: `tests/eval/__init__.py`, `tests/eval/queries.yaml`, `tests/eval/test_recall.py`

### Description

Implement the `recall@k` eval harness: a YAML file of labeled queries with expected doc IDs, and a pytest test that runs each query through the live retrieval stack and measures recall at k=1, 3, 10. Pytest-skips if Qdrant is unreachable.

### Test Plan

#### Test File(s)
- `tests/eval/test_recall.py`, `tests/eval/queries.yaml`

#### Test Scenarios

##### `Recall measurement`

- **computes recall@1, recall@3, recall@10 per query** — GIVEN a query with expected doc_ids, WHEN retrieval runs, THEN measures whether each expected doc_id appears in top-1, top-3, top-10
- **aggregates across all queries** — reports mean recall@1, @3, @10

##### `Infrastructure guard`

- **pytest.skip if Qdrant unreachable** — GIVEN Qdrant is down, WHEN test module loads, THEN pytest.skip is triggered

### Implementation Notes

- **Module:** `tests/eval/test_recall.py` (new)
- **queries.yaml format:**
  ```yaml
  - query: "health benefits of Mediterranean diet"
    expected_doc_ids: ["mediterranean-diet"]
  - query: "AI machine learning applications"
    expected_doc_ids: ["artificial-intelligence"]
  ```
- **Metric:** `recall@k = |expected ∩ retrieved_top_k| / |expected|` per query, averaged.
- **Libraries:** `pyyaml` (for loading queries).

### Scope Boundaries

- Do NOT implement generation evals (RAGAS) — Out of Scope.
- Only implement: `queries.yaml`, `test_recall.py` with recall computation + skip guard.

### Files Expected

**New files:**
- `tests/eval/__init__.py`
- `tests/eval/queries.yaml`
- `tests/eval/test_recall.py`

---

## Task T23: Test Rewiring + KB Deletion

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T15 (researcher rewrite), T17 (orchestrator wiring)
> **Satisfies REQs:** R-NF-BackCompat; Decision 19
> **Footprint slice:** Deleted: `src/blog_mas/knowledge_base.py`, `tests/test_knowledge_base.py`. Modified: any remaining imports of `knowledge_base`.

### Description

Delete `knowledge_base.py` and its test file. Remove any remaining imports of `knowledge_base` across the codebase (the researcher was rewritten in T15, but `cli.py` may still import `get_available_topics`). Ensure the full existing test suite is green after deletion.

### Test Plan

#### Test File(s)
- No new test file. Validation is: the full test suite passes.

#### Test Scenarios

_(Validation: `pytest` exits 0.)_

- **no remaining imports of `knowledge_base`** — `grep -r "knowledge_base" src/` returns nothing
- **full test suite green** — `pytest` passes
- **test_knowledge_base.py is gone** — file does not exist

### Implementation Notes

- **Files to delete:** `src/blog_mas/knowledge_base.py`, `tests/test_knowledge_base.py`.
- **Files to check for imports:** `cli.py` (currently imports `get_available_topics`), any other file that references `knowledge_base`.
- **cli.py fix:** `print_welcome()` currently lists topics from the KB. After deletion, either remove the topic listing or read from `data/knowledge/*.md` file names. Prefer file-name listing (simple).

### Scope Boundaries

- Do NOT add new test logic — only delete and fix imports.
- Only implement: delete files, fix broken imports, verify suite green.

### Files Expected

**Deleted files:**
- `src/blog_mas/knowledge_base.py`
- `tests/test_knowledge_base.py`

**Modified files:**
- `src/blog_mas/cli.py` — remove `knowledge_base` import, fix `print_welcome`

---

## Task T24: Observability Wiring (LangSmith + structlog)

> **Status:** done
> **Effort:** s
> **Priority:** low
> **Depends on:** None
> **Satisfies REQs:** R-Observability; F20
> **Footprint slice:** New: `src/blog_mas/rag/observability.py`, `tests/rag/test_observability.py`

### Description

Wire structlog for structured local logs across RAG modules and configure LangSmith tracing with no-op fallback when `LANGCHAIN_API_KEY` is unset. Existing runtime agents continue to use stdlib logging — structlog is scoped to `rag/`.

### Test Plan

#### Test File(s)
- `tests/rag/test_observability.py`

#### Test Scenarios

##### `structlog`

- **configures structlog with stage/query/latency fields** — GIVEN structlog is configured, WHEN a log record is emitted, THEN it carries structured key-value pairs

##### `LangSmith`

- **tracer is no-op when LANGCHAIN_API_KEY unset** — GIVEN the env var is absent, WHEN tracer is initialized, THEN it does not raise and does not block _(verifies F20)_
- **tracer is active when key present** — GIVEN the env var is set, WHEN tracer is initialized, THEN LangSmith tracing is enabled

### Implementation Notes

- **Module:** `rag/observability.py` (new)
- **Key decisions:**
  - Decision 18 (LangSmith + structlog).
  - structlog scoped to `rag/` — runtime agents keep stdlib logging.
- **Libraries:** `structlog` (added to pyproject.toml), `langsmith` (transitive dep via langchain).
- **Integration:** RAG modules call `get_logger(stage="chunking")` etc. to get a bound structlog logger.

### Scope Boundaries

- Do NOT migrate runtime agents to structlog — explicitly Out of Scope.
- Do NOT modify `logging_config.py` — stdlib config preserved.
- Only implement: structlog config, LangSmith no-op wiring, `get_logger` helper.

### Files Expected

**New files:**
- `src/blog_mas/rag/observability.py`
- `tests/rag/test_observability.py`

**Modified files:**
- `pyproject.toml` — add `structlog`

**Must NOT modify:**
- `src/blog_mas/logging_config.py` — stdlib config preserved
- `src/blog_mas/agents/**` — no structlog in agents

---

### TDD Sequence (G3–G11)

**Dependency order:**

```
G3 (T9) ─────────────────────────────────────────┐
G4 (T10, T11) ← depends on G1+G2+T9              │
G5 (T12, T13) ← T12 first, then T13              │
G6 (T14, T15, T16) ← depends on T9, T12, T13     │
G7 (T17) ← depends on T13, T14, T15, T16         │
G8 (T18, T19) ← independent, can run early       │
G9 (T20, T21) ← depends on T10, T11, T22         │
G10 (T22, T23) ← T22 depends on T9+T10; T23 last │
G11 (T24) ← independent, can run anytime         │
```

**Recommended serial order:**
T9 → T10 → T11 → T12 → T13 → T14 → T15 → T16 → T17 → T18 → T19 → T20 → T22 → T21 → T23 → T24

**Can parallelize:** T18/T19 (data files) and T24 (observability) are independent of everything else. T14/T15/T16 can run in parallel after T9+T12+T13.
