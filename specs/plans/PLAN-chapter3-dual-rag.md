# Plan: Chapter 3 — Dual-RAG Multi-Agent Blog System

> **Date:** 2026-05-05
> **Project source:** Standalone (extends existing `blog-mas/` codebase from Chapters 1 & 2)
> **Estimated tasks:** 35–45
> **Planning session:** detailed
> **Source chapter:** `chapter_3` (Deep Teaching Guide: Building the Context-Aware Multi-Agent System)

## Summary

Extend the existing `blog-mas` LangGraph multi-agent system from a hardcoded Python-dict knowledge base to a production-grade **Dual RAG** architecture: factual retrieval over a Qdrant vector store of chunked Markdown knowledge, plus procedural retrieval of "Semantic Blueprints" that dictate writing style. A new **Context Librarian** agent runs in parallel with the upgraded **Researcher**, both orchestrated by an extended **Intake** node that performs LLM-based goal decomposition. The ingestion side is a separate LangGraph-based pipeline implementing structural + recursive + agentic-proposition + Anthropic-style contextual chunking, with hybrid (dense + sparse) retrieval and cross-encoder reranking at runtime.

## Requirements

### Functional Requirements

1. **Dual RAG retrieval at runtime**: every blog generation must retrieve (a) factual chunks from a `knowledge` namespace and (b) one matching Semantic Blueprint from a `blueprints` namespace, both before the Writer runs.
2. **Goal decomposition**: the existing `intake` node must, in addition to producing a `BlogSpec`, emit `intent_query` (style/tone descriptor) and `topic_query` (factual subject) as LangGraph state fields.
3. **Context Librarian agent (NEW)**: a new node that takes `intent_query`, performs semantic search over the blueprints namespace, validates the result against a `Blueprint` Pydantic schema, and returns the matched blueprint (or a neutral fallback) on the state.
4. **Researcher upgrade**: replace `lookup_topic()` dict access with a real Qdrant retrieval over the knowledge namespace, then synthesize chunks into bullet points via LLM with explicit anti-hallucination instructions and per-chunk citations.
5. **Writer upgrade**: accept `(facts, blueprint)` and dynamically inject the validated blueprint JSON into its system prompt.
6. **Parallel retrieval**: Librarian and Researcher run concurrently via `asyncio.gather(..., return_exceptions=True)`; Writer waits for both.
7. **Validator preserved**: the existing validator + revision loop (max 3) survives Chapter 3 unchanged.
8. **Two-phase architecture**: ingestion (Phase 1) and runtime (Phase 2) are independent LangGraph applications with separate entrypoints.
9. **Hybrid chunking pipeline (Phase 1)**: structural → recursive → agentic-proposition → contextual stages, in that order.
10. **Hybrid retrieval (Phase 2)**: dense vector search + BM25 sparse search, fused via Reciprocal Rank Fusion, then cross-encoder reranked, with small-to-big parent expansion when a child (proposition) chunk wins.
11. **Idempotent ingestion**: content-hash chunk IDs; re-running ingestion on unchanged sources is a no-op upsert.
12. **Rebuild path**: `--rebuild` flag fully drops and re-creates the Qdrant collection with async-completion polling.
13. **Knowledge corpus**: existing 5 topics from `knowledge_base.py` migrated verbatim to `data/knowledge/*.md` (one file per topic).
14. **Blueprint corpus**: 6 seeded blueprints — `technical-deep-dive`, `executive-summary`, `casual-explainer`, `tutorial-stepwise`, `news-brief`, `opinion-essay` — each with dummy description + structured JSON blueprint.
15. **CLI surface**: `blog-mas ingest [--rebuild] [--path]`, `blog-mas ingest-blueprints [--rebuild]`, `blog-mas` (runtime, unchanged invocation).
16. **Eval harness**: pytest-integrated `recall@k` test suite under `tests/eval/` with a labeled `queries.yaml`.
17. **Observability**: LangSmith tracing + structlog structured local logs across both graphs.
18. **Graceful degradation**: every retrieval failure mode resolves to a neutral default rather than a crash.

### Non-Functional Requirements

1. **Production-grade reliability**: every external call (HF Inference embeddings, HF Inference LLM, Qdrant) wrapped in tenacity retries with exponential backoff + jitter.
2. **Idempotency**: re-running ingestion against unchanged data must not produce duplicate vectors.
3. **Resumability**: ingestion failures resume from the last completed stage via LangGraph checkpointer.
4. **Latency budget (Phase 2)**: parallel Librarian + Researcher retrieval should keep total runtime ≤ 1.5× the single-retrieval baseline.
5. **Security**: blueprints retrieved from the vector store are untrusted input — must pass Pydantic schema validation and an injection-marker scan before being injected into LLM prompts.
6. **Observability**: every stage of both graphs emits structured logs with timing, retrieval scores, and chunk counts; LangSmith tracing covers both ingestion and runtime.
7. **Cost control**: contextualization and proposition LLM calls reuse a single batched HF endpoint client; embeddings batched at ≤100 per call with adaptive halving on size errors.
8. **Configurability**: chunking parameters (target tokens, overlap, strategy toggles), retrieval parameters (top-k, score threshold, RRF k, rerank top-n), and provider selection are all environment-driven with sensible defaults.
9. **Backward compatibility**: existing `blog-mas` test suite must continue to pass after Chapter 3 changes (with the dict-based knowledge base swapped for a fake Qdrant client in tests).

## Behaviors

### Phase 1 — Ingestion

**Why ingestion is a separate graph (not a script):**
- Per-document state machine enables resume-from-failure at scale (1000s of docs).
- Per-stage tracing in LangSmith mirrors runtime tracing — one tool, one mental model.
- Conditional branching: skip proposition extraction for chunks whose `content_hash` is already indexed.
- Parallel fan-out across documents, fan-in for batched upserts.

**Ingestion graph topology** (descriptive, not implementation):

```
load_docs
  └─► fan_out_per_doc
        ├─► structural_split        (markdown headers → sections)
        ├─► recursive_split          (token-budgeted, only if section > target)
        ├─► proposition_extract      (LLM, applied to ALL parent chunks)
        ├─► contextualize            (LLM, prepends situating context per chunk)
        └─► embed_batch              (HF Inference, batched, retried)
  └─► fan_in
  └─► upsert_qdrant                  (idempotent via content-hash IDs)
  └─► verify_index_stats             (assert vector_count delta matches expected)
```

**Blueprint ingestion graph (separate, simpler):**

```
load_blueprints
  └─► validate_schema                (Pydantic Blueprint model)
  └─► embed_descriptions             (HF Inference; only `description` is embedded)
  └─► upsert_qdrant                  (full blueprint JSON in payload)
```

**Why rules matter:**

- **Description-only embedding for blueprints** is non-obvious but critical. The full JSON blueprint contains keys like `style_guide`, `participants`, `instruction` that are not what users semantically search for. Users search by intent ("I want a suspenseful tone"), and that intent lives in the natural-language `description` field. Embedding the JSON would pollute similarity scores with structural keywords.
- **Content-hash chunk IDs** are what make the system idempotent. Without them, every ingestion run would either duplicate vectors or require destructive deletion.
- **Async deletion polling** is required because Qdrant's `delete_collection` returns before storage actually frees. Re-creating immediately after the API returns can race with the deletion and lose data. Same gotcha as Pinecone in the chapter, different API surface.
- **Contextual Retrieval (Anthropic style)** prepends 50–100 tokens of "this chunk is from section X of doc Y, discussing Z" before embedding. Single biggest retrieval-quality win on prose with implicit references ("the diet", "this approach"). Worth the LLM cost.
- **Proposition extraction** decomposes parent chunks into atomic factoids that embed more precisely. At retrieval, when a proposition wins, we fetch its parent chunk for fuller context (small-to-big). This combines retrieval precision with generation recall.
- **Hybrid (dense + sparse) retrieval with RRF fusion + cross-encoder reranking** is the modern production stack. Anthropic's own benchmarks show ~49% retrieval-error reduction over naive dense-only.

### Phase 2 — Runtime

**Updated runtime graph topology:**

```
intake (extended)
  └─► [parallel via asyncio.gather]
        ├─► librarian   (procedural RAG: blueprint by intent_query)
        └─► researcher  (factual RAG: chunks by topic_query → LLM synthesis)
  └─► writer            (consumes blueprint + facts)
  └─► validator         (existing; revision loop preserved)
  └─► END (or retry → writer, max 3)
```

**Why rules matter:**

- **Parallelization of Librarian + Researcher**: they have no data dependency on each other. Sequential execution wastes ~half the retrieval latency. `asyncio.gather(..., return_exceptions=True)` lets one branch fail without nuking the other.
- **Score thresholding on the Librarian**: a low-quality blueprint match is worse than no match. Below threshold (default 0.7), fall back to neutral default.
- **Blueprint Pydantic validation at retrieval time**: the vector store is treated as untrusted input. A malformed or attacker-injected blueprint would otherwise be concatenated into the Writer's system prompt — classic prompt injection vector.
- **Citations in research synthesis**: source IDs are preserved per bullet. When the Validator catches a hallucination, you can trace it to a specific chunk.
- **Goal decomposition lives in `intake`** (not a separate node): keeps the public state contract tight, and the `BlogSpec` already holds the structured fields (`tone`, `goal`, `topic`, `audience`) needed to derive `intent_query` and `topic_query` cheaply with a single follow-up LLM call.

**What's optional vs required:**

- Required for v1: all 4 chunking stages, hybrid retrieval, reranker, proposition extraction on all parent chunks, contextual retrieval on all chunks.
- Required for v1: LangSmith tracing wiring (no-op if `LANGCHAIN_API_KEY` unset).
- Optional / future: incremental ingestion that diffs a manifest to delete orphaned chunks (covered by `--rebuild` for now).
- Optional / future: alternative chunkers as pluggable strategies (interface designed for this; default strategy locked in).

**Common mistakes a developer would make on first attempt:**

- Embedding the full blueprint JSON instead of just the `description` (kills semantic search).
- Forgetting to store original chunk text in payload alongside the vector (you'd retrieve a vector but have nothing to show the LLM).
- Recreating the Qdrant collection without polling for deletion completion (race condition, intermittent data loss).
- Running Librarian and Researcher sequentially in `async` code (correctness fine, latency wasted).
- Trusting the retrieved blueprint JSON and concatenating it directly into the system prompt (prompt injection).
- Setting `top_k=1` with no score threshold (always returns *something*, even garbage).
- Character-based chunking instead of token-based (chunks become unpredictable relative to embedding model limits).
- Skipping the overlap parameter or using zero overlap (sentence-boundary information loss).
- Not deduplicating by content hash on re-ingestion (vector store grows unboundedly).
- Forgetting the `text.replace("\n", " ")` normalization before embedding (embedding quality drops measurably for HF/OpenAI text models).

## Detailed Specifications

### Component 1: `Blueprint` Schema (Security-Hardened)

**Purpose:** Pydantic model that all retrieved blueprints must validate against before being injected into any LLM prompt.

**Interface:**

A `Blueprint` model with the following typed fields, all bounded:
- `id: str` — corresponds to Qdrant point ID
- `description: str` — natural language summary; bounded length
- `scene_goal: str` — bounded
- `style_guide: str` — bounded
- `participants: list[Participant]` — bounded count and per-item length
- `instruction: str` — bounded; injection-marker scan applied
- `metadata: dict[str, str | int | bool] | None` — only primitive values

**Behavior:**
- Loaded from Qdrant payload field `blueprint_json` (a JSON string).
- Parsed → validated → injection-scanned. Failure at any stage = fall back to neutral default + structured error log.

**Validation rules:**
- Total serialized blueprint size ≤ 8 KB (well under typical metadata limits).
- `instruction` field must not contain: `{{`, `}}`, `<script`, `</script`, `<|`, `|>` (template/prompt-injection markers).
- All string fields stripped of leading/trailing whitespace.

**Error scenarios:**

| Condition | Expected Behavior |
|---|---|
| `blueprint_json` missing from Qdrant payload | Fall back to neutral; log `librarian.fallback reason=missing_payload` |
| JSON parse failure | Fall back to neutral; log `librarian.fallback reason=json_parse_error` |
| Pydantic validation failure | Fall back to neutral; log `librarian.fallback reason=schema_violation field=<name>` |
| Injection marker detected | Fall back to neutral; log as `security_event blueprint_id=<id>` |
| Top match score < threshold (default 0.7) | Fall back to neutral; log `librarian.fallback reason=low_score score=<value>` |

### Component 2: Chunking Pipeline

**Purpose:** Convert a Markdown document into a list of embeddable, retrievable chunks with rich payload metadata.

**Stages (executed in order, all required for v1):**

**Stage 1 — Structural split (no LLM):**
- Split on Markdown headers using a header hierarchy splitter (H1, H2, H3).
- Each section retains its `headings_path` (e.g. `["Mediterranean Diet", "Health Effects"]`) in payload.
- Sections shorter than `min_section_tokens` (default 50) are merged forward into the next section.

**Stage 2 — Recursive split (no LLM):**
- For sections exceeding `target_chunk_tokens` (default 400), apply recursive character splitting with separator hierarchy `["\n\n", "\n", ". ", " "]` and token budget 400 / overlap 50.
- Token counts measured with `tiktoken cl100k_base` for consistency with the chapter's reference; embedding model uses its own tokenizer for actual embedding (HF Inference handles this).
- Output of stages 1+2 is the set of **parent chunks**.

**Stage 3 — Agentic proposition extraction (LLM):**
- Applied to **all** parent chunks (per Q7 decision).
- LLM prompt: "Extract atomic, self-contained factual propositions from the following text. Each proposition must be a single declarative sentence understandable in isolation. Return strict JSON: `{\"propositions\": [str, ...]}`."
- Each proposition becomes a **child chunk** linked to the parent via `parent_id` payload field.
- On JSON parse failure: discard propositions for that parent, keep parent indexed solo, log warning.
- LLM uses HF Inference (Q8: same provider as the rest of the stack).

**Stage 4 — Contextual Retrieval (LLM, Anthropic-style):**
- Applied to every chunk (parent and child) before embedding.
- LLM prompt receives the full document (or the surrounding 8k-token window if doc exceeds it) plus the chunk; outputs 50–100 tokens of "situating context" describing where the chunk sits in the document and what implicit references resolve to.
- The contextualized text = `f"{situating_context}\n\n{raw_text}"`.
- The **contextualized text is what gets embedded.** The **raw text is what gets returned to the Writer** (so the LLM downstream sees clean source material, not retrieval-time scaffolding).
- For docs > 8k tokens: pre-flight splits the doc into windows; chunks are contextualized against their containing window, not the full doc.

**Stage 5 — Embed + index:**
- Batched HF Inference calls (≤100 chunks per batch, adaptive halving on 4xx).
- Tenacity retry: `wait_random_exponential(min=1, max=60)`, `stop_after_attempt(6)`, with jitter.
- Newline normalization (`text.replace("\n", " ")`) applied immediately before embedding.

**Payload schema per Qdrant point:**

| Field | Purpose |
|---|---|
| `raw_text` | Original chunk text; what's returned to the Researcher's LLM |
| `contextualized_text` | What was actually embedded; useful for debugging |
| `parent_id` | For child propositions; points to parent's Qdrant ID |
| `doc_id` | Stable per-document identifier (e.g. file path) |
| `headings_path` | List of ancestor headings |
| `content_hash` | sha256 of `(doc_id, chunk_index, raw_text)`; used as Qdrant point ID |
| `chunk_type` | `"parent"` or `"proposition"` |
| `chunk_index` | Order within document |
| `created_at` | ISO timestamp |

**Validation rules:**

- Token count per chunk (raw_text): 50 ≤ N ≤ 600 (sanity bound).
- `content_hash` must be unique per (doc_id, chunk_index, raw_text) — reused as Qdrant point ID for idempotency.
- `parent_id` must reference an existing parent chunk in the same ingestion run.

### Component 3: Hybrid Retrieval

**Purpose:** Given a query string, return the top-K most relevant chunks from a Qdrant namespace using dense + sparse + rerank.

**Behavior:**

1. **Dense search**: embed the query (HF Inference); Qdrant cosine search; return top `dense_top_n` (default 20).
2. **Sparse search**: BM25 over `raw_text` via Qdrant's sparse vector support (using `fastembed` BM25); return top `sparse_top_n` (default 20).
3. **Reciprocal Rank Fusion**: combine the two ranked lists with RRF (default `k=60`) into a unified top `fusion_top_n` (default 20).
4. **Cross-encoder rerank**: pass the fused top-N + query to `BAAI/bge-reranker-base` (via HF Inference); return reranked top-K (default 3 for runtime, 5 for blueprints).
5. **Small-to-big expansion** (knowledge namespace only): if a winning chunk has `chunk_type="proposition"`, fetch its parent by `parent_id` and substitute the parent's `raw_text` for context (deduplicating if the parent is also in the result set).

**Configuration (env-driven):**

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DENSE_TOP_N` | 20 | Dense candidates |
| `RAG_SPARSE_TOP_N` | 20 | Sparse candidates |
| `RAG_FUSION_TOP_N` | 20 | After RRF |
| `RAG_RERANK_TOP_K` | 3 | After rerank (knowledge) |
| `RAG_BLUEPRINT_TOP_K` | 5 | After rerank (blueprints) |
| `RAG_LIBRARIAN_SCORE_THRESHOLD` | 0.7 | Below = fallback neutral |
| `RAG_RRF_K` | 60 | RRF smoothing constant |

**Error scenarios:**

| Condition | Expected Behavior |
|---|---|
| Dense backend errors | Soft-fail; proceed with sparse-only; log degraded mode |
| Sparse backend errors | Soft-fail; proceed with dense-only; log degraded mode |
| Both backends error | Hard fail; Researcher returns "No data found"; pipeline continues |
| Reranker backend errors | Soft-fail; return RRF top-K unmodified; log degraded mode |
| Empty result set | Return `[]`; caller (Researcher / Librarian) handles fallback |

### Component 4: Context Librarian Agent (NEW)

**Purpose:** Procedural RAG. Takes `intent_query`, returns a validated `Blueprint` (or neutral fallback).

**Interface:** A LangGraph node consuming and producing `BlogState`.

**Behavior:**

1. Read `intent_query` from state.
2. Call hybrid retrieval against the `blueprints` namespace, top-K = 5.
3. If top result's score < `RAG_LIBRARIAN_SCORE_THRESHOLD`, fall back to neutral.
4. Parse top result's `blueprint_json` payload → validate with `Blueprint` Pydantic → injection-scan.
5. On any failure → fall back to neutral.
6. Write `blueprint`, `blueprint_match_score`, `blueprint_alternatives` (top 3 IDs), `blueprint_fallback_reason` (None on success) to state.

**Neutral fallback blueprint:**

```
{
  "id": "blueprint_neutral_default",
  "description": "Neutral content generation fallback.",
  "scene_goal": "Inform the reader.",
  "style_guide": "Clear, neutral, factual prose. Standard paragraph structure.",
  "participants": [],
  "instruction": "Generate the content based strictly on the provided research findings, in clear neutral prose."
}
```

### Component 5: Researcher Agent (Upgraded)

**Purpose:** Factual RAG. Replaces dict lookup with real Qdrant retrieval over the `knowledge` namespace, then synthesizes via LLM with citation tracking.

**Behavior:**

1. Read `topic_query` from state (preferred) or fall back to `BlogSpec.topic`.
2. Hybrid retrieve top-3 chunks from `knowledge` namespace; expand propositions to parents.
3. Build user prompt:
   ```
   Topic: {topic_query}
   Audience: {audience}
   Goal: {goal}
   Sources:
   [Source {chunk_id}]: {raw_text}
   ---
   [Source {chunk_id}]: {raw_text}
   ...
   ```
4. System prompt: existing `RESEARCHER_SYSTEM_PROMPT`, extended with: "Include `[Source <id>]` citations after each bullet point. Do not add information not present in the sources."
5. Output: existing `ResearchSummary` Pydantic model, with the `source` field populated as a comma-joined list of cited chunk IDs.

**Error scenarios:**

| Condition | Expected Behavior |
|---|---|
| Empty retrieval | Return `ResearchSummary(topic=..., bullet_points=[], source="none")`; pipeline continues; Writer sees empty facts and the blueprint instructs "no data" handling |
| All chunks below score threshold | Same as above |
| LLM synthesis fails after retries | Propagate exception; LangGraph error handling logs and ends the run gracefully |

### Component 6: Writer Agent (Upgraded)

**Purpose:** Combine factual research + procedural blueprint into final draft.

**Behavior:**

1. Read `research_summary` and `blueprint` from state.
2. Build system prompt by injecting the validated blueprint (already parsed and re-serialized to canonical JSON) into a fixed scaffold:
   ```
   You are an expert content generation AI.
   --- SEMANTIC BLUEPRINT (JSON) ---
   {canonical_blueprint_json}
   --- END SEMANTIC BLUEPRINT ---
   The blueprint defines HOW you write; the research defines WHAT you write about.
   Adhere strictly to the blueprint's instructions, style guides, and goals.
   ```
3. User prompt: research findings (bullet points + citations).
4. Output: existing `BlogDraft` Pydantic model.

**Why canonical re-serialization:** the blueprint string injected into the prompt is `Blueprint.model_dump_json()` of the *validated* model, not the raw payload string from Qdrant. This guarantees the prompt sees only fields and values that passed validation.

### Component 7: Intake Agent (Extended for Goal Decomposition)

**Purpose:** Parse `raw_input` into a `BlogSpec` (existing behavior) AND derive `intent_query` + `topic_query` (new).

**Behavior:**

1. Existing: parse raw input → `BlogSpec`.
2. New: call LLM with structured output (Pydantic `GoalDecomposition` model) to produce:
   - `intent_query: str` — natural-language descriptor of style/tone, derived from `BlogSpec.tone + goal + audience`
   - `topic_query: str` — concise factual query, derived from `BlogSpec.topic`
3. Both stored on state.

**Validation:**
- Both queries must be non-empty after `.strip()`.
- On LLM failure or empty queries: retry once with stricter prompt; on second failure, fall back to deterministic derivation: `intent_query = f"{tone} {goal} for {audience}"`, `topic_query = topic`.

**Why one combined node, not a separate `goal_decomposer`:** the `BlogSpec` already encodes the structured fields needed; a separate node would just shuttle state for one extra LLM call. Co-locating keeps the graph topology minimal and the contract atomic ("intake produces everything downstream nodes need").

### Component 8: CLI Surface

**Purpose:** Operator-facing entrypoints.

| Command | Purpose |
|---|---|
| `blog-mas` | Phase 2 runtime, unchanged invocation |
| `blog-mas ingest [--rebuild] [--path data/knowledge/]` | Phase 1 knowledge ingestion |
| `blog-mas ingest-blueprints [--rebuild]` | Phase 1 blueprint ingestion |
| `blog-mas eval [--queries tests/eval/queries.yaml]` | Run recall@k harness |

**Behavior:**

- `--rebuild` triggers collection drop + async-completion poll + recreate.
- Without `--rebuild`, ingestion is incremental via content-hash IDs (idempotent upsert).
- Each command acquires a Qdrant collection-level lock (a metadata key `ingestion_lock_held_by={hostname}-{pid}`) to prevent concurrent ingestion runs corrupting each other; fail-fast if held.

### Component 9: Eval Harness (`recall@k`)

**Purpose:** Production-grade RAG must ship with retrieval evals.

**Behavior:**

- `tests/eval/queries.yaml` defines a list of `{query, expected_doc_ids: [...]}` records.
- `tests/eval/test_recall.py` runs each query through the live retrieval stack, measures whether each `expected_doc_id` appears in the top-K results.
- Reports `recall@1`, `recall@3`, `recall@10` aggregated across the test set.
- Integrates with LangSmith datasets (when env vars present) so runs are tracked over time.
- Pytest-skip if Qdrant unreachable (so CI without infra doesn't fail).

### Component 10: Module Layout

```
src/blog_mas/
├── rag/
│   ├── __init__.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── structural.py       # Stage 1
│   │   ├── recursive.py        # Stage 2
│   │   ├── propositions.py     # Stage 3
│   │   └── contextual.py       # Stage 4
│   ├── embedding.py            # HF Inference batched embeddings + retries
│   ├── vector_store.py         # Qdrant client wrapper, namespaces, upsert, hybrid search
│   ├── retrieval.py            # RRF fusion + reranking + small-to-big
│   ├── ingestion_graph.py      # LangGraph for knowledge ingestion
│   ├── blueprint_graph.py      # LangGraph for blueprint ingestion
│   ├── blueprints.py           # Blueprint Pydantic schema + injection-scan + neutral default
│   └── ingest_cli.py           # CLI entrypoint glue
├── agents/
│   ├── librarian.py            # NEW
│   ├── intake.py               # extended for goal decomposition
│   ├── researcher.py           # rewritten to use rag.retrieval
│   ├── writer.py               # rewritten to consume blueprint
│   └── validator.py            # unchanged
├── state.py                    # extended with intent_query, topic_query, blueprint, etc.
├── orchestrator.py             # add librarian node + parallel branch
└── ... (existing modules)

data/
├── knowledge/                  # *.md files (5 from existing dict)
└── blueprints/                 # *.json files (6 seeded blueprints with dummy data)

tests/
├── eval/
│   ├── queries.yaml
│   └── test_recall.py
└── ... (existing tests, updated to use FakeVectorStore)
```

## Key Constraints

| Constraint | Why It Matters |
|---|---|
| All retrieved blueprints must pass `Blueprint` Pydantic validation + injection-marker scan before LLM injection | Vector store is untrusted input. Direct concatenation of payload strings into prompts is a prompt-injection vector. Skipping this means an attacker who can write to Qdrant has arbitrary prompt control. |
| Chunk Qdrant point IDs must be content-hash based | Without this, re-ingestion duplicates vectors. Pinecone/Qdrant don't deduplicate by vector value — only by ID. |
| `--rebuild` must poll for deletion completion before recreate | Qdrant collection deletion is async. Recreate-immediately-after-delete races and silently drops data. The chapter teaches this for Pinecone; Qdrant has the same gotcha with different surface. |
| Embedding dimension must match collection dimension | Dim drift on model swap silently corrupts retrieval. Detect mismatch on startup; refuse non-rebuild upserts. |
| Description-only embedding for blueprints | Embedding the full JSON pollutes similarity with structural keywords; users search by intent (English), not by JSON syntax. |
| `raw_text` (not `contextualized_text`) is what reaches the Writer's LLM | Contextualization is retrieval scaffolding, not source material. Showing the LLM "This chunk is from section X..." would distort generation. |
| Librarian and Researcher run in parallel via `asyncio.gather` | Sequential execution wastes ~half the retrieval latency for zero correctness gain. |
| Tenacity retries wrap every external call | HF Inference + Qdrant both have transient failures at ~1% rate. Without retries, a 4-call pipeline has ~4% per-request failure. |
| LangSmith tracing must degrade silently when API key absent | A missing env var must never block a runtime request. |
| Pre-flight token check before contextualization | Docs > context window cannot be contextualized whole; must be windowed. Skipping this triggers silent quality cliffs at large doc sizes. |

## Edge Cases & Failure Modes

| # | Scenario | Decision | Rationale |
|---|---|---|---|
| 1 | Qdrant unreachable on ingest CLI startup | Fail-fast with structured error | Better to crash on a known config issue than half-ingest |
| 2 | Qdrant unreachable on runtime startup | Runtime graph proceeds with neutral blueprint + "No data found" facts; structured error logged | "Millions of users" target requires graceful degradation |
| 3 | Embedding-dim drift (collection has wrong dim for current model) | Refuse upsert; require explicit `--rebuild` | Silent re-embedding with mismatched dim corrupts the index |
| 4 | HF Inference rate limit (429) | Tenacity exponential backoff with jitter, 6 attempts | Standard pattern; jitter avoids thundering herd |
| 5 | HF Inference batch too large (413/400) | Adaptive halving (100 → 50 → 25 → 12 → ...) | Graceful adaptation to changing endpoint limits |
| 6 | Empty retrieval (no chunks above threshold) | Researcher returns `bullet_points=[]`, `source="none"`; Writer + blueprint must handle "no data" case in instruction | Pipeline continues; failure is observable downstream |
| 7 | Librarian top score < 0.7 | Neutral default blueprint; `fallback_reason="low_score"` in state | Bad match is worse than no match for procedural RAG |
| 8 | Blueprint payload missing/malformed JSON | Neutral default; structured error log | Defense-in-depth against vector-store corruption |
| 9 | Blueprint fails Pydantic validation | Neutral default; log field that violated schema | Schema is the security boundary |
| 10 | Blueprint contains injection markers | Neutral default; log as `security_event` | Prompt injection prevention |
| 11 | Goal decomposition LLM returns invalid JSON | Retry once with stricter prompt; on second failure, deterministic fallback derivation from BlogSpec fields | Always produce valid intent/topic queries |
| 12 | Concurrent ingestion runs against same collection | Distributed lock via collection metadata key; second run fails-fast | Prevents partial-overwrite corruption |
| 13 | Partial ingestion failure mid-run | LangGraph checkpointer resumes from last completed stage on next invocation | Required for 1000s-of-docs scale |
| 14 | Doc removed from `data/knowledge/` between ingest runs | Documented limitation: incremental mode leaves orphaned chunks; `--rebuild` is the supported path for deletions | Manifest-diff incremental delete is Phase 2 work |
| 15 | Reranker returns fewer candidates than requested | Fall back to RRF-fused top-K; log degraded mode | Quality degrades gracefully |
| 16 | Dense backend errors but sparse succeeds | Proceed with sparse-only; log degraded retrieval | Hybrid handles asymmetric failure |
| 17 | Both dense + sparse error | Researcher returns empty; Writer proceeds with blueprint-only output | Avoids hard pipeline failure |
| 18 | Contextualization LLM exceeds context window | Pre-flight: split doc into 8k-token windows; chunk contextualized against its window | Prevents silent quality drop on large docs |
| 19 | Proposition extraction returns malformed JSON | Discard propositions for that parent; index parent solo; log warning | One bad parse shouldn't lose the parent chunk |
| 20 | Blueprint corpus retrieval returns 0 results (empty namespace) | Neutral default; log warning advising `ingest-blueprints` | First-run UX |
| 21 | Parallel Librarian + Researcher: one fails | `asyncio.gather(return_exceptions=True)`; Writer proceeds with available context; log which branch failed | Partial success > total failure |
| 22 | LangSmith API unreachable | Tracing degrades silently to local logs only | Tracing must never block requests |
| 23 | Existing test suite uses dict-based knowledge base | Introduce `FakeVectorStore` test double in `tests/conftest.py`; existing tests rewired to use it; behavior preserved | Backward compatibility |
| 24 | Two knowledge docs have identical content | content_hash collision is correct: same content → same ID → one Qdrant point | Natural deduplication |
| 25 | Knowledge doc lacks any Markdown headers | Structural split returns the whole doc as one section; recursive split handles the rest | Robust to varied input formats |
| 26 | Blueprint JSON exceeds 8 KB serialized | Pydantic validation fails on size; ingestion rejects the blueprint with explicit error | Avoids pathological prompt sizes |
| 27 | Researcher gets `topic_query=""` (empty) | Validation in intake catches this; intake retries; pipeline never reaches Researcher with empty query | Defense-in-depth at boundary |
| 28 | Validator's revision loop triggers; Writer needs to regenerate | Existing flow preserved; blueprint + facts already in state; Writer re-runs with same context + revision feedback | Chapter 3 changes don't affect revision semantics |

## Decisions Log

| # | Decision | Alternatives Considered | Chosen Because |
|---|---|---|---|
| 1 | Qdrant as vector DB | Pinecone (chapter), pgvector, Chroma | Self-hosted via Docker (free dev), production-grade, faithful semantic match to chapter's namespace pattern via Qdrant collections |
| 2 | HF Inference embeddings | OpenAI, local sentence-transformers, pluggable provider | Keeps single-provider stack with existing `HF_TOKEN`; no new credentials |
| 3 | Hybrid chunking: structural → recursive → propositions (all parents) → contextual | Faithful chapter (sliding window only), sentence-aware only, recursive only, semantic clustering | User explicitly requested hybrid; structural-first leverages free signal in markdown; contextual retrieval (Anthropic) gives biggest measured retrieval-quality win on prose |
| 4 | Hybrid retrieval (dense + BM25) + cross-encoder reranker, day one | Dense-only v1 + hybrid as Phase 2 | User chose full pipeline now; matches "millions of users" production target |
| 5 | Same HF Inference model for chunking-stage LLMs (propositions, contextualization) | Anthropic API for contextualization (cost/quality), defer contextualization to Phase 2 | Single-provider consistency; cost mitigated by batched calls |
| 6 | Goal decomposition lives inside extended `intake` node | Separate `goal_decomposer` node | Co-located with BlogSpec parsing; minimal graph topology; one LLM round-trip |
| 7 | Idempotent content-hash IDs + `--rebuild` flag | Always nuke-and-repave (chapter), always incremental | Production needs both: cheap reruns by default, escape hatch for breaking changes |
| 8 | `recall@k` hand-rolled eval harness | RAGAS, deferred eval | Lightweight; pytest-integrated; LangSmith dataset compatible; doesn't require labeled answers, just expected doc IDs |
| 9 | LangSmith + structlog observability | Plain logging, OpenTelemetry, Langfuse | LangSmith integrates natively with LangGraph already in use |
| 10 | Blueprint Pydantic validation + injection-marker scan at retrieval time | Trust retrieved blueprints, validate at ingest only | Vector store is untrusted boundary; ingest-time validation insufficient if store is later modified |
| 11 | `src/blog_mas/rag/` module layout | Separate `blog_mas_rag` package, embed in `knowledge_base.py` | Cohesive with existing package; clear domain boundary; easy testing |
| 12 | Separate LangGraph for ingestion (with checkpointer) | Plain async script, Prefect/Dagster, Airflow | Per-doc state machine, resume-from-failure, parity with runtime tracing in LangSmith, pedagogically consistent |
| 13 | Migrate existing 5 dict topics to `data/knowledge/*.md` | Generate fresh corpus, BYO corpus | Preserves backward compat; gives the eval harness a known-good baseline |
| 14 | 6 blueprints with dummy data: technical-deep-dive, executive-summary, casual-explainer, tutorial-stepwise, news-brief, opinion-essay | Smaller set, larger set, defer | Covers blog-domain breadth without bloating the v1 corpus |
| 15 | Validator + revision loop preserved (chapter drops it) | Drop validator (chapter-faithful) | User's existing system has it; chapter explicitly notes "you'd add it back for production" |
| 16 | Apply proposition extraction to ALL parent chunks | Selective by fact-density heuristic, pluggable strategy | User chose all-chunks; simpler v1; selectivity can be added later behind same interface |
| 17 | Parallel Librarian + Researcher via `asyncio.gather(return_exceptions=True)` | Sequential, parallel-but-fail-fast | Latency win + partial-failure tolerance |
| 18 | Cross-encoder reranker: `BAAI/bge-reranker-base` via HF Inference | Cohere Rerank, no reranker | Stays on HF stack; production-quality reranker |
| 19 | RRF for sparse+dense fusion | Linear weighted combination, learned fusion | RRF is parameter-light, robust, well-understood |
| 20 | Top-K = 3 (knowledge), top-K = 5 (blueprints), threshold 0.7 (librarian) | k=1 (chapter), variable | Matches chapter's improved-Librarian pattern; small enough to fit Writer's context budget |

## Scope Boundaries

### In Scope (v1)

- Two LangGraph applications: ingestion + runtime
- Full chunking pipeline (structural → recursive → propositions all-parents → contextual)
- Hybrid retrieval (dense + BM25 sparse) + cross-encoder reranker + small-to-big parent expansion
- Context Librarian agent with neutral fallback + Pydantic-validated blueprints + injection-marker scan
- Researcher upgrade with citation tracking
- Writer upgrade consuming validated blueprint
- Intake extension for goal decomposition (`intent_query`, `topic_query`)
- Parallel Librarian + Researcher via `asyncio.gather`
- Idempotent ingestion via content-hash IDs + `--rebuild` flag with async-deletion polling
- 5 knowledge docs migrated from existing dict to `data/knowledge/*.md`
- 6 seeded blueprints with dummy data in `data/blueprints/*.json`
- LangSmith tracing + structlog structured logs
- `recall@k` eval harness in `tests/eval/`
- Updated test suite using `FakeVectorStore` test double
- CLI commands: `blog-mas`, `blog-mas ingest`, `blog-mas ingest-blueprints`, `blog-mas eval`

### Out of Scope (deferred / future)

- **Manifest-diff incremental delete** of orphaned chunks (current path: `--rebuild`). Reason: complexity vs benefit; rebuild is fast at this corpus size.
- **Pluggable chunking-strategy registry** beyond the locked-in default. Reason: interface is designed for this, but v1 ships one strategy.
- **RAGAS-based generation evals** (faithfulness, answer-relevance). Reason: needs labeled answer set; recall@k is the v1 quality bar.
- **Reranker model swap to Cohere or proprietary**. Reason: HF reranker is sufficient for v1.
- **Multi-tenant namespace isolation** beyond knowledge/blueprints. Reason: not needed for current product surface.
- **Streaming generation in Writer**. Reason: orthogonal to RAG architecture; can be added without disturbing this plan.
- **Web UI for blueprint authoring**. Reason: developer-controlled corpus is fine for v1; non-engineer authoring is a Phase 2 product question.
- **A/B testing of blueprints**. Reason: requires telemetry pipeline beyond LangSmith.
- **Caching layer for embeddings/retrievals at runtime**. Reason: premature optimization; add when latency budget is breached.

## Dependencies

### Depends On (must exist before this work starts)

- Existing `blog-mas` Chapter 1+2 implementation (LangGraph, MCP models, intake/researcher/writer/validator agents).
- Local Docker (or remote-hosted) Qdrant instance reachable via env-configured URL.
- HF Inference API token (`HF_TOKEN`) with access to:
  - Embedding model (e.g. `BAAI/bge-small-en-v1.5`)
  - Reranker model (`BAAI/bge-reranker-base`)
  - LLM model already used by blog-mas (for propositions + contextualization + agents)
- Optional: LangSmith API key (`LANGCHAIN_API_KEY`) for tracing; degrades gracefully if absent.

### Depended On By (other work waiting for this)

- Future Chapter 4+ work that will likely add tools, planner-executor patterns, or memory — all building on the dual-RAG foundation.
- Future production deployment work (containerization, secrets management, health checks).

## Architecture Notes

**Two-graph architecture is the central organizing principle.** Phase 1 (ingestion) is rare and expensive; Phase 2 (runtime) is frequent and cheap. They share the `vector_store` and `embedding` modules but have independent state schemas, independent checkpointers, and independent CLI entrypoints.

**Dual RAG splits "what" from "how" at the data layer.** Two Qdrant collections (`knowledge`, `blueprints`) play the role of Pinecone namespaces in the chapter. The Researcher only ever queries `knowledge`; the Librarian only ever queries `blueprints`. Cross-namespace queries are architecturally impossible.

**Goal decomposition is the first sign of LLM-as-planner in this codebase.** The intake node now uses one LLM call to design what subsequent agents do. This is foundational for any future ReAct or plan-and-execute extension.

**Contextual Retrieval is the highest-leverage chunking decision.** It costs one extra LLM call per chunk at ingestion time but lifts retrieval quality measurably (Anthropic's published benchmarks). The plan locks it in for v1 because retrieval quality compounds through every downstream stage.

**Blueprint Pydantic validation is a security boundary, not a developer-experience nicety.** The vector store is treated as untrusted input. A blueprint that doesn't validate is dropped, period. This is non-negotiable.

**LangGraph for ingestion isn't gold-plating — it's how we get checkpointed resumes and unified tracing.** A plain async script would work for 5 docs but not for 1000s.

**The existing test suite stays green.** A `FakeVectorStore` test double is introduced in `tests/conftest.py`; tests that previously asserted against the dict-based knowledge base are rewired to assert against the fake. No production code paths fork on test mode.

## Open Questions

- **HF Inference reranker model availability/latency**: `BAAI/bge-reranker-base` should be available on HF Inference, but if rate limits or availability are issues at integration time, we may need to swap to local inference via `sentence-transformers`.
  - **Impact if unresolved:** Reranker stage degrades to "RRF-only" (already handled by error path) until alternative is wired.
  - **Suggested default:** Start with HF Inference; if issues surface during implementation, add local fallback as a follow-up task.

- **Knowledge corpus markdown formatting**: existing dict topics are paragraph prose without explicit headers. Migration to `*.md` may benefit from synthesizing H2 headers per logical paragraph to give Stage 1 (structural split) more signal.
  - **Impact if unresolved:** Stage 1 emits whole-doc-as-one-section; Stage 2 (recursive) handles all the splitting. Quality fine, just less header-aware metadata.
  - **Suggested default:** Migrate verbatim first (no synthetic headers); revisit if recall@k harness shows weak topical separation.

---

_This plan is the input for the generate-tasks skill._
_Review this document, then run: "Generate task from plan: specs/plans/PLAN-chapter3-dual-rag.md"_
