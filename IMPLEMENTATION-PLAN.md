# Implementation Plan — Code Review Issues (Critical/High/Medium)

> Generated from `CODE-REVIEW-FULL-2026-05-03.md`
> Covers 1 Critical + 14 High + 27 Medium issues
> Scope: All critical/high, most medium. Low-severity items deferred.

---

## Decisions Summary

| # | Decision | Choice |
|---|----------|--------|
| D1 | Dead code strategy | Integrate all (retry + MCP) into pipeline |
| D2 | MCP integration pattern | Guard-based validation (Pydantic at agent entry) |
| D3 | Retry scope | Wrap chain.ainvoke only |
| D4 | Error handling | Layered specific exception catches |
| D5 | Logging | print() for CLI, logging for internals |
| D6 | Sync wrapper | Replace with asyncio.run() |
| D7 | Agent boilerplate | Extract shared helper function |
| D8 | Test scope | T1/T2/T3 (high) + refactor updates only |
| D9 | Prompt safety (S3) | Strip directive patterns via sanitize_feedback() |
| D10 | Dependencies | C1 only (move test deps to dev group) |
| D11 | Dead error field | Remove from BlogState |
| D12 | Pydantic constraints | Add max_length / max_items to models |
| D13 | Empty input | Guard at orchestrator + fix lookup_topic |
| D14 | Topic list DRY | Export get_available_topics() from KB |
| D15 | Config/docs | Fix all four (C3, C4, C5, C6-C7) |
| D16 | MCP protocol | S5 sender validation + S6 generic errors + Q8 Pydantic model |
| D17 | Input safety (S4) | Rely on Pydantic constraints only |
| D18 | Retry minor fixes | Skip (E5, E6) |

---

## Phase 0: Security — Rotate Token (S1)

**Before any code changes.**

| Step | Action | File |
|------|--------|------|
| 0.1 | Rotate the HuggingFace token at huggingface.co/settings/tokens | External |
| 0.2 | Replace `.env` value with `HF_TOKEN="your_huggingface_api_token_here"` | `.env` |
| 0.3 | Check git remotes: `git log --all --full-history -- .env` | CLI |
| 0.4 | Set file permissions: `chmod 600 .env` | CLI |

**Verification**: Confirm `.env` contains only a placeholder. Confirm no token in git history.

---

## Phase 1: Configuration & Housekeeping

No code dependencies. All changes are isolated.

### 1A: pyproject.toml cleanup (C1, C3, C4)

| Step | Action | Detail |
|------|--------|--------|
| 1A.1 | Move pytest deps to dev group | `pytest>=9.0.3`, `pytest-asyncio>=1.3.0` → `[dependency-groups] dev = [...]` |
| 1A.2 | Replace placeholder description | `"Multi-agent blog generation system using LangGraph"` |
| 1A.3 | Fix entry point | `blog-mas = "blog_mas.cli:main"` (bypass `__init__.py` re-export) |

### 1B: Documentation & env (C5, C6-C7)

| Step | Action | Detail |
|------|--------|--------|
| 1B.1 | Create `.env.example` | `HF_TOKEN="your_huggingface_api_token_here"` |
| 1B.2 | Remove `run_ins.txt` from `.gitignore` | Line: `run_ins.txt` — delete it |
| 1B.3 | Delete `run_ins.txt` | Content already in README.md |

### 1C: Remove dead state field (Q6)

| Step | Action | Detail |
|------|--------|--------|
| 1C.1 | Remove `error: str \| None` from `BlogState` | `state.py` |
| 1C.2 | Update any test that references `state["error"]` | Search tests for `"error"` key usage |

**Verification**: `uv sync --group dev` succeeds. `uv run pytest tests/ -v` passes (may have failures from removed field — fix in 1C.2).

---

## Phase 2: Core Infrastructure

Foundation for agent refactoring in Phase 3.

### 2A: Logging setup (E4, E8)

| Step | Action | File |
|------|--------|------|
| 2A.1 | Create `src/blog_mas/logging_config.py` | Configure `logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)` |
| 2A.2 | Add `import logging; logger = logging.getLogger(__name__)` to all non-CLI source files | `orchestrator.py`, `llm.py`, `knowledge_base.py`, `retry.py`, `agents/*.py`, `mcp/*.py` |
| 2A.3 | Replace all `print()` with `logger.info()` / `logger.debug()` / `logger.warning()` in non-CLI files | All files except `cli.py` |
| 2A.4 | Replace `print()` in `mcp/protocol.py` validation with `logger.warning()` | `mcp/protocol.py:19, 24, 32` |
| 2A.5 | Call `logging_config` setup from `cli.py:main()` before `asyncio.run()` | `cli.py` |

**Files to update**: `orchestrator.py`, `llm.py`, `knowledge_base.py`, `retry.py`, `agents/intake.py`, `agents/researcher.py`, `agents/writer.py`, `agents/validator.py`, `mcp/protocol.py`

**Do NOT change**: `cli.py` print statements (user-facing output stays as print).

### 2B: Knowledge base cleanup (Q7, Q9)

| Step | Action | File |
|------|--------|------|
| 2B.1 | Add `get_available_topics() -> list[str]` returning `_KNOWLEDGE_BASE.keys()` | `knowledge_base.py` |
| 2B.2 | Fix `lookup_topic("")` — add length guard: reject empty/whitespace-only strings, return `None` | `knowledge_base.py:146-156` |
| 2B.3 | Update `cli.py:print_welcome()` to import and use `get_available_topics()` | `cli.py` |

### 2C: MCP protocol improvements (S5, S6, Q8)

| Step | Action | File |
|------|--------|------|
| 2C.1 | Define `MCPEnvelope(BaseModel)` in `mcp/protocol.py` with fields: `protocol_version`, `sender`, `content`, `metadata` | `mcp/protocol.py` |
| 2C.2 | Update `create_mcp_message()` to return `MCPEnvelope` instead of `dict` | `mcp/protocol.py` |
| 2C.3 | Add sender format validation to `validate_mcp_envelope()` — require `sender` to match `r"^[a-z][a-z0-9_-]*$"` pattern | `mcp/protocol.py` |
| 2C.4 | Replace specific error messages with generic `"Invalid envelope format"`, log details at debug level | `mcp/protocol.py` |
| 2C.5 | Update `test_mcp_protocol.py` for new return type and validation behavior | `tests/test_mcp_protocol.py` |

### 2D: Pydantic model constraints (S7)

| Step | Action | File |
|------|--------|--------|
| 2D.1 | Add `Field(max_length=200)` to `BlogSpec.topic` | `mcp/models.py` |
| 2D.2 | Add `Field(max_length=100)` to string fields: `BlogSpec.audience`, `BlogSpec.tone`, `BlogSpec.goal` | `mcp/models.py` |
| 2D.3 | Add `Field(max_length=500)` to `BlogSpec.constraints` items | `mcp/models.py` |
| 2D.4 | Add `Field(max_length=100)` to `ResearchRequest` string fields | `mcp/models.py` |
| 2D.5 | Add `Field(max_length=1000)` to `ResearchSummary.bullet_points` items, `max_items=10` to list | `mcp/models.py` |
| 2D.6 | Add `Field(max_length=50)` to `ValidationVerdict.verdict` | `mcp/models.py` |
| 2D.7 | Add `Field(max_length=10000)` to `BlogDraft.body` | `mcp/models.py` |

### 2E: Orchestrator input validation (E7)

| Step | Action | File |
|------|--------|--------|
| 2E.1 | At the top of `run_pipeline_async()`, validate `raw_input`: if `None` or `strip()` is empty, return error immediately | `orchestrator.py` |
| 2E.2 | Same for `blog_spec` when provided directly | `orchestrator.py` |

**Verification**: `uv run pytest tests/ -v` — existing tests may need updates for logging and new validation behavior.

---

## Phase 3: Agent Refactoring

The core structural change. Depends on Phase 2.

### 3A: Extract shared agent helper (Q5)

| Step | Action | File |
|------|--------|--------|
| 3A.1 | Create `src/blog_mas/agent_helpers.py` with `async def run_agent_chain(config, model_cls, system_prompt, user_message, agent_name)` | New file |
| 3A.2 | Helper extracts LLM from config with guard: `llm = config.get("configurable", {}).get("llm"); if llm is None: raise ValueError(f"[{agent_name}] No LLM configured")` | `agent_helpers.py` |
| 3A.3 | Helper creates parser, builds chain, invokes with retry: `await retry_handler(lambda: chain.ainvoke(messages), agent_name)` | `agent_helpers.py` |
| 3A.4 | Helper parses output: `result = parser.parse(llm_response.content)` | `agent_helpers.py` |
| 3A.5 | Helper returns parsed result | `agent_helpers.py` |

**Helper signature**:
```python
async def run_agent_chain(
    config: dict,
    model_cls: type[BaseModel],
    system_prompt: str,
    user_message: str,
    agent_name: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> BaseModel:
```

### 3B: Add guard-based validation to agents (Q2/Q3 integration, E2, E3)

| Step | Action | File |
|------|--------|--------|
| 3B.1 | **intake_node**: Add null guard on `state.get("raw_input")`. Call `run_agent_chain()` instead of manual chain building. | `agents/intake.py` |
| 3B.2 | **research_node**: Add guard `state.get("blog_spec"); if None: raise ValueError(...)`. Validate with `ResearchRequest` model. Call `run_agent_chain()`. | `agents/researcher.py` |
| 3B.3 | **write_node**: Add guard on `state.get("blog_spec")` and `state.get("research_summary")`. Validate with `WriterInput` model. Call `run_agent_chain()`. | `agents/writer.py` |
| 3B.4 | **validate_node**: Add guard on `state.get("draft")` and `state.get("research_summary")`. Validate with `ValidationInput` model. Call `run_agent_chain()`. | `agents/validator.py` |

**Guard pattern for each agent**:
```python
blog_spec = state.get("blog_spec")
if blog_spec is None:
    raise ValueError(f"[{agent_name}] Upstream agent failed — no blog spec in state")
```

**Pydantic guard pattern**:
```python
ResearchRequest(topic=blog_spec.topic, audience=blog_spec.audience, goal=blog_spec.goal)
# If this raises ValidationError, it propagates to orchestrator's catch
```

### 3C: Wire retry_handler (Q1)

Already handled by 3A.3 — the shared helper wraps `chain.ainvoke` with `retry_handler`.

### 3D: Add sanitize_feedback (S3)

| Step | Action | File |
|------|--------|--------|
| 3D.1 | Add `sanitize_feedback(feedback: str) -> str` to `agent_helpers.py` | `agent_helpers.py` |
| 3D.2 | Strip lines starting with directive patterns: `"Ignore"`, `"System:"`, `"You are"`, `"NEW INSTRUCTION"`, `"Forget"`, `"Disregard"` (case-insensitive) | `agent_helpers.py` |
| 3D.3 | Call `sanitize_feedback()` in `write_node` before passing feedback to `WRITER_REVISION_SYSTEM_PROMPT.format(feedback=...)` | `agents/writer.py` |

**Verification**: `uv run pytest tests/ -v` — many agent tests will need updates for the new helper. Fix them.

---

## Phase 4: Error Handling

Depends on Phase 3 (agents now raise specific exceptions).

### 4A: Layered exception handling in orchestrator (E1, S2)

| Step | Action | File |
|------|--------|--------|
| 4A.1 | Replace broad `except Exception` in `run_pipeline_async()` with layered catches | `orchestrator.py` |
| 4A.2 | Catch `pydantic.ValidationError` → log + return `"Failed to parse agent output"` | `orchestrator.py` |
| 4A.3 | Catch `RuntimeError` (from retry_handler exhausted) → log + return `"LLM call failed after retries"` | `orchestrator.py` |
| 4A.4 | Catch `ValueError` (from our guards) → log + return `"Pipeline configuration error"` | `orchestrator.py` |
| 4A.5 | Catch `ConnectionError` → log + return `"Network error. Please try again."` | `orchestrator.py` |
| 4A.6 | Catch `Exception` as last resort → `logger.exception()` + return `"Unexpected error"` | `orchestrator.py` |
| 4A.7 | All catches use `logger.exception()` or `logger.error()` to preserve traceback | `orchestrator.py` |

### 4B: Fix sync wrapper (Q4)

| Step | Action | File |
|------|--------|--------|
| 4B.1 | Replace `asyncio.get_event_loop().run_until_complete(coro)` with `asyncio.run(coro)` | `orchestrator.py:run_pipeline()` |

**Verification**: `uv run pytest tests/ -v` — orchestrator tests need updates for new exception types.

---

## Phase 5: Test Coverage

Depends on Phases 3-4 (production code must be stable before writing new tests).

### 5A: Test create_llm (T1)

| Step | Action | File |
|------|--------|--------|
| 5A.1 | Create `tests/test_llm.py` | New file |
| 5A.2 | Test default model (Qwen/Qwen2.5-7B-Instruct, temperature=0.3) | `tests/test_llm.py` |
| 5A.3 | Test custom model override via parameter | `tests/test_llm.py` |
| 5A.4 | Test temperature forwarding | `tests/test_llm.py` |
| 5A.5 | Test API key fallback (param → env var → None) | `tests/test_llm.py` |

**Approach**: Mock `HuggingFaceEndpoint` and `ChatHuggingFace` with `unittest.mock.patch`.

### 5B: Test run_pipeline_async (T2)

| Step | Action | File |
|------|--------|------|
| 5B.1 | Add tests calling `run_pipeline_async()` directly (not `graph.ainvoke()`) | `tests/test_orchestrator.py` |
| 5B.2 | Test success path — verify result dict has `{"success": True, "draft": ...}` | `tests/test_orchestrator.py` |
| 5B.3 | Test ValueError from guards — verify `{"success": False, "error": "Pipeline configuration error"}` | `tests/test_orchestrator.py` |
| 5B.4 | Test ValidationError from Pydantic — verify error message | `tests/test_orchestrator.py` |
| 5B.5 | Test RuntimeError from retry exhaustion — verify error message | `tests/test_orchestrator.py` |
| 5B.6 | Test generic Exception — verify `{"success": False, "error": "Unexpected error"}` | `tests/test_orchestrator.py` |

### 5C: Test async_main / main (T3)

| Step | Action | File |
|------|--------|--------|
| 5C.1 | Add tests for `async_main()` and `main()` | `tests/test_cli.py` |
| 5C.2 | Test normal flow: mock `builtins.input` → valid topic → pipeline runs → result displayed | `tests/test_cli.py` |
| 5C.3 | Test EOF handling: input raises `EOFError` → clean exit | `tests/test_cli.py` |
| 5C.4 | Test KeyboardInterrupt: input raises `KeyboardInterrupt` → clean exit | `tests/test_cli.py` |
| 5C.5 | Test quit commands: "quit", "exit" → loop ends | `tests/test_cli.py` |

### 5D: Update existing tests for refactored agents

| Step | Action | Detail |
|------|--------|--------|
| 5D.1 | Update agent tests to work with `run_agent_chain()` helper | Mock the helper or provide proper config |
| 5D.2 | Update retry handler tests — parameter name unchanged (we kept `callable`) | No change needed |
| 5D.3 | Update MCP protocol tests for new `MCPEnvelope` return type and sender validation | `tests/test_mcp_protocol.py` |
| 5D.4 | Update orchestrator tests for new exception handling | `tests/test_orchestrator.py` |
| 5D.5 | Fix any tests broken by BlogState `error` field removal | Search for `state["error"]` usage |

---

## Phase 6: Final Verification

| Step | Action |
|------|--------|
| 6.1 | Run full test suite: `uv run pytest tests/ -v --tb=short` |
| 6.2 | Verify test count is reasonable (should be ~90-100 after additions) |
| 6.3 | Run the CLI manually: `uv run blog-mas` with a test topic |
| 6.4 | Verify no real tokens in any file: `grep -r "hf_" --include="*.py" --include="*.env"` |
| 6.5 | Verify `.env.example` exists and `.env` has only placeholder |
| 6.6 | Verify `run_ins.txt` is removed and `.gitignore` is updated |

---

## Files Changed Summary

| File | Phase | Changes |
|------|-------|---------|
| `.env` | 0 | Replace token with placeholder |
| `.env.example` | 1 | New file |
| `.gitignore` | 1 | Remove `run_ins.txt` line |
| `run_ins.txt` | 1 | Delete |
| `pyproject.toml` | 1 | C1, C3, C4 |
| `src/blog_mas/state.py` | 1 | Remove `error` field |
| `src/blog_mas/logging_config.py` | 2 | New file |
| `src/blog_mas/agent_helpers.py` | 3 | New file (shared helper + sanitize_feedback) |
| `src/blog_mas/knowledge_base.py` | 2 | Add `get_available_topics()`, fix `lookup_topic("")` |
| `src/blog_mas/cli.py` | 2 | Import topics from KB, add logging setup call |
| `src/blog_mas/orchestrator.py` | 2,4 | Input validation, layered catches, asyncio.run() |
| `src/blog_mas/llm.py` | 2 | Add logging, move import |
| `src/blog_mas/retry.py` | 2 | Add logging |
| `src/blog_mas/agents/intake.py` | 3 | Use helper, add guard |
| `src/blog_mas/agents/researcher.py` | 3 | Use helper, add guard, validate with ResearchRequest |
| `src/blog_mas/agents/writer.py` | 3 | Use helper, add guard, validate with WriterInput, sanitize feedback |
| `src/blog_mas/agents/validator.py` | 3 | Use helper, add guard, validate with ValidationInput |
| `src/blog_mas/mcp/protocol.py` | 2 | MCPEnvelope model, sender validation, generic errors, logging |
| `src/blog_mas/mcp/models.py` | 2 | Add Field constraints |
| `tests/test_llm.py` | 5 | New file |
| `tests/test_orchestrator.py` | 4,5 | Update exception tests, add run_pipeline_async tests |
| `tests/test_cli.py` | 5 | Add async_main/main tests |
| `tests/test_mcp_protocol.py` | 2,5 | Update for MCPEnvelope + sender validation |
| `tests/test_intake_agent.py` | 5 | Update for helper |
| `tests/test_researcher_agent.py` | 5 | Update for helper + ResearchRequest guard |
| `tests/test_writer_agent.py` | 5 | Update for helper + WriterInput guard + sanitize |
| `tests/test_validator_agent.py` | 5 | Update for helper + ValidationInput guard |
| `tests/conftest.py` | 5 | Update mocks if needed |

**Total**: 3 new files, ~20 modified files, ~5 test files with significant changes.

---

## Issue Coverage Matrix

| Issue | Severity | Phase | Status |
|-------|----------|-------|--------|
| S1 | Critical | 0 | Rotate token |
| S2 | High | 4 | Layered catches + generic messages |
| S3 | High | 3 | sanitize_feedback() |
| Q1 | High | 3 | Wire retry via helper |
| Q2 | High | 3 | Integrate MCP models as guards |
| Q3 | High | 3 | Integrate MCP models as guards |
| Q4 | High | 4 | asyncio.run() |
| T1 | High | 5 | test_llm.py |
| T2 | High | 5 | test_orchestrator.py additions |
| T3 | High | 5 | test_cli.py additions |
| E1 | High | 4 | Layered specific catches |
| E2 | High | 3 | LLM config guard in helper |
| E3 | High | 3 | Null guards in each agent |
| C1 | High | 1 | Move to dev group |
| S4 | Medium | — | Skipped (Pydantic constraints sufficient) |
| S5 | Medium | 2 | Sender validation |
| S6 | Medium | 2 | Generic error messages |
| S7 | Medium | 2 | Field constraints |
| Q5 | Medium | 3 | Shared agent helper |
| Q6 | Medium | 1 | Remove error field |
| Q7 | Medium | 2 | get_available_topics() |
| Q8 | Medium | 2 | MCPEnvelope Pydantic model |
| Q9 | Medium | 2 | lookup_topic empty string guard |
| E4 | Medium | 2 | Logging setup |
| E5 | Medium | — | Skipped (user decision) |
| E6 | Medium | — | Skipped (user decision) |
| E7 | Medium | 2 | Orchestrator input validation |
| E8 | Medium | 2 | MCP print → logging.warning |
| C2 | Medium | — | Skipped (user decision) |
| C3 | Medium | 1 | Project description |
| C4 | Medium | 1 | Entry point fix |
| C5 | Medium | 1 | .env.example |
| C6 | Medium | 1 | Clean up run_ins.txt |
| C7 | Medium | 1 | Remove from .gitignore |

**Covered**: 1/1 Critical, 14/14 High, 22/27 Medium
**Deferred**: 5 Medium (S4, E5, E6, C2, + S4 effectively handled by S7)
**Not in scope**: All 17 Low + 3 Manual
