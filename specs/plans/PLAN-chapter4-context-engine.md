# PLAN — Chapter 4 Context Engine on `blog-mas`

## 0. Mission

Implement chapter 4 of *Context Engineering for MAS* — the **Context Engine** (Planner → Executor → Tracer pattern) — on top of the existing `blog-mas` project. The Context Engine replaces a hardcoded multi-agent workflow with one that **designs its own JSON execution plan per user goal** at runtime.

The user's goal is pedagogical: practice and fully grasp the chapter 4 idea by building it end-to-end on a real codebase. Faithfulness to the chapter's design is more important than novelty.

**Implementation style:** straight-through (no TDD). Write tests after each module is complete to verify, but do not red-green-refactor.

---

## 1. Background — what `blog-mas` already has

`blog-mas` is a multi-agent blog generation system built through chapters 1–3 of the book. The relevant existing pieces:

### Existing agents (in `src/blog_mas/agents/`)
| Agent | File | Role |
|---|---|---|
| Intake | `intake.py` | Decomposes raw user request into `BlogSpec` + `intent_query`/`topic_query` |
| Librarian | `librarian.py` | Hybrid-searches Qdrant for a matching style **Blueprint** (procedural RAG) |
| Researcher | `researcher.py` | Hybrid-searches Qdrant for facts, then synthesizes a `ResearchSummary` (factual RAG) |
| Writer | `writer.py` | Combines Blueprint + ResearchSummary + BlogSpec into a `BlogDraft` |
| Validator | `validator.py` | Fact-checks the draft against the research summary, returns `ValidationVerdict` |

All agent functions are **`async`** and take `(state: BlogState, config: RunnableConfig) -> dict` (LangGraph node signature).

### Existing infrastructure
- **`src/blog_mas/orchestrator.py`** — LangGraph `StateGraph` with hardcoded topology:
  `intake → [librarian ∥ research] → write → validate → END/retry` (max 3 revision retries).
- **`src/blog_mas/state.py`** — `BlogState` (TypedDict) is the LangGraph state.
- **`src/blog_mas/mcp/models.py`** — Pydantic content models: `BlogSpec`, `ResearchSummary`, `BlogDraft`, `ValidationVerdict`, `WriterInput`, `ValidationInput`, `GoalDecomposition`.
- **`src/blog_mas/llm.py`** — `create_llm()` returns a `ChatOpenAI` pointed at local LM Studio (`http://127.0.0.1:1234`, default model `qwen/qwen3.5-9b`).
- **`src/blog_mas/agent_helpers.py`** — `run_agent_chain()` wraps LLM call + Pydantic parsing + retry.
- **`src/blog_mas/retry.py`** — `retry_handler()`.
- **`src/blog_mas/cli.py`** — interactive CLI; subcommands `ingest`, `ingest-blueprints`, `eval`.
- **`src/blog_mas/rag/`** — Qdrant store, embedder, reranker, hybrid search, blueprint/knowledge ingestion.

### Constraints
- Keep the existing LangGraph orchestrator **untouched and working** — engine is added alongside, not as a replacement. Existing tests must stay green.
- Reuse `create_llm()` for the Planner.
- Adapters wrap existing agents — **do not rewrite the agents themselves**.

---

## 2. Chapter 4 in one paragraph

Chapter 3's orchestrator hardcoded the workflow. Chapter 4 replaces the hardcoded orchestrator with a **Context Engine** that runs three phases:

1. **Plan** — an LLM reads the user's goal and a description of the available agents, and produces a step-by-step JSON plan.
2. **Execute** — a generic loop walks the plan, resolves `$$STEP_N_OUTPUT$$` placeholder references against a state dict (context chaining), and calls each agent.
3. **Reflect (Trace)** — a logger records the plan, every step's planned input, resolved input, output, status, and timings.

Five components: **Planner**, **Executor**, **Agent Registry** (catalog + capability descriptions), **Specialist Agents**, **Tracer**. The Writer is upgraded to a **dual-mode** function: it can either generate fresh content from `facts` (a `ResearchSummary`) **or** rewrite `previous_content` in a new style.

---

## 3. Design decisions (already locked)

| Decision | Choice | Rationale |
|---|---|---|
| Agent integration | **Thin adapters** wrapping existing LangGraph nodes into `(mcp_message) → mcp_message` shape | Zero risk to existing code, faithful to chapter 4's MCP-style handlers |
| Planner LLM | Reuse `create_llm()` | Same model as the rest of the stack; no extra config |
| User interface | New `--engine` flag on existing CLI; default behavior unchanged | Lets user A/B compare chapter-3 hardcoded vs chapter-4 dynamic |
| Tracer persistence | In-memory + optional `--save-trace` JSON dump | Matches the chapter's "improved version" |
| Validation as a step | **Leave it to the Planner** to decide whether a plan needs a validation step | Faithful to chapter 4 — the Planner owns what-to-do decisions |
| Build order | Straight-through (no TDD) | User preference; tests written after to verify |

---

## 4. New file layout

All new code lives under `src/blog_mas/engine/`. Tests under `tests/engine/`.

```
src/blog_mas/engine/
├── __init__.py
├── mcp_envelope.py     # create_mcp_message helper (dict-style envelope)
├── resolver.py         # resolve_dependencies — the $$STEP_N_OUTPUT$$ substitutor
├── tracer.py           # ExecutionTrace (in-memory + save-to-JSON)
├── registry.py         # AgentRegistry: name→handler + capability description
├── validators.py       # validate_plan: structural plan validation
├── planner.py          # planner(goal, capabilities) → JSON plan
├── executor.py         # execute(plan, registry, state, trace)
├── context_engine.py   # context_engine(goal) — top-level entry point
└── agent_adapters.py   # wraps existing async LangGraph nodes into MCP handlers

tests/engine/
├── __init__.py
├── test_mcp_envelope.py
├── test_resolver.py
├── test_tracer.py
├── test_registry.py
├── test_validators.py
├── test_planner.py        # uses fake LLM
├── test_executor.py       # uses stub agents
├── test_context_engine.py # end-to-end with stubs
└── test_writer_rewrite.py # Writer's new previous_content mode
```

**Modified files:**
- `src/blog_mas/agents/writer.py` — add `previous_content` rewrite mode.
- `src/blog_mas/cli.py` — add `--engine` and `--save-trace PATH` flags.

**No modifications** to: `orchestrator.py`, `state.py`, `mcp/models.py`, `llm.py`, `retry.py`, any RAG files, any other agent files.

---

## 5. Module-by-module specification

### 5.1 `engine/mcp_envelope.py`

The book uses a simple dict envelope `{"sender": str, "content": dict}`. blog-mas elsewhere uses Pydantic models — engine layer needs the dict-style envelope to match the chapter exactly.

```python
def create_mcp_message(sender: str, content: dict) -> dict:
    """Wrap content in a minimal MCP envelope (chapter 2 pattern)."""
    return {"sender": sender, "content": content}
```

That's the entire module.

---

### 5.2 `engine/resolver.py`

Implements **context chaining** — the `$$STEP_N_OUTPUT$$` placeholder substitution.

```python
import copy

def resolve_dependencies(input_params: dict, state: dict) -> dict:
    """Replace $$REF$$ placeholders in input_params with values from state.

    - Uses copy.deepcopy so the original plan is never mutated.
    - Recursively walks dicts and lists.
    - Whole-string placeholders only (e.g. "$$STEP_1_OUTPUT$$") — partial
      substitution is intentionally NOT supported (chapter 4 simplification).
    - Raises ValueError if a referenced key is not in state.
    """
    resolved = copy.deepcopy(input_params)

    def resolve(value):
        if isinstance(value, str) and value.startswith("$$") and value.endswith("$$"):
            ref_key = value[2:-2]
            if ref_key not in state:
                raise ValueError(
                    f"Dependency Error: Reference {ref_key} not found in execution state."
                )
            return state[ref_key]
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return resolve(resolved)
```

---

### 5.3 `engine/tracer.py`

```python
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

class ExecutionTrace:
    """Records the full plan-execute-reflect lifecycle for a single goal."""

    def __init__(self, goal: str):
        self.trace_id = str(uuid.uuid4())
        self.goal = goal
        self.plan = None
        self.steps = []
        self.status = "Initialized"
        self.final_output = None
        self.started_at = datetime.utcnow().isoformat()
        self.start_time = time.time()
        self.duration = None

    def log_plan(self, plan):
        self.plan = plan
        self.status = "Running"

    def log_step(self, step_num, agent, planned_input, resolved_input, output):
        self.steps.append({
            "step": step_num,
            "agent": agent,
            "planned_input": planned_input,
            "resolved_context": resolved_input,
            "output": output,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def finalize(self, status: str, final_output=None):
        self.status = status
        self.final_output = final_output
        self.duration = time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "goal": self.goal,
            "plan": self.plan,
            "steps": self.steps,
            "status": self.status,
            "final_output": self.final_output,
            "started_at": self.started_at,
            "duration_seconds": self.duration,
        }

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{self.trace_id}.json"
        # Pydantic objects in steps may not be JSON-serializable directly —
        # use default=str to fall back to repr for unknown types.
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return out
```

---

### 5.4 `engine/registry.py`

The chapter's "improved version" — declarative registration with auto-generated capability description.

```python
from dataclasses import dataclass
from typing import Awaitable, Callable

# Handler shape: takes an MCP envelope dict, returns an MCP envelope dict (async).
Handler = Callable[[dict], Awaitable[dict]]

@dataclass
class AgentSpec:
    handler: Handler
    role: str
    inputs: dict[str, str]  # input_key -> human-readable description+type
    output: str             # description of the output payload

class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, AgentSpec] = {}

    def register(self, name: str, handler: Handler, role: str,
                 inputs: dict[str, str], output: str) -> None:
        self._agents[name] = AgentSpec(handler, role, inputs, output)

    def get_handler(self, name: str) -> Handler:
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found in registry.")
        return self._agents[name].handler

    def has(self, name: str) -> bool:
        return name in self._agents

    def required_input_keys(self, name: str) -> set[str]:
        return set(self._agents[name].inputs.keys())

    def get_capabilities_description(self) -> str:
        lines = ["Available Agents and their required inputs:"]
        for i, (name, spec) in enumerate(self._agents.items(), 1):
            lines.append(f"\n{i}. AGENT: {name}")
            lines.append(f"   ROLE: {spec.role}")
            lines.append("   INPUTS:")
            for key, desc in spec.inputs.items():
                lines.append(f'   - "{key}": {desc}')
            lines.append(f"   OUTPUT: {spec.output}")
        return "\n".join(lines)
```

The default registry is built at runtime via a factory `build_default_registry()` in `agent_adapters.py` (see §5.9).

---

### 5.5 `engine/validators.py`

Structural plan validation, run *before* execution to fail fast.

```python
def validate_plan(plan: list, registry) -> None:
    """Raise ValueError on the first malformed step. Otherwise return None.

    Checks:
    - plan is a non-empty list of dicts
    - each step has integer 'step' (1-indexed, sequential), string 'agent', dict 'input'
    - 'agent' exists in registry
    - any $$STEP_N_OUTPUT$$ reference points to a strictly earlier step
    - required input keys for the agent are present (best-effort warning, not error,
      because a Writer step legitimately uses either 'facts' OR 'previous_content')
    """
    if not isinstance(plan, list) or not plan:
        raise ValueError("Plan must be a non-empty list.")

    for i, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {i}: must be a dict.")
        if step.get("step") != i:
            raise ValueError(f"Step {i}: 'step' must equal {i} (got {step.get('step')!r}).")
        agent = step.get("agent")
        if not isinstance(agent, str) or not registry.has(agent):
            raise ValueError(f"Step {i}: unknown agent {agent!r}.")
        inp = step.get("input")
        if not isinstance(inp, dict):
            raise ValueError(f"Step {i}: 'input' must be a dict.")
        _check_refs(i, inp)

def _check_refs(step_num: int, value) -> None:
    if isinstance(value, str) and value.startswith("$$STEP_") and value.endswith("_OUTPUT$$"):
        try:
            ref_num = int(value[len("$$STEP_"):-len("_OUTPUT$$")])
        except ValueError as e:
            raise ValueError(f"Step {step_num}: malformed reference {value!r}.") from e
        if ref_num >= step_num:
            raise ValueError(
                f"Step {step_num}: forward/self reference to step {ref_num}."
            )
    elif isinstance(value, dict):
        for v in value.values():
            _check_refs(step_num, v)
    elif isinstance(value, list):
        for v in value:
            _check_refs(step_num, v)
```

---

### 5.6 `engine/planner.py`

The Planner. Builds a system prompt with role, capability description, rules, and few-shot examples, then asks the LLM for a JSON list.

**Key implementation notes:**
- Uses `langchain_core.messages.SystemMessage` + `HumanMessage` like the rest of blog-mas.
- The LM Studio backend may not honor strict JSON mode, so the prompt itself instructs `Output ONLY a valid JSON list`. Parsing is robust:
  1. Try `json.loads` directly.
  2. If that fails, extract the largest `[...]` substring with a regex and try again.
  3. If the result is a dict with a `"plan"` key, unwrap it.
- Wraps the LLM call with `retry_handler` (reusing `blog_mas.retry`).

```python
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from blog_mas.retry import retry_handler

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT_TEMPLATE = """\
You are the strategic core of the Context Engine for a multi-agent blog generation system.
Analyze the user's high-level goal and produce a structured Execution Plan that uses the available agents.

--- AVAILABLE CAPABILITIES ---
{capabilities}
--- END CAPABILITIES ---

INSTRUCTIONS:
1. The plan MUST be a JSON list of objects, where each object is a "step".
2. Each step MUST have integer "step" (starting at 1, contiguous), string "agent" matching one of the agents above, and dict "input".
3. Use Context Chaining: when a step needs the output of an earlier step, reference it as "$$STEP_N_OUTPUT$$" (a single string).
4. Do NOT forward-reference. Step N may only reference steps 1..N-1.
5. The Writer agent has TWO modes:
   - Fresh generation: provide "research_summary" (use this when writing from facts).
   - Rewriting: provide "previous_content" (use this when adapting an existing draft to a new style).
   In both modes, the Writer also requires "blueprint" and "blog_spec".
6. Multi-step blog generation typically goes: Intake → Librarian and Researcher (in any order) → Writer → optionally Validator.
7. For "rewrite in a different tone" goals, plan TWO Librarian retrievals (one per blueprint), TWO Writer steps (mode 1 then mode 2), then optionally Validator.
8. Output ONLY a valid JSON list. No prose, no markdown fences.

EXAMPLE GOAL: "Write a casual blog about climate change for general readers."
EXAMPLE PLAN:
[
  {{"step": 1, "agent": "Intake", "input": {{"raw_input": "Write a casual blog about climate change for general readers."}}}},
  {{"step": 2, "agent": "Librarian", "input": {{"intent_query": "$$STEP_1_OUTPUT$$.intent_query"}}}},
  {{"step": 3, "agent": "Researcher", "input": {{"topic_query": "$$STEP_1_OUTPUT$$.topic_query", "blog_spec": "$$STEP_1_OUTPUT$$.blog_spec"}}}},
  {{"step": 4, "agent": "Writer", "input": {{"blueprint": "$$STEP_2_OUTPUT$$", "research_summary": "$$STEP_3_OUTPUT$$", "blog_spec": "$$STEP_1_OUTPUT$$.blog_spec"}}}},
  {{"step": 5, "agent": "Validator", "input": {{"draft": "$$STEP_4_OUTPUT$$", "research_summary": "$$STEP_3_OUTPUT$$"}}}}
]

EXAMPLE GOAL: "Write a technical deep-dive on space exploration, then rewrite it as a casual explainer."
EXAMPLE PLAN:
[
  {{"step": 1, "agent": "Intake", "input": {{"raw_input": "Write a technical deep-dive on space exploration."}}}},
  {{"step": 2, "agent": "Librarian", "input": {{"intent_query": "technical deep-dive blueprint"}}}},
  {{"step": 3, "agent": "Researcher", "input": {{"topic_query": "$$STEP_1_OUTPUT$$.topic_query", "blog_spec": "$$STEP_1_OUTPUT$$.blog_spec"}}}},
  {{"step": 4, "agent": "Writer", "input": {{"blueprint": "$$STEP_2_OUTPUT$$", "research_summary": "$$STEP_3_OUTPUT$$", "blog_spec": "$$STEP_1_OUTPUT$$.blog_spec"}}}},
  {{"step": 5, "agent": "Librarian", "input": {{"intent_query": "casual explainer blueprint"}}}},
  {{"step": 6, "agent": "Writer", "input": {{"blueprint": "$$STEP_5_OUTPUT$$", "previous_content": "$$STEP_4_OUTPUT$$", "blog_spec": "$$STEP_1_OUTPUT$$.blog_spec"}}}},
  {{"step": 7, "agent": "Validator", "input": {{"draft": "$$STEP_6_OUTPUT$$", "research_summary": "$$STEP_3_OUTPUT$$"}}}}
]
"""

_JSON_LIST_RE = re.compile(r"\[.*\]", re.DOTALL)

async def plan(goal: str, capabilities: str, llm) -> list:
    """Generate a JSON execution plan for `goal` using `llm`.

    Returns the parsed plan (a list of step dicts). Raises ValueError on
    unparseable output.
    """
    if llm is None:
        raise ValueError("[Planner] No LLM configured")

    system_prompt = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(capabilities=capabilities)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=goal)]

    logger.info("[Engine: Planner] Analyzing goal and generating execution plan...")
    raw = await retry_handler(
        lambda: llm.ainvoke(messages),
        agent_name="Planner",
        max_retries=3,
        base_delay=2.0,
    )
    text = raw.content if hasattr(raw, "content") else str(raw)
    plan_obj = _parse_plan_text(text)
    logger.info("[Engine: Planner] Plan generated with %d steps.", len(plan_obj))
    return plan_obj

def _parse_plan_text(text: str) -> list:
    """Robustly extract a JSON list from LLM output."""
    candidates = [text]
    match = _JSON_LIST_RE.search(text)
    if match:
        candidates.append(match.group(0))

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("plan"), list):
            return obj["plan"]
        if isinstance(obj, list):
            return obj

    raise ValueError(f"Planner did not return parseable JSON. Raw output: {text!r}")
```

#### Note on dotted path references

The example plans above use refs like `"$$STEP_1_OUTPUT$$.intent_query"` to pull a sub-field of a previous output. This is a **light extension** beyond the book's strict whole-string rule. To support it, `resolver.py`'s `resolve()` function needs an extra branch:

```python
# Inside resolve(), BEFORE the plain $$...$$ branch:
if isinstance(value, str) and value.startswith("$$") and "$$." in value:
    ref_part, _, attr_path = value[2:].partition("$$.")
    if ref_part not in state:
        raise ValueError(f"Dependency Error: Reference {ref_part} not found in execution state.")
    obj = state[ref_part]
    for attr in attr_path.split("."):
        if isinstance(obj, dict):
            obj = obj[attr]
        else:
            obj = getattr(obj, attr)
    return obj
```

Add this **inside `engine/resolver.py`** alongside the whole-string branch. Update the resolver's docstring accordingly. Tests must cover both forms.

---

### 5.7 `engine/executor.py`

```python
import logging

from blog_mas.engine.mcp_envelope import create_mcp_message
from blog_mas.engine.resolver import resolve_dependencies

logger = logging.getLogger(__name__)

async def execute(plan: list, registry, trace) -> tuple[dict, object | None]:
    """Walk the plan, calling each agent. Returns (state, final_output).

    On per-step failure: log to trace, finalize trace as Failed, re-raise.
    """
    state: dict = {}

    for step in plan:
        step_num = step["step"]
        agent_name = step["agent"]
        planned_input = step["input"]

        logger.info("[Engine: Executor] Starting Step %d: %s", step_num, agent_name)
        try:
            handler = registry.get_handler(agent_name)
            resolved_input = resolve_dependencies(planned_input, state)
            mcp_in = create_mcp_message("Engine", resolved_input)
            mcp_out = await handler(mcp_in)
            output = mcp_out["content"]
            state[f"STEP_{step_num}_OUTPUT"] = output
            trace.log_step(step_num, agent_name, planned_input, resolved_input, output)
            logger.info("[Engine: Executor] Step %d completed.", step_num)
        except Exception as e:
            logger.exception("[Engine: Executor] Step %d (%s) failed: %s", step_num, agent_name, e)
            trace.finalize(f"Failed at Step {step_num}: {e}")
            raise

    final_output = state.get(f"STEP_{len(plan)}_OUTPUT")
    return state, final_output
```

---

### 5.8 `engine/context_engine.py`

Top-level entry point. Wires planner + validator + executor + tracer with graceful failure.

```python
import logging

from blog_mas.engine.planner import plan as planner_plan
from blog_mas.engine.executor import execute
from blog_mas.engine.tracer import ExecutionTrace
from blog_mas.engine.validators import validate_plan

logger = logging.getLogger(__name__)

async def run_context_engine(
    goal: str,
    registry,
    llm,
    *,
    save_trace_dir: str | None = None,
):
    """Plan → validate → execute → trace.

    Returns (final_output, trace). final_output is None on failure;
    trace.status describes what happened.
    """
    logger.info("=== [Context Engine] Starting New Task === Goal: %s", goal)
    trace = ExecutionTrace(goal)

    # Phase 1: Plan
    try:
        capabilities = registry.get_capabilities_description()
        plan = await planner_plan(goal, capabilities, llm)
        validate_plan(plan, registry)
        trace.log_plan(plan)
    except Exception as e:
        logger.exception("[Context Engine] Planning failed: %s", e)
        trace.finalize(f"Failed during Planning: {e}")
        if save_trace_dir:
            trace.save(save_trace_dir)
        return None, trace

    # Phase 2: Execute
    try:
        _state, final_output = await execute(plan, registry, trace)
    except Exception:
        # execute() already finalized the trace as Failed.
        if save_trace_dir:
            trace.save(save_trace_dir)
        return None, trace

    trace.finalize("Success", final_output)
    if save_trace_dir:
        trace.save(save_trace_dir)
    logger.info("=== [Context Engine] Task Complete ===")
    return final_output, trace
```

---

### 5.9 `engine/agent_adapters.py`

Bridges the chapter-4 dict-style handler shape `(mcp_message: dict) → mcp_message: dict` to blog-mas's existing `(state: BlogState, config: RunnableConfig) → dict` LangGraph nodes.

Each adapter:
1. Takes the dict input from `mcp_message["content"]`.
2. Constructs a minimal `BlogState`-shaped dict with the required fields.
3. Calls the existing async node with a synthetic `config` carrying llm/store/embedder/reranker.
4. Wraps the relevant return value in `create_mcp_message("AgentName", ...)`.

```python
from blog_mas.agents.intake import intake_node
from blog_mas.agents.librarian import librarian_node
from blog_mas.agents.researcher import research_node
from blog_mas.agents.writer import write_node
from blog_mas.agents.validator import validate_node
from blog_mas.engine.mcp_envelope import create_mcp_message
from blog_mas.engine.registry import AgentRegistry
from blog_mas.rag.blueprints import NEUTRAL_BLUEPRINT


def _config(llm, store, embedder, reranker):
    return {"configurable": {
        "llm": llm, "store": store, "embedder": embedder, "reranker": reranker,
    }}


def make_intake_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        raw_input = msg["content"]["raw_input"]
        state = {"raw_input": raw_input, "revision_count": 0}
        out = await intake_node(state, _config(llm, store, embedder, reranker))
        # Return ALL fields the Planner may want to reference downstream.
        return create_mcp_message("Intake", {
            "blog_spec": out["blog_spec"],
            "intent_query": out["intent_query"],
            "topic_query": out["topic_query"],
        })
    return handler


def make_librarian_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        intent_query = msg["content"].get("intent_query")
        state = {"intent_query": intent_query, "revision_count": 0}
        out = await librarian_node(state, _config(llm, store, embedder, reranker))
        # The Librarian returns a dict with 'blueprint' and metadata.
        # Planner needs the blueprint object directly as the "OUTPUT" of this step.
        return create_mcp_message("Librarian", out["blueprint"])
    return handler


def make_researcher_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        state = {
            "blog_spec": content["blog_spec"],
            "topic_query": content.get("topic_query"),
            "revision_count": 0,
        }
        out = await research_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Researcher", out["research_summary"])
    return handler


def make_writer_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        blueprint = content["blueprint"] or NEUTRAL_BLUEPRINT
        blog_spec = content["blog_spec"]

        # Dual-mode dispatch — exactly one of research_summary OR previous_content.
        research_summary = content.get("research_summary")
        previous_content = content.get("previous_content")

        state = {
            "blog_spec": blog_spec,
            "blueprint": blueprint,
            "research_summary": research_summary,
            "previous_content": previous_content,
            "draft": None,
            "revision_feedback": None,
            "revision_count": 0,
        }
        out = await write_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Writer", out["draft"])
    return handler


def make_validator_handler(llm, store, embedder, reranker):
    async def handler(msg: dict) -> dict:
        content = msg["content"]
        state = {
            "research_summary": content["research_summary"],
            "draft": content["draft"],
            "revision_count": 0,
        }
        out = await validate_node(state, _config(llm, store, embedder, reranker))
        return create_mcp_message("Validator", out["verdict"])
    return handler


def build_default_registry(llm, store, embedder, reranker) -> AgentRegistry:
    """Build the default registry of all five blog-mas agents."""
    reg = AgentRegistry()

    reg.register(
        "Intake",
        make_intake_handler(llm, store, embedder, reranker),
        role="Decomposes a free-text user request into a structured BlogSpec plus intent_query (style) and topic_query (subject).",
        inputs={"raw_input": "(String) The user's original blog request."},
        output="A dict with keys 'blog_spec' (BlogSpec), 'intent_query' (string), 'topic_query' (string).",
    )

    reg.register(
        "Librarian",
        make_librarian_handler(llm, store, embedder, reranker),
        role="Retrieves a Semantic Blueprint (style/structure instructions) from the Qdrant blueprint collection via hybrid search.",
        inputs={"intent_query": "(String/Reference) A descriptive phrase of the desired style or format."},
        output="A Blueprint object (Pydantic model) describing the target voice, structure, and constraints.",
    )

    reg.register(
        "Researcher",
        make_researcher_handler(llm, store, embedder, reranker),
        role="Retrieves and synthesizes factual information from the Qdrant knowledge collection.",
        inputs={
            "topic_query": "(String/Reference) The subject matter to research.",
            "blog_spec": "(Reference) The BlogSpec produced by Intake (provides topic, audience, goal context).",
        },
        output="A ResearchSummary object containing topic, bullet_points, and source citations.",
    )

    reg.register(
        "Writer",
        make_writer_handler(llm, store, embedder, reranker),
        role="Generates or rewrites a blog draft. Has TWO modes — fresh generation from research, or rewriting an existing draft in a new style.",
        inputs={
            "blueprint": "(Reference) The Blueprint from a Librarian step (style instructions).",
            "blog_spec": "(Reference) The BlogSpec from Intake.",
            "research_summary": "(Reference, OPTIONAL) ResearchSummary from a Researcher step. Use this for FRESH content generation.",
            "previous_content": "(Reference, OPTIONAL) A prior BlogDraft from an earlier Writer step. Use this for REWRITING in a new style.",
        },
        output="A BlogDraft object with title, body, and word_count.",
    )

    reg.register(
        "Validator",
        make_validator_handler(llm, store, embedder, reranker),
        role="Fact-checks a BlogDraft against a ResearchSummary and returns a pass/fail verdict.",
        inputs={
            "draft": "(Reference) The BlogDraft from a Writer step.",
            "research_summary": "(Reference) The ResearchSummary the draft should be grounded in.",
        },
        output="A ValidationVerdict object with verdict ('pass'|'fail') and reason.",
    )

    return reg
```

**Important:** the chapter-4 plan does not include an explicit revision loop — if the Validator says fail, the Context Engine returns failure with the trace. The user's chapter-3 LangGraph still has the retry loop; this is a deliberate philosophical difference and worth pointing out in code comments.

---

### 5.10 Modify `agents/writer.py` — add dual-mode dispatch

Current Writer requires a `research_summary` and raises if missing. Add `previous_content` as an alternative source.

**Diff sketch (apply to existing file):**

1. After loading `blog_spec` and `blueprint`, replace the strict `research_summary` check with this:

```python
research_summary = state.get("research_summary")
previous_content = state.get("previous_content")

if research_summary is None and previous_content is None:
    raise ValueError(
        "[Writer] Upstream agent failed — Writer requires either "
        "'research_summary' (fresh mode) or 'previous_content' (rewrite mode)."
    )
```

2. Build `user_message` differently per mode. Keep the existing `research_summary` branch as-is. Add a new branch when `previous_content` is set (e.g., a `BlogDraft`):

```python
if previous_content is not None and research_summary is None:
    # Rewrite mode — feed the previous draft body as source material.
    prev_body = previous_content.body if hasattr(previous_content, "body") else str(previous_content)
    lines = [
        "PREVIOUS CONTENT (rewrite this in the new blueprint's style):",
        prev_body,
        "",
        "Blog specification:",
        f"- Tone: {blog_spec.tone}",
        f"- Audience: {blog_spec.audience}",
        f"- Goal: {blog_spec.goal}",
    ]
    if blog_spec.constraints:
        lines.append(f"- Constraints: {', '.join(blog_spec.constraints)}")
    user_message = "\n".join(lines)
    # Reuse WRITER_SYSTEM_PROMPT — the blueprint scaffold is already injected.
    system_prompt = WRITER_SYSTEM_PROMPT.format(blueprint_scaffold=scaffold)
else:
    # existing research_summary path, unchanged
    ...
```

3. Update `BlogState` in `state.py`? **No.** Adding fields to `BlogState` would touch the LangGraph orchestrator's contract. Instead, the Writer reads `previous_content` via `state.get("previous_content")`, which works on a plain dict whether or not the key is declared in the TypedDict. Adapters pass plain dicts so this is fine. **Do not modify `state.py`.**

4. The Writer's existing **revision** branch (`is_revision = revision_feedback is not None`) is unchanged — it still applies in chapter-3 mode.

5. Validate via the existing `WriterInput` Pydantic model only when `research_summary` is present (otherwise skip — `WriterInput` requires `research_summary`).

---

### 5.11 Modify `cli.py` — add `--engine` and `--save-trace` flags

Add to `build_parser()`:

```python
parser.add_argument(
    "--engine",
    action="store_true",
    help="Route requests through the chapter 4 Context Engine (dynamic plan-and-execute) "
         "instead of the default LangGraph orchestrator.",
)
parser.add_argument(
    "--save-trace",
    metavar="DIR",
    default=None,
    help="When --engine is set, save the execution trace JSON to this directory.",
)
```

In `async_main()`, branch the per-request call:

```python
if args.engine:
    from blog_mas.engine.agent_adapters import build_default_registry
    from blog_mas.engine.context_engine import run_context_engine
    registry = build_default_registry(llm, store, embedder, reranker=None)

    while True:
        # ... read user_input, validate ...
        final, trace = await run_context_engine(
            goal=validated, registry=registry, llm=llm,
            save_trace_dir=args.save_trace,
        )
        _display_engine_result(final, trace)
else:
    # existing run_pipeline_async loop
```

Add a small `_display_engine_result(final, trace)` helper that prints:
- Plan summary (steps + agents)
- Status
- Duration
- The final output if it's a `BlogDraft` (title + body), otherwise `repr(final)`

`build_parser()` currently uses subparsers. Make `--engine` and `--save-trace` **top-level** flags so they coexist with the no-subcommand interactive mode. They are inert when a subcommand (`ingest`, `eval`, etc.) is used.

`reranker` argument: the CLI doesn't currently construct a reranker — leave it as `None` in the registry factory; `librarian_node` will see `reranker=None`. Confirm the existing `librarian_node` handles `reranker=None` before relying on this; if it doesn't, build a no-op reranker stub. **(Sub-step: open `librarian.py` and `hybrid_search` to verify the None path works; if not, instantiate the existing reranker the same way `async_main` already does for the LangGraph path — mirror that exactly.)**

---

## 6. Tests (post-implementation)

Place under `tests/engine/`. Run with `uv run pytest tests/engine/ -v`.

### 6.1 `test_mcp_envelope.py`
- `create_mcp_message` returns `{"sender": ..., "content": ...}` shape.

### 6.2 `test_resolver.py`
- Whole-string `$$STEP_1_OUTPUT$$` resolves to state value.
- Dotted form `$$STEP_1_OUTPUT$$.field` resolves to attribute/key.
- Nested dicts and lists are walked recursively.
- Missing reference raises `ValueError`.
- `deepcopy` invariance: original `input_params` is not mutated after resolution.
- Non-placeholder values pass through unchanged (ints, plain strings, booleans, None).

### 6.3 `test_tracer.py`
- `log_plan` sets status to `Running`.
- `log_step` appends with all expected keys.
- `finalize("Success", out)` sets status, final_output, duration.
- `to_dict()` is JSON-serializable with `default=str`.
- `save(tmp_path)` writes a file named `{trace_id}.json`.

### 6.4 `test_registry.py`
- Register + `get_handler` round-trip.
- `get_handler` on unknown name raises.
- `get_capabilities_description` includes role, all input keys, and output for each agent.

### 6.5 `test_validators.py`
- Empty plan → ValueError.
- Step number not contiguous → ValueError.
- Unknown agent → ValueError.
- Forward reference (`$$STEP_3_OUTPUT$$` inside step 2) → ValueError.
- Self-reference → ValueError.
- Valid plan passes silently.

### 6.6 `test_planner.py`
Use a fake LLM whose `ainvoke` returns a `MagicMock` with `.content = "<json>"`.
- Returns a list when LLM returns a JSON list.
- Unwraps `{"plan": [...]}` shape.
- Strips surrounding prose: `"Here is the plan: [..]"` still parses.
- Unparseable output raises `ValueError`.

### 6.7 `test_executor.py`
Use stub handlers (no LLM, no Qdrant).
- Linear 3-step plan with chaining works; final state has all `STEP_N_OUTPUT` keys.
- Step failure propagates; `trace.status` starts with `"Failed at Step"`.
- `$$STEP_N_OUTPUT$$` references resolve to actual prior outputs.

### 6.8 `test_context_engine.py`
- Patch `planner.plan` to return a hardcoded plan; use stub registry.
- End-to-end: `(final, trace)` returned, status `Success`, plan logged, all steps logged.
- Planning failure → `final is None`, trace status starts with `"Failed during Planning"`.
- `save_trace_dir` writes a JSON file.

### 6.9 `test_writer_rewrite.py`
- Writer in `previous_content` mode produces a `BlogDraft` (use a fake LLM returning a valid `BlogDraft` JSON).
- Writer raises when both `research_summary` and `previous_content` are missing.
- Writer in `research_summary` mode (existing behavior) still works — regression check.

---

## 7. Implementation order (straight-through)

Execute in this order. After each step, run `uv run pytest tests/ -v` to confirm nothing regressed.

1. **`engine/mcp_envelope.py`** — trivial.
2. **`engine/resolver.py`** — including the dotted-path branch.
3. **`engine/tracer.py`**.
4. **`engine/registry.py`**.
5. **`engine/validators.py`**.
6. **Modify `agents/writer.py`** — add `previous_content` mode. Run existing writer tests to confirm fresh-mode regression-free.
7. **`engine/agent_adapters.py`** — verify reranker handling in `cli.py`'s existing path first; mirror it.
8. **`engine/planner.py`**.
9. **`engine/executor.py`**.
10. **`engine/context_engine.py`**.
11. **Modify `cli.py`** — add `--engine` and `--save-trace` flags + display helper.
12. **Tests under `tests/engine/`** — write per §6.
13. **Smoke test**: with LM Studio + Qdrant running, `uv run blog-mas --engine` → enter a goal like *"Write a casual blog about climate change, then rewrite it as a technical deep-dive."* Verify a 7-step plan is produced, executed, and the trace logs all steps.

---

## 8. Acceptance criteria

- [ ] All new files created at the paths in §4.
- [ ] All existing tests still pass (`uv run pytest tests/ -v`).
- [ ] New `tests/engine/` tests all pass.
- [ ] `uv run blog-mas` (no flag) behaves **identically** to before.
- [ ] `uv run blog-mas --engine` accepts a free-text goal, prints the generated plan, executes it through the existing agents, and prints the final output.
- [ ] `uv run blog-mas --engine --save-trace ./traces` writes a `{uuid}.json` per request.
- [ ] No edits to `orchestrator.py`, `state.py`, `mcp/models.py`, `llm.py`, `retry.py`, `agent_helpers.py`, RAG modules, or any agent file other than `writer.py`.
- [ ] The Planner's system prompt contains: role, dynamic capabilities injection, the Context Chaining rules, and at least two few-shot examples (one simple, one with rewrite).
- [ ] The Tracer captures plan, every step's planned + resolved input + output, status, duration.
- [ ] `validate_plan` is called before execution and rejects malformed plans.

---

## 9. Open notes for the implementer

- **LM Studio JSON mode**: Qwen via LM Studio does not always honor strict JSON mode. The Planner relies on prompt instructions + robust regex extraction. If parsing fails repeatedly in practice, raise a clear error with the raw output — do not silently fall back to a hardcoded plan.
- **Pydantic objects in trace JSON**: `ExecutionTrace.save()` uses `default=str`, which falls back to repr for unknown types. That's fine for human inspection. If structured trace analysis is needed later, switch to `pydantic.BaseModel.model_dump()` per known type.
- **No replanning**: chapter 4 is pure plan-and-execute. If a step fails or the Validator says fail, the engine returns failure. The chapter-3 LangGraph orchestrator's revision loop remains the only retry path. Do not try to add replanning here — that is chapter 5+ territory.
- **Reranker in `cli.py`**: confirm how the existing CLI builds the reranker for the LangGraph path and mirror exactly when constructing the engine registry. Don't invent a new wiring.
- **Don't import from `__pycache__`**: ignore stale `.pyc` files in the repo.

---

## 10. Reference — chapter 4's core concepts (for the implementer's understanding)

- **Plan-Execute-Reflect**: an LLM produces a JSON plan; a generic executor runs it; a tracer logs everything.
- **Plans as data, not code**: JSON is auditable, loggable, and safe to receive from an LLM.
- **`$$STEP_N_OUTPUT$$` context chaining**: a string-based reference protocol resolved against a state dict before each agent call. Like spreadsheet `=A1`.
- **Agent Registry**: name→handler map plus a natural-language capability description that becomes part of the Planner's prompt — "prompt engineering for the Planner."
- **Separation of what vs. how**: Planner decides what sequence of steps. Executor carries it out. Specialist agents don't know they are part of a plan.
- **Tracer**: required for debugging, auditing, and (per chapter) regulatory compliance.

The full chapter is at `../chapter_4` (sibling of `blog-mas/`) for reference.
