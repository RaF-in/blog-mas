# Plan: Multi-Agent Blog Generation System (Chapters 1 & 2)

> **Date:** 2026-05-01
> **Project source:** Standalone — Context Engineering for MAS book, Chapters 1 & 2
> **Estimated tasks:** 10-14
> **Planning session:** Detailed

## Summary

Build an interactive CLI application that implements a multi-agent blog generation system using LangChain, LangGraph, and async Python. The system takes a raw user request, decomposes it into a structured specification, then runs it through a pipeline of specialized agents (Researcher, Writer, Validator) coordinated by an Orchestrator via a standardized MCP message protocol. This covers Chapter 1's context engineering principles (L4-L5 prompt design, structured input decomposition, context chaining) and Chapter 2's multi-agent system concepts (MAS, MCP, hub-and-spoke orchestration, resilience, reliability, self-correcting loops).

## Requirements

### Functional Requirements
1. Interactive CLI loop that accepts raw blog requests from the user, processes them through the agent pipeline, and displays the final blog post
2. Intake step that decomposes a raw user request into a structured Pydantic model (topic, audience, tone, goal, constraints) using an LLM
3. Orchestrator that receives the structured spec and coordinates all agent calls in sequence — agents never communicate directly
4. Researcher agent that looks up a topic in a simulated knowledge base dictionary (4-5 topics) and synthesizes the retrieved info into structured bullet points via an LLM
5. Writer agent that transforms research bullet points into a polished blog post with a title, using an LLM
6. Validator agent that fact-checks the Writer's draft against the Researcher's summary and returns a structured pass/fail verdict with reason
7. Self-correcting revision loop: if Validator fails the draft, Writer receives the feedback and revises, up to 3 maximum revisions
8. MCP message envelope wrapping every inter-agent message with four fields: protocol_version, sender, content, metadata
9. Two-layer validation: MCP envelope validation at every message boundary, plus per-agent Pydantic content validation on inputs and outputs
10. Retry logic with exponential backoff (3 retries, base delays 2s/4s/8s) for all LLM calls
11. Verbose CLI output showing each pipeline step as it happens (agent activation, MCP validation, retry attempts, validation verdicts, revision feedback)
12. Pipeline continues even when the Researcher finds no matching topic in the knowledge base — Writer works with whatever the Researcher produces

### Non-Functional Requirements
1. Fully async — all LLM calls use async/await via LangChain's async interface (ainvoke)
2. Uses HuggingFace free Inference API via LangChain's HuggingFace integration with the most capable free model available (e.g., Mistral 7B or Llama 3.1 8B)
3. Graceful error handling — when all retries are exhausted, print a clear error message and exit gracefully (no stack traces to the user)
4. Input validation at the CLI level before anything enters the pipeline

## Behaviors

### CLI Interaction

**Interactive loop behavior:**
- On startup, print a welcome message and list available knowledge base topics (so the user knows what's available)
- Prompt the user for a blog request
- Process the request through the full pipeline with verbose step-by-step output
- Display the final blog post (or error message if validation loop exhausted)
- Prompt for the next request
- User types `exit` or `quit` to end the session

**Input validation:**
- Empty input or whitespace-only: print "Please provide a blog topic" and re-prompt immediately — nothing enters the pipeline
- Input exceeding 500 characters: print a message about the character limit and re-prompt
- Valid input: pass to the Intake agent

### Pipeline Execution (Verbose Output)

Each step prints its activity so the user sees the system working. Example flow:
```
[Intake] Decomposing your request...
[Intake] Structured spec: topic="Mediterranean diet", audience="fitness enthusiasts", ...
[MCP] Message from Intake validated successfully.
[Orchestrator] Delegating to Researcher...
[Researcher] Looking up topic: "Mediterranean diet"
[Researcher] Topic found in knowledge base. Synthesizing...
[MCP] Message from Researcher validated successfully.
[Orchestrator] Delegating to Writer...
[Writer] Drafting blog post...
[MCP] Message from Writer validated successfully.
[Orchestrator] Delegating to Validator...
[Validator] Fact-checking draft against research...
[Validator] Verdict: PASS
[MCP] Message from Validator validated successfully.
[Orchestrator] Validation PASSED. Here is your blog post:

--- BLOG POST ---
(final content)
--- END ---
```

On validation failure with revision:
```
[Validator] Verdict: FAIL — "Draft claims omega-3 reduces cancer risk, not supported by research summary"
[Orchestrator] Requesting revision (attempt 2 of 3)...
[Writer] Revising draft based on feedback...
```

### Agent Prompt Design (Chapter 1 L4-L5 Applied)

**Why this matters:** The entire lesson of Chapter 1 is that structured, goal-oriented, role-based prompts produce dramatically better output than bare prompts. Every agent's system prompt must demonstrate L4-L5 principles.

**Each agent system prompt must include:**
- Explicit role definition (who the agent is)
- Explicit task description (what it must do)
- Output format specification (exact shape expected)
- Constraints and negative constraints (what NOT to do)

**Common mistakes to avoid:**
- Writing vague system prompts like "You are helpful" — this is L1, defeats the chapter's purpose
- Forgetting to specify output format — smaller HF models need very explicit formatting instructions
- Not including negative constraints — LLMs love to add unrequested content

### Self-Correcting Loop

**Behavior:**
- Writer receives research summary on first pass
- If Validator returns "fail", Writer receives the original research summary PLUS the Validator's feedback reason on the next pass
- Maximum 3 revision attempts
- On pass at any point: output the approved draft and break the loop
- On exhausting all 3 revisions: print error "Failed to produce validated content after 3 revisions. Please try again." and return to the prompt — do NOT output any draft

**Why no partial output on failure:** The Validator exists to ensure factual consistency. Outputting a draft that failed 3 validation checks would undermine the entire purpose of the Validator agent.

### Retry Logic

**Behavior:**
- Applies to every LLM call (Intake, Researcher, Writer, Validator)
- 3 attempts maximum
- Exponential backoff: 2s after first failure, 4s after second, 8s after third
- Retries on: network errors (timeout, rate limit, 503 cold start) AND content validation failures (LLM returned text that doesn't match expected Pydantic model)
- On retry, print: `[Retry] Attempt 2/3 for [AgentName] — retrying in 4s...`
- On all retries exhausted: print clear error identifying which agent failed and exit gracefully

### MCP Message Protocol

**Envelope structure (every message, no exceptions):**
```python
{
    "protocol_version": "1.0",
    "sender": str,       # e.g., "Orchestrator", "ResearcherAgent"
    "content": Any,      # payload — shape varies per agent
    "metadata": dict     # e.g., {"task_id": "...", "source": "..."}
}
```

**Envelope validation:** every message must have all four keys, must be a dict, sender must be non-empty string.

**Content shapes per agent (validated by Pydantic):**

| From → To | Content Shape |
|-----------|---------------|
| User → Intake | raw string (CLI input) |
| Intake → Orchestrator | `BlogSpec(topic, audience, tone, goal, constraints)` |
| Orchestrator → Researcher | `ResearchRequest(topic, audience, goal)` |
| Researcher → Orchestrator | `ResearchSummary(topic, bullet_points, source)` |
| Orchestrator → Writer | `WriterInput(research_summary, blog_spec, revision_feedback=None)` |
| Writer → Orchestrator | `BlogDraft(title, body, word_count)` |
| Orchestrator → Validator | `ValidationInput(research_summary, draft)` |
| Validator → Orchestrator | `ValidationVerdict(verdict: "pass"\|"fail", reason)` |

### Simulated Knowledge Base

**4-5 hardcoded topics with substantial content (3-4 paragraphs each) so the Researcher has enough material to synthesize:**
- Mediterranean diet
- Artificial intelligence
- Climate change
- Space exploration
- Mental health

**On topic not found:** Researcher returns a ResearchSummary with bullet_points indicating no information was found. Pipeline continues — Writer works with whatever it gets.

## Detailed Specifications

### MCP Message Factory

**Purpose:** Single source of truth for creating MCP-compliant messages.

**Interface:** Accepts sender (str), content (Any), optional metadata (dict). Returns a dict with all four MCP fields.

**Behavior:**
- Always sets protocol_version to "1.0"
- If metadata is None, defaults to empty dict
- Never uses mutable default argument for metadata

### MCP Envelope Validator

**Purpose:** Validates that any dict conforms to the MCP envelope schema.

**Behavior:**
- Checks message is a dict
- Checks all four required keys exist: protocol_version, sender, content, metadata
- Checks sender is a non-empty string
- Returns True/False with logged details on failure

**Error Scenarios:**

| Condition | Expected Behavior |
|-----------|-------------------|
| Message is not a dict (e.g., None from failed LLM call) | Log "MCP Validation Failed: Message is not a dictionary", return False |
| Missing key | Log "MCP Validation Failed: Missing key '{key}' in message from {sender}", return False |
| Empty sender | Log "MCP Validation Failed: Empty sender field", return False |

### Per-Agent Content Validator

**Purpose:** Validates that an agent's input or output content matches its expected Pydantic model.

**Behavior:**
- Each agent defines its input and output Pydantic models
- Before an agent processes input: validate content against input model
- After an agent produces output: validate content against output model
- On validation failure: triggers retry (same as network failure retry path)

### Async Retry Handler

**Purpose:** Wraps any async LLM call with retry logic and exponential backoff.

**Interface:** Takes an async callable, max retries (default 3), base delay (default 2s).

**Behavior:**
- On success: return result immediately
- On failure (network or content validation): log attempt number, wait exponential delay, retry
- Delays: 2s, 4s, 8s (base_delay * 2^attempt)
- On final failure: raise a clear exception with agent name and failure reason
- Print retry status: `[Retry] Attempt {n}/3 for {agent_name} — retrying in {delay}s...`

### Intake Agent

**Purpose:** Decompose raw user text into structured BlogSpec.

**System prompt design (L4-L5):**
- Role: "You are a content planning specialist"
- Task: "Extract the blog specification from the user's request"
- Output format: explicit JSON schema matching BlogSpec fields
- Constraints: "If a field is not mentioned, use a sensible default. Do not invent topics not implied by the request."

**Input:** raw string from CLI
**Output:** BlogSpec Pydantic model (topic, audience, tone, goal, constraints)

**Default values when user doesn't specify:**
- audience: "general readers"
- tone: "informative and engaging"
- goal: "educate the reader"
- constraints: empty list

### Researcher Agent

**Purpose:** Look up topic in simulated knowledge base and synthesize into bullet points.

**System prompt design (L4-L5):**
- Role: "You are a research analyst"
- Task: "Synthesize the provided information into 3-4 concise, factual bullet points"
- Output format: explicit bullet point format
- Constraints: "Only use information from the provided source material. Do not add external knowledge. Do not speculate."

**Input:** ResearchRequest (from Orchestrator)
**Output:** ResearchSummary (topic, bullet_points list, source indicator)

**Knowledge base lookup:** case-insensitive fuzzy match on topic against dictionary keys. Falls back to "No information found" message.

### Writer Agent

**Purpose:** Transform research into a polished blog post.

**System prompt design (L4-L5):**
- Role: "You are a skilled content writer for a health and wellness blog"
- Task: "Write a short, engaging blog post (approximately 150-200 words) with a catchy title based on the research points provided"
- Output format: title on first line, then body
- Constraints: "Only use facts from the provided research. Do not add claims not supported by the research summary. Match the requested tone and audience."

**First pass input:** research summary + blog spec
**Revision pass input:** research summary + blog spec + validator feedback
**Output:** BlogDraft (title, body, word_count)

### Validator Agent

**Purpose:** Fact-check the Writer's draft against the Researcher's summary.

**System prompt design (L4-L5):**
- Role: "You are a meticulous fact-checker"
- Task: "Determine if every claim in the DRAFT is supported by the SOURCE SUMMARY"
- Output format: JSON with verdict ("pass" or "fail") and reason
- Constraints: "Respond ONLY with the JSON. If all claims are supported, verdict is 'pass'. If any claim is unsupported or fabricated, verdict is 'fail' with a specific explanation of what's wrong."

**Input:** ValidationInput (research_summary + draft)
**Output:** ValidationVerdict (verdict: Literal["pass", "fail"], reason: str)

### Orchestrator

**Purpose:** Central coordinator. Receives structured spec from Intake, delegates to agents in sequence, manages the validation loop, handles failures.

**Behavior flow:**
1. Receive BlogSpec from Intake
2. Create MCP message to Researcher with ResearchRequest
3. Validate Researcher's response (envelope + content)
4. If Researcher response invalid after retries → exit gracefully
5. Create MCP message to Writer with WriterInput (research + spec)
6. Validate Writer's response (envelope + content)
7. If Writer response invalid after retries → exit gracefully
8. Create MCP message to Validator with ValidationInput (research + draft)
9. Validate Validator's response (envelope + content)
10. If verdict is "pass" → return the blog draft
11. If verdict is "fail" and revisions remaining → go to step 5 with feedback appended
12. If max revisions (3) exhausted → print failure message, return to CLI prompt

**Hub-and-spoke enforcement:** The Orchestrator is the ONLY component that calls agents. Agents receive MCP messages and return MCP messages. No agent references or calls another agent.

## Key Constraints

| Constraint | Why It Matters |
|------------|----------------|
| All inter-agent communication goes through MCP envelopes | Without this, the system devolves into ad-hoc function calls and you lose traceability, validation boundaries, and the ability to swap agents |
| Agents never communicate directly — only through Orchestrator | Hub-and-spoke keeps complexity O(n) not O(n^2). Adding a new agent means one new connection, not N new connections |
| Every MCP message validated at every boundary | A malformed message from one agent silently corrupts all downstream agents. Validation catches this at the source |
| Per-agent Pydantic models for content | LLMs (especially smaller free models) frequently return unexpected formats. Without content validation, garbage propagates silently |
| Max 3 revisions on validation loop | Without a cap, a struggling LLM could loop forever. Every feedback loop needs a ceiling |
| No draft output on validation failure | Outputting an unvalidated draft undermines the Validator's purpose and teaches bad habits for production systems |
| Async throughout — no blocking LLM calls | I/O-bound calls blocking the event loop is the #1 performance anti-pattern in Python AI systems. Building the async habit now pays off when you add parallel agents later |
| L4-L5 prompt design for every agent | The whole point of Chapter 1. Vague prompts defeat the learning objective |

## Edge Cases & Failure Modes

| Scenario | Decision | Rationale |
|----------|----------|-----------|
| Empty or whitespace-only CLI input | Re-prompt immediately with "Please provide a blog topic" | Nothing should enter the pipeline without valid input |
| Input exceeds 500 characters | Re-prompt with character limit message | Prevents blowing up LLM context window on small HF models |
| Topic not found in simulated knowledge base | Pipeline continues — Researcher returns "no info found" summary | Lets user see how the system handles sparse data; Writer works with whatever it gets |
| LLM returns empty string | Treated as content validation failure, triggers retry | Empty output is never valid for any agent |
| LLM returns text that doesn't parse into expected Pydantic model | Triggers retry with exponential backoff (2s, 4s, 8s) | Smaller HF models frequently need multiple attempts for structured output |
| All 3 retries exhausted for any agent | Print error identifying which agent failed, exit gracefully | User needs to know where the pipeline broke |
| MCP envelope validation fails | Log which field is missing and which agent produced it, exit gracefully | Structural failures indicate a code bug, not a transient issue |
| Validator returns "fail" on all 3 revision attempts | Print "Failed to produce validated content after 3 revisions", return to CLI prompt | Don't output unvalidated content |
| Writer produces identical bad output across revisions | Max revision cap handles this — exits after 3 attempts | Feedback injection gives Writer a chance to improve, but cap prevents infinite loops |
| HuggingFace 503 (model cold start) | Retry with backoff handles this naturally | Cold starts are transient — model loads within seconds |
| HuggingFace 429 (rate limit) | Retry with backoff handles this naturally | Exponential backoff is the standard pattern for rate limits |
| Validator gives ambiguous verdict | Pydantic model enforces verdict is Literal["pass", "fail"] — ambiguity eliminated | Structured output beats substring matching |

## Decisions Log

| # | Decision | Alternatives Considered | Chosen Because |
|---|----------|------------------------|----------------|
| 1 | Single unified CLI app for both chapters | Separate apps per chapter | Natural progression — Ch2 extends Ch1. One app shows the evolution |
| 2 | Blog generation domain only | Meeting analysis + blog generation | User preference — blog generation covers all Ch1 & Ch2 concepts |
| 3 | Pydantic structured intake instead of SRL visualizer | Explicit SRL decomposition step with linguistic labels | Production systems use structured input parsing, not linguistic SRL. Captures the lesson (structure beats ambiguity) in a transferable way |
| 4 | L4-L5 principles applied in prompts, not demonstrated as 5 levels side-by-side | Side-by-side mode showing L1 through L5 | Applying the principles is more valuable than demonstrating them academically |
| 5 | HuggingFace free Inference API via LangChain | Direct OpenAI API (book default) | User preference — free tier. Rate limits make retry logic genuinely useful |
| 6 | Async throughout, no multi-threading | Sync with threading, or mixed | Pipeline is sequential — async handles I/O-bound LLM calls correctly. Threading adds complexity with no parallel work to justify it |
| 7 | Per-agent Pydantic content validation | Envelope-only validation (as in book) | Production pattern — explicit contracts between agents. Catches format issues from smaller models. Prepares for future chapters adding more agents |
| 8 | 3 retries with exponential backoff (2s, 4s, 8s) | Fixed delay (book default: 3 retries, 5s fixed) | Exponential backoff is the industry standard — avoids hammering a recovering API |
| 9 | Retry on content validation failure (same path as network failure) | Fail immediately on bad content | Smaller HF models frequently produce format issues on first attempt but succeed on retry |
| 10 | Max 3 revision attempts (bumped from book's 2) | Keep book's default of 2 | HF free models are less capable than GPT-4; extra revision improves success rate |
| 11 | No output on validation loop exhaustion | Output last draft with warning | Unvalidated output undermines the Validator pattern |
| 12 | Interactive CLI loop with verbose output | Single-shot CLI; quiet with --verbose flag | Learning/practice app — seeing the pipeline work is the point |
| 13 | 4-5 simulated KB topics | Single topic (book default) | Interactive loop needs variety to stay useful across multiple requests |
| 14 | Hardcoded simulated dictionary for Researcher | Local JSON/YAML file | Stays true to the book. Future chapters will upgrade this |

## Scope Boundaries

### In Scope
- Interactive CLI loop with input validation
- Intake agent with Pydantic structured output (BlogSpec)
- MCP message envelope: create, validate at every boundary
- Per-agent Pydantic content models for inputs and outputs
- Orchestrator with hub-and-spoke coordination
- Researcher agent with simulated dictionary knowledge base (4-5 topics)
- Writer agent with L4-L5 engineered prompts
- Validator agent with structured pass/fail verdict
- Self-correcting revision loop (max 3 revisions)
- Async LLM calls via LangChain + HuggingFace
- Retry with exponential backoff on network and content validation failures
- Verbose step-by-step CLI output
- Graceful error handling throughout

### Out of Scope
- Meeting analysis pipeline (not needed — blog generation covers all concepts)
- SRL visualizer / matplotlib stemma diagrams (replaced by structured intake)
- Five-level side-by-side demonstration mode (L4-L5 applied, not demonstrated)
- Multi-threading (no parallel agents in current scope)
- Real knowledge base / vector DB / RAG (future chapters)
- UI of any kind (practice phase — CLI only)
- Persistent state between sessions
- Logging to file (CLI print is sufficient for practice)
- Authentication or API key management beyond basic env var
- Unit tests (can be added separately if desired)

## Dependencies

### Depends On (must exist before this work starts)
- [uv](https://docs.astral.sh/uv/) installed globally
- HuggingFace API token — user needs a free account and token set as environment variable
- Python 3.11+ (managed by uv)
- Dependencies installed via `uv add`: langchain, langchain-huggingface, pydantic, pytest, pytest-asyncio

### Depended On By (other work waiting for this)
- Chapter 3+ implementations — this system is the foundation that future chapters will extend (real RAG replacing simulated DB, additional agents, etc.)

## Architecture Notes

**Hub-and-spoke pattern:** The Orchestrator is the single coordinator. Every agent is a standalone async function that takes an MCP message and returns an MCP message. This uniformity is what makes agents composable and swappable.

**Two-layer validation pattern:** MCP envelope validation happens at every message boundary (generic, same check everywhere). Per-agent Pydantic content validation happens at each agent's input and output (specific, different model per agent). Both must pass before the pipeline continues.

**Retry scope:** The async retry handler wraps individual agent calls, not the entire pipeline. If the Researcher fails after 3 retries, the pipeline exits — it doesn't retry from the beginning.

**Context chaining realized:** Each agent's output becomes the next agent's input, mediated by the Orchestrator. This is Chapter 1's context chaining implemented as a system, not just sequential prompt calls.

**Revision loop data flow:** On revision, the Writer receives the original research summary + blog spec + the Validator's feedback reason. This is context accumulation — each iteration adds information, not replaces it.

## Open Questions (if any)

- **Which specific HuggingFace model to use** — depends on what's available and performing well on the free tier at implementation time. Suggest trying Mistral 7B Instruct first, falling back to Llama 3.1 8B if needed.
  - **Impact if unresolved:** Minor — can be swapped via LangChain config without code changes.
  - **Suggested default:** `mistralai/Mistral-7B-Instruct-v0.3` or latest available.

---

# Tasks

## Task T1: Project Setup + MCP Protocol + Pydantic Models

> **Status:** done
> **Effort:** m
> **Priority:** critical
> **Depends on:** None

### Description

Scaffold the Python project with pyproject.toml, create the package structure (`src/blog_mas/`, `tests/`), and implement the MCP message protocol layer. This includes the message factory that creates compliant envelopes, the envelope validator that checks every message boundary, and all Pydantic content models that define the typed contracts between agents. Every subsequent task depends on this layer.

### Test Plan

#### Test File(s)
- `tests/test_mcp_protocol.py`

#### Test Scenarios

##### MCP Message Factory

- **creates valid envelope with all four fields** — GIVEN sender "TestAgent" and content {"key": "value"} WHEN create_mcp_message is called THEN returned dict has keys protocol_version="1.0", sender="TestAgent", content={"key": "value"}, metadata={}
- **defaults metadata to empty dict when not provided** — GIVEN only sender and content WHEN create_mcp_message is called THEN metadata is an empty dict
- **sets protocol_version to "1.0" always** — GIVEN any call WHEN create_mcp_message is called THEN protocol_version is always "1.0"
- **does not share mutable default across calls** — GIVEN two calls to create_mcp_message without metadata WHEN the first result's metadata is mutated THEN the second result's metadata is still an empty dict (no aliasing)
- **accepts custom metadata** — GIVEN sender, content, and metadata={"task_id": "abc"} WHEN create_mcp_message is called THEN metadata equals {"task_id": "abc"}

##### MCP Envelope Validator

- **returns True for valid envelope** — GIVEN a dict with all four MCP fields and non-empty sender WHEN validate_mcp_envelope is called THEN returns True
- **returns False when message is None** — GIVEN None WHEN validate_mcp_envelope is called THEN returns False and logs "Message is not a dictionary"
- **returns False when message is a string** — GIVEN "not a dict" WHEN validate_mcp_envelope is called THEN returns False and logs "Message is not a dictionary"
- **returns False when protocol_version key missing** — GIVEN envelope without protocol_version WHEN validate_mcp_envelope is called THEN returns False and logs missing key
- **returns False when sender key missing** — GIVEN envelope without sender WHEN validate_mcp_envelope is called THEN returns False and logs missing key
- **returns False when content key missing** — GIVEN envelope without content WHEN validate_mcp_envelope is called THEN returns False and logs missing key
- **returns False when metadata key missing** — GIVEN envelope without metadata WHEN validate_mcp_envelope is called THEN returns False and logs missing key
- **returns False when sender is empty string** — GIVEN envelope with sender="" WHEN validate_mcp_envelope is called THEN returns False and logs "Empty sender field"

##### Pydantic Content Models

- **BlogSpec creates with all fields** — GIVEN topic="AI", audience="tech", tone="neutral", goal="inform", constraints=["no jargon"] WHEN BlogSpec is instantiated THEN all fields are set correctly
- **BlogSpec rejects missing topic** — GIVEN no topic WHEN BlogSpec is instantiated THEN raises ValidationError
- **ResearchRequest creates with required fields** — GIVEN topic, audience, goal WHEN ResearchRequest is instantiated THEN fields are set
- **ResearchSummary creates with bullet_points and source** — GIVEN topic, bullet_points=["point1", "point2"], source="kb" WHEN ResearchSummary is instantiated THEN fields are set
- **WriterInput defaults revision_feedback to None** — GIVEN research_summary and blog_spec only WHEN WriterInput is instantiated THEN revision_feedback is None
- **WriterInput accepts optional revision_feedback** — GIVEN research_summary, blog_spec, and revision_feedback="fix claims" WHEN WriterInput is instantiated THEN revision_feedback is "fix claims"
- **BlogDraft creates with title, body, word_count** — GIVEN title="Test", body="Content here", word_count=2 WHEN BlogDraft is instantiated THEN fields are set
- **ValidationInput creates with research_summary and draft** — GIVEN a ResearchSummary and a BlogDraft WHEN ValidationInput is instantiated THEN fields are set
- **ValidationVerdict accepts verdict "pass"** — GIVEN verdict="pass", reason="all good" WHEN ValidationVerdict is instantiated THEN no error
- **ValidationVerdict accepts verdict "fail"** — GIVEN verdict="fail", reason="unsupported claim" WHEN ValidationVerdict is instantiated THEN no error
- **ValidationVerdict rejects other verdict values** — GIVEN verdict="maybe" WHEN ValidationVerdict is instantiated THEN raises ValidationError

### Implementation Notes

- **Layer:** Protocol/contract layer — foundation for all inter-agent communication
- **Pattern reference:** Pydantic v2 BaseModel with Literal types for constrained strings
- **Key decisions:** Decision #7 (per-agent Pydantic content validation), Decision #1 (single unified app)
- **Libraries:** pydantic>=2.0 (added via `uv add pydantic`)
- **Folder structure:** `src/blog_mas/mcp/protocol.py` (factory + validator), `src/blog_mas/mcp/models.py` (all Pydantic models)
- **Project init:** `uv init --package blog-mas` in the project root, then `uv add pydantic langchain langchain-huggingface pytest pytest-asyncio`

### Scope Boundaries

- Do NOT implement any agent logic — this is purely the message protocol and data contracts
- Do NOT import LangChain or any LLM libraries — models are plain Pydantic, no LLM calls
- Do NOT add serialization/deserialization beyond what Pydantic provides natively
- Only implement the message factory, envelope validator, and the 7 content models listed in the MCP Protocol section

### Files Expected

**New files:**
- `pyproject.toml` (generated by `uv init`, dependencies added via `uv add`)
- `uv.lock` (auto-generated by uv)
- `src/blog_mas/__init__.py`
- `src/blog_mas/mcp/__init__.py`
- `src/blog_mas/mcp/protocol.py` (create_mcp_message, validate_mcp_envelope)
- `src/blog_mas/mcp/models.py` (BlogSpec, ResearchRequest, ResearchSummary, WriterInput, BlogDraft, ValidationInput, ValidationVerdict)
- `tests/__init__.py`
- `tests/conftest.py` (empty initially, shared fixtures added as needed)
- `tests/test_mcp_protocol.py`

**Modified files:**
- None (first task)

**Must NOT modify:**
- `specs/` directory

---

## Task T2: Async Retry Handler

> **Status:** done
> **Effort:** s
> **Priority:** critical
> **Depends on:** None

### Description

Implement a generic async retry handler that wraps any async callable (LLM calls) with exponential backoff. This handles both network failures (timeouts, rate limits, 503 cold starts) and content validation failures (LLM returning unparseable output). Every agent uses this handler, making it a critical shared utility.

### Test Plan

#### Test File(s)
- `tests/test_retry_handler.py`

#### Test Scenarios

##### Successful Calls

- **returns result immediately on first success** — GIVEN an async callable that succeeds WHEN retry_handler is called THEN result is returned with no retry attempts
- **works with async callables** — GIVEN an async function returning a value WHEN retry_handler awaits it THEN the result is the awaited value

##### Retry Behavior

- **retries once and succeeds on second attempt** — GIVEN a callable that fails once then succeeds WHEN retry_handler is called THEN it returns the second result after one retry
- **retries with correct exponential delays** — GIVEN a callable that fails 3 times WHEN retry_handler is called THEN delays between retries are 2s, 4s, 8s (base_delay * 2^attempt)
- **raises exception after all retries exhausted** — GIVEN a callable that always fails WHEN retry_handler is called THEN it raises a clear exception after 3 total attempts

##### Error Types

- **handles network-style errors** — GIVEN a callable raising ConnectionError/TimeoutError WHEN retry_handler is called THEN it retries with backoff
- **handles content validation errors** — GIVEN a callable raising a ValidationError (Pydantic parse failure) WHEN retry_handler is called THEN it retries with backoff

##### Output

- **prints retry status on each attempt** — GIVEN a callable that fails once for agent "TestAgent" WHEN retry happens THEN stdout contains "[Retry] Attempt 2/3 for TestAgent — retrying in 4s..."

### Implementation Notes

- **Layer:** Utility layer — stateless async function
- **Pattern reference:** Standard exponential backoff: delay = base_delay * (2 ** attempt)
- **Key decisions:** Decision #8 (exponential backoff 2s/4s/8s), Decision #9 (retry on content validation failure same as network)
- **Libraries:** asyncio (asyncio.sleep for delays), pytest-asyncio for testing (added via `uv add --dev pytest-asyncio`)
- **File:** `src/blog_mas/retry.py`

### Scope Boundaries

- Do NOT implement any agent-specific logic — this is a generic wrapper
- Do NOT add jitter to backoff (not in the plan)
- Do NOT log to file — print to stdout only
- Only implement: the retry handler function with configurable max_retries (default 3) and base_delay (default 2)

### Files Expected

**New files:**
- `src/blog_mas/retry.py`
- `tests/test_retry_handler.py`

**Modified files:**
- None

**Must NOT modify:**
- `src/blog_mas/mcp/` (T1 owns this)

### TDD Sequence

1. Test: returns result on first success → implement happy path
2. Test: retries once and succeeds → implement retry loop
3. Test: exponential delays → implement backoff calculation
4. Test: raises after exhaustion → implement exception on final failure
5. Test: error types → ensure both network and validation errors trigger retry
6. Test: prints retry status → add print statements

---

## Task T3: Simulated Knowledge Base

> **Status:** done
> **Effort:** xs
> **Priority:** high
> **Depends on:** None

### Description

Create a hardcoded dictionary knowledge base with 5 substantial topics and a lookup function that supports case-insensitive and fuzzy matching. The Researcher agent depends on this as its data source. This is the simplest standalone task and can be done in parallel with T1 and T2.

### Test Plan

#### Test File(s)
- `tests/test_knowledge_base.py`

#### Test Scenarios

##### Topic Lookup

- **returns content for exact topic match** — GIVEN "Mediterranean diet" WHEN lookup is called THEN returns the full content string for that topic
- **returns content for case-insensitive match** — GIVEN "mediterranean diet" or "MEDITERRANEAN DIET" WHEN lookup is called THEN returns the same content
- **returns content for partial/fuzzy match** — GIVEN "diet" or "climate" WHEN lookup is called THEN returns content for the matching topic (e.g., "Mediterranean diet" or "Climate change")
- **returns None for unknown topic** — GIVEN "quantum physics" (not in KB) WHEN lookup is called THEN returns None

##### Content Requirements

- **contains exactly 5 topics** — WHEN knowledge base is inspected THEN it has 5 entries
- **each topic has substantial content** — WHEN each entry's content is checked THEN every entry has at least 3 paragraphs of text (not empty, not placeholder)

### Implementation Notes

- **Layer:** Data layer — pure data, no LLM calls
- **Pattern reference:** Simple dict with string keys and string values, plus a lookup function
- **Key decisions:** Decision #14 (hardcoded dictionary), Decision #13 (4-5 topics, using 5)
- **Topics (from plan):** Mediterranean diet, Artificial intelligence, Climate change, Space exploration, Mental health
- **File:** `src/blog_mas/knowledge_base.py`

### Scope Boundaries

- Do NOT use external files (JSON/YAML) — hardcoded dict as per Decision #14
- Do NOT add vector search or embedding — plain string matching only
- Do NOT connect to any external data source
- Only implement: the dictionary and a single lookup function

### Files Expected

**New files:**
- `src/blog_mas/knowledge_base.py`
- `tests/test_knowledge_base.py`

**Modified files:**
- None

**Must NOT modify:**
- Any other source files

---

## Task T4: Intake Agent

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T1, T2

### Description

Implement the Intake agent that takes a raw user string from the CLI and decomposes it into a structured BlogSpec using an LLM call with an L4-L5 engineered system prompt. This is the first agent in the pipeline and demonstrates Chapter 1's core lesson: structured prompts produce structured output. Uses the retry handler for resilience.

### Test Plan

#### Test File(s)
- `tests/test_intake_agent.py`

#### Test Scenarios

##### Happy Path

- **decomposes valid request into BlogSpec with all fields** — GIVEN "Write about the Mediterranean diet for athletes" WHEN intake_agent processes it THEN returned content is a valid BlogSpec with topic containing "Mediterranean diet" and audience containing "athletes"
- **applies defaults for unspecified fields** — GIVEN "Write about AI" (no audience, tone, goal specified) WHEN intake_agent processes it THEN returned BlogSpec has audience="general readers", tone="informative and engaging", goal="educate the reader", constraints=[]
- **wraps output in MCP envelope** — GIVEN any valid input WHEN intake_agent returns THEN the result is an MCP envelope with sender="IntakeAgent", content is a BlogSpec, and protocol_version="1.0"

##### Failure and Retry

- **retries on LLM network failure** — GIVEN LLM raises ConnectionError on first call THEN retry_handler kicks in and retries (mock the LLM to fail then succeed)
- **retries on LLM returning unparseable content** — GIVEN LLM returns "not json" (can't parse into BlogSpec) WHEN intake_agent processes it THEN retry is triggered
- **raises clear error after exhausting retries** — GIVEN LLM always fails WHEN all retries are exhausted THEN raises an exception identifying "IntakeAgent" as the failed agent

### Implementation Notes

- **Layer:** Agent layer — async function taking MCP message, returning MCP message
- **Pattern reference:** All agents follow the same pattern: receive MCP message → extract input → call LLM → validate output → return MCP message
- **Key decisions:** Decision #3 (Pydantic structured intake), Decision #4 (L4-L5 applied in prompts)
- **Libraries:** langchain, langchain-huggingface, pydantic
- **System prompt must include:** Role ("content planning specialist"), Task ("extract blog specification"), Output format (explicit JSON schema), Constraints ("use defaults for missing fields")
- **Files:** `src/blog_mas/agents/intake.py`, prompt in `src/blog_mas/prompts.py`

### Scope Boundaries

- Do NOT add multi-turn clarification — single-pass decomposition only
- Do NOT validate input length (that's the CLI's job in T9)
- Do NOT implement the LLM call directly — use LangChain's async interface (ainvoke)
- Only implement: the intake agent function, its system prompt, and input/output wrapping

### Files Expected

**New files:**
- `src/blog_mas/agents/__init__.py`
- `src/blog_mas/agents/intake.py`
- `src/blog_mas/prompts.py` (INTAKE_SYSTEM_PROMPT constant)
- `tests/test_intake_agent.py`

**Modified files:**
- None (imports T1 and T2, doesn't modify them)

**Must NOT modify:**
- `src/blog_mas/mcp/` (T1)
- `src/blog_mas/retry.py` (T2)

### TDD Sequence

1. Test: wraps output in MCP envelope → implement basic function signature + MCP wrapping
2. Test: decomposes valid request → implement LLM call + BlogSpec parsing
3. Test: applies defaults → verify prompt produces defaults
4. Test: retries on network failure → wire retry handler
5. Test: retries on bad content → wire content validation into retry path
6. Test: raises after exhaustion → verify final error behavior

---

## Task T5: Researcher Agent

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T1, T2, T3

### Description

Implement the Researcher agent that looks up a topic in the simulated knowledge base and uses an LLM to synthesize the retrieved information into 3-4 concise factual bullet points. When the topic isn't found, it returns a "no info found" summary and the pipeline continues — it does NOT fail.

### Test Plan

#### Test File(s)
- `tests/test_researcher_agent.py`

#### Test Scenarios

##### Happy Path

- **looks up known topic and returns ResearchSummary with bullet_points** — GIVEN a ResearchRequest with topic="Mediterranean diet" WHEN researcher_agent processes it THEN returned content is a ResearchSummary with a non-empty bullet_points list
- **includes source indicator in output** — GIVEN a known topic WHEN researcher_agent returns THEN ResearchSummary.source indicates "knowledge_base"
- **wraps output in MCP envelope** — GIVEN any valid input WHEN researcher_agent returns THEN result is an MCP envelope with sender="ResearcherAgent"

##### Topic Not Found

- **returns "no info found" summary for unknown topic** — GIVEN a ResearchRequest with topic="quantum physics" WHEN researcher_agent processes it THEN returns ResearchSummary with bullet_points indicating no information was found (pipeline continues, no error raised)

##### Failure and Retry

- **retries on LLM network failure** — GIVEN LLM raises ConnectionError THEN retry_handler retries
- **retries on LLM returning unparseable content** — GIVEN LLM returns garbage THEN retry is triggered
- **raises clear error after exhausting retries** — GIVEN LLM always fails THEN raises exception identifying "ResearcherAgent"

### Implementation Notes

- **Layer:** Agent layer — async function taking MCP message, returning MCP message
- **Pattern reference:** Same agent pattern as T4
- **Key decisions:** Decision #14 (hardcoded KB), plan requirement that pipeline continues on topic not found
- **Libraries:** langchain, langchain-huggingface
- **System prompt must include:** Role ("research analyst"), Task ("synthesize provided information into 3-4 concise factual bullet points"), Output format (bullet points), Constraints ("only use provided source material, do not add external knowledge")
- **Files:** `src/blog_mas/agents/researcher.py`, prompt in `src/blog_mas/prompts.py`

### Scope Boundaries

- Do NOT call external APIs or real databases — simulated KB only
- Do NOT fail the pipeline on topic not found — return "no info" and continue
- Do NOT add vector search — plain string matching via knowledge_base.py
- Only implement: researcher agent function, its system prompt, KB lookup + LLM synthesis

### Files Expected

**New files:**
- `src/blog_mas/agents/researcher.py`
- `tests/test_researcher_agent.py`

**Modified files:**
- `src/blog_mas/prompts.py` (add RESEARCHER_SYSTEM_PROMPT)

**Must NOT modify:**
- `src/blog_mas/mcp/` (T1)
- `src/blog_mas/retry.py` (T2)
- `src/blog_mas/knowledge_base.py` (T3)

---

## Task T6: Writer Agent

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T1, T2

### Description

Implement the Writer agent that transforms research bullet points into a polished blog post with a title. On first pass it receives research + blog spec. On revision passes it additionally receives the Validator's feedback. The system prompt must enforce L4-L5 principles and constrain the Writer to only use facts from the research summary.

### Test Plan

#### Test File(s)
- `tests/test_writer_agent.py`

#### Test Scenarios

##### First Pass

- **generates BlogDraft with title, body, and word_count** — GIVEN WriterInput with research_summary and blog_spec WHEN writer_agent processes it THEN returns a BlogDraft with non-empty title and non-empty body
- **respects tone and audience in prompt** — GIVEN blog_spec with tone="humorous" and audience="children" WHEN writer_agent constructs its prompt THEN the prompt includes the tone and audience parameters

##### Revision Pass

- **incorporates feedback into revised draft** — GIVEN WriterInput with revision_feedback="Remove unsupported claim about X" WHEN writer_agent processes it THEN the prompt includes the feedback and the returned BlogDraft addresses it

##### MCP Wrapping

- **wraps output in MCP envelope** — GIVEN any valid input WHEN writer_agent returns THEN result is an MCP envelope with sender="WriterAgent"

##### Failure and Retry

- **retries on LLM network failure** — GIVEN LLM raises ConnectionError THEN retry_handler retries
- **retries on LLM returning unparseable content** — GIVEN LLM returns garbage THEN retry is triggered
- **raises clear error after exhausting retries** — GIVEN LLM always fails THEN raises exception identifying "WriterAgent"

### Implementation Notes

- **Layer:** Agent layer — async function taking MCP message, returning MCP message
- **Pattern reference:** Same agent pattern as T4/T5
- **Key decisions:** Decision #4 (L4-L5 applied in prompts), plan specifies ~150-200 word output
- **Libraries:** langchain, langchain-huggingface
- **System prompt must include:** Role ("skilled content writer"), Task ("write short engaging blog post 150-200 words with catchy title"), Output format ("title on first line, then body"), Constraints ("only use facts from provided research, match requested tone and audience")
- **Files:** `src/blog_mas/agents/writer.py`, prompt in `src/blog_mas/prompts.py`

### Scope Boundaries

- Do NOT add word count enforcement beyond what the prompt requests — LLM output length is approximate
- Do NOT implement multi-turn editing — single-pass per revision
- Do NOT generate HTML or markdown formatting — plain text output
- Only implement: writer agent function, its system prompt (first pass + revision variants), LLM call

### Files Expected

**New files:**
- `src/blog_mas/agents/writer.py`
- `tests/test_writer_agent.py`

**Modified files:**
- `src/blog_mas/prompts.py` (add WRITER_SYSTEM_PROMPT, WRITER_REVISION_SYSTEM_PROMPT)

**Must NOT modify:**
- `src/blog_mas/mcp/` (T1)
- `src/blog_mas/retry.py` (T2)
- Other agent files

---

## Task T7: Validator Agent

> **Status:** done
> **Effort:** s
> **Priority:** high
> **Depends on:** T1, T2

### Description

Implement the Validator agent that fact-checks the Writer's draft against the Researcher's summary. Returns a structured pass/fail verdict with a reason. The Pydantic model enforces verdict is strictly "pass" or "fail" (Literal type), eliminating ambiguity. This agent drives the self-correcting loop in the Orchestrator.

### Test Plan

#### Test File(s)
- `tests/test_validator_agent.py`

#### Test Scenarios

##### Happy Path

- **returns pass verdict when draft matches research** — GIVEN a ValidationInput where the draft's claims are supported by the research summary WHEN validator_agent processes it THEN returns ValidationVerdict with verdict="pass" and a reason
- **returns fail verdict with specific reason for unsupported claims** — GIVEN a ValidationInput where the draft contains a claim not in the research summary WHEN validator_agent processes it THEN returns ValidationVerdict with verdict="fail" and a reason describing the unsupported claim

##### MCP Wrapping

- **wraps output in MCP envelope** — GIVEN any valid input WHEN validator_agent returns THEN result is an MCP envelope with sender="ValidatorAgent"

##### Constraint Enforcement

- **rejects ambiguous verdict via Pydantic model** — GIVEN the Pydantic model ValidationVerdict WHEN instantiated with verdict="maybe" THEN raises ValidationError (this is a model-level test, not an LLM output test)

##### Failure and Retry

- **retries on LLM network failure** — GIVEN LLM raises ConnectionError THEN retry_handler retries
- **retries on LLM returning unparseable content** — GIVEN LLM returns non-JSON THEN retry is triggered
- **raises clear error after exhausting retries** — GIVEN LLM always fails THEN raises exception identifying "ValidatorAgent"

### Implementation Notes

- **Layer:** Agent layer — async function taking MCP message, returning MCP message
- **Pattern reference:** Same agent pattern as T4-T6
- **Key decisions:** Decision #11 (no output on validation failure), Pydantic Literal enforcement for verdict
- **Libraries:** langchain, langchain-huggingface
- **System prompt must include:** Role ("meticulous fact-checker"), Task ("determine if every claim in DRAFT is supported by SOURCE SUMMARY"), Output format (JSON with verdict and reason), Constraints ("respond ONLY with JSON, pass if all supported, fail with specific explanation if not")
- **Files:** `src/blog_mas/agents/validator.py`, prompt in `src/blog_mas/prompts.py`

### Scope Boundaries

- Do NOT implement scoring or grading — binary pass/fail only
- Do NOT add multiple validation criteria — only fact-checking against research summary
- Do NOT modify the ValidationVerdict model — use the one from T1
- Only implement: validator agent function, its system prompt, LLM call with structured output parsing

### Files Expected

**New files:**
- `src/blog_mas/agents/validator.py`
- `tests/test_validator_agent.py`

**Modified files:**
- `src/blog_mas/prompts.py` (add VALIDATOR_SYSTEM_PROMPT)

**Must NOT modify:**
- `src/blog_mas/mcp/` (T1)
- `src/blog_mas/retry.py` (T2)
- Other agent files

---

## Task T8: Orchestrator (Hub-and-Spoke + Revision Loop)

> **Status:** done
> **Effort:** l
> **Priority:** critical
> **Depends on:** T4, T5, T6, T7

### Description

Implement the central Orchestrator that coordinates all agents in hub-and-spoke fashion. Receives a BlogSpec from Intake, delegates to Researcher → Writer → Validator in sequence, validates every message at every boundary, and manages the self-correcting revision loop (max 3 revisions). On validation pass, returns the draft. On 3 failed revisions, returns failure with no draft. This is the largest task as it wires everything together.

### Test Plan

#### Test File(s)
- `tests/test_orchestrator.py`

#### Test Scenarios

##### Full Happy Path

- **runs complete pipeline and returns blog draft** — GIVEN a valid BlogSpec WHEN orchestrator runs THEN it calls Researcher → Writer → Validator in order, Validator returns "pass", and orchestrator returns the BlogDraft
- **delegates to agents in correct sequence** — GIVEN a valid BlogSpec WHEN orchestrator runs THEN agent call order is exactly: researcher, writer, validator (verify call order via mocks)

##### MCP Validation at Boundaries

- **validates MCP envelope at every message boundary** — GIVEN each agent returns a message WHEN orchestrator receives it THEN validate_mcp_envelope is called on every response
- **validates Pydantic content at every boundary** — GIVEN each agent's response WHEN orchestrator processes it THEN content is validated against the expected Pydantic model

##### Validation Pass

- **returns blog draft immediately on pass** — GIVEN Validator returns verdict="pass" WHEN orchestrator receives it THEN the BlogDraft from Writer is returned (not re-generated)

##### Revision Loop

- **sends feedback to Writer on validation fail** — GIVEN Validator returns verdict="fail" with reason="unsupported claim" WHEN orchestrator processes it THEN Writer is called again with revision_feedback set to the reason
- **accumulates context across revisions** — GIVEN revision attempt 2 WHEN Writer is called THEN it receives original research_summary + blog_spec + revision_feedback (not just feedback alone)
- **respects max 3 revisions** — GIVEN Validator returns "fail" 3 times WHEN orchestrator runs THEN it stops after 3 revision attempts and returns failure
- **returns failure with no draft after 3 failed revisions** — GIVEN all 3 revisions fail WHEN orchestrator completes THEN returns a failure indicator (not a BlogDraft)

##### Agent Failure

- **exits gracefully when any agent fails after retries** — GIVEN Researcher (or any agent) exhausts retries and raises WHEN orchestrator catches it THEN returns a clear error identifying the failed agent (no stack trace)

##### Hub-and-Spoke Enforcement

- **agents never reference each other** — GIVEN the orchestrator code WHEN inspected THEN no agent module imports another agent module — all coordination goes through orchestrator

### Implementation Notes

- **Layer:** Orchestration layer — the hub in hub-and-spoke
- **Pattern reference:** Hub-and-spoke from Chapter 2. Orchestrator is the ONLY component that calls agents.
- **Key decisions:** Decision #11 (no output on validation failure), Decision #10 (max 3 revisions), revision loop data flow from Architecture Notes
- **Libraries:** asyncio, agent modules from T4-T7
- **File:** `src/blog_mas/orchestrator.py`
- **Critical design:** Revision loop sends WriterInput with revision_feedback ONLY on revision passes (None on first pass). Context accumulation: Writer always gets research_summary + blog_spec, plus feedback if revising.

### Scope Boundaries

- Do NOT implement parallel agent execution — sequential pipeline only
- Do NOT add new Pydantic models — use models from T1
- Do NOT implement input validation — that's the CLI (T9)
- Do NOT print to stdout — orchestrator returns results, CLI handles display
- Only implement: the run_pipeline async function that coordinates all agents, validates messages, and manages the revision loop

### Files Expected

**New files:**
- `src/blog_mas/orchestrator.py`
- `tests/test_orchestrator.py`

**Modified files:**
- None (imports agents and MCP modules, doesn't modify them)

**Must NOT modify:**
- `src/blog_mas/mcp/` (T1)
- `src/blog_mas/retry.py` (T2)
- `src/blog_mas/knowledge_base.py` (T3)
- `src/blog_mas/agents/` (T4-T7)

### TDD Sequence

1. Test: runs complete happy path → implement basic sequential flow with mocks
2. Test: correct agent sequence → verify call order
3. Test: validates MCP envelope at boundaries → add envelope validation calls
4. Test: validates content at boundaries → add Pydantic content validation
5. Test: returns draft on pass → implement pass handling
6. Test: sends feedback on fail → implement revision loop
7. Test: accumulates context across revisions → verify WriterInput construction
8. Test: respects max 3 revisions → implement revision cap
9. Test: no draft after 3 failures → implement failure return
10. Test: graceful exit on agent failure → implement exception handling

---

## Task T9: CLI Application (Interactive Loop + Integration)

> **Status:** done
> **Effort:** m
> **Priority:** high
> **Depends on:** T8

### Description

Implement the interactive CLI loop that serves as the user-facing entry point. Prints a welcome message with available KB topics, accepts raw blog requests, validates input (empty, whitespace, length), runs the full pipeline via the Orchestrator with verbose step-by-step output, displays the final blog post or error, and loops until the user types exit/quit. This is the integration task that wires everything into a runnable application.

### Test Plan

#### Test File(s)
- `tests/test_cli.py`

#### Test Scenarios

##### Startup

- **prints welcome message with KB topics on startup** — GIVEN the CLI starts WHEN main loop begins THEN stdout contains a welcome message and lists the 5 knowledge base topics

##### Input Validation

- **rejects empty input** — GIVEN user presses Enter (empty string) WHEN CLI reads input THEN prints "Please provide a blog topic" and re-prompts
- **rejects whitespace-only input** — GIVEN user types "   " (spaces only) WHEN CLI reads input THEN prints "Please provide a blog topic" and re-prompts
- **rejects input over 500 characters** — GIVEN a 501+ character string WHEN CLI reads input THEN prints a message about the character limit and re-prompts
- **accepts valid input** — GIVEN "Write about the Mediterranean diet" WHEN CLI reads input THEN passes it to the Intake agent

##### Pipeline Integration

- **runs full pipeline on valid input** — GIVEN valid input WHEN CLI processes it THEN Intake → Orchestrator → full pipeline runs and returns a blog post
- **displays final blog post with markers** — GIVEN pipeline succeeds WHEN CLI displays output THEN blog post is wrapped in "--- BLOG POST ---" and "--- END ---" markers
- **displays verbose step-by-step output** — GIVEN pipeline is running WHEN each agent activates THEN stdout shows "[AgentName] doing X..." messages
- **displays error on pipeline failure** — GIVEN pipeline fails (agent exhausts retries) WHEN CLI handles the error THEN prints a clear error message with no stack trace

##### Loop Behavior

- **loops back to prompt after completing a request** — GIVEN a successful pipeline run WHEN blog post is displayed THEN CLI prompts for the next request
- **exits cleanly on "quit"** — GIVEN user types "quit" WHEN CLI reads input THEN program exits gracefully
- **exits cleanly on "exit"** — GIVEN user types "exit" WHEN CLI reads input THEN program exits gracefully

### Implementation Notes

- **Layer:** Presentation layer — CLI I/O only, no business logic
- **Pattern reference:** Standard Python CLI loop with input()/print()
- **Key decisions:** Decision #12 (interactive loop with verbose output), input validation rules from Behaviors section
- **Libraries:** asyncio (to run async orchestrator from sync CLI), sys (for exit)
- **File:** `src/blog_mas/main.py`
- **Verbose output:** Each step prints its activity — the agent functions themselves print their status (e.g., "[Researcher] Looking up topic..."), so the CLI mainly handles the outer flow (welcome, prompt, displaying result)
- **Entry point:** `uv run python -m blog_mas` or `uv run blog-mas` (if console script configured in pyproject.toml)

### Scope Boundaries

- Do NOT implement the verbose output inside the CLI — agents print their own status, CLI displays the final result
- Do NOT add argparse or command-line arguments — interactive loop only
- Do NOT persist state between sessions
- Do NOT add logging to file — print only
- Do NOT add --verbose flag — always verbose (this is a learning/practice app)
- Only implement: input loop, validation, pipeline invocation, result display, exit handling

### Files Expected

**New files:**
- `src/blog_mas/main.py`
- `tests/test_cli.py`

**Modified files:**
- `src/blog_mas/__init__.py` (add entry point if needed)

**Must NOT modify:**
- `src/blog_mas/orchestrator.py` (T8)
- `src/blog_mas/agents/` (T4-T7)
- `src/blog_mas/mcp/` (T1)
- `src/blog_mas/retry.py` (T2)
- `src/blog_mas/knowledge_base.py` (T3)

### TDD Sequence

1. Test: prints welcome with topics → implement startup display
2. Test: rejects empty input → implement empty/whitespace check
3. Test: rejects long input → implement 500-char limit
4. Test: accepts valid input → implement handoff to pipeline
5. Test: displays blog post with markers → implement result display
6. Test: displays error without stack trace → implement error handling
7. Test: loops after completion → implement loop structure
8. Test: exits on quit/exit → implement exit handling
