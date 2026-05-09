"""Planner: LLM generates a JSON execution plan from a goal and agent capabilities."""

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
6. Multi-step blog generation typically goes: Intake -> Librarian and Researcher (in any order) -> Writer -> optionally Validator.
7. For "rewrite in a different tone" goals, plan TWO Librarian retrievals (one per blueprint), TWO Writer steps (mode 1 then mode 2), then optionally Validator.
8. TOKEN MANAGEMENT — When the user supplies a large document or text block, or when the Researcher output is likely to be very detailed, insert a Summarizer step between the Researcher and the Writer. Set "summary_objective" to a specific extraction goal, not just "summarize". Pass the Summarizer output to the Writer using the "summary_result" key.
9. Output ONLY a valid JSON list. No prose, no markdown fences.

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

EXAMPLE GOAL: "First, summarize the following text about the Juno probe to extract only the key scientific mission facts and instruments. Then write a short, suspenseful blog post about the probe's arrival at Jupiter for a general audience.\\n\\n--- TEXT TO USE ---\\nJuno is a NASA space probe orbiting Jupiter, launched August 5, 2011. Its mission is to study Jupiter's origins, interior structure, deep atmosphere, and magnetosphere. Juno carries nine scientific instruments including microwave radiometer, magnetometer, and gravity science payload..."
EXAMPLE PLAN:
[
  {{"step": 1, "agent": "Intake", "input": {{"raw_input": "Write a suspenseful blog post about Juno probe's arrival at Jupiter for a general audience."}}}},
  {{"step": 2, "agent": "Librarian", "input": {{"intent_query": "suspenseful science storytelling blueprint"}}}},
  {{"step": 3, "agent": "Summarizer", "input": {{"text_to_summarize": "Juno is a NASA space probe orbiting Jupiter, launched August 5, 2011. Its mission is to study Jupiter's origins, interior structure, deep atmosphere, and magnetosphere. Juno carries nine scientific instruments including microwave radiometer, magnetometer, and gravity science payload...", "summary_objective": "Extract the key scientific mission facts, instrument names, and the most dramatic mission events relevant to the probe's arrival at Jupiter."}}}},
  {{"step": 4, "agent": "Writer", "input": {{"blueprint": "$$STEP_2_OUTPUT$$", "summary_result": "$$STEP_3_OUTPUT$$", "blog_spec": "$$STEP_1_OUTPUT$$.blog_spec"}}}},
  {{"step": 5, "agent": "Validator", "input": {{"draft": "$$STEP_4_OUTPUT$$", "research_summary": "$$STEP_3_OUTPUT$$.summary"}}}}
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
