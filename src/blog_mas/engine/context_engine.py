"""Context Engine: top-level entry point powered by LangGraph StateGraph."""

import logging

from blog_mas.engine.engine_graph import build_engine_graph
from blog_mas.engine.engine_state import EngineState

logger = logging.getLogger(__name__)


async def run_context_engine(
    goal: str,
    registry,
    llm,
    *,
    save_trace_dir: str | None = None,
):
    """Plan -> validate -> execute -> trace via LangGraph.

    Returns (final_output, trace). final_output is None on failure;
    trace.status describes what happened.
    """
    logger.info("=== [Context Engine] Starting New Task === Goal: %s", goal)

    graph = build_engine_graph()

    initial_state: EngineState = {
        "goal": goal,
        "plan": None,
        "current_step": 0,
        "step_outputs": {},
        "trace": None,
        "final_output": None,
        "error": None,
    }

    config = {
        "configurable": {
            "llm": llm,
            "registry": registry,
        },
    }

    final_state = await graph.ainvoke(initial_state, config=config)

    trace = final_state["trace"]
    final_output = final_state.get("final_output")

    if save_trace_dir and trace:
        trace.save(save_trace_dir)

    return final_output, trace
