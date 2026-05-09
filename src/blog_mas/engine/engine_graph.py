"""LangGraph StateGraph for the Context Engine: Plan → Validate → Execute (loop) → Finalize."""

import logging

from langgraph.graph import END, StateGraph

from blog_mas.engine.engine_state import EngineState
from blog_mas.engine.mcp_envelope import create_mcp_message
from blog_mas.engine.planner import plan as planner_plan
from blog_mas.engine.resolver import resolve_dependencies
from blog_mas.engine.tracer import ExecutionTrace
from blog_mas.engine.validators import validate_plan

logger = logging.getLogger(__name__)


# ── Node functions ──────────────────────────────────────────────────────────


async def plan_node(state: EngineState, config) -> dict:
    """Generate a JSON execution plan from the user's goal."""
    llm = config["configurable"]["llm"]
    registry = config["configurable"]["registry"]

    trace = ExecutionTrace(state["goal"])
    capabilities = registry.get_capabilities_description()

    logger.info("[Engine: Planner] Analyzing goal and generating execution plan...")
    plan = await planner_plan(state["goal"], capabilities, llm)
    trace.log_plan(plan)
    logger.info("[Engine: Planner] Plan generated with %d steps.", len(plan))

    return {"plan": plan, "trace": trace, "current_step": 0, "step_outputs": {}}


async def validate_plan_node(state: EngineState, config) -> dict:
    """Validate the plan structure. Catches errors to route to handle_error."""
    registry = config["configurable"]["registry"]

    try:
        validate_plan(state["plan"], registry)
        return {}
    except ValueError as e:
        logger.error("[Engine: Validator] Plan validation failed: %s", e)
        return {"error": f"Plan validation failed: {e}"}


async def execute_step_node(state: EngineState, config) -> dict:
    """Execute a single plan step. Increments current_step and logs to trace."""
    registry = config["configurable"]["registry"]
    step_idx = state["current_step"]
    step = state["plan"][step_idx]

    step_num = step["step"]
    agent_name = step["agent"]
    planned_input = step["input"]

    logger.info("[Engine: Executor] Starting Step %d: %s", step_num, agent_name)
    try:
        handler = registry.get_handler(agent_name)
        resolved_input = resolve_dependencies(planned_input, state["step_outputs"])
        mcp_in = create_mcp_message("Engine", resolved_input)
        mcp_out = await handler(mcp_in)
        output = mcp_out["content"]

        new_outputs = {**state["step_outputs"], f"STEP_{step_num}_OUTPUT": output}

        trace = state["trace"]
        trace.log_step(step_num, agent_name, planned_input, resolved_input, output)

        logger.info("[Engine: Executor] Step %d completed.", step_num)
        return {"current_step": step_idx + 1, "step_outputs": new_outputs, "trace": trace}
    except Exception as e:
        logger.exception("[Engine: Executor] Step %d (%s) failed: %s", step_num, agent_name, e)
        return {"error": f"Failed at Step {step_num}: {e}"}


async def finalize_node(state: EngineState, config) -> dict:
    """Extract final output and finalize the trace."""
    plan = state["plan"]
    final_output = state["step_outputs"].get(f"STEP_{len(plan)}_OUTPUT")

    trace = state["trace"]
    trace.finalize("Success", final_output)
    logger.info("=== [Context Engine] Task Complete ===")

    return {"final_output": final_output, "trace": trace}


async def handle_error_node(state: EngineState, config) -> dict:
    """Finalize the trace with error status."""
    trace = state["trace"]
    error = state.get("error", "Unknown error")
    trace.finalize(error)
    logger.error("=== [Context Engine] Task Failed: %s ===", error)

    return {"trace": trace}


# ── Conditional edge functions ──────────────────────────────────────────────


def route_after_validate(state: EngineState) -> str:
    if state.get("error"):
        return "handle_error"
    return "execute_step"


def route_after_execute(state: EngineState) -> str:
    if state.get("error"):
        return "handle_error"
    if state["current_step"] < len(state["plan"]):
        return "execute_step"
    return "finalize"


# ── Graph builder ───────────────────────────────────────────────────────────


def build_engine_graph() -> StateGraph:
    """Build and compile the Context Engine LangGraph.

    Topology:
        plan → validate_plan → execute_step ↔ execute_step (loop)
                           ↘ handle_error → END
                                               ↘ finalize → END
    """
    graph = StateGraph(EngineState)

    graph.add_node("plan", plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("execute_step", execute_step_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("handle_error", handle_error_node)

    graph.set_entry_point("plan")

    graph.add_edge("plan", "validate_plan")

    graph.add_conditional_edges(
        "validate_plan",
        route_after_validate,
        {"execute_step": "execute_step", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "execute_step",
        route_after_execute,
        {"execute_step": "execute_step", "finalize": "finalize", "handle_error": "handle_error"},
    )

    graph.add_edge("finalize", END)
    graph.add_edge("handle_error", END)

    return graph.compile()
