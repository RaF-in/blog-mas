"""Executor: walks a JSON plan, resolving $$STEP_N_OUTPUT$$ references and calling each agent."""

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
