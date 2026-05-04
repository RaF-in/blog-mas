"""Orchestrator: LangGraph StateGraph with revision loop."""

import asyncio
import logging

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from blog_mas.agents.intake import intake_node
from blog_mas.agents.researcher import research_node
from blog_mas.agents.validator import validate_node
from blog_mas.agents.writer import write_node
from blog_mas.state import BlogState

logger = logging.getLogger(__name__)

MAX_REVISIONS = 3


def should_continue(state: BlogState) -> str:
    """Conditional edge: retry write if validation failed, otherwise end."""
    verdict = state.get("verdict")
    if verdict is None or verdict.verdict == "pass":
        return "end"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "end"
    logger.info(
        "[Orchestrator] Requesting revision "
        "(attempt %d of %d)...",
        state["revision_count"], MAX_REVISIONS,
    )
    return "retry"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph blog generation pipeline."""
    graph = StateGraph(BlogState)

    graph.add_node("intake", intake_node)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "validate")
    graph.add_conditional_edges(
        "validate",
        should_continue,
        {"retry": "write", "end": END},
    )

    return graph.compile()


def run_pipeline(
    blog_spec=None,
    raw_input: str | None = None,
    llm=None,
    max_retries: int = 3,
    base_delay: float = 2,
):
    """Sync wrapper. Delegates to async."""
    return asyncio.run(
        run_pipeline_async(
            blog_spec=blog_spec,
            raw_input=raw_input,
            llm=llm,
        )
    )


async def run_pipeline_async(
    blog_spec=None,
    raw_input: str | None = None,
    llm=None,
) -> dict:
    """Run the full pipeline via LangGraph and return a result dict."""
    if raw_input is not None and not raw_input.strip():
        return {"success": False, "error": "Input cannot be empty"}

    if blog_spec is None and (raw_input is None or not raw_input.strip()):
        return {"success": False, "error": "Input cannot be empty"}

    graph = build_graph()

    initial_state = {
        "raw_input": raw_input or "",
        "blog_spec": blog_spec,
        "research_summary": None,
        "draft": None,
        "verdict": None,
        "revision_feedback": None,
        "revision_count": 0,
    }

    config = {"configurable": {"llm": llm}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except ValidationError as e:
        logger.error("Parse error: %s", e)
        return {"success": False, "error": "Failed to parse agent output"}
    except RuntimeError as e:
        logger.error("LLM call failed: %s", e)
        return {"success": False, "error": "LLM call failed after retries"}
    except ValueError as e:
        logger.error("Pipeline configuration error: %s", e)
        return {"success": False, "error": "Pipeline configuration error"}
    except ConnectionError as e:
        logger.error("Network error: %s", e)
        return {"success": False, "error": "Network error. Please try again."}
    except Exception:
        logger.exception("Unexpected pipeline error")
        return {"success": False, "error": "Unexpected error"}

    verdict = final_state.get("verdict")
    draft = final_state.get("draft")

    if verdict and verdict.verdict == "pass" and draft:
        logger.info("[Orchestrator] Validation PASSED.")
        return {"success": True, "draft": draft}

    error_msg = (
        f"Failed to produce validated content after {MAX_REVISIONS} revisions. "
        "Please try again."
    )
    logger.warning(error_msg)
    return {"success": False, "error": error_msg}
