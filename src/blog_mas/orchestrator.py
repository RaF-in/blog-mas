"""Orchestrator: LangGraph StateGraph with revision loop."""

from langgraph.graph import END, StateGraph

from blog_mas.agents.intake import intake_node
from blog_mas.agents.researcher import research_node
from blog_mas.agents.validator import validate_node
from blog_mas.agents.writer import write_node
from blog_mas.state import BlogState

MAX_REVISIONS = 3


def should_continue(state: BlogState) -> str:
    """Conditional edge: retry write if validation failed, otherwise end."""
    verdict = state.get("verdict")
    if verdict is None or verdict.verdict == "pass":
        return "end"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "end"
    print(
        f"[Orchestrator] Requesting revision "
        f"(attempt {state['revision_count']} of {MAX_REVISIONS})..."
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
    """Sync wrapper kept for backward-compat with tests. Delegates to async."""
    import asyncio

    return asyncio.get_event_loop().run_until_complete(
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
    graph = build_graph()

    initial_state = {
        "raw_input": raw_input or "",
        "blog_spec": blog_spec,
        "research_summary": None,
        "draft": None,
        "verdict": None,
        "revision_feedback": None,
        "revision_count": 0,
        "error": None,
    }

    config = {"configurable": {"llm": llm}}

    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        return {"success": False, "error": str(e)}

    verdict = final_state.get("verdict")
    draft = final_state.get("draft")

    if verdict and verdict.verdict == "pass" and draft:
        print("[Orchestrator] Validation PASSED.")
        return {"success": True, "draft": draft}

    revision_count = final_state.get("revision_count", 0)
    error_msg = (
        f"Failed to produce validated content after {MAX_REVISIONS} revisions. "
        "Please try again."
    )
    print(error_msg)
    return {"success": False, "error": error_msg}
