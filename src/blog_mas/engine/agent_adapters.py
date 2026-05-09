"""Thin adapters: wrap existing LangGraph nodes into (mcp_message) -> mcp_message handlers."""

from blog_mas.agents.intake import intake_node
from blog_mas.agents.librarian import librarian_node
from blog_mas.agents.researcher import research_node
from blog_mas.agents.validator import validate_node
from blog_mas.agents.writer import write_node
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
