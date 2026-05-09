"""CLI: interactive loop + ingestion subcommands."""

import argparse
import asyncio
from pathlib import Path

from blog_mas.llm import create_llm
from blog_mas.logging_config import setup_logging
from blog_mas.orchestrator import run_pipeline_async

MAX_INPUT_LENGTH = 500


def build_parser():
    parser = argparse.ArgumentParser(prog="blog-mas", description="Multi-agent blog generation system")

    # Chapter 4 Context Engine flags (top-level, coexist with subcommands)
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

    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest knowledge markdown files")
    p_ingest.add_argument("--path", default="data/knowledge", help="Source directory")
    p_ingest.add_argument("--rebuild", action="store_true", help="Drop and recreate collection")

    p_bp = sub.add_parser("ingest-blueprints", help="Ingest blueprint JSON files")
    p_bp.add_argument("--rebuild", action="store_true", help="Drop and recreate collection")

    p_eval = sub.add_parser("eval", help="Run recall evaluation")
    p_eval.add_argument("--queries", default=None, help="Path to queries JSON")

    return parser


KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"


def get_available_topics() -> list[str]:
    return sorted(
        p.stem.replace("-", " ").title()
        for p in KNOWLEDGE_DIR.glob("*.md")
    )


def print_welcome():
    print("=" * 50)
    print("  Multi-Agent Blog Generation System")
    print("=" * 50)
    print()
    print("Available topics:")
    for t in get_available_topics():
        print(f"  - {t}")
    print()
    print("Type 'exit' or 'quit' to end the session.")
    print()


def validate_input(user_input: str) -> str | None:
    if not user_input or not user_input.strip():
        print("Please provide a blog topic")
        return None
    if len(user_input) > MAX_INPUT_LENGTH:
        print(
            f"Input too long ({len(user_input)} characters). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )
        return None
    return user_input


def should_exit(user_input: str) -> bool:
    return user_input.strip().lower() in ("exit", "quit")


def display_result(result: dict):
    if result["success"]:
        draft = result["draft"]
        print()
        print("--- BLOG POST ---")
        print(f"Title: {draft.title}")
        print()
        print(draft.body)
        print("--- END ---")
        print()
    else:
        print(f"\nError: {result['error']}\n")


async def process_request(raw_input: str, llm=None) -> dict:
    """Run the full LangGraph pipeline for a single user request."""
    if llm is None:
        llm = create_llm()
    return await run_pipeline_async(raw_input=raw_input, llm=llm)


async def async_main(llm=None, store=None, embedder=None, use_engine=False, save_trace_dir=None):
    print_welcome()

    if llm is None:
        llm = create_llm()
    if store is None:
        from blog_mas.rag.vector_store import QdrantStore
        store = QdrantStore()
    if embedder is None:
        from blog_mas.rag.embedding import EmbeddingClient
        embedder = EmbeddingClient()

    if use_engine:
        await _engine_loop(llm, store, embedder, save_trace_dir)
    else:
        await _default_loop(llm, store, embedder)


async def _default_loop(llm, store, embedder):
    """Existing LangGraph orchestrator interactive loop."""
    while True:
        try:
            user_input = input("Blog request > ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if should_exit(user_input):
            print("Goodbye!")
            break

        validated = validate_input(user_input)
        if validated is None:
            continue

        result = await run_pipeline_async(raw_input=validated, llm=llm, store=store, embedder=embedder)
        display_result(result)


async def _engine_loop(llm, store, embedder, save_trace_dir):
    """Chapter 4 Context Engine interactive loop."""
    from blog_mas.engine.agent_adapters import build_default_registry
    from blog_mas.engine.context_engine import run_context_engine

    registry = build_default_registry(llm, store, embedder, reranker=None)

    while True:
        try:
            user_input = input("Engine goal > ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if should_exit(user_input):
            print("Goodbye!")
            break

        validated = validate_input(user_input)
        if validated is None:
            continue

        final, trace = await run_context_engine(
            goal=validated, registry=registry, llm=llm,
            save_trace_dir=save_trace_dir,
        )
        _display_engine_result(final, trace)


def _display_engine_result(final, trace):
    """Print the Context Engine result summary."""
    print()
    print("--- CONTEXT ENGINE RESULT ---")
    if trace.plan:
        print(f"Plan ({len(trace.plan)} steps):")
        for step in trace.plan:
            print(f"  Step {step['step']}: {step['agent']}")
    print(f"Status: {trace.status}")
    print(f"Duration: {trace.duration:.2f}s")
    if final is not None:
        if hasattr(final, "title") and hasattr(final, "body"):
            print()
            print("Title:", final.title)
            print()
            print(final.body)
        else:
            print("Output:", final)
    print("--- END ---")
    print()


def main():
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        from blog_mas.rag.ingest_cli import cmd_ingest
        cmd_ingest(args)
    elif args.command == "ingest-blueprints":
        from blog_mas.rag.ingest_cli import cmd_ingest_blueprints
        cmd_ingest_blueprints(args)
    elif args.command == "eval":
        from blog_mas.rag.ingest_cli import cmd_eval
        cmd_eval(args)
    else:
        asyncio.run(async_main(
            use_engine=args.engine,
            save_trace_dir=args.save_trace,
        ))


if __name__ == "__main__":
    main()
