"""CLI: interactive loop + ingestion subcommands."""

import argparse
import asyncio
from pathlib import Path

from blog_mas.llm import create_llm
from blog_mas.logging_config import setup_logging
from blog_mas.orchestrator import run_pipeline_async

MAX_INPUT_LENGTH = 500

# Chapter 6 demo — Juno probe text (mirrors the book's §3 Code Block 6 example).
# Embedded here so `--demo-summarizer` works without user-supplied text.
_JUNO_DEMO_TEXT = """\
Juno is a NASA space probe orbiting the planet Jupiter. It was launched from Cape
Canaveral Air Force Station on August 5, 2011, as part of the New Frontiers program.
The spacecraft entered Jupiter orbit on July 4, 2016, after a five-year journey.
Juno's primary mission is to investigate Jupiter's origins, interior structure, deep
atmosphere and magnetosphere. The probe carries nine scientific instruments: a microwave
radiometer (MWR) to probe below the cloud tops, a magnetometer (MAG), a gravity science
radio instrument, an ultraviolet spectrograph (UVS), an infrared auroral mapper (JIRAM),
the Jovian auroral distributions experiment (JADE) and three other particle detectors.
A key discovery came in 2021 when Juno confirmed the existence of persistent cyclones at
both poles, each thousands of kilometres wide. Measurements of Jupiter's gravitational
field revealed an unexpectedly deep atmosphere where jet streams extend to depths of
3,000 kilometres. The mission was extended twice — first in 2021 and again in 2025 —
to allow observations of Ganymede, Europa, and Io. Total mission cost is approximately
$1.1 billion. The probe completes one orbit every 53 days and transmits ~40 GB of data
per orbit, all processed by the Deep Space Network.
"""

_JUNO_DEMO_GOAL = (
    "First, summarize the following text about the Juno probe to extract only the key "
    "scientific mission facts and instruments. Then write a short, suspenseful blog post "
    "about the probe's dangerous arrival at Jupiter for a general audience.\n\n"
    "--- TEXT TO USE ---\n" + _JUNO_DEMO_TEXT
)


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

    # Chapter 6 demo flag
    parser.add_argument(
        "--demo-summarizer",
        action="store_true",
        help=(
            "Chapter 6 demo: run a pre-canned Juno-probe goal through the Context Engine "
            "to demonstrate the Summarizer agent and print token-reduction analytics. "
            "Implies --engine."
        ),
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
    """Print the Context Engine result summary.

    Chapter 6 §7 — when a Summarizer step ran, compute_token_savings pulls
    the efficiency metrics from the trace and prints them as a one-liner so
    the business value is immediately visible: "Token reduction: 56.5% ...".
    """
    from blog_mas.tokens import compute_token_savings

    print()
    print("--- CONTEXT ENGINE RESULT ---")
    if trace.plan:
        print(f"Plan ({len(trace.plan)} steps):")
        for step in trace.plan:
            print(f"  Step {step['step']}: {step['agent']}")
    print(f"Status: {trace.status}")
    print(f"Duration: {trace.duration:.2f}s")

    # Chapter 6 §7 — token-reduction analytics (only shown when Summarizer ran)
    savings = compute_token_savings(trace)
    if savings:
        print()
        print("--- TOKEN REDUCTION ANALYTICS ---")
        for s in savings["summarizer_steps"]:
            print(
                f"  Step {s['step']} (Summarizer): "
                f"{s['input_tokens']} → {s['output_tokens']} tokens  "
                f"({s['reduction_percent']:.1f}% reduction)"
            )
        print(
            f"  Overall: {savings['total_input_tokens']} → "
            f"{savings['total_output_tokens']} tokens  "
            f"({savings['overall_reduction_pct']:.1f}% reduction)"
        )
        print("---------------------------------")

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
    elif getattr(args, "demo_summarizer", False):
        # Chapter 6 demo — run the Juno probe goal through the Context Engine.
        # This is the book's §3 Code Block 6/7 made runnable as a CLI flag.
        asyncio.run(_run_demo_summarizer(save_trace_dir=args.save_trace))
    else:
        asyncio.run(async_main(
            use_engine=args.engine,
            save_trace_dir=args.save_trace,
        ))


async def _run_demo_summarizer(save_trace_dir=None):
    """Chapter 6 demo: Summarizer → Writer pipeline on the Juno probe text.

    Mirrors the book's §3 Code Block 6 example:
      - Large technical text supplied directly in the goal
      - Planner recognises it and routes through Summarizer → Writer
      - Token-reduction analytics printed after the run

    Run with:  uv run blog-mas --demo-summarizer
    """
    from blog_mas.engine.agent_adapters import build_default_registry
    from blog_mas.engine.context_engine import run_context_engine
    from blog_mas.rag.embedding import EmbeddingClient
    from blog_mas.rag.vector_store import QdrantStore

    llm = create_llm()
    store = QdrantStore()
    embedder = EmbeddingClient()
    registry = build_default_registry(llm, store, embedder, reranker=None)

    print("=" * 60)
    print("  Chapter 6 Demo — Summarizer Agent")
    print("=" * 60)
    print()
    print("Goal:")
    print(_JUNO_DEMO_GOAL[:300] + "...\n")

    final, trace = await run_context_engine(
        goal=_JUNO_DEMO_GOAL,
        registry=registry,
        llm=llm,
        save_trace_dir=save_trace_dir,
    )
    _display_engine_result(final, trace)


if __name__ == "__main__":
    main()
