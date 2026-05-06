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


async def async_main(llm=None):
    print_welcome()

    if llm is None:
        llm = create_llm()

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

        result = await run_pipeline_async(raw_input=validated, llm=llm)
        display_result(result)


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
        asyncio.run(async_main())


if __name__ == "__main__":
    main()
