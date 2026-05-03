"""CLI: interactive loop for the multi-agent blog generation system."""

import asyncio

from blog_mas.llm import create_llm
from blog_mas.orchestrator import run_pipeline_async

MAX_INPUT_LENGTH = 500


def print_welcome():
    print("=" * 50)
    print("  Multi-Agent Blog Generation System")
    print("=" * 50)
    print()
    print("Available topics:")
    topics = [
        "Mediterranean diet",
        "Artificial intelligence",
        "Climate change",
        "Space exploration",
        "Mental health",
    ]
    for t in topics:
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
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
