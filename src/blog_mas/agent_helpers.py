"""Shared helpers for agent nodes: chain invocation with retry and feedback sanitization."""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from blog_mas.retry import retry_handler
from blog_mas.tokens import count_tokens

logger = logging.getLogger(__name__)

# Warn when a single prompt exceeds this many tokens — not a hard block,
# just the "fuel gauge" alarm from Chapter 5 §C.
_TOKEN_WARN_THRESHOLD = 8_000

_DIRECTIVE_PATTERNS = re.compile(
    r"^(Ignore|System:|You are|NEW INSTRUCTION|Forget|Disregard)",
    re.IGNORECASE,
)


async def run_agent_chain(
    config: dict,
    model_cls: type[BaseModel],
    system_prompt: str,
    user_message: str,
    agent_name: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> BaseModel:
    llm = config.get("configurable", {}).get("llm")
    if llm is None:
        raise ValueError(f"[{agent_name}] No LLM configured")

    parser = PydanticOutputParser(pydantic_object=model_cls)
    chain = llm | parser
    full_prompt = system_prompt + "\n\n" + parser.get_format_instructions()
    messages = [
        SystemMessage(content=full_prompt),
        HumanMessage(content=user_message),
    ]

    # Chapter 5 §C — pre-flight token check.  We count the combined prompt so
    # the logs show exactly how much "fuel" each agent call consumes.
    prompt_tokens = count_tokens(full_prompt + user_message)
    logger.info("[%s] Prompt tokens: %d", agent_name, prompt_tokens)
    if prompt_tokens > _TOKEN_WARN_THRESHOLD:
        logger.warning(
            "[%s] Large prompt: %d tokens (threshold=%d). "
            "Consider adding a Summarizer step upstream.",
            agent_name, prompt_tokens, _TOKEN_WARN_THRESHOLD,
        )

    result = await retry_handler(
        lambda: chain.ainvoke(messages), agent_name, max_retries, base_delay
    )

    # Log output size so we can measure total consumption per step.
    output_tokens = count_tokens(str(result))
    logger.info("[%s] Output tokens: %d", agent_name, output_tokens)

    return result


def sanitize_feedback(feedback: str) -> str:
    lines = feedback.splitlines()
    cleaned = [line for line in lines if not _DIRECTIVE_PATTERNS.match(line.strip())]
    return "\n".join(cleaned)
