"""LLM factory: creates a ChatOpenAI instance pointed at local LM Studio."""

import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen/qwen3.5-9b"
DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234"


def create_llm(
    model: str | None = None,
    temperature: float = 0.3,
    base_url: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI LLM pointed at local LM Studio.

    Args:
        model: Model name as loaded in LM Studio (defaults to qwen2.5-7b-instruct).
        temperature: Sampling temperature.
        base_url: LM Studio base URL. Falls back to LM_STUDIO_URL env var,
                  then http://127.0.0.1:1234.

    Returns:
        A ChatOpenAI instance ready for .with_structured_output().
    """
    url = (base_url or os.environ.get("LM_STUDIO_URL") or DEFAULT_LM_STUDIO_URL).rstrip("/")
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        base_url=f"{url}/v1",
        api_key="lm-studio",
    )
