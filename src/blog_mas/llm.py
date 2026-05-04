"""LLM factory: creates a ChatHuggingFace instance for the pipeline."""

import logging
import os

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def create_llm(
    model: str | None = None,
    temperature: float = 0.3,
    api_key: str | None = None,
) -> ChatHuggingFace:
    """Create a ChatHuggingFace LLM for structured output.

    Args:
        model: HuggingFace model repo ID (defaults to Qwen2.5-7B-Instruct).
        temperature: Sampling temperature.
        api_key: HuggingFace API token. Falls back to HF_TOKEN from .env.

    Returns:
        A ChatHuggingFace instance ready for .with_structured_output().
    """
    token = api_key or os.environ.get("HF_TOKEN")
    endpoint = HuggingFaceEndpoint(
        repo_id=model or DEFAULT_MODEL,
        task="text-generation",
        temperature=temperature,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=endpoint)
