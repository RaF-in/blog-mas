"""HF Inference batched embeddings with tenacity retries and adaptive halving.

Single entrypoint every other module uses to produce embeddings — chunking,
retrieval, and ingestion graphs all depend on it.
"""

import logging
import os

from huggingface_hub import InferenceClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_BATCH_SIZE = 100


class TransientError(Exception):
    """Raised for retryable HTTP errors (429, 502, 503)."""
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message)
        self.status_code = status_code


class EmbeddingClient:
    """Wraps HF Inference text-embedding endpoint with retries and adaptive halving."""

    def __init__(
        self,
        model: str | None = None,
        token: str | None = None,
    ) -> None:
        self._model = model or DEFAULT_EMBEDDING_MODEL
        self._token = token or os.environ.get("HF_TOKEN")
        self._client = InferenceClient(token=self._token)

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, TransientError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _call_endpoint(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.feature_extraction(texts, model=self._model)
            return response
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            if status in (429, 502, 503):
                raise TransientError(status, str(exc)) from exc
            raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, handling rate limits and size errors.

        Normalizes newlines to spaces before sending. Retries on transient
        errors (429, connection) with exponential backoff. Adaptively halves
        batch on 413/400 size errors.
        """
        if not texts:
            return []

        normalized = [t.replace("\n", " ") for t in texts]
        return self._embed_with_halving(normalized, _DEFAULT_BATCH_SIZE)

    def _embed_with_halving(
        self, texts: list[str], batch_size: int
    ) -> list[list[float]]:
        results: list[list[float]] = []
        i = 0
        while i < len(texts):
            end = min(i + batch_size, len(texts))
            batch = texts[i:end]
            try:
                vectors = self._call_endpoint(batch)
                if hasattr(vectors, "tolist"):
                    vectors = vectors.tolist()
                results.extend(vectors)
                i = end
            except Exception as exc:
                if self._is_size_error(exc) and len(batch) > 1:
                    half = max(1, len(batch) // 2)
                    logger.debug(
                        "embedding adaptive halving batch_size=%d -> %d",
                        len(batch),
                        half,
                    )
                    results.extend(self._embed_with_halving(batch, half))
                    i = end
                else:
                    raise
        return results

    @staticmethod
    def _is_size_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        if status in (413, 400):
            return True
        msg = str(exc).lower()
        return "too large" in msg or "batch size" in msg or "payload" in msg
