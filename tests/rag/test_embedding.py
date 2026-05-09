"""Tests for rag/embedding.py — batched embeddings, retries, adaptive halving."""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from blog_mas.rag.embedding import EmbeddingClient

DIM = 384


def _fake_vectors(n: int) -> list[list[float]]:
    import random
    rng = random.Random(42)
    return [[rng.random() for _ in range(DIM)] for _ in range(n)]


def _fake_response(vectors: list[list[float]]) -> MagicMock:
    """Build a fake httpx.Response that looks like an OpenAI embeddings response."""
    resp = MagicMock(spec=httpx.Response)
    data = [{"embedding": v, "index": i} for i, v in enumerate(vectors)]
    resp.json.return_value = {"data": data}
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


class FakeHTTPStatusError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message)
        response = MagicMock()
        response.status_code = status_code
        self.response = response


class TestHappyPath:
    def test_embeds_batch_to_vectors(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()
        client._client.post.return_value = _fake_response(_fake_vectors(5))

        result = client.embed_batch(["a", "b", "c", "d", "e"])
        assert len(result) == 5
        assert all(len(v) == DIM for v in result)

    def test_empty_batch_returns_empty(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()
        result = client.embed_batch([])
        assert result == []
        client._client.post.assert_not_called()


class TestNewlineNormalization:
    def test_replaces_newlines_with_spaces(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()
        client._client.post.return_value = _fake_response(_fake_vectors(1))

        client.embed_batch(["hello\nworld\nfoo"])
        call_args = client._client.post.call_args[1]["json"]["input"]
        assert call_args == ["hello world foo"]


class TestRetries429:
    def test_retries_on_429_and_succeeds(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()
        err = httpx.HTTPStatusError(
            "429",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        client._client.post.side_effect = [
            err,
            err,
            _fake_response(_fake_vectors(2)),
        ]

        result = client.embed_batch(["a", "b"])
        assert len(result) == 2

    def test_gives_up_after_6_transient_errors(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()
        client._client.post.side_effect = ConnectionError("down")

        with pytest.raises(ConnectionError):
            client.embed_batch(["a"])


class TestAdaptiveHalving:
    def test_halves_on_413(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()

        call_count = 0

        def fake_post(url, json=None):
            nonlocal call_count
            call_count += 1
            n = len(json["input"])
            if n > 1:
                raise httpx.HTTPStatusError(
                    "413", request=MagicMock(), response=MagicMock(status_code=413)
                )
            return _fake_response(_fake_vectors(n))

        client._client.post = fake_post

        result = client.embed_batch(["a", "b"])
        assert len(result) == 2
        assert call_count > 1

    def test_halves_recursively(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()

        sizes = []

        def fake_post(url, json=None):
            n = len(json["input"])
            sizes.append(n)
            if n > 1:
                raise httpx.HTTPStatusError(
                    "413", request=MagicMock(), response=MagicMock(status_code=413)
                )
            return _fake_response(_fake_vectors(n))

        client._client.post = fake_post

        result = client.embed_batch(["a", "b", "c", "d"])
        assert len(result) == 4
        assert sizes[0] == 4

    def test_fails_when_single_item_rejected(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._base_url = "http://localhost:1234"
        client._client = MagicMock()
        client._client.post.side_effect = httpx.HTTPStatusError(
            "413", request=MagicMock(), response=MagicMock(status_code=413)
        )

        with pytest.raises(httpx.HTTPStatusError):
            client.embed_batch(["a"])


class TestBaseUrlResolution:
    @patch.dict(os.environ, {"LM_STUDIO_URL": "http://custom:9999"})
    def test_reads_base_url_from_env(self):
        client = EmbeddingClient()
        assert client._base_url == "http://custom:9999"

    def test_constructor_base_url_overrides_env(self):
        client = EmbeddingClient(base_url="http://explicit:5555")
        assert client._base_url == "http://explicit:5555"

    def test_defaults_to_localhost_1234(self):
        with patch.dict(os.environ, {}, clear=True):
            client = EmbeddingClient()
            assert client._base_url == "http://127.0.0.1:1234"
