"""Tests for rag/embedding.py — batched embeddings, retries, adaptive halving."""

import os
import random
from unittest.mock import MagicMock, patch

import pytest

from blog_mas.rag.embedding import EmbeddingClient

DIM = 384


def _fake_vectors(n: int) -> list[list[float]]:
    rng = random.Random(42)
    return [[rng.random() for _ in range(DIM)] for _ in range(n)]


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        super().__init__(message)
        self.status_code = status_code


class FakeSizeError(Exception):
    """Simulates a size-related error without status_code attr."""
    pass


class TestHappyPath:
    def test_embeds_batch_to_vectors(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()
        client._client.feature_extraction.return_value = _fake_vectors(5)

        result = client.embed_batch(["a", "b", "c", "d", "e"])
        assert len(result) == 5
        assert all(len(v) == DIM for v in result)

    def test_empty_batch_returns_empty(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()
        result = client.embed_batch([])
        assert result == []
        client._client.feature_extraction.assert_not_called()


class TestNewlineNormalization:
    def test_replaces_newlines_with_spaces(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()
        client._client.feature_extraction.return_value = _fake_vectors(1)

        client.embed_batch(["hello\nworld\nfoo"])
        call_args = client._client.feature_extraction.call_args[0][0]
        assert call_args == ["hello world foo"]


class TestRetries429:
    def test_retries_on_429_and_succeeds(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()
        err = FakeHTTPError(429)
        client._client.feature_extraction.side_effect = [
            err,
            err,
            _fake_vectors(2),
        ]

        result = client.embed_batch(["a", "b"])
        assert len(result) == 2

    def test_gives_up_after_6_transient_errors(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()
        client._client.feature_extraction.side_effect = ConnectionError("down")

        with pytest.raises(ConnectionError):
            client.embed_batch(["a"])


class TestAdaptiveHalving:
    def test_halves_on_413(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()

        call_count = 0

        def fake_embed(texts, model=None):
            nonlocal call_count
            call_count += 1
            if len(texts) > 1:
                raise FakeHTTPError(413)
            return _fake_vectors(len(texts))

        client._client.feature_extraction = fake_embed

        result = client.embed_batch(["a", "b"])
        assert len(result) == 2
        assert call_count > 1

    def test_halves_recursively(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()

        sizes = []

        def fake_embed(texts, model=None):
            sizes.append(len(texts))
            if len(texts) > 1:
                raise FakeHTTPError(413)
            return _fake_vectors(len(texts))

        client._client.feature_extraction = fake_embed

        result = client.embed_batch(["a", "b", "c", "d"])
        assert len(result) == 4
        assert sizes[0] == 4

    def test_fails_when_single_item_rejected(self):
        client = EmbeddingClient.__new__(EmbeddingClient)
        client._model = "fake"
        client._token = "fake"
        client._client = MagicMock()
        client._client.feature_extraction.side_effect = FakeHTTPError(413)

        with pytest.raises(FakeHTTPError):
            client.embed_batch(["a"])


class TestTokenResolution:
    @patch.dict(os.environ, {"HF_TOKEN": "env-token"})
    def test_reads_hf_token_from_env(self):
        client = EmbeddingClient()
        assert client._token == "env-token"

    def test_constructor_token_overrides_env(self):
        client = EmbeddingClient(token="explicit-token")
        assert client._token == "explicit-token"
