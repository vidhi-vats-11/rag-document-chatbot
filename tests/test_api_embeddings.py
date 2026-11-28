"""EMBEDDING_PROVIDER=api: embeddings come from an online API (mocked here, no real requests)."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest
from pydantic import SecretStr

from app.core.config import EmbeddingProvider
from app.core.exceptions import LLMNotConfiguredError, LLMRateLimitError
from app.core.container import build_container
from app.services.api_embeddings import ApiSentenceEncoder


class FakeEmbeddingsAPI:
    """Stands in for `client.embeddings`; returns 2-number vectors and records each request."""

    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[list[str]] = []
        self.error = error

    def create(self, model: str, input: list[str]):
        if self.error:
            raise self.error
        self.requests.append(list(input))
        # Return items out of order: the encoder must sort them by `index`.
        items = [SimpleNamespace(index=i, embedding=[3.0, 4.0 * (i + 1)]) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(items)))


def _encoder(settings, api: FakeEmbeddingsAPI) -> ApiSentenceEncoder:
    settings.llm_api_key = SecretStr("test-key")
    encoder = ApiSentenceEncoder("gemini-embedding-001", settings)
    encoder._client = SimpleNamespace(embeddings=api)
    return encoder


def test_vectors_are_normalised_and_kept_in_order(settings):
    api = FakeEmbeddingsAPI()
    vectors = _encoder(settings, api).encode(["a", "b"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert np.allclose(vectors[0], [0.6, 0.8])  # [3, 4] / 5 belongs to the first text


def test_results_without_index_keep_their_order(settings):
    # Gemini's OpenAI-compatible endpoint returns index=None (found with the real API).
    class NoIndexAPI(FakeEmbeddingsAPI):
        def create(self, model, input):
            return SimpleNamespace(data=[SimpleNamespace(index=None, embedding=[1.0, float(i)]) for i in range(len(input))])

    vectors = _encoder(settings, NoIndexAPI()).encode(["a", "b"])
    assert np.allclose(vectors[0], [1.0, 0.0])


def test_many_texts_are_sent_in_batches(settings):
    api = FakeEmbeddingsAPI()
    vectors = _encoder(settings, api).encode([f"t{i}" for i in range(250)], batch_size=500)
    assert len(vectors) == 250
    assert [len(r) for r in api.requests] == [100, 100, 50]  # the API accepts at most 100 at once


def test_rate_limit_becomes_a_friendly_error(settings):
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    api = FakeEmbeddingsAPI(error=openai.RateLimitError("quota", response=response, body=None))
    with pytest.raises(LLMRateLimitError):
        _encoder(settings, api).encode(["a"])


def test_api_mode_needs_a_key(settings):
    settings.llm_api_key = None
    with pytest.raises(LLMNotConfiguredError):
        ApiSentenceEncoder("gemini-embedding-001", settings)


def test_api_mode_turns_re_ranking_off(settings, fake_llm):
    settings.embedding_provider = EmbeddingProvider.API
    settings.llm_api_key = SecretStr("test-key")
    container = build_container(settings, llm=fake_llm, use_configured_llm=False)
    assert container.chat_service.retrieval.reranker is None
    assert type(container.embedder._loader("gemini-embedding-001")).__name__ == "ApiSentenceEncoder"
