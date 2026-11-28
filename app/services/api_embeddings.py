"""Embeddings from an online API (OpenAI-compatible, e.g. Google Gemini) instead of a local model.

Used when EMBEDDING_PROVIDER=api, e.g. on Vercel, where there is no room to download and run
local models. It uses the same provider and key as the LLM (LLM_BASE_URL, LLM_API_KEY).

Trade-off: no model files and almost no memory, but every upload and every question costs an
API request (Gemini counts these separately from answer requests).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np

from app.core.config import Settings
from app.core.exceptions import LLMNotConfiguredError, LLMProviderError, LLMRateLimitError, LLMTimeoutError

logger = logging.getLogger(__name__)

_MAX_TEXTS_PER_REQUEST = 100  # Gemini accepts at most 100 texts in one embedding request


class ApiSentenceEncoder:
    """Same `encode(texts)` method as the local encoder: one L2-normalised vector per text."""

    def __init__(self, model_name: str, settings: Settings) -> None:
        import openai

        if not settings.llm_configured:
            raise LLMNotConfiguredError("EMBEDDING_PROVIDER=api needs LLM_API_KEY (and LLM_BASE_URL) in .env.")
        self._openai = openai
        self._client = openai.OpenAI(
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.model_name = model_name

    def encode(self, inputs: Iterable[str], batch_size: int = 32, **_: Any) -> np.ndarray:
        texts = list(inputs)
        step = max(1, min(batch_size, _MAX_TEXTS_PER_REQUEST))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), step):
            vectors.extend(self._request(texts[start : start + step]))
        array = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.where(norms == 0, 1.0, norms)

    def _request(self, texts: list[str]) -> list[list[float]]:
        openai = self._openai
        try:
            response = self._client.embeddings.create(model=self.model_name, input=texts)
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("The embedding service took too long to respond. Please try again.") from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError("The embedding service is rate-limited. Please wait a moment and retry.") from exc
        except openai.AuthenticationError as exc:
            raise LLMNotConfiguredError("The API key was rejected by the embedding service. Check LLM_API_KEY.") from exc
        except openai.NotFoundError as exc:
            raise LLMProviderError(f"Embedding model '{self.model_name}' was not found. Check EMBEDDING_MODEL.") from exc
        except openai.APIStatusError as exc:
            logger.error("Embedding API error %s", exc.status_code)
            raise LLMProviderError("The embedding service returned an error.") from exc
        except openai.APIConnectionError as exc:
            raise LLMProviderError("Could not reach the embedding service.") from exc
        items = list(response.data)
        # OpenAI numbers each result with `index`; Gemini leaves it empty and keeps the input order.
        if all(isinstance(item.index, int) for item in items):
            items.sort(key=lambda item: item.index)
        if len(items) != len(texts):
            raise LLMProviderError("The embedding service returned the wrong number of vectors.")
        return [item.embedding for item in items]
