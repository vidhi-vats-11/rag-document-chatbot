"""LLM provider adapters.

Each adapter turns (system prompt, user message) into answer text and maps the
provider SDK's errors onto our own exception types, so the rest of the app
never depends on a specific vendor's SDK.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.core.config import LLMProvider, Settings
from app.core.exceptions import (
    LLMNotConfiguredError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

# "Overloaded" / "unavailable" responses are temporary; tell the user to retry.
_BUSY_STATUSES = {500, 502, 503, 504, 529}
_BUSY_MESSAGE = "The AI service is busy right now. Please try again in a moment."


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_message: str, model: str | None = None) -> str:
        """`model` overrides the configured model for this one request."""
        ...


class AnthropicClient:
    """Claude via the official `anthropic` SDK (Messages API)."""

    def __init__(self, settings: Settings) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,  # SDK retries 408/409/429/5xx with backoff
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._effort = settings.llm_effort
        self._fallbacks = settings.anthropic_enable_fallbacks

    def complete(self, system_prompt: str, user_message: str, model: str | None = None) -> str:
        anthropic = self._anthropic
        model = model or self._model
        kwargs: dict = {
            "model": model,
            "max_tokens": self._max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "output_config": {"effort": self._effort},
        }
        if self._fallbacks:
            # If a safety classifier declines, the API retries on a suitable fallback model.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        try:
            response = self._client.beta.messages.create(**kwargs)
        except anthropic.APITimeoutError as exc:
            raise LLMTimeoutError("The language model took too long to respond. Please try again.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMRateLimitError("The language model is rate-limited. Please wait a moment and retry.") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMNotConfiguredError("The LLM API key was rejected. Check LLM_API_KEY.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMProviderError(f"LLM model '{model}' was not found. Check LLM_MODEL.") from exc
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error %s: %s", exc.status_code, getattr(exc, "type", ""))
            if exc.status_code in _BUSY_STATUSES:
                raise LLMProviderError(_BUSY_MESSAGE) from exc
            raise LLMProviderError("The language model provider returned an error.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMProviderError("Could not reach the language model provider.") from exc

        if response.stop_reason == "refusal":
            raise LLMProviderError("The language model declined to answer this request.")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMProviderError("The language model returned an empty answer.")
        return text


class OpenAICompatibleClient:
    """OpenAI, or any server exposing the OpenAI Chat Completions API (e.g. Ollama)."""

    def __init__(self, settings: Settings) -> None:
        import openai

        self._openai = openai
        self._client = openai.OpenAI(
            # Local servers such as Ollama accept any non-empty key.
            api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed",
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens

    def complete(self, system_prompt: str, user_message: str, model: str | None = None) -> str:
        openai = self._openai
        model = model or self._model
        try:
            response = self._client.chat.completions.create(
                model=model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except openai.APITimeoutError as exc:
            raise LLMTimeoutError("The language model took too long to respond. Please try again.") from exc
        except openai.RateLimitError as exc:
            raise LLMRateLimitError("The language model is rate-limited. Please wait a moment and retry.") from exc
        except openai.AuthenticationError as exc:
            raise LLMNotConfiguredError("The LLM API key was rejected. Check LLM_API_KEY.") from exc
        except openai.NotFoundError as exc:
            raise LLMProviderError(f"LLM model '{model}' was not found. Check LLM_MODEL_CHOICES.") from exc
        except openai.APIStatusError as exc:
            logger.error("OpenAI-compatible API error %s", exc.status_code)
            if exc.status_code in _BUSY_STATUSES:
                raise LLMProviderError(_BUSY_MESSAGE) from exc
            raise LLMProviderError("The language model provider returned an error.") from exc
        except openai.APIConnectionError as exc:
            raise LLMProviderError("Could not reach the language model provider.") from exc

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LLMProviderError("The language model returned an empty answer.")
        return text


def build_llm_client(settings: Settings) -> LLMClient | None:
    """Return a client, or None when the LLM is not configured (chat will report 503)."""
    if not settings.llm_configured:
        logger.warning(
            "LLM is not configured (set LLM_API_KEY). Uploads work; /chat/ask will return 503."
        )
        return None
    if settings.llm_provider is LLMProvider.ANTHROPIC:
        return AnthropicClient(settings)
    return OpenAICompatibleClient(settings)
