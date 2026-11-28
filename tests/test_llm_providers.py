"""Provider adapters: SDK calls are mocked, so no network or API key is needed."""

from types import SimpleNamespace

import anthropic
import pytest

from app.core.config import LLMProvider, Settings
from app.core.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from app.services.llm_providers import (
    AnthropicClient,
    OpenAICompatibleClient,
    build_llm_client,
)


def _settings(**overrides) -> Settings:
    base = dict(_env_file=None, llm_provider=LLMProvider.ANTHROPIC, llm_model="anthropic-test-model",
                llm_api_key="test-key-not-real", llm_max_retries=0)
    return Settings(**{**base, **overrides})


def _fake_create(monkeypatch, client: AnthropicClient, result=None, error=None):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        if error:
            raise error
        return result

    monkeypatch.setattr(client._client.beta.messages, "create", create)
    return captured


def _request():
    import httpx2  # the HTTP library used by anthropic>=1.0

    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_build_llm_client_none_without_key():
    assert build_llm_client(_settings(llm_api_key=None)) is None
    assert build_llm_client(_settings(llm_api_key="  ")) is None


def test_build_llm_client_picks_provider():
    assert isinstance(build_llm_client(_settings()), AnthropicClient)
    openai_settings = _settings(llm_provider=LLMProvider.OPENAI, llm_model="gpt-test")
    assert isinstance(build_llm_client(openai_settings), OpenAICompatibleClient)


def test_openai_compatible_local_server_needs_no_key():
    s = _settings(llm_provider=LLMProvider.OPENAI, llm_api_key=None, llm_base_url="http://localhost:11434/v1")
    assert s.llm_configured
    assert isinstance(build_llm_client(s), OpenAICompatibleClient)


def test_anthropic_success_joins_text_blocks(monkeypatch):
    client = AnthropicClient(_settings())
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Refunds take "),
            SimpleNamespace(type="text", text="30 days [1]."),
        ],
    )
    captured = _fake_create(monkeypatch, client, result=response)

    assert client.complete("sys", "user msg") == "Refunds take 30 days [1]."
    assert captured["model"] == "anthropic-test-model"
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "user msg"}]
    assert captured["output_config"] == {"effort": "low"}
    assert captured["fallbacks"] == "default"


def test_anthropic_fallbacks_can_be_disabled(monkeypatch):
    client = AnthropicClient(_settings(anthropic_enable_fallbacks=False))
    response = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok")])
    captured = _fake_create(monkeypatch, client, result=response)
    client.complete("s", "u")
    assert "fallbacks" not in captured and "betas" not in captured


def test_anthropic_refusal_raises(monkeypatch):
    client = AnthropicClient(_settings())
    _fake_create(monkeypatch, client, result=SimpleNamespace(stop_reason="refusal", content=[]))
    with pytest.raises(LLMProviderError, match="declined"):
        client.complete("s", "u")


def test_anthropic_timeout_maps(monkeypatch):
    client = AnthropicClient(_settings())
    _fake_create(monkeypatch, client, error=anthropic.APITimeoutError(request=_request()))
    with pytest.raises(LLMTimeoutError):
        client.complete("s", "u")


def test_anthropic_rate_limit_maps(monkeypatch):
    import httpx2

    client = AnthropicClient(_settings())
    response = httpx2.Response(429, request=_request())
    _fake_create(monkeypatch, client, error=anthropic.RateLimitError("rate limited", response=response, body=None))
    with pytest.raises(LLMRateLimitError):
        client.complete("s", "u")


def test_anthropic_server_error_maps(monkeypatch):
    import httpx2

    client = AnthropicClient(_settings())
    response = httpx2.Response(500, request=_request())
    _fake_create(monkeypatch, client, error=anthropic.InternalServerError("boom", response=response, body=None))
    with pytest.raises(LLMProviderError, match="busy"):
        client.complete("s", "u")


def test_openai_compatible_success(monkeypatch):
    client = OpenAICompatibleClient(_settings(llm_provider=LLMProvider.OPENAI, llm_model="gpt-test"))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" Answer [1] "))])
    captured = {}
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: captured.update(kw) or response)

    assert client.complete("sys", "usr") == "Answer [1]"
    assert captured["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["model"] == "gpt-test"
