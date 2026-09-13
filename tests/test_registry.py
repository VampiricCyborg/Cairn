"""Tests for cairn.providers.registry.get_provider."""

import pytest

from cairn.providers.anthropic import AnthropicProvider
from cairn.providers.base import ProviderUnavailableError
from cairn.providers.mock import MockProvider
from cairn.providers.ollama import OllamaProvider
from cairn.providers.openai import OpenAIProvider
from cairn.providers.registry import get_provider


def test_no_provider_table_defaults_to_mock() -> None:
    assert isinstance(get_provider({}), MockProvider)


def test_name_mock_dispatches_to_mock_provider() -> None:
    assert isinstance(get_provider({"provider": {"name": "mock"}}), MockProvider)


def test_name_anthropic_dispatches_when_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    provider = get_provider({"provider": {"name": "anthropic", "model": "claude-sonnet-4-6"}})

    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-sonnet-4-6"


def test_name_anthropic_without_api_key_raises_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ProviderUnavailableError, match="ANTHROPIC_API_KEY"):
        get_provider({"provider": {"name": "anthropic"}})


def test_name_ollama_dispatches_with_config_overrides() -> None:
    provider = get_provider(
        {
            "provider": {
                "name": "ollama",
                "model": "qwen2.5-coder:14b",
                "base_url": "http://localhost:1234",
            },
            "redaction": {"deny_globs": [".env*"], "patterns": ["sk-[A-Za-z0-9]{20,}"]},
            "reflect": {"min_session_turns": 6},
        }
    )

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5-coder:14b"
    assert provider.base_url == "http://localhost:1234"
    assert provider.min_session_turns == 6


def test_name_openai_dispatches_with_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

    provider = get_provider(
        {
            "provider": {
                "name": "openai",
                "model": "llama-3.1-8b-instant",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY",
            }
        }
    )

    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "llama-3.1-8b-instant"
    assert provider.base_url == "https://api.groq.com/openai/v1"
    assert provider.api_key == "groq-test-key"


def test_name_openai_without_model_raises_clearly() -> None:
    with pytest.raises(ProviderUnavailableError, match="model"):
        get_provider({"provider": {"name": "openai", "base_url": "https://api.groq.com/openai/v1"}})


def test_name_openai_without_base_url_raises_clearly() -> None:
    with pytest.raises(ProviderUnavailableError, match="base_url"):
        get_provider({"provider": {"name": "openai", "model": "llama-3.1-8b-instant"}})


def test_unknown_provider_name_raises_clearly() -> None:
    with pytest.raises(ProviderUnavailableError, match="bogus"):
        get_provider({"provider": {"name": "bogus"}})
