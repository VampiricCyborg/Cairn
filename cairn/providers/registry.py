"""Resolves `config.toml`'s `[provider]` table into a `Provider` instance.

Kept apart from `cairn.core.config` (which only reads `[provider].name`, for
callers like `doctor` that don't need a live provider) because this module
does real dispatch work: reading redaction/reflect settings other provider
constructors need, reading API keys from the environment, and raising
`ProviderUnavailableError` for a provider that can't actually be used.
"""

import os
from typing import Any

from cairn.providers.anthropic import AnthropicProvider
from cairn.providers.base import Provider, ProviderUnavailableError
from cairn.providers.mock import MockProvider
from cairn.providers.ollama import OllamaProvider
from cairn.providers.openai import OpenAIProvider

DEFAULT_PROVIDER_NAME = "mock"
_DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
_KNOWN_PROVIDERS = ("anthropic", "openai", "ollama", "mock")


def _table(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def get_provider(config: dict[str, Any]) -> Provider:
    """Dispatch the full parsed `config.toml` document to a `Provider`.

    Reads `[provider].name` to choose the implementation, `[redaction]` and
    `[reflect].min_session_turns` to construct it consistently with the rest
    of the pipeline, and (for `openai`) the environment variable named by
    `[provider].api_key_env`. Raises `ProviderUnavailableError` -- never a
    raw exception from a missing key or an unreachable server -- for a
    provider that is unconfigured, misconfigured, or unreachable; the caller
    decides whether that means failing loudly or falling back.
    """

    provider_cfg = _table(config, "provider")
    name = provider_cfg.get("name") or DEFAULT_PROVIDER_NAME

    if name == "mock":
        return MockProvider()

    redaction_cfg = _table(config, "redaction")
    deny_globs = redaction_cfg.get("deny_globs") or []
    patterns = redaction_cfg.get("patterns") or []
    min_session_turns = _table(config, "reflect").get("min_session_turns")
    max_output_tokens = provider_cfg.get("max_output_tokens")
    model = provider_cfg.get("model")

    common_kwargs: dict[str, Any] = {"deny_globs": deny_globs, "patterns": patterns}
    if isinstance(min_session_turns, int):
        common_kwargs["min_session_turns"] = min_session_turns
    if isinstance(max_output_tokens, int):
        common_kwargs["max_output_tokens"] = max_output_tokens

    if name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderUnavailableError(
                "provider 'anthropic' is configured but ANTHROPIC_API_KEY is not set"
            )
        if isinstance(model, str) and model:
            common_kwargs["model"] = model
        return AnthropicProvider(**common_kwargs)

    if name == "ollama":
        base_url = provider_cfg.get("base_url")
        if isinstance(model, str) and model:
            common_kwargs["model"] = model
        if isinstance(base_url, str) and base_url:
            common_kwargs["base_url"] = base_url
        return OllamaProvider(**common_kwargs)

    if name == "openai":
        if not isinstance(model, str) or not model:
            raise ProviderUnavailableError(
                "provider 'openai' requires [provider].model to be set in config.toml"
            )
        base_url = provider_cfg.get("base_url")
        if not isinstance(base_url, str) or not base_url:
            raise ProviderUnavailableError(
                "provider 'openai' requires [provider].base_url to be set in config.toml"
            )
        api_key_env = provider_cfg.get("api_key_env") or _DEFAULT_API_KEY_ENV
        api_key = os.environ.get(api_key_env)
        return OpenAIProvider(model=model, base_url=base_url, api_key=api_key, **common_kwargs)

    raise ProviderUnavailableError(
        f"unknown provider {name!r} in config.toml; expected one of {', '.join(_KNOWN_PROVIDERS)}"
    )
