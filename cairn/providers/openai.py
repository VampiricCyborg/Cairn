"""OpenAI-compatible provider implementation (OpenAI, Groq, Together, etc.)."""

import logging

import httpx

from cairn.core.models import Entry, SessionTrace
from cairn.core.redactor import Redactor
from cairn.core.reflector import build_reflector_prompt
from cairn.core.salience import find_salient_spans, is_reflectable, render_salient_excerpt
from cairn.providers._json_schema_common import (
    DEFAULT_MIN_SESSION_TURNS,
    build_evidence,
    candidates_json_schema,
    parse_candidates,
    raw_candidates_to_entries,
)
from cairn.providers.base import ProviderUnavailableError

logger = logging.getLogger(__name__)

_PROVIDER_NAME = "openai"
_SCHEMA_NAME = "candidate_entries"


class OpenAIProvider:
    """Extracts candidate entries through any OpenAI-compatible
    `/chat/completions` endpoint (OpenAI itself, Groq, Together, etc.).

    `model` and `base_url` are both required: this class's whole point is
    pointing at whichever endpoint the config specifies, so there is no
    sensible default for either. Like `OllamaProvider`, there is no forced
    tool-use here -- the model is asked to conform to
    `candidate_entry_json_schema()` via `response_format`, and its response
    is parsed with the shared repair logic in `_json_schema_common`.

    The client is injectable for tests; by default an `httpx.Client` is
    created on first use.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        max_output_tokens: int = 2000,
        deny_globs: list[str] | None = None,
        patterns: list[str] | None = None,
        min_session_turns: int = DEFAULT_MIN_SESSION_TURNS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.min_session_turns = min_session_turns
        self._redactor = Redactor(deny_globs=deny_globs or [], patterns=patterns or [])
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url)
        return self._client

    def extract(
        self, trace: SessionTrace, known: list[Entry], max_candidates: int
    ) -> list[tuple[Entry, str]]:
        if max_candidates <= 0:
            return []

        if not self.api_key:
            raise ProviderUnavailableError(
                f"no API key configured for {self.base_url} -- set the environment variable "
                "named by [provider].api_key_env in config.toml"
            )

        redacted = self._redactor.redact_trace(trace)
        if not is_reflectable(redacted, self.min_session_turns):
            return []

        excerpt = render_salient_excerpt(redacted, find_salient_spans(redacted))
        prompt = build_reflector_prompt(excerpt, known)

        try:
            response = self.client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_output_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": _SCHEMA_NAME,
                            "schema": candidates_json_schema(),
                            "strict": True,
                        },
                    },
                },
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"could not reach {self.base_url}: {exc}") from exc

        if response.status_code in (401, 403):
            raise ProviderUnavailableError(
                f"{self.base_url} rejected the API key (HTTP {response.status_code}) -- check "
                "the environment variable named by [provider].api_key_env"
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"request to {self.base_url} failed: HTTP {response.status_code}: {exc}"
            ) from exc

        content = _message_content(response.json())
        raw_candidates = parse_candidates(content, provider=_PROVIDER_NAME)

        evidence = build_evidence(redacted, excerpt)
        return raw_candidates_to_entries(raw_candidates, redacted, evidence, max_candidates)


def _message_content(data: object) -> str:
    """The assistant message text from a `/chat/completions` response body,
    or `""` if the shape isn't what's expected -- treated by
    `parse_candidates` as unparsable JSON, not raised."""

    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
    logger.warning(
        "%s: response has no choices[0].message.content; treating as zero candidates",
        _PROVIDER_NAME,
    )
    return ""
