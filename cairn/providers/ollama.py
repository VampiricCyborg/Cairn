"""Ollama provider implementation, for local/air-gapped use."""

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

DEFAULT_MODEL = "qwen2.5-coder:14b"
DEFAULT_BASE_URL = "http://localhost:11434"
_PROVIDER_NAME = "ollama"


class OllamaProvider:
    """Extracts candidate entries with a locally-hosted model through
    Ollama's `/api/chat` endpoint.

    Ollama has no forced tool-use, so the model is asked to conform to
    `candidate_entry_json_schema()` via the `format` field, and its response
    is parsed with the shared repair logic in `_json_schema_common` rather
    than trusted outright.

    The client is injectable for tests; by default an `httpx.Client` is
    created on first use, so a provider that only ever sees unreflectable
    traces never needs a running Ollama server.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        *,
        max_output_tokens: int = 2000,
        deny_globs: list[str] | None = None,
        patterns: list[str] | None = None,
        min_session_turns: int = DEFAULT_MIN_SESSION_TURNS,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
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

        redacted = self._redactor.redact_trace(trace)
        if not is_reflectable(redacted, self.min_session_turns):
            return []

        excerpt = render_salient_excerpt(redacted, find_salient_spans(redacted))
        prompt = build_reflector_prompt(excerpt, known)

        try:
            response = self.client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": candidates_json_schema(),
                    "stream": False,
                    "options": {"num_predict": self.max_output_tokens},
                },
            )
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"could not reach Ollama at {self.base_url} -- is `ollama serve` running?"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"Ollama at {self.base_url} returned HTTP {response.status_code}: {exc}"
            ) from exc

        content = _message_content(response.json())
        raw_candidates = parse_candidates(content, provider=_PROVIDER_NAME)

        evidence = build_evidence(redacted, excerpt)
        return raw_candidates_to_entries(raw_candidates, redacted, evidence, max_candidates)


def _message_content(data: object) -> str:
    """The assistant message text from an `/api/chat` response body, or `""`
    if the shape isn't what's expected -- treated by `parse_candidates` as
    unparsable JSON, not raised."""

    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
    logger.warning(
        "%s: response has no message.content; treating as zero candidates", _PROVIDER_NAME
    )
    return ""
