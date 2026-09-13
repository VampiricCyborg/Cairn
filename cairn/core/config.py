"""Minimal reader for `.cairn/config.toml`: just enough to pick a provider.

Narrow on purpose: the rest of `config.toml` (redaction, budgets, review
settings) is read by whichever module owns that concern once it needs it,
not centralized here.
"""

import logging
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER_NAME = "mock"


def load_provider_name(cairn_root: Path) -> str:
    """The configured `[provider].name`, or `DEFAULT_PROVIDER_NAME` if
    `config.toml` is missing, unparsable, or doesn't set it."""

    config_path = cairn_root / "config.toml"
    if not config_path.is_file():
        return DEFAULT_PROVIDER_NAME

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("could not parse %s: %s", config_path, exc)
        return DEFAULT_PROVIDER_NAME

    provider = data.get("provider")
    if isinstance(provider, dict):
        name = provider.get("name")
        if isinstance(name, str) and name:
            return name
    return DEFAULT_PROVIDER_NAME
