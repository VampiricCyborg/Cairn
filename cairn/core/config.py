"""Minimal reader for `.cairn/config.toml`: the provider name and the
human-approval gate.

Narrow on purpose: the rest of `config.toml` (redaction, budgets) is read by
whichever module owns that concern once it needs it, not centralized here.
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


class HumanApprovalRequiredError(Exception):
    """Raised when something would write to `entries/` but `config.toml` does
    not set `[review] require_human_approval = true`."""


def _human_approval_problem(config_path: Path) -> str | None:
    """Why `[review].require_human_approval` is not enabled in `config_path`,
    or `None` if it is the boolean `true`."""

    if not config_path.is_file():
        return "config.toml does not exist"

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return f"config.toml could not be read ({exc})"

    review = data.get("review")
    if not isinstance(review, dict) or "require_human_approval" not in review:
        return "[review] require_human_approval is not set"

    value = review["require_human_approval"]
    if value is True:
        return None
    if value is False:
        return "[review] require_human_approval is false"
    return f"[review] require_human_approval must be the boolean true, not {value!r}"


def enforce_human_approval(cairn_root: Path) -> None:
    """Refuse, with `HumanApprovalRequiredError`, unless `config.toml` sets
    `[review] require_human_approval = true`.

    v0 has no auto-approval mode: `false`, an absent key, a non-boolean
    value, or a missing or unreadable `config.toml` all refuse, because the
    only supported way into `entries/` is an explicit human decision. Called
    by every module that writes the trusted store (see
    `tests/test_store_write_surface.py`), before it shows or touches
    anything.
    """

    config_path = cairn_root / "config.toml"
    problem = _human_approval_problem(config_path)
    if problem is not None:
        raise HumanApprovalRequiredError(
            f"refusing to write to entries/: {problem}. Set `require_human_approval = true` "
            f"under [review] in {config_path}; v0 has no auto-approval mode, so nothing may "
            "reach the trusted store without it."
        )
