"""Tests for cairn.core.config."""

from pathlib import Path

import pytest

from cairn.core.config import (
    DEFAULT_PROVIDER_NAME,
    HumanApprovalRequiredError,
    enforce_human_approval,
    load_provider_name,
)


def test_returns_default_when_config_missing(tmp_path: Path) -> None:
    assert load_provider_name(tmp_path) == DEFAULT_PROVIDER_NAME


def test_reads_configured_provider_name(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('[provider]\nname = "anthropic"\n', encoding="utf-8")

    assert load_provider_name(tmp_path) == "anthropic"


def test_falls_back_to_default_on_malformed_toml(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("not = [valid toml", encoding="utf-8")

    assert load_provider_name(tmp_path) == DEFAULT_PROVIDER_NAME


def test_falls_back_to_default_when_provider_table_missing(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('spec_version = "0.1.0"\n', encoding="utf-8")

    assert load_provider_name(tmp_path) == DEFAULT_PROVIDER_NAME


def test_enforce_human_approval_passes_only_for_boolean_true(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        "[review]\nrequire_human_approval = true\n", encoding="utf-8"
    )

    enforce_human_approval(tmp_path)  # does not raise


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        (None, "does not exist"),
        ("not = [valid toml", "could not be read"),
        ("[review]\n", "not set"),
        ("[review]\nrequire_human_approval = false\n", "is false"),
        ('[review]\nrequire_human_approval = "true"\n', "boolean true"),
    ],
)
def test_enforce_human_approval_refuses_and_says_why(
    tmp_path: Path, contents: str | None, reason: str
) -> None:
    config_path = tmp_path / "config.toml"
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(HumanApprovalRequiredError) as excinfo:
        enforce_human_approval(tmp_path)

    message = str(excinfo.value)
    assert reason in message
    assert "require_human_approval" in message
    assert str(config_path) in message
