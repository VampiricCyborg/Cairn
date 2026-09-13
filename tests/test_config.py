"""Tests for cairn.core.config."""

from pathlib import Path

from cairn.core.config import DEFAULT_PROVIDER_NAME, load_provider_name


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
