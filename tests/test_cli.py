"""CLI tests: argument handling and the config check, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfsummarizer import cli


def test_check_config_masks_the_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    key = "live-" + "k" * 40
    monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.setattr("pdfsummarizer.config.load_dotenv", lambda *args, **kwargs: False)

    assert cli.main(["--check-config"]) == 0

    output = capsys.readouterr().out
    assert "api_key_usable" in output
    assert "True" in output
    assert key not in output


def test_check_config_fails_without_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("pdfsummarizer.config.load_dotenv", lambda *args, **kwargs: False)

    assert cli.main(["--check-config"]) == 1

    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_missing_path_is_a_usage_error(capsys) -> None:
    assert cli.main([]) == 2
    assert "PDF path is required" in capsys.readouterr().err


def test_unknown_file_is_reported(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "nope.pdf"

    assert cli.main([str(missing)]) == 2
    assert "file not found" in capsys.readouterr().err


def test_broken_pdf_is_reported(tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"definitely not a pdf")

    assert cli.main([str(broken)]) == 1
    assert "Could not read the PDF" in capsys.readouterr().err
