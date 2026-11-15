"""UI smoke tests with Streamlit's AppTest: no browser, no network."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-" + "x" * 32)
    instance = AppTest.from_file(APP, default_timeout=60)
    instance.run()
    return instance


def test_app_boots_without_exception(app: AppTest) -> None:
    assert not app.exception
    assert "PDF Summarizer" in app.title[0].value


def test_sidebar_exposes_every_control(app: AppTest) -> None:
    assert [box.label for box in app.sidebar.selectbox] == ["Model", "Summary style"]
    assert [box.label for box in app.sidebar.checkbox] == [
        "Analyze document structure",
        "Extract key quotes",
        "OCR scanned pages",
    ]
    assert [slider.label for slider in app.sidebar.slider] == ["Max tokens per chunk"]


def test_model_choices_come_from_settings(app: AppTest) -> None:
    model = app.sidebar.selectbox[0]

    assert "gemini-3.1-flash-lite" in model.options
    assert model.value == "gemini-3.1-flash-lite"


def test_missing_key_shows_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("pdfsummarizer.config.load_dotenv", lambda *args, **kwargs: False)

    instance = AppTest.from_file(APP, default_timeout=60)
    instance.run()

    assert not instance.exception
    assert any("GEMINI_API_KEY is missing" in error.value for error in instance.error)
    assert not instance.sidebar.selectbox
