"""Opt-in live test against the real Gemini API.

    python -m pytest -m live

Excluded by default (``addopts = -m 'not live'`` in pyproject.toml).
"""

from __future__ import annotations

import pytest

from pdfsummarizer.config import Settings, load_settings
from pdfsummarizer.pdf import PdfProcessor
from pdfsummarizer.summarize import SummarizerService

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings() -> Settings:
    config = load_settings()
    if not config.has_api_key:
        pytest.skip("GEMINI_API_KEY is not set")
    return config


def test_brief_summary_of_one_chunk(settings: Settings, text_pdf: bytes) -> None:
    processor = PdfProcessor(settings)
    extraction = processor.extract(text_pdf)
    chunks = processor.chunk(extraction.text, max_tokens=2000)

    result = SummarizerService(settings).summarize(
        file_name="sample.pdf",
        text=extraction.text,
        chunks=chunks[:1],
        summary_type="brief",
    )

    assert result.successful_chunks == 1
    assert len(result.combined_summary) > 20
    assert result.model == settings.model
