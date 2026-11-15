"""Tests for prompt templates and style handling (no network)."""

from __future__ import annotations

import pytest

from pdfsummarizer.config import SUMMARY_TYPES
from pdfsummarizer.prompts import (
    ANALYSIS,
    CHUNK_SUMMARY,
    COMBINE,
    QUOTES,
    STYLES,
    SYSTEM,
    style_for,
)


def test_every_summary_type_has_a_style() -> None:
    assert set(SUMMARY_TYPES) == set(STYLES)
    assert all(instruction.strip() for instruction in STYLES.values())


def test_unknown_style_falls_back_to_detailed() -> None:
    assert style_for("nonsense") == STYLES["detailed"]


@pytest.mark.parametrize("summary_type", SUMMARY_TYPES)
def test_chunk_summary_renders_style_and_text(summary_type: str) -> None:
    messages = CHUNK_SUMMARY.invoke(
        {"style": style_for(summary_type), "text": "THE DOCUMENT"}
    ).to_messages()

    assert messages[0].content == SYSTEM
    assert style_for(summary_type) in messages[1].content
    assert "THE DOCUMENT" in messages[1].content


def test_combine_renders_summaries() -> None:
    messages = COMBINE.invoke({"style": STYLES["brief"], "summaries": "SECTION ONE"}).to_messages()

    assert "SECTION ONE" in messages[1].content
    assert "cohesive" in messages[1].content


def test_analysis_and_quotes_render_text() -> None:
    analysis = ANALYSIS.invoke({"text": "THE DOCUMENT"}).to_messages()
    quotes = QUOTES.invoke({"text": "THE DOCUMENT"}).to_messages()

    assert "THE DOCUMENT" in analysis[1].content
    assert "THE DOCUMENT" in quotes[1].content


def test_system_prompt_forbids_invention() -> None:
    assert "never invent" in SYSTEM
