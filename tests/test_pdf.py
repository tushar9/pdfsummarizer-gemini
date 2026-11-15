"""Tests for extraction, cleaning and token-budgeted chunking (no network)."""

from __future__ import annotations

import io

import pytest

from pdfsummarizer.config import Settings
from pdfsummarizer.pdf import (
    EmptyDocumentError,
    PdfError,
    PdfProcessor,
    clean_text,
    count_tokens,
)


@pytest.fixture()
def processor() -> PdfProcessor:
    return PdfProcessor(Settings(ocr_enabled=False, max_tokens_per_chunk=200))


def test_extract_reads_all_pages(processor: PdfProcessor, text_pdf: bytes) -> None:
    result = processor.extract(text_pdf)

    assert result.metadata.num_pages == 3
    assert result.text.startswith("Chapter 1")
    assert "auditable" in result.text
    assert result.ocr_pages == ()


def test_extract_accepts_path_and_file_like(
    processor: PdfProcessor, text_pdf: bytes, tmp_path
) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(text_pdf)

    from_path = processor.extract(path)
    from_file = processor.extract(io.BytesIO(text_pdf))

    assert from_path.text == from_file.text == processor.extract(text_pdf).text


def test_metadata_has_page_count(processor: PdfProcessor, text_pdf: bytes) -> None:
    metadata = processor.metadata(text_pdf)

    assert metadata.num_pages == 3
    assert metadata.title == "Unknown"


def test_chunk_respects_token_budget(processor: PdfProcessor, text_pdf: bytes) -> None:
    text = processor.extract(text_pdf).text
    chunks = processor.chunk(text, max_tokens=200)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 200 for chunk in chunks)
    assert "".join(chunks).replace(" ", "").startswith("Chapter1")


def test_chunk_keeps_every_word(processor: PdfProcessor) -> None:
    text = " ".join(f"Sentence number {n} about chunking." for n in range(200))
    chunks = processor.chunk(text, max_tokens=100)

    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 100 for chunk in chunks)
    for chunk in chunks:
        assert chunk.strip()


def test_chunk_empty_text_raises(processor: PdfProcessor) -> None:
    with pytest.raises(EmptyDocumentError):
        processor.chunk("   \n\t ")


def test_invalid_pdf_raises(processor: PdfProcessor) -> None:
    with pytest.raises(PdfError):
        processor.extract(b"not a pdf at all")


def test_clean_text_normalises_whitespace() -> None:
    raw = "Hello   world\r\n\n\n\nnext\x07 line hyph-\nenated"

    assert clean_text(raw) == "Hello world\n\nnext line hyphenated"


def test_ocr_is_not_used_for_pages_with_text(
    processor: PdfProcessor, text_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("OCR should not be called for a text PDF")

    monkeypatch.setattr("pdfsummarizer.ocr.ocr_pages", fail)

    result = PdfProcessor(Settings(ocr_enabled=True)).extract(text_pdf)

    assert result.ocr_pages == ()
    assert result.warnings == ()


def test_missing_tesseract_is_reported_not_raised(
    processor: PdfProcessor, make_pdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    scanned = make_pdf([""])  # a page with no text layer
    monkeypatch.setattr("pdfsummarizer.ocr.is_available", lambda options=None: False)

    result = PdfProcessor(Settings(ocr_enabled=True)).extract(scanned)

    assert result.ocr_pages == ()
    assert any("Tesseract" in warning for warning in result.warnings)


def test_count_tokens_is_positive() -> None:
    assert count_tokens("hello world") > 0
