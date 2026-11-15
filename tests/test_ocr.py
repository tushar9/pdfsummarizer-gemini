"""OCR fallback tests.

The ``ocr`` marker is skipped automatically when Tesseract is not installed, so
the suite still runs on machines that only have the Python dependencies.
"""

from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from pdfsummarizer import ocr
from pdfsummarizer.config import Settings
from pdfsummarizer.pdf import PdfProcessor

OCR_TEXT = "The quick brown fox jumps over the lazy dog"

tesseract_available = ocr.is_available()
requires_ocr = pytest.mark.skipif(not tesseract_available, reason="Tesseract is not installed")


def _font(size: int = 44) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _scanned_page(text: str) -> bytes:
    """A page that is only an image, i.e. no text layer at all."""
    image = Image.new("RGB", (1240, 400), "white")
    ImageDraw.Draw(image).text((60, 120), text, fill="black", font=_font())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _scanned_pdf(lines: list[str]) -> bytes:
    document = pymupdf.open()
    for line in lines:
        page = document.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(40, 100, 555, 266), stream=_scanned_page(line))
    data = document.tobytes()
    document.close()
    return data


@requires_ocr
@pytest.mark.ocr
def test_scanned_page_is_recovered_by_ocr() -> None:
    result = PdfProcessor(Settings(ocr_enabled=True)).extract(_scanned_pdf([OCR_TEXT]))

    assert result.ocr_pages == (0,)
    assert "quick brown fox" in result.text.lower()


@requires_ocr
@pytest.mark.ocr
def test_ocr_page_cap_is_reported() -> None:
    settings = Settings(ocr_enabled=True, ocr_max_pages=1)
    result = PdfProcessor(settings).extract(_scanned_pdf([OCR_TEXT, OCR_TEXT]))

    assert result.ocr_pages == (0,)
    assert any("OCR_MAX_PAGES" in warning for warning in result.warnings)


def test_ocr_can_be_disabled() -> None:
    result = PdfProcessor(Settings(ocr_enabled=False)).extract(_scanned_pdf([OCR_TEXT]))

    assert result.ocr_pages == ()
    assert result.text == ""


def test_failing_page_becomes_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pdfsummarizer.ocr.is_available", lambda options=None: True)
    monkeypatch.setattr("pdfsummarizer.ocr.ocr_pages", lambda doc, idx, opts: ({}, ["boom"]))

    result = PdfProcessor(Settings(ocr_enabled=True)).extract(_scanned_pdf([OCR_TEXT]))

    assert result.ocr_pages == ()
    assert result.warnings == ("boom",)


def test_explicit_missing_binary_is_not_available() -> None:
    options = ocr.OcrOptions(tesseract_cmd="C:/definitely/not/here/tesseract.exe")

    assert ocr.find_tesseract(options.tesseract_cmd) is None
    assert ocr.is_available(options) is False
