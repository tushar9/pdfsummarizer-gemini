"""OCR fallback for PDF pages that have no extractable text layer.

Pages are rendered with PyMuPDF (a pure wheel, so no Poppler install) and the
recognition itself is done by Tesseract through ``pytesseract``. When the
Tesseract binary is missing, callers get a clear reason instead of an exception
so a mixed PDF still summarizes its text pages.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

#: Common install locations on Windows; ``shutil.which`` covers PATH installs.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


@dataclass(frozen=True)
class OcrOptions:
    """How much OCR work is allowed for one document."""

    dpi: int = 200
    lang: str = "eng"
    max_pages: int = 30
    tesseract_cmd: str | None = None


def find_tesseract(explicit: str | None = None) -> str | None:
    """Locate the Tesseract executable, preferring an explicit path."""
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("tesseract")
    if found:
        return found
    per_user = Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    for candidate in (*_WINDOWS_CANDIDATES, str(per_user)):
        if Path(candidate).exists():
            return candidate
    return None


def configure(options: OcrOptions) -> str | None:
    """Point pytesseract at the binary and return its path (``None`` if missing)."""
    path = find_tesseract(options.tesseract_cmd)
    if path:
        pytesseract.pytesseract.tesseract_cmd = path
    return path


def is_available(options: OcrOptions | None = None) -> bool:
    """True when the Tesseract binary exists and responds."""
    options = options or OcrOptions()
    if configure(options) is None:
        return False
    try:
        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False
    return True


def render_page(document: pymupdf.Document, index: int, dpi: int) -> Image.Image:
    """Rasterize one page at ``dpi``."""
    pixmap = document[index].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def ocr_page(document: pymupdf.Document, index: int, options: OcrOptions) -> str:
    """OCR a single page; raises whatever pytesseract raises."""
    image = render_page(document, index, options.dpi)
    return pytesseract.image_to_string(image, lang=options.lang).strip()


def ocr_pages(
    document: pymupdf.Document, indexes: list[int], options: OcrOptions
) -> tuple[dict[int, str], list[str]]:
    """OCR the given page indexes, collecting per-page failures as warnings."""
    texts: dict[int, str] = {}
    warnings: list[str] = []
    for index in indexes:
        try:
            texts[index] = ocr_page(document, index, options)
        except Exception as exc:  # noqa: BLE001 - one bad page must not stop the run
            logger.warning("OCR failed on page %d: %s", index + 1, exc)
            warnings.append(f"OCR failed on page {index + 1}: {exc}")
    return texts, warnings
