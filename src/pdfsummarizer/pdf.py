"""PDF text extraction, cleaning, token counting and chunking.

``pypdf`` handles the text layer, :mod:`pdfsummarizer.ocr` fills in pages that
have none, and LangChain's ``RecursiveCharacterTextSplitter`` splits the result
by *tokens* (measured with tiktoken) so every chunk fits a model's context.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import pymupdf
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from pdfsummarizer import ocr
from pdfsummarizer.config import Settings, load_settings

logger = logging.getLogger(__name__)

#: Pages with less text than this are treated as having no text layer.
MIN_TEXT_LAYER_CHARS = 30

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TRAILING_SPACES = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE = re.compile(r" ?\n ?")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


class PdfError(RuntimeError):
    """Raised when a PDF cannot be read at all."""


class EmptyDocumentError(PdfError):
    """Raised when a PDF yields no usable text (scanned pages, OCR unavailable)."""


@dataclass(frozen=True)
class PdfMetadata:
    num_pages: int
    title: str = "Unknown"
    author: str = "Unknown"
    subject: str = "Unknown"


@dataclass
class ExtractionResult:
    text: str
    metadata: PdfMetadata
    ocr_pages: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def used_ocr(self) -> bool:
        return bool(self.ocr_pages)


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - offline machines have no encoding cache
        logger.warning("tiktoken encoding unavailable (%s); approximating tokens", exc)
        return None


def count_tokens(text: str) -> int:
    """Token count for ``text``; falls back to a rough estimate when offline."""
    encoding = _encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text, disallowed_special=()))


def clean_text(text: str) -> str:
    """Drop control characters, re-join hyphenated line breaks, tidy whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _TRAILING_SPACES.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


class PdfProcessor:
    """Turns a PDF (path, bytes or file-like object) into clean text and chunks."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()

    # -- input handling ---------------------------------------------------
    @staticmethod
    def read(source: str | Path | bytes | bytearray | BinaryIO) -> bytes:
        """Normalize any supported source into PDF bytes."""
        if isinstance(source, (bytes, bytearray)):
            return bytes(source)
        if isinstance(source, (str, Path)):
            return Path(source).read_bytes()
        if hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read()
            if isinstance(data, str):
                raise PdfError("PDF sources must be opened in binary mode")
            return bytes(data)
        raise PdfError(f"Unsupported PDF source: {type(source)!r}")

    @staticmethod
    def _open(data: bytes) -> PdfReader:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                raise PdfError("This PDF is password-protected; remove the password first.")
        except PdfError:
            raise
        except Exception as exc:  # noqa: BLE001 - pypdf raises a wide range
            raise PdfError(f"Could not read the PDF: {exc}") from exc
        return reader

    # -- metadata ---------------------------------------------------------
    def metadata(self, source: str | Path | bytes | bytearray | BinaryIO) -> PdfMetadata:
        reader = self._open(self.read(source))
        info = reader.metadata or {}

        def field(key: str) -> str:
            value = info.get(key)
            return str(value).strip() if value else "Unknown"

        return PdfMetadata(
            num_pages=len(reader.pages),
            title=field("/Title"),
            author=field("/Author"),
            subject=field("/Subject"),
        )

    # -- extraction -------------------------------------------------------
    def extract(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        *,
        ocr_enabled: bool | None = None,
    ) -> ExtractionResult:
        """Extract text, OCR-ing pages that have no text layer when allowed."""
        data = self.read(source)
        reader = self._open(data)

        warnings: list[str] = []
        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            try:
                pages.append(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - skip the page, keep the rest
                logger.warning("Skipping page %d: %s", number, exc)
                warnings.append(f"Could not read page {number}: {exc}")
                pages.append("")

        ocr_texts: dict[int, str] = {}
        scanned = [
            index
            for index, page_text in enumerate(pages)
            if len(page_text.strip()) < MIN_TEXT_LAYER_CHARS
        ]
        enabled = self.settings.ocr_enabled if ocr_enabled is None else ocr_enabled
        if scanned and enabled:
            ocr_texts, ocr_warnings = self._ocr_pages(data, scanned)
            warnings.extend(ocr_warnings)
            for index, text in ocr_texts.items():
                pages[index] = text

        metadata = PdfMetadata(
            num_pages=len(pages),
            title=str((reader.metadata or {}).get("/Title") or "Unknown"),
            author=str((reader.metadata or {}).get("/Author") or "Unknown"),
            subject=str((reader.metadata or {}).get("/Subject") or "Unknown"),
        )
        text = clean_text("\n\n".join(pages))
        return ExtractionResult(
            text=text,
            metadata=metadata,
            ocr_pages=tuple(sorted(ocr_texts)),
            warnings=tuple(warnings),
        )

    def _ocr_pages(self, data: bytes, indexes: list[int]) -> tuple[dict[int, str], list[str]]:
        options = ocr.OcrOptions(
            dpi=self.settings.ocr_dpi,
            lang=self.settings.ocr_lang,
            max_pages=self.settings.ocr_max_pages,
            tesseract_cmd=self.settings.tesseract_cmd,
        )
        if not ocr.is_available(options):
            return {}, [
                "OCR skipped: the Tesseract binary was not found (see README > OCR setup)."
            ]

        targets = indexes[: options.max_pages]
        warnings: list[str] = []
        if len(indexes) > len(targets):
            warnings.append(
                f"OCR limited to the first {len(targets)} scanned pages (OCR_MAX_PAGES)."
            )

        try:
            document = pymupdf.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            return {}, [f"OCR skipped: could not render the PDF ({exc})."]

        try:
            texts, failures = ocr.ocr_pages(document, targets, options)
        finally:
            document.close()
        return texts, warnings + failures

    # -- chunking ---------------------------------------------------------
    def chunk(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Split ``text`` into chunks of at most ``max_tokens`` tokens."""
        if not text.strip():
            raise EmptyDocumentError(
                "No extractable text. The PDF is probably scanned images — enable OCR "
                "or install Tesseract (see README > OCR setup)."
            )
        limit = max_tokens or self.settings.max_tokens_per_chunk
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            chunk_size=limit,
            chunk_overlap=min(200, max(0, limit // 10)),
            length_function=count_tokens,
            keep_separator=True,
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    def prepare(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        *,
        max_tokens: int | None = None,
        ocr_enabled: bool | None = None,
    ) -> tuple[ExtractionResult, list[str]]:
        """Extract and chunk in one call; raises :class:`EmptyDocumentError` if empty."""
        result = self.extract(source, ocr_enabled=ocr_enabled)
        return result, self.chunk(result.text, max_tokens)
