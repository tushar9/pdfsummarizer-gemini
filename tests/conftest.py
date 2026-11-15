"""Shared test fixtures.

PDFs are generated on the fly with PyMuPDF, so the test suite needs no binary
assets and works offline.
"""

from __future__ import annotations

from collections.abc import Callable

import pymupdf
import pytest

PARAGRAPH = (
    "Retrieval augmented generation combines a document index with a language model. "
    "The index stores embeddings of short passages, and a query is matched against "
    "those embeddings to fetch the most relevant context. The language model then "
    "answers using only the retrieved passages, which keeps answers grounded and "
    "auditable. Chunking strategy matters: chunks that are too small lose context, "
    "while chunks that are too large dilute the signal and waste tokens. "
)


@pytest.fixture(scope="session")
def make_pdf() -> Callable[[list[str]], bytes]:
    """Factory that renders each string as one page of a PDF."""

    def _make(pages: list[str]) -> bytes:
        document = pymupdf.open()
        for body in pages:
            page = document.new_page()
            page.insert_textbox(pymupdf.Rect(60, 60, 535, 780), body, fontsize=10)
        data = document.tobytes()
        document.close()
        return data

    return _make


@pytest.fixture(scope="session")
def text_pdf(make_pdf: Callable[[list[str]], bytes]) -> bytes:
    """Three text pages, long enough to need several chunks at a small budget."""
    return make_pdf([f"Chapter {n}\n\n{PARAGRAPH * 2}" for n in (1, 2, 3)])
