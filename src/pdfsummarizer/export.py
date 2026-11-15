"""Plain-text and CSV rendering of a :class:`~pdfsummarizer.summarize.SummaryResult`.

Deliberately free of Streamlit imports so the CLI, tests and any future web
front-end can reuse it.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from pdfsummarizer.summarize import SummaryResult

CSV_COLUMNS = (
    "Chunk",
    "Status",
    "Original Chars",
    "Summary Chars",
    "Original Tokens",
    "Compression Ratio",
)


def _page_list(pages: tuple[int, ...]) -> str:
    return ", ".join(str(page + 1) for page in pages)


def summary_to_text(result: SummaryResult) -> str:
    """Build the full plain-text report for a finished run."""
    lines = [
        "PDF Summary Report",
        "=" * 18,
        "",
        f"File: {result.file_name}",
        f"Model: {result.model}",
        f"Summary type: {result.summary_type.replace('_', ' ').title()}",
        f"Chunks: {result.total_chunks} "
        f"({result.successful_chunks} succeeded, {result.failed_chunks} failed)",
        f"Processing time: {result.elapsed_seconds}s",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
    ]
    if result.ocr_pages:
        lines.append(f"OCR pages: {_page_list(result.ocr_pages)}")
    if result.warnings:
        lines += ["", "WARNINGS", "-" * 8, *(f"- {warning}" for warning in result.warnings)]

    lines += ["", "FINAL SUMMARY", "=" * 13, "", result.combined_summary]

    if result.analysis is not None:
        heading = "DOCUMENT ANALYSIS" if result.analysis.ok else "DOCUMENT ANALYSIS (failed)"
        lines += ["", heading, "=" * len(heading), "", result.analysis.text]

    if result.quotes:
        lines += ["", "KEY QUOTES", "=" * 10, *(f'- "{quote}"' for quote in result.quotes)]

    lines += ["", "CHUNK SUMMARIES", "=" * 15]
    for chunk in result.chunks:
        suffix = "" if chunk.ok else " (FAILED)"
        lines += ["", f"Chunk {chunk.chunk_number}{suffix}:", chunk.summary]

    return "\n".join(lines).rstrip() + "\n"


def chunk_rows(result: SummaryResult) -> list[dict[str, object]]:
    """One row of per-chunk statistics."""
    return [
        {
            "Chunk": chunk.chunk_number,
            "Status": "Success" if chunk.ok else "Error",
            "Original Chars": chunk.original_chars,
            "Summary Chars": chunk.summary_chars,
            "Original Tokens": chunk.original_tokens,
            "Compression Ratio": (
                round(chunk.summary_chars / chunk.original_chars, 2)
                if chunk.original_chars
                else 0
            ),
        }
        for chunk in result.chunks
    ]


def summary_to_csv(result: SummaryResult) -> str:
    """Per-chunk statistics as CSV text (usable with ``st.download_button``)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(chunk_rows(result))
    return buffer.getvalue()
