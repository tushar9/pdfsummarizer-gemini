"""Export tests: the report and CSV must stay usable without Streamlit."""

from __future__ import annotations

import csv
import io

from pdfsummarizer.export import chunk_rows, summary_to_csv, summary_to_text
from pdfsummarizer.summarize import ChunkSummary, DocumentAnalysis, SummaryResult


def sample_result() -> SummaryResult:
    return SummaryResult(
        file_name="report.pdf",
        summary_type="bullet_points",
        model="gemini-3.1-flash-lite",
        chunks=[
            ChunkSummary(1, "- first point", True, 400, 40, 100),
            ChunkSummary(2, "Error processing chunk: boom", False, 200, 0, 50, "boom"),
        ],
        combined_summary="Final combined text.",
        analysis=DocumentAnalysis("It is a report.", True),
        quotes=["quote one"],
        ocr_pages=(2,),
        warnings=("Chunk(s) 2 failed and were left out of the final summary.",),
        elapsed_seconds=12.5,
    )


def test_text_report_contains_every_section() -> None:
    report = summary_to_text(sample_result())

    assert "File: report.pdf" in report
    assert "Summary type: Bullet Points" in report
    assert "Chunks: 2 (1 succeeded, 1 failed)" in report
    assert "OCR pages: 3" in report
    assert "Final combined text." in report
    assert "DOCUMENT ANALYSIS" in report
    assert 'KEY QUOTES' in report
    assert 'Chunk 2 (FAILED):' in report
    assert report.endswith("\n")


def test_csv_has_header_and_one_row_per_chunk() -> None:
    rows = list(csv.DictReader(io.StringIO(summary_to_csv(sample_result()))))

    assert [row["Chunk"] for row in rows] == ["1", "2"]
    assert rows[0]["Status"] == "Success"
    assert rows[1]["Status"] == "Error"
    assert rows[0]["Compression Ratio"] == "0.1"


def test_chunk_rows_handle_empty_chunks() -> None:
    result = sample_result()
    result.chunks = [ChunkSummary(1, "", False, 0, 0, 0, "boom")]

    assert chunk_rows(result)[0]["Compression Ratio"] == 0
