"""Streamlit rendering helpers.

All Streamlit calls used by the app live here (or in :mod:`.app`), which keeps
the pipeline modules importable and testable without a browser.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from pdfsummarizer import ocr
from pdfsummarizer.config import SUMMARY_TYPES, Settings
from pdfsummarizer.export import summary_to_csv, summary_to_text
from pdfsummarizer.pdf import ExtractionResult, PdfMetadata
from pdfsummarizer.summarize import SummaryResult


@dataclass(frozen=True)
class RunOptions:
    """What the user picked in the sidebar."""

    model: str
    summary_type: str
    max_tokens: int
    include_analysis: bool
    include_quotes: bool
    ocr_enabled: bool


def sidebar(settings: Settings) -> RunOptions:
    """Render the sidebar and return the chosen options."""
    st.sidebar.header("Settings")

    model = st.sidebar.selectbox(
        "Model",
        settings.models,
        index=settings.models.index(settings.model),
        help="Gemini model used for every LLM call",
    )
    summary_type = st.sidebar.selectbox(
        "Summary style",
        SUMMARY_TYPES,
        index=SUMMARY_TYPES.index("detailed"),
        format_func=lambda name: name.replace("_", " ").title(),
        help="How the final summary should be written",
    )
    max_tokens = st.sidebar.slider(
        "Max tokens per chunk",
        min_value=500,
        max_value=16000,
        value=settings.max_tokens_per_chunk,
        step=500,
        help="Smaller chunks mean more requests but stay well inside the context window",
    )
    include_analysis = st.sidebar.checkbox(
        "Analyze document structure", value=True, help="One extra request"
    )
    include_quotes = st.sidebar.checkbox(
        "Extract key quotes", value=False, help="One extra request"
    )
    ocr_enabled = st.sidebar.checkbox(
        "OCR scanned pages",
        value=settings.ocr_enabled,
        help="Runs Tesseract only on pages without a text layer. Slower on scans.",
    )

    options = ocr.OcrOptions(
        dpi=settings.ocr_dpi,
        lang=settings.ocr_lang,
        max_pages=settings.ocr_max_pages,
        tesseract_cmd=settings.tesseract_cmd,
    )
    if ocr.is_available(options):
        st.sidebar.caption("Tesseract found - scanned PDFs supported")
    else:
        st.sidebar.caption("Tesseract not found - scanned PDFs cannot be read")
    st.sidebar.caption(f"API key loaded ({len(settings.api_key)} chars)")

    return RunOptions(
        model=model,
        summary_type=summary_type,
        max_tokens=max_tokens,
        include_analysis=include_analysis,
        include_quotes=include_quotes,
        ocr_enabled=ocr_enabled,
    )


def empty_state() -> None:
    """What to show before a file is uploaded."""
    st.info(
        "Upload a PDF to get started. Text is extracted, split into chunks, "
        "summarized, then merged."
    )


def render_metadata(metadata: PdfMetadata) -> None:
    """Four small cards with the PDF's own metadata."""
    left, right = st.columns(2)
    left.metric("Pages", metadata.num_pages)
    left.caption(f"Title: {metadata.title}")
    right.caption(f"Author: {metadata.author}")
    right.caption(f"Subject: {metadata.subject}")


def render_extraction(extraction: ExtractionResult, chunks: list[str], max_tokens: int) -> None:
    """Report what was extracted, including any OCR recovery."""
    render_metadata(extraction.metadata)

    st.caption(
        f"{len(extraction.text):,} characters - {len(chunks)} chunk(s) at "
        f"{max_tokens:,} tokens each"
    )
    if extraction.ocr_pages:
        pages = ", ".join(str(page + 1) for page in extraction.ocr_pages)
        st.info(f"Recovered {len(extraction.ocr_pages)} scanned page(s) with OCR: {pages}")
    for warning in extraction.warnings:
        st.warning(warning)


def render_result(result: SummaryResult) -> None:
    """Show the final summary, optional extras and per-chunk detail."""
    for warning in result.warnings:
        st.warning(warning)

    left, middle, right, far = st.columns(4)
    left.metric("Chunks", result.total_chunks)
    middle.metric("Succeeded", result.successful_chunks)
    right.metric("Time", f"{result.elapsed_seconds}s")
    far.metric("Size", f"{result.compression_ratio:.0%}")

    st.subheader("Final summary")
    st.write(result.combined_summary)

    if result.analysis is not None:
        label = "Document analysis" if result.analysis.ok else "Document analysis (failed)"
        with st.expander(label, expanded=result.analysis.ok):
            st.write(result.analysis.text)

    if result.quotes:
        st.subheader("Key quotes")
        for quote in result.quotes:
            st.markdown(f"> {quote}")

    with st.expander(f"Per-chunk summaries ({result.total_chunks})"):
        for chunk in result.chunks:
            suffix = "" if chunk.ok else " - failed"
            st.markdown(f"**Chunk {chunk.chunk_number}{suffix}**")
            st.write(chunk.summary)
            st.caption(
                f"{chunk.original_chars:,} chars -> {chunk.summary_chars:,} chars "
                f"({chunk.original_tokens:,} tokens in)"
            )
            st.divider()


def render_downloads(result: SummaryResult) -> None:
    """Text report and CSV statistics side by side."""
    stem = result.file_name.rsplit(".", 1)[0] or "summary"
    left, right = st.columns(2)
    left.download_button(
        "Download summary (.txt)",
        data=summary_to_text(result),
        file_name=f"{stem}-summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
    right.download_button(
        "Download chunk stats (.csv)",
        data=summary_to_csv(result),
        file_name=f"{stem}-chunks.csv",
        mime="text/csv",
        use_container_width=True,
    )
