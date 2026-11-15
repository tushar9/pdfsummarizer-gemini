"""Streamlit UI for the PDF summarizer.

Run with ``streamlit run app.py``. Results live in ``st.session_state`` so that
clicking a download button does not re-run the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import streamlit as st

from pdfsummarizer.config import ConfigError, Settings, load_settings
from pdfsummarizer.pdf import EmptyDocumentError, PdfError, PdfProcessor
from pdfsummarizer.summarize import SummarizerError, SummarizerService
from pdfsummarizer.ui import components

logger = logging.getLogger(__name__)


def _load_settings() -> Settings | None:
    try:
        return load_settings()
    except ConfigError as exc:
        st.error(str(exc))
        return None


def main() -> None:
    st.set_page_config(page_title="PDF Summarizer", page_icon="📄", layout="wide")

    st.title("📄 PDF Summarizer")
    st.caption("LangChain + Gemini map-reduce pipeline, with an OCR fallback for scanned pages.")

    settings = _load_settings()
    if settings is None:
        return
    if not settings.has_api_key:
        st.error("GEMINI_API_KEY is missing or incomplete.")
        st.info(
            "Copy `.env.example` to `.env`, add a key from "
            "https://aistudio.google.com/apikey, then restart the app."
        )
        return

    options = components.sidebar(settings)

    uploaded = st.file_uploader("Choose a PDF file", type="pdf", help="Text or scanned PDF")
    if uploaded is None:
        components.empty_state()
        return

    # A new file invalidates anything cached from the previous one.
    if st.session_state.get("file_id") != uploaded.file_id:
        st.session_state.file_id = uploaded.file_id
        for key in ("result", "extraction", "chunks"):
            st.session_state.pop(key, None)

    if st.button("Summarize", type="primary"):
        _run(uploaded, settings, options)

    if "result" in st.session_state:
        result = st.session_state.result
        components.render_result(result)
        components.render_downloads(result)


def _run(uploaded, settings: Settings, options: components.RunOptions) -> None:
    """Extract, chunk and summarize, then store the result in session state."""
    active = replace(
        settings, model=options.model, max_tokens_per_chunk=options.max_tokens
    )
    processor = PdfProcessor(active)

    try:
        with st.spinner("Extracting text..."):
            extraction = processor.extract(uploaded, ocr_enabled=options.ocr_enabled)
        with st.spinner("Splitting into chunks..."):
            chunks = processor.chunk(extraction.text, options.max_tokens)
    except (PdfError, EmptyDocumentError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - never show a raw traceback
        logger.exception("Extraction failed")
        st.error(f"Could not read this PDF: {exc}")
        return

    components.render_extraction(extraction, chunks, options.max_tokens)

    try:
        with st.spinner(f"Summarizing {len(chunks)} chunk(s) with {options.model}..."):
            result = SummarizerService(active).summarize(
                file_name=uploaded.name,
                text=extraction.text,
                chunks=chunks,
                summary_type=options.summary_type,
                include_analysis=options.include_analysis,
                include_quotes=options.include_quotes,
                ocr_pages=extraction.ocr_pages,
                warnings=extraction.warnings,
            )
    except SummarizerError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001 - auth/quota errors land here
        logger.exception("Summarization failed")
        st.error(f"Summarization failed: {exc}")
        st.info("Check your API key, quota and network connection, then try again.")
        return

    st.session_state.result = result
    st.success("Done.")


if __name__ == "__main__":
    main()
