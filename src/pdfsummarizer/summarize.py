"""Map-reduce orchestration.

``SummarizerService`` turns a list of chunks into per-chunk summaries, then
merges the successful ones into a final summary. A chunk that fails is recorded
and skipped instead of aborting the document; only a total failure raises.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from pdfsummarizer.chains import Chains, build_chains
from pdfsummarizer.config import Settings, load_settings
from pdfsummarizer.pdf import count_tokens
from pdfsummarizer.prompts import style_for

logger = logging.getLogger(__name__)

#: Above this many chunks, warn about cost/latency before the run.
MAX_CHUNKS_WARNING = 40

#: Characters of text sent to the analysis and quote prompts (keeps them cheap).
PREVIEW_CHARS = 3000


class SummarizerError(RuntimeError):
    """Raised when the model produced nothing usable for a document."""


@dataclass
class ChunkSummary:
    chunk_number: int
    summary: str
    ok: bool
    original_chars: int
    summary_chars: int
    original_tokens: int
    error: str | None = None


@dataclass
class DocumentAnalysis:
    text: str
    ok: bool


@dataclass
class SummaryResult:
    file_name: str
    summary_type: str
    model: str
    chunks: list[ChunkSummary]
    combined_summary: str
    analysis: DocumentAnalysis | None = None
    quotes: list[str] = field(default_factory=list)
    ocr_pages: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def successful_chunks(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.ok)

    @property
    def failed_chunks(self) -> int:
        return self.total_chunks - self.successful_chunks

    @property
    def original_chars(self) -> int:
        return sum(chunk.original_chars for chunk in self.chunks)

    @property
    def compression_ratio(self) -> float:
        """Summary characters per original character (lower is more compressed)."""
        summary_chars = sum(chunk.summary_chars for chunk in self.chunks)
        return round(summary_chars / self.original_chars, 3) if self.original_chars else 0.0


class SummarizerService:
    """Runs the map-reduce pipeline against a Gemini model."""

    def __init__(self, settings: Settings | None = None, chains: Chains | None = None):
        self.settings = settings or load_settings()
        self.chains = chains or build_chains(self.settings)

    # -- map --------------------------------------------------------------
    def summarize_chunks(
        self, chunks: list[str], summary_type: str = "detailed"
    ) -> list[ChunkSummary]:
        """Summarize every chunk concurrently, isolating per-chunk failures."""
        style = style_for(summary_type)
        inputs = [{"style": style, "text": chunk} for chunk in chunks]
        responses = self.chains.map.batch(
            inputs,
            config={"max_concurrency": self.settings.max_concurrency},
            return_exceptions=True,
        )

        summaries: list[ChunkSummary] = []
        for number, (chunk, response) in enumerate(zip(chunks, responses, strict=True), start=1):
            failed = isinstance(response, BaseException)
            summary = f"Error processing chunk: {response}" if failed else str(response)
            if failed:
                logger.warning("Chunk %d failed: %s", number, response)
            summaries.append(
                ChunkSummary(
                    chunk_number=number,
                    summary=summary,
                    ok=not failed,
                    original_chars=len(chunk),
                    summary_chars=0 if failed else len(summary),
                    original_tokens=count_tokens(chunk),
                    error=str(response) if failed else None,
                )
            )
        return summaries

    # -- reduce -----------------------------------------------------------
    def combine(self, chunks: list[ChunkSummary], summary_type: str) -> str:
        """Merge chunk summaries into one; falls back to a plain concatenation."""
        valid = [chunk.summary for chunk in chunks if chunk.ok and chunk.summary.strip()]
        if not valid:
            first_error = next((chunk.error for chunk in chunks if chunk.error), "unknown error")
            raise SummarizerError(
                f"All {len(chunks)} chunk(s) failed. First error: {first_error}"
            )
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(f"Section {i}: {text}" for i, text in enumerate(valid, start=1))
        try:
            return self.chains.reduce.invoke(
                {"style": style_for(summary_type), "summaries": sections}
            )
        except Exception as exc:  # noqa: BLE001 - a merged summary is still useful
            logger.warning("Combine step failed (%s); returning joined sections", exc)
            return f"Combined Summary:\n\n{sections}"

    # -- optional extras ---------------------------------------------------
    def analyze(self, text: str) -> DocumentAnalysis:
        try:
            return DocumentAnalysis(text=self.chains.analysis.invoke({"text": text}), ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analysis failed: %s", exc)
            return DocumentAnalysis(text=f"Analysis unavailable: {exc}", ok=False)

    def extract_quotes(self, text: str) -> list[str]:
        try:
            response = self.chains.quotes.invoke({"text": text})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quote extraction failed: %s", exc)
            return []
        cleaned = (
            re.sub(r"^\s*(\d+[.)]|[-*\u2022])\s*", "", line).strip(" \"\u201c\u201d")
            for line in response.splitlines()
        )
        return [line for line in cleaned if line][:10]

    # -- whole document ----------------------------------------------------
    def summarize(
        self,
        *,
        file_name: str,
        text: str,
        chunks: list[str],
        summary_type: str = "detailed",
        include_analysis: bool = False,
        include_quotes: bool = False,
        ocr_pages: tuple[int, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> SummaryResult:
        """Run map + reduce (+ optional extras) and time the whole thing."""
        started = time.perf_counter()
        notes = list(warnings)
        if len(chunks) > MAX_CHUNKS_WARNING:
            notes.append(
                f"{len(chunks)} chunks is a lot; expect several minutes and notable API usage."
            )

        chunk_summaries = self.summarize_chunks(chunks, summary_type)
        combined = self.combine(chunk_summaries, summary_type)
        if any(not chunk.ok for chunk in chunk_summaries):
            failed = ", ".join(str(chunk.chunk_number) for chunk in chunk_summaries if not chunk.ok)
            notes.append(f"Chunk(s) {failed} failed and were left out of the final summary.")

        preview = text[:PREVIEW_CHARS]
        analysis = self.analyze(preview) if include_analysis else None
        quotes = self.extract_quotes(preview) if include_quotes else []

        return SummaryResult(
            file_name=file_name,
            summary_type=summary_type,
            model=self.settings.model,
            chunks=chunk_summaries,
            combined_summary=combined,
            analysis=analysis,
            quotes=quotes,
            ocr_pages=ocr_pages,
            warnings=tuple(notes),
            elapsed_seconds=round(time.perf_counter() - started, 1),
        )
