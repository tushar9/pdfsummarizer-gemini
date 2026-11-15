"""Headless entry point.

    python -m pdfsummarizer.cli document.pdf --style brief --analysis
    python -m pdfsummarizer.cli --check-config

Progress goes to stderr, the report to stdout, and every message is ASCII so it
renders in a Windows console.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from pdfsummarizer.config import SUMMARY_TYPES, ConfigError, Settings, load_settings
from pdfsummarizer.export import summary_to_csv, summary_to_text
from pdfsummarizer.pdf import EmptyDocumentError, PdfError, PdfProcessor
from pdfsummarizer.summarize import SummarizerError, SummarizerService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdfsummarizer",
        description="Summarize a PDF with Gemini through a LangChain map-reduce pipeline.",
    )
    parser.add_argument("pdf", type=Path, nargs="?", help="PDF file to summarize")
    parser.add_argument(
        "--style",
        choices=SUMMARY_TYPES,
        default="detailed",
        help="summary style (default: detailed)",
    )
    parser.add_argument("--model", help="override GEMINI_MODEL")
    parser.add_argument("--max-tokens", type=int, help="override MAX_TOKENS_PER_CHUNK")
    parser.add_argument("--analysis", action="store_true", help="also analyze document structure")
    parser.add_argument("--quotes", action="store_true", help="also extract key quotes")
    parser.add_argument("--ocr", dest="ocr", action="store_true", default=None, help="force OCR on")
    parser.add_argument("--no-ocr", dest="ocr", action="store_false", help="disable OCR")
    parser.add_argument("--ocr-max-pages", type=int, help="override OCR_MAX_PAGES")
    parser.add_argument("--out", type=Path, help="write the text report to this file")
    parser.add_argument("--csv", type=Path, help="write per-chunk statistics to this file")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="print the resolved settings (key masked) and exit",
    )
    return parser


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    updates: dict[str, object] = {}
    if args.model:
        updates["model"] = args.model
    if args.max_tokens:
        updates["max_tokens_per_chunk"] = args.max_tokens
    if args.ocr is not None:
        updates["ocr_enabled"] = args.ocr
    if args.ocr_max_pages:
        updates["ocr_max_pages"] = args.ocr_max_pages
    return replace(settings, **updates) if updates else settings


def _print_config(settings: Settings) -> int:
    for key, value in settings.redacted().items():
        print(f"{key:<20} {value}")
    print(f"{'api_key_usable':<20} {settings.has_api_key}")
    if not settings.has_api_key:
        print("error: set GEMINI_API_KEY in .env (see .env.example)", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = _apply_overrides(load_settings(), args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check_config:
        return _print_config(settings)

    if args.pdf is None:
        print("error: a PDF path is required (or use --check-config)", file=sys.stderr)
        return 2
    if not args.pdf.exists():
        print(f"error: file not found: {args.pdf}", file=sys.stderr)
        return 2

    processor = PdfProcessor(settings)
    try:
        extraction, chunks = processor.prepare(args.pdf, ocr_enabled=args.ocr)
    except (PdfError, EmptyDocumentError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"file       {args.pdf.name}")
    print(f"pages      {extraction.metadata.num_pages}")
    print(f"characters {len(extraction.text)}")
    print(f"chunks     {len(chunks)}")
    if extraction.ocr_pages:
        pages = ", ".join(str(page + 1) for page in extraction.ocr_pages)
        print(f"ocr pages  {pages}")
    for warning in extraction.warnings:
        print(f"warning    {warning}")
    print(f"model      {settings.model}", file=sys.stderr)
    print("summarizing...", file=sys.stderr)

    try:
        result = SummarizerService(settings).summarize(
            file_name=args.pdf.name,
            text=extraction.text,
            chunks=chunks,
            summary_type=args.style,
            include_analysis=args.analysis,
            include_quotes=args.quotes,
            ocr_pages=extraction.ocr_pages,
            warnings=extraction.warnings,
        )
    except SummarizerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface auth/quota problems plainly
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report = summary_to_text(result)
    print()
    print(result.combined_summary)

    if result.analysis is not None and result.analysis.ok:
        print()
        print("DOCUMENT ANALYSIS")
        print(result.analysis.text)

    if result.quotes:
        print()
        print("KEY QUOTES")
        for quote in result.quotes:
            print(f'- "{quote}"')

    print()
    print(
        f"[{result.successful_chunks}/{result.total_chunks} chunks, "
        f"{result.elapsed_seconds}s, {result.compression_ratio:.0%} of original size]",
        file=sys.stderr,
    )

    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"report written to {args.out}", file=sys.stderr)
    if args.csv:
        args.csv.write_text(summary_to_csv(result), encoding="utf-8")
        print(f"csv written to {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
