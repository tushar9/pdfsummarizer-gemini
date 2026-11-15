# Walkthrough

A code-level tour of what happens between clicking **Summarize** and getting a summary, plus the
reasoning behind the design choices.

## 1. Settings are resolved once

`pdfsummarizer/config.py` → `load_settings()`

1. `python-dotenv` reads `.env` **without overriding** variables that already exist, so CI and
   shells win.
2. Every value is parsed and range-checked (`TEMPERATURE` 0–2, `MAX_CONCURRENCY` 1–16, …). A typo
   raises `ConfigError` with the variable name instead of failing deep inside a model call.
3. The result is a frozen `Settings` dataclass. `Settings.redacted()` is the only representation
   that is ever printed, and it shows the key's length, never the key.

`cli.py --check-config` prints exactly that: model, masked key, and the OCR knobs.

## 2. Extraction: text first, OCR second

`pdfsummarizer/pdf.py` → `PdfProcessor.extract()`

1. The source (path, bytes, or a Streamlit `UploadedFile`) is normalized to bytes by
   `PdfProcessor.read()`, so the rest of the code has one input type.
2. `pypdf` extracts text page by page. A page that raises is skipped and recorded as a warning;
   the document continues.
3. Any page with **fewer than 30 characters** (`MIN_TEXT_LAYER_CHARS`) is treated as "no text
   layer". If OCR is enabled, those page indexes go to `ocr.py`:

   - `find_tesseract()` checks `TESSERACT_CMD`, then `PATH`, then the standard Windows paths.
   - PyMuPDF rasterizes only those pages at `OCR_DPI` (200 by default).
   - `pytesseract.image_to_string()` reads each one; a failing page becomes a warning.
   - `OCR_MAX_PAGES` caps the work, and hitting the cap is reported.
4. If Tesseract is missing, extraction still returns the text pages plus a warning — the run does
   not crash on a machine without the binary.
5. Pages are joined and cleaned by `clean_text()`: control characters removed, words re-joined
   across hyphenated line breaks (`summa-\nrize` → `summarize`), spaces around newlines collapsed,
   and runs of blank lines reduced to one.

The threshold is deliberately low: a real page of prose has hundreds of characters, so normal PDFs
never pay the OCR cost.

## 3. Chunking by tokens, not characters

`PdfProcessor.chunk()` uses LangChain's `RecursiveCharacterTextSplitter` with
`length_function=count_tokens` and separators ordered `paragraph → line → sentence → word →
character`. It therefore prefers to cut on paragraph boundaries and only falls back to hard cuts
for pathological input.

- `MAX_TOKENS_PER_CHUNK` (8000) is the budget per chunk.
- Overlap is `min(200, budget // 10)` tokens so a sentence split across a boundary is still
  readable in one of the two chunks.
- `count_tokens()` uses tiktoken's `cl100k_base` as an approximation for Gemini's tokenizer. If the
  encoding file is unavailable (offline machine, first run), it degrades to `len(text) // 4`
  instead of failing.

## 4. Prompts live in one file

`pdfsummarizer/prompts.py`

- `SYSTEM` fixes the persona: use only the provided text, never invent facts, no preamble.
- `STYLES` maps each summary type to its instruction text; `style_for()` falls back to `detailed`
  (with a log warning) if an unknown type is passed.
- Four `ChatPromptTemplate`s share the system message: `CHUNK_SUMMARY`, `COMBINE`, `ANALYSIS`,
  `QUOTES`.
- The document is wrapped in `<document>…</document>` tags. This makes the boundary between
  instructions and content unambiguous, which reduces prompt-injection style drift from the PDF
  text itself.

## 5. Chains are built once per run

`pdfsummarizer/chains.py` → `build_chains()`

```python
model = llm or build_llm(settings)
parser = RunnableLambda(message_text)
retry = {"stop_after_attempt": 3, "wait_exponential_jitter": True}

Chains(
    map=(CHUNK_SUMMARY | model | parser).with_retry(**retry),
    reduce=(COMBINE | model | parser).with_retry(**retry),
    analysis=(ANALYSIS | model | parser).with_retry(**retry),
    quotes=(QUOTES | model | parser).with_retry(**retry),
)
```

- LCEL pipes (`prompt | model | parser`) keep the flow declarative and make the model swappable:
  tests pass `FakeListChatModel` and the code path is identical.
- `message_text()` handles Gemini 3's block-list content. `message.text` is a `str` subclass but
  also callable (calling it is deprecated), so the parser only accepts the string form.
- `.with_retry(...)` retries transient 429/5xx errors with jittered exponential backoff. Permanent
  errors (bad key, unknown model) surface immediately.

## 6. Map: summarize chunks concurrently

`pdfsummarizer/summarize.py` → `SummarizerService.summarize_chunks()`

```python
responses = self.chains.map.batch(
    inputs,
    config={"max_concurrency": self.settings.max_concurrency},
    return_exceptions=True,
)
```

- `return_exceptions=True` is the important part: a failed chunk becomes a `ChunkSummary(ok=False,
  error=…)` instead of aborting the batch.
- Each entry records character and token counts, which feed the metrics and the CSV export.
- Default concurrency is 2 because free-tier keys rate-limit quickly; raise it on a paid tier.

## 7. Reduce: merge without repetition

`SummarizerService.combine()`

- Zero valid summaries → `SummarizerError` with the first underlying error message (the UI and CLI
  show it directly).
- Exactly one valid summary → returned as-is; no extra model call.
- Otherwise the summaries are labelled `Section 1…N` and sent through the `COMBINE` prompt.
- If the reduce call fails, the joined sections are returned under a `Combined Summary:` heading.
  Losing the merge is annoying; losing the whole document is worse.

## 8. Optional analysis and quotes

Both run on the first `PREVIEW_CHARS` (3000) characters of the document — cheap, and enough for
structure and representative quotes. Analysis failures are stored as `DocumentAnalysis(ok=False)`
and rendered as "failed" rather than raising. Quotes are parsed line by line with numbering,
bullets and wrapping quotes stripped, capped at 10.

## 9. Results are plain dataclasses

`SummaryResult` carries the per-chunk list, combined summary, optional extras, OCR pages, warnings
and elapsed time. Properties (`successful_chunks`, `compression_ratio`, …) compute the metrics, and
`export.py` renders it as text or CSV with no Streamlit import — which is why the CLI, tests and UI
all share the same formatting.

## 10. UI state survives reruns

`ui/app.py`

- Streamlit re-runs the whole script on every interaction, including a download click. The result
  is therefore stored in `st.session_state.result` and only recomputed when the file changes
  (`file_id` comparison) or the button is pressed again.
- Extraction and summarization are wrapped in `st.spinner`, and every failure path shows
  `st.error` with an actionable sentence — never a raw traceback.
- Sidebar choices become a frozen `RunOptions`, which is applied to `Settings` via
  `dataclasses.replace`, so one code path serves CLI and UI.

## 11. Testing strategy

| Layer | How it is tested |
| --- | --- |
| Cleaning, chunk budgets | generated PDFs via PyMuPDF, asserted against `count_tokens` |
| Prompts | rendered and inspected for style text and document placement |
| Pipeline | `FakeListChatModel` and hand-built `Chains` (including failure cases) |
| OCR | real Tesseract on a rendered image, plus monkeypatched failure paths |
| Exports | parsed back with `csv.DictReader` |
| UI | `streamlit.testing.v1.AppTest` boots the app and inspects the sidebar |
| Live | one opt-in test that calls Gemini (`-m live`) |

The default suite makes no network calls.

## 12. Design decisions and trade-offs

| Decision | Why | Cost |
| --- | --- | --- |
| `src/` layout + editable install | imports are unambiguous and packaging is standard | one extra `pip install -e .` |
| tiktoken for sizing | token-accurate budgets beat character heuristics | an approximation of Gemini's tokenizer |
| OCR on demand, per page | scanned PDFs work without slowing normal ones | needs a system binary |
| Retry + concurrency 2 | survives free-tier rate limits | slower on large documents |
| Per-chunk failure isolation | partial summaries are more useful than nothing | the final summary may miss a section |
| stdlib `csv` instead of pandas | one fewer heavy dependency | none at this scale |
| Text/CSV export in `export.py` | CLI and UI share one implementation | none |

## 13. Extending it

- **New style**: add an entry to `STYLES` in `prompts.py` and to `SUMMARY_TYPES` in `config.py`
  (an assertion keeps them in sync).
- **New model**: set `GEMINI_MODEL`, or add it to `AVAILABLE_MODELS` in `config.py` to show it in
  the sidebar.
- **Different provider**: replace `build_llm()` in `chains.py` with any LangChain chat model; the
  prompts, pipeline and tests are provider-agnostic.
