# PDF Summarizer (LangChain + Gemini)

An end-to-end PDF summarizer: extract text (with an OCR fallback for scanned
pages), split it into token-budgeted chunks, summarize each chunk with Google
Gemini through LangChain, and combine the results into one final summary.

Available as a Streamlit app and as a headless CLI.

> Setup, usage, architecture and troubleshooting are documented below as the
> project is built out.
