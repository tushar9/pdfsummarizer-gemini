"""LangChain chains for the map-reduce summarization pipeline.

Four chains share one chat model: per-chunk summaries (map), the merged summary
(reduce), document analysis and quote extraction. Each is wrapped in
``with_retry`` so transient 429/5xx responses do not fail a whole document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from pdfsummarizer.config import Settings
from pdfsummarizer.prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES

logger = logging.getLogger(__name__)

RETRY_ATTEMPTS = 3


def message_text(message: BaseMessage) -> str:
    """Extract plain text from a chat response.

    Gemini 3 returns content as a list of blocks; ``.text`` joins the text ones,
    while ``str()`` on the accessor would keep a wrapper type.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text.strip()
    if text is not None and not callable(text):
        return str(text).strip()

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def build_llm(settings: Settings) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model for these settings."""
    return ChatGoogleGenerativeAI(
        model=settings.model,
        temperature=settings.temperature,
        google_api_key=settings.require_api_key(),
    )


@dataclass(frozen=True)
class Chains:
    """The four runnables used by :class:`pdfsummarizer.summarize.SummarizerService`."""

    map: Runnable
    reduce: Runnable
    analysis: Runnable
    quotes: Runnable


def build_chains(settings: Settings, llm: BaseChatModel | None = None) -> Chains:
    """Build the chains, retrying transient model errors three times."""
    model = llm or build_llm(settings)
    parser = RunnableLambda(message_text)
    retry = {"stop_after_attempt": RETRY_ATTEMPTS, "wait_exponential_jitter": True}

    return Chains(
        map=(CHUNK_SUMMARY | model | parser).with_retry(**retry),
        reduce=(COMBINE | model | parser).with_retry(**retry),
        analysis=(ANALYSIS | model | parser).with_retry(**retry),
        quotes=(QUOTES | model | parser).with_retry(**retry),
    )
