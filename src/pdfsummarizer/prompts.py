"""Prompt templates.

One system persona, per-style instructions, and four chat templates (chunk
summary, combine, analysis, quotes). Keeping them in one module makes the
summarizer's behaviour easy to read and easy to test.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from pdfsummarizer.config import SUMMARY_TYPES

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a precise document summarizer. Use only information in the provided "
    "text; never invent facts, numbers or names. Write in clear English. "
    "Respond with the requested content only: no introductions or closing remarks."
)

#: Instruction text injected as ``{style}`` for each supported summary type.
STYLES: dict[str, str] = {
    "brief": "2-3 sentences focusing on the main points and key takeaways.",
    "detailed": (
        "A detailed summary. Include:\n"
        "- Main topics and themes\n"
        "- Key arguments or findings\n"
        "- Important details and examples\n"
        "- Conclusions or recommendations"
    ),
    "bullet_points": (
        "Clear, organized bullet points. "
        "Group related information together and use sub-bullets where appropriate."
    ),
    "executive": (
        "An executive summary suitable for business leaders. Include:\n"
        "- Key insights and findings\n"
        "- Strategic implications\n"
        "- Actionable recommendations\n"
        "- Risk factors or considerations"
    ),
}

assert set(STYLES) == set(SUMMARY_TYPES), "STYLES and SUMMARY_TYPES disagree"


def _template(human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", human)])


def style_for(summary_type: str) -> str:
    """Instruction text for ``summary_type``, falling back to ``detailed``."""
    if summary_type not in STYLES:
        logger.warning("Unknown summary type %r, using 'detailed'", summary_type)
        return STYLES["detailed"]
    return STYLES[summary_type]


CHUNK_SUMMARY = _template(
    "Summarize the document below in this style:\n{style}\n\n<document>\n{text}\n</document>"
)

COMBINE = _template(
    "Below are summaries of consecutive sections of one document. Merge them into one "
    "cohesive summary in this style:\n{style}\n\nRemove repetition.\n\n"
    "<summaries>\n{summaries}\n</summaries>"
)

ANALYSIS = _template(
    "Analyze the structure and content of the document below. Provide:\n\n"
    "1. Document Type (research paper, report, article, etc.)\n"
    "2. Main Topics/Themes\n"
    "3. Key Sections or Chapters\n"
    "4. Target Audience\n"
    "5. Overall Purpose\n\n"
    "<document>\n{text}\n</document>"
)

QUOTES = _template(
    "Extract 5-10 key quotes that best represent the main ideas of the document below. "
    "Copy each quote verbatim, exactly as written in the text, joining any line-wrapped "
    "text into a single line. "
    "Output one quote per line, with no numbering, bullets or surrounding quotation marks.\n\n"
    "<document>\n{text}\n</document>"
)
