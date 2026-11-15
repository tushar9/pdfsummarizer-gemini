"""PDF summarizer: a LangChain + Gemini map-reduce pipeline with an OCR fallback."""

from pdfsummarizer.config import AVAILABLE_MODELS, Settings, load_settings

__all__ = ["AVAILABLE_MODELS", "Settings", "load_settings"]
__version__ = "1.0.0"
