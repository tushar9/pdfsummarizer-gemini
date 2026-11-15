"""Runtime configuration.

Every knob is read from the environment (or a local ``.env``) once, validated,
and exposed as an immutable :class:`Settings` object, so the rest of the code
never touches ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

#: Models offered in the UI. A ``GEMINI_MODEL`` set in the environment is always
#: allowed too, even when it is not in this list.
AVAILABLE_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
)

DEFAULT_MODEL = "gemini-3.1-flash-lite"

#: Summary styles understood by :mod:`pdfsummarizer.prompts`.
SUMMARY_TYPES: tuple[str, ...] = ("brief", "detailed", "bullet_points", "executive")

API_KEY_MIN_LENGTH = 20


class ConfigError(RuntimeError):
    """Raised when the environment is missing something required to run."""


def _raw(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _get_bool(name: str, default: bool) -> bool:
    raw = _raw(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _get_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = _raw(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated, immutable view of the environment."""

    api_key: str = field(default="", repr=False)
    model: str = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens_per_chunk: int = 8000
    max_concurrency: int = 2
    ocr_enabled: bool = True
    ocr_dpi: int = 200
    ocr_max_pages: int = 30
    ocr_lang: str = "eng"
    tesseract_cmd: str | None = None

    @property
    def has_api_key(self) -> bool:
        """True when a plausibly complete key is configured."""
        return len(self.api_key) >= API_KEY_MIN_LENGTH

    @property
    def models(self) -> tuple[str, ...]:
        """Model choices for the UI, with the configured model first."""
        if self.model in AVAILABLE_MODELS:
            return AVAILABLE_MODELS
        return (self.model, *AVAILABLE_MODELS)

    def require_api_key(self) -> str:
        """Return the key, or raise a message a human can act on."""
        if not self.api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env, add your key "
                "from https://aistudio.google.com/apikey, and restart."
            )
        if not self.has_api_key:
            raise ConfigError(
                "GEMINI_API_KEY looks incomplete. Create a new key at "
                "https://aistudio.google.com/apikey."
            )
        return self.api_key

    def redacted(self) -> dict[str, object]:
        """Safe-to-print summary; the key itself never leaves this object."""
        return {
            "model": self.model,
            "api_key": f"set ({len(self.api_key)} chars)" if self.api_key else "missing",
            "temperature": self.temperature,
            "max_tokens_per_chunk": self.max_tokens_per_chunk,
            "max_concurrency": self.max_concurrency,
            "ocr_enabled": self.ocr_enabled,
            "ocr_dpi": self.ocr_dpi,
            "ocr_max_pages": self.ocr_max_pages,
            "ocr_lang": self.ocr_lang,
            "tesseract_cmd": self.tesseract_cmd or "(auto-detect)",
        }


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Load ``.env`` (if present) and build validated settings.

    Values already present in the real environment win over the file, which is
    what you want in CI or when a shell exports ``GEMINI_API_KEY``.
    """
    load_dotenv(dotenv_path=env_file, override=False)

    return Settings(
        api_key=_raw("GEMINI_API_KEY") or "",
        model=_raw("GEMINI_MODEL") or DEFAULT_MODEL,
        temperature=_get_float("TEMPERATURE", 0.3, low=0.0, high=2.0),
        max_tokens_per_chunk=_get_int(
            "MAX_TOKENS_PER_CHUNK", 8000, low=500, high=100_000
        ),
        max_concurrency=_get_int("MAX_CONCURRENCY", 2, low=1, high=16),
        ocr_enabled=_get_bool("OCR_ENABLED", True),
        ocr_dpi=_get_int("OCR_DPI", 200, low=100, high=600),
        ocr_max_pages=_get_int("OCR_MAX_PAGES", 30, low=1, high=1000),
        ocr_lang=_raw("OCR_LANG") or "eng",
        tesseract_cmd=_raw("TESSERACT_CMD"),
    )
