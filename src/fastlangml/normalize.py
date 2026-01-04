"""Text normalization and preprocessing for language detection.

Provides unicode normalization, script detection, and heuristics
for determining if text has enough linguistic content for detection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class TextStats:
    """Statistics about a text sample for detection heuristics."""

    n_chars: int
    """Total character count."""

    n_letters: int
    """Count of Unicode letters (L category)."""

    letter_ratio: float
    """Ratio of letters to total characters."""

    unique_letters: int
    """Count of unique letters."""

    script: str | None
    """Dominant Unicode script, if detectable."""

    is_mostly_ascii: bool
    """Whether text is predominantly ASCII."""

    has_cjk: bool
    """Whether text contains CJK characters."""


def compute_text_stats(text: str) -> TextStats:
    """Compute statistics about text for detection heuristics.

    Args:
        text: Input text to analyze.

    Returns:
        TextStats with character counts, ratios, and script info.
    """
    if not text:
        return TextStats(
            n_chars=0,
            n_letters=0,
            letter_ratio=0.0,
            unique_letters=0,
            script=None,
            is_mostly_ascii=True,
            has_cjk=False,
        )

    n_chars = len(text)
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    n_letters = len(letters)
    unique_letters = len(set(letters))
    letter_ratio = n_letters / n_chars if n_chars > 0 else 0.0

    # Check ASCII proportion
    ascii_count = sum(1 for c in text if ord(c) < 128)
    is_mostly_ascii = (ascii_count / n_chars) > 0.8 if n_chars > 0 else True

    # Check for CJK
    has_cjk = any(
        "\u4e00" <= c <= "\u9fff"  # CJK Unified
        or "\u3040" <= c <= "\u309f"  # Hiragana
        or "\u30a0" <= c <= "\u30ff"  # Katakana
        or "\uac00" <= c <= "\ud7af"  # Hangul
        for c in text
    )

    # Detect dominant script
    script = detect_dominant_script(text)

    return TextStats(
        n_chars=n_chars,
        n_letters=n_letters,
        letter_ratio=letter_ratio,
        unique_letters=unique_letters,
        script=script,
        is_mostly_ascii=is_mostly_ascii,
        has_cjk=has_cjk,
    )


def detect_dominant_script(text: str) -> str | None:
    """Detect the dominant Unicode script in text.

    Args:
        text: Input text to analyze.

    Returns:
        Script name (e.g., 'Latin', 'Cyrillic', 'Han') or None.
    """
    if not text:
        return None

    script_counts: dict[str, int] = {}

    for char in text:
        if unicodedata.category(char).startswith("L"):
            try:
                script_name = unicodedata.name(char, "").split()[0]
                # Normalize script names
                if script_name in ("CJK", "HIRAGANA", "KATAKANA"):
                    script_name = "CJK"
                elif "LATIN" in unicodedata.name(char, ""):
                    script_name = "Latin"
                elif "CYRILLIC" in unicodedata.name(char, ""):
                    script_name = "Cyrillic"
                elif "ARABIC" in unicodedata.name(char, ""):
                    script_name = "Arabic"
                elif "HEBREW" in unicodedata.name(char, ""):
                    script_name = "Hebrew"
                elif "GREEK" in unicodedata.name(char, ""):
                    script_name = "Greek"
                elif "DEVANAGARI" in unicodedata.name(char, ""):
                    script_name = "Devanagari"
                elif "HANGUL" in unicodedata.name(char, ""):
                    script_name = "Hangul"
                elif "THAI" in unicodedata.name(char, ""):
                    script_name = "Thai"

                script_counts[script_name] = script_counts.get(script_name, 0) + 1
            except (ValueError, KeyError):
                continue

    if not script_counts:
        return None

    dominant = max(script_counts.items(), key=lambda x: x[1])
    return dominant[0]


def normalize_text(text: str, *, strip_noise: bool = True) -> str:
    """Normalize text for language detection.

    Performs:
    - Unicode NFC normalization
    - Whitespace normalization
    - Optionally removes noise (URLs, emails) while preserving linguistic content

    Args:
        text: Input text to normalize.
        strip_noise: Whether to remove URLs, emails, etc.

    Returns:
        Normalized text.
    """
    if not text:
        return ""

    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Normalize whitespace (but preserve newlines as spaces)
    text = re.sub(r"\s+", " ", text)

    if strip_noise:
        # Remove URLs
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", " ", text)

        # Remove @mentions and #hashtags (but keep content after)
        text = re.sub(r"[@#]\w+", " ", text)

        # Normalize multiple spaces again
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_linguistic(text: str, min_letter_ratio: float = 0.3) -> tuple[bool, str | None]:
    """Check if text has enough linguistic content for detection.

    Args:
        text: Input text to check.
        min_letter_ratio: Minimum ratio of letters to total chars.

    Returns:
        Tuple of (is_linguistic, reason_if_not).
    """
    stats = compute_text_stats(text)

    if stats.n_chars == 0:
        return False, "empty_text"

    if stats.n_letters == 0:
        return False, "non_linguistic"

    if stats.letter_ratio < min_letter_ratio:
        return False, "non_linguistic"

    return True, None


def is_sufficient_length(
    text: str, min_chars: int = 3, min_letters: int = 2
) -> tuple[bool, str | None]:
    """Check if text has sufficient length for detection.

    Args:
        text: Input text to check.
        min_chars: Minimum total characters required.
        min_letters: Minimum letters required.

    Returns:
        Tuple of (is_sufficient, reason_if_not).
    """
    stats = compute_text_stats(text)

    if stats.n_chars < min_chars:
        return False, "too_little_text"

    if stats.n_letters < min_letters:
        return False, "too_little_text"

    return True, None


# Language tag normalization
LANG_NORMALIZATION: dict[str, str] = {
    # Standard normalizations
    "zh-cn": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zho": "zh",
    "cmn": "zh",
    "yue": "zh-Yue",
    "por": "pt",
    "spa": "es",
    "fra": "fr",
    "deu": "de",
    "eng": "en",
    "rus": "ru",
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
    "unknown": "und",
    "": "und",
}


def normalize_lang_tag(tag: str) -> str:
    """Normalize a language tag to consistent format.

    Uses BCP-47-like tags:
    - Lowercase 2-letter codes: 'en', 'es', 'fr'
    - Script variants: 'zh-Hans', 'zh-Hant'
    - 'und' for undetermined

    Args:
        tag: Input language tag.

    Returns:
        Normalized language tag.
    """
    if not tag:
        return "und"

    tag = tag.lower().strip()

    # Check normalization map
    if tag in LANG_NORMALIZATION:
        return LANG_NORMALIZATION[tag]

    # Handle zh-cn, zh-tw variants
    if tag.startswith("zh"):
        if "hans" in tag or "cn" in tag or "simplified" in tag:
            return "zh-Hans"
        if "hant" in tag or "tw" in tag or "traditional" in tag:
            return "zh-Hant"
        return "zh"

    # Return first 2 chars for standard ISO codes
    if len(tag) >= 2:
        return tag[:2]

    return "und"
