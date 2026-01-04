"""Script-based fast filtering for language detection.

Uses Unicode script detection to quickly narrow down possible languages
based on the writing system used. This is especially useful for:
- Very short strings where statistical methods may fail
- Reducing confusion between unrelated languages
- Fast pre-filtering before heavier detection methods
"""

from __future__ import annotations

import bisect
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Counter


class Script(Enum):
    """Unicode script categories relevant for language detection."""

    LATIN = "latin"
    CYRILLIC = "cyrillic"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    GREEK = "greek"
    CJK = "cjk"  # Chinese, Japanese, Korean (Han + Hangul + Kana)
    DEVANAGARI = "devanagari"
    BENGALI = "bengali"
    TAMIL = "tamil"
    TELUGU = "telugu"
    THAI = "thai"
    GEORGIAN = "georgian"
    ARMENIAN = "armenian"
    ETHIOPIC = "ethiopic"
    HANGUL = "hangul"  # Korean
    HIRAGANA = "hiragana"  # Japanese
    KATAKANA = "katakana"  # Japanese
    HAN = "han"  # Chinese characters
    UNKNOWN = "unknown"
    MIXED = "mixed"


# Mapping from script to likely languages (ISO 639-1)
SCRIPT_TO_LANGUAGES: dict[Script, set[str]] = {
    Script.CYRILLIC: {"ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "mn", "tg", "uz"},
    Script.ARABIC: {"ar", "fa", "ur", "ps", "sd", "ug", "ku"},
    Script.HEBREW: {"he", "yi"},
    Script.GREEK: {"el"},
    Script.DEVANAGARI: {"hi", "mr", "ne", "sa"},
    Script.BENGALI: {"bn", "as"},
    Script.TAMIL: {"ta"},
    Script.TELUGU: {"te"},
    Script.THAI: {"th"},
    Script.GEORGIAN: {"ka"},
    Script.ARMENIAN: {"hy"},
    Script.ETHIOPIC: {"am", "ti"},
    Script.HANGUL: {"ko"},
    Script.HIRAGANA: {"ja"},
    Script.KATAKANA: {"ja"},
    Script.HAN: {"zh", "ja"},  # Can be Chinese or Japanese
    Script.LATIN: set(),  # Too many languages use Latin
}

# Unicode ranges for script detection
SCRIPT_RANGES: dict[Script, list[tuple[int, int]]] = {
    Script.CYRILLIC: [(0x0400, 0x04FF), (0x0500, 0x052F)],
    Script.ARABIC: [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
    Script.HEBREW: [(0x0590, 0x05FF)],
    Script.GREEK: [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    Script.DEVANAGARI: [(0x0900, 0x097F)],
    Script.BENGALI: [(0x0980, 0x09FF)],
    Script.TAMIL: [(0x0B80, 0x0BFF)],
    Script.TELUGU: [(0x0C00, 0x0C7F)],
    Script.THAI: [(0x0E00, 0x0E7F)],
    Script.GEORGIAN: [(0x10A0, 0x10FF)],
    Script.ARMENIAN: [(0x0530, 0x058F)],
    Script.ETHIOPIC: [(0x1200, 0x137F)],
    Script.HANGUL: [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    Script.HIRAGANA: [(0x3040, 0x309F)],
    Script.KATAKANA: [(0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    Script.HAN: [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF)],
}

# Precomputed sorted index for O(log n) script lookup via binary search
# Each entry: (start_code, end_code, script)
_SCRIPT_INDEX: list[tuple[int, int, Script]] = sorted(
    [(start, end, script) for script, ranges in SCRIPT_RANGES.items() for start, end in ranges],
    key=lambda x: x[0]
)
_SCRIPT_STARTS: list[int] = [entry[0] for entry in _SCRIPT_INDEX]


def _get_char_script(char: str) -> Script:
    """Determine the script of a single character using O(log n) binary search."""
    code = ord(char)

    # Binary search for the script range
    idx = bisect.bisect_right(_SCRIPT_STARTS, code) - 1
    if idx >= 0:
        start, end, script = _SCRIPT_INDEX[idx]
        if start <= code <= end:
            return script

    # Check if Latin (fallback for characters not in precomputed ranges)
    if char.isalpha():
        # Fast path: ASCII letters are Latin
        if char.isascii():
            return Script.LATIN
        # Use unicodedata for non-ASCII Latin characters
        try:
            name = unicodedata.name(char, "")
            if "LATIN" in name:
                return Script.LATIN
        except ValueError:
            pass

    return Script.UNKNOWN


def detect_script(text: str) -> tuple[Script, float]:
    """
    Detect the dominant script in a text.

    Args:
        text: Input text

    Returns:
        Tuple of (dominant_script, proportion)
    """
    if not text:
        return Script.UNKNOWN, 0.0

    # Count scripts for each character
    script_counts: Counter[Script] = Counter()
    total_chars = 0

    for char in text:
        if char.isalpha():
            script = _get_char_script(char)
            if script != Script.UNKNOWN:
                script_counts[script] += 1
                total_chars += 1

    if total_chars == 0:
        return Script.UNKNOWN, 0.0

    # Get most common script
    most_common = script_counts.most_common(1)
    if most_common:
        script, count = most_common[0]
        proportion = count / total_chars

        # Check if it's mixed (no clear majority)
        if proportion < 0.7 and len(script_counts) > 1:
            return Script.MIXED, proportion

        return script, proportion

    return Script.UNKNOWN, 0.0


@dataclass
class ScriptFilter:
    """
    Fast script-based pre-filter for language detection.

    Analyzes the Unicode scripts present in text to quickly narrow down
    possible languages before running full detection.
    """

    min_script_proportion: float = 0.7
    """Minimum proportion of text in dominant script to apply filter."""

    def filter_languages(
        self, text: str, candidate_languages: set[str] | None = None
    ) -> set[str] | None:
        """
        Filter candidate languages based on script detection.

        Args:
            text: Input text
            candidate_languages: Optional set of allowed languages

        Returns:
            Filtered set of possible languages, or None if no filtering applied
        """
        script, proportion = detect_script(text)

        if script in (Script.UNKNOWN, Script.MIXED, Script.LATIN):
            # Can't filter on Latin (too many languages) or unknown
            return candidate_languages

        if proportion < self.min_script_proportion:
            return candidate_languages

        script_langs = SCRIPT_TO_LANGUAGES.get(script, set())
        if not script_langs:
            return candidate_languages

        if candidate_languages:
            # Intersect with allowed languages
            filtered = candidate_languages & script_langs
            return filtered if filtered else candidate_languages

        return script_langs

    def get_script_hint(self, text: str) -> dict[str, float] | None:
        """
        Get language hints based on script detection.

        Args:
            text: Input text

        Returns:
            Dict mapping language codes to boost scores, or None
        """
        script, proportion = detect_script(text)

        if script in (Script.UNKNOWN, Script.MIXED, Script.LATIN):
            return None

        if proportion < self.min_script_proportion:
            return None

        script_langs = SCRIPT_TO_LANGUAGES.get(script, set())
        if not script_langs:
            return None

        # Return equal boost for all languages using this script
        boost = 0.2 * proportion  # Scale boost by how dominant the script is
        return {lang: boost for lang in script_langs}

    def is_japanese(self, text: str) -> bool:
        """Check if text appears to be Japanese (mix of scripts)."""
        script_counts: Counter[Script] = Counter()

        for char in text:
            if char.isalpha():
                script = _get_char_script(char)
                script_counts[script] += 1

        # Japanese typically mixes Hiragana, Katakana, and Han
        has_kana = script_counts[Script.HIRAGANA] > 0 or script_counts[Script.KATAKANA] > 0
        has_han = script_counts[Script.HAN] > 0

        return has_kana or (has_han and not script_counts[Script.HANGUL])

    def is_korean(self, text: str) -> bool:
        """Check if text appears to be Korean."""
        script, proportion = detect_script(text)
        return script == Script.HANGUL and proportion > 0.3

    def is_chinese(self, text: str) -> bool:
        """Check if text appears to be Chinese (Han only, no Kana)."""
        script_counts: Counter[Script] = Counter()

        for char in text:
            if char.isalpha():
                script = _get_char_script(char)
                script_counts[script] += 1

        total = sum(script_counts.values())
        if total == 0:
            return False

        has_han = script_counts[Script.HAN] > 0
        has_kana = script_counts[Script.HIRAGANA] > 0 or script_counts[Script.KATAKANA] > 0
        has_hangul = script_counts[Script.HANGUL] > 0

        # Chinese uses Han without Kana or Hangul
        return has_han and not has_kana and not has_hangul
