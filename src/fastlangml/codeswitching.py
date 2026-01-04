"""Code-switching detection for mixed-language messages.

Detects when a message contains multiple languages, like:
- "That's muy importante" (English + Spanish)
- "I need to faire les courses" (English + French)
- "Das ist so cool, right?" (German + English)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastlangml.backends import Backend


@dataclass
class CodeSwitchSpan:
    """A span of text in a specific language within a mixed-language message."""

    text: str
    """The text content of this span."""

    language: str
    """Detected language code (ISO 639-1)."""

    confidence: float
    """Confidence score for this span's language."""

    start: int
    """Start character position in original text."""

    end: int
    """End character position in original text."""


@dataclass
class CodeSwitchResult:
    """Result of code-switching detection."""

    is_mixed: bool
    """True if multiple languages were detected."""

    primary_language: str
    """The dominant language in the message."""

    secondary_languages: list[str]
    """Other languages detected (if mixed)."""

    spans: list[CodeSwitchSpan]
    """Language spans within the message."""

    language_distribution: dict[str, float]
    """Proportion of text in each language."""

    confidence: float
    """Overall confidence in the analysis."""

    @property
    def languages(self) -> list[str]:
        """All languages detected, primary first."""
        return [self.primary_language] + self.secondary_languages


@dataclass
class CodeSwitchDetector:
    """Detects code-switching (language mixing) in text.

    Code-switching is common in multilingual communities:
    - Spanglish: "Vamos to the store"
    - Franglais: "C'est so boring"
    - Hinglish: "Main office ja raha hoon"

    This detector identifies when a message contains multiple languages
    and segments it by language.

    Example:
        >>> detector = CodeSwitchDetector()
        >>> result = detector.detect("That's muy importante for us")
        >>> result.is_mixed
        True
        >>> result.primary_language
        "en"
        >>> result.secondary_languages
        ["es"]
    """

    min_segment_length: int = 3
    """Minimum characters for a segment to be considered."""

    min_confidence_threshold: float = 0.6
    """Minimum confidence to count a segment."""

    word_level: bool = True
    """If True, detect at word level. If False, use longer segments."""

    _backend: Backend | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the detector with a backend."""
        if self._backend is None:
            self._load_default_backend()

    def _load_default_backend(self):
        """Load the fastest available backend for quick word-level detection."""
        from fastlangml.backends import create_backend, get_available_backends

        available = get_available_backends()
        # Prefer fasttext for speed
        for name in ["fasttext", "langdetect", "lingua"]:
            if name in available:
                try:
                    self._backend = create_backend(name)
                    break
                except Exception:
                    pass

    def detect(self, text: str) -> CodeSwitchResult:
        """Detect code-switching in text.

        Args:
            text: Input text to analyze

        Returns:
            CodeSwitchResult with language breakdown
        """
        if not text or not text.strip():
            return CodeSwitchResult(
                is_mixed=False,
                primary_language="und",
                secondary_languages=[],
                spans=[],
                language_distribution={},
                confidence=0.0,
            )

        if self.word_level:
            return self._detect_word_level(text)
        else:
            return self._detect_segment_level(text)

    def _detect_word_level(self, text: str) -> CodeSwitchResult:
        """Detect code-switching at word level."""
        # Tokenize into words with positions
        words_with_pos = []
        for match in re.finditer(r"\b\w+\b", text):
            word = match.group()
            if len(word) >= self.min_segment_length:
                words_with_pos.append((word, match.start(), match.end()))

        if not words_with_pos:
            return CodeSwitchResult(
                is_mixed=False,
                primary_language="und",
                secondary_languages=[],
                spans=[],
                language_distribution={},
                confidence=0.0,
            )

        # Detect language for each word
        spans: list[CodeSwitchSpan] = []
        lang_counts: dict[str, int] = {}

        for word, start, end in words_with_pos:
            if self._backend is None:
                continue

            try:
                result = self._backend.detect(word)
                lang = result.language
                conf = result.confidence

                if conf >= self.min_confidence_threshold and lang != "unknown":
                    spans.append(
                        CodeSwitchSpan(
                            text=word,
                            language=lang,
                            confidence=conf,
                            start=start,
                            end=end,
                        )
                    )
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
            except Exception:
                pass

        return self._build_result(text, spans, lang_counts)

    def _detect_segment_level(self, text: str) -> CodeSwitchResult:
        """Detect code-switching at segment level (sentence/clause)."""
        # Split on punctuation and conjunctions
        segments = re.split(r"[.!?;,]\s*|\s+(?:and|but|or|y|et|und|e)\s+", text)
        segments = [s.strip() for s in segments if s.strip()]

        spans: list[CodeSwitchSpan] = []
        lang_counts: dict[str, int] = {}
        pos = 0

        for segment in segments:
            if len(segment) < self.min_segment_length:
                pos = text.find(segment, pos) + len(segment)
                continue

            if self._backend is None:
                continue

            try:
                result = self._backend.detect(segment)
                lang = result.language
                conf = result.confidence

                start = text.find(segment, pos)
                end = start + len(segment)
                pos = end

                if conf >= self.min_confidence_threshold and lang != "unknown":
                    spans.append(
                        CodeSwitchSpan(
                            text=segment,
                            language=lang,
                            confidence=conf,
                            start=start,
                            end=end,
                        )
                    )
                    lang_counts[lang] = lang_counts.get(lang, 0) + len(segment)
            except Exception:
                pos = text.find(segment, pos) + len(segment)

        return self._build_result(text, spans, lang_counts)

    def _build_result(
        self,
        text: str,
        spans: list[CodeSwitchSpan],
        lang_counts: dict[str, int],
    ) -> CodeSwitchResult:
        """Build the final result from spans and counts."""
        if not lang_counts:
            return CodeSwitchResult(
                is_mixed=False,
                primary_language="und",
                secondary_languages=[],
                spans=spans,
                language_distribution={},
                confidence=0.0,
            )

        # Calculate distribution
        total = sum(lang_counts.values())
        distribution = {lang: count / total for lang, count in lang_counts.items()}

        # Sort by proportion
        sorted_langs = sorted(distribution.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_langs[0][0]
        secondary = [lang for lang, _ in sorted_langs[1:] if lang != primary]

        # Determine if mixed (more than one language with significant presence)
        is_mixed = len(sorted_langs) > 1 and sorted_langs[1][1] >= 0.15

        # Calculate overall confidence
        avg_conf = sum(s.confidence for s in spans) / len(spans) if spans else 0.0

        return CodeSwitchResult(
            is_mixed=is_mixed,
            primary_language=primary,
            secondary_languages=secondary,
            spans=spans,
            language_distribution=distribution,
            confidence=avg_conf,
        )

    def get_language_spans(self, text: str) -> list[tuple[str, str]]:
        """Get simplified (text, language) pairs for each span.

        Args:
            text: Input text to analyze

        Returns:
            List of (text_segment, language) tuples
        """
        result = self.detect(text)
        return [(span.text, span.language) for span in result.spans]

    def is_code_switched(self, text: str) -> bool:
        """Quick check if text contains code-switching.

        Args:
            text: Input text to check

        Returns:
            True if multiple languages detected
        """
        return self.detect(text).is_mixed


# Common code-switching patterns by language pair
CODE_SWITCH_PATTERNS: dict[tuple[str, str], list[str]] = {
    # Spanglish patterns
    ("en", "es"): [
        r"\b(very|so|really)\s+(bueno|malo|loco|rico)",
        r"\b(muy|tan)\s+(cool|nice|good|bad)",
        r"\b(I|you|we|they)\s+(tengo|tienes|quiero|necesito)",
        r"\b(el|la|los|las)\s+\w+\s+(is|are|was|were)",
    ],
    # Franglais patterns
    ("en", "fr"): [
        r"\b(très|trop|si)\s+(cool|nice|good)",
        r"\b(c'est|il est|elle est)\s+(so|very|really)",
        r"\b(I|you|we)\s+(suis|es|sommes|êtes)",
    ],
    # Hinglish patterns
    ("en", "hi"): [
        r"\b(main|tu|hum)\s+\w+\s+(will|can|must)",
        r"\b(I|you|we)\s+(karna|chahta|jata)",
    ],
    # Denglish (German + English)
    ("en", "de"): [
        r"\b(das|es|ich)\s+(ist|bin)\s+(so|very|really)",
        r"\b(sehr|total|echt)\s+(cool|nice|good)",
    ],
}


def detect_code_switching_pattern(text: str) -> tuple[str, str] | None:
    """Detect if text matches known code-switching patterns.

    Args:
        text: Input text to check

    Returns:
        Tuple of (lang1, lang2) if pattern matched, None otherwise
    """
    text_lower = text.lower()
    for (lang1, lang2), patterns in CODE_SWITCH_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return (lang1, lang2)
    return None


__all__ = [
    "CodeSwitchDetector",
    "CodeSwitchResult",
    "CodeSwitchSpan",
    "CODE_SWITCH_PATTERNS",
    "detect_code_switching_pattern",
]
