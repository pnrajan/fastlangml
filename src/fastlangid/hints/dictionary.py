"""Runtime dictionary for word-to-language hints.

Allows users to specify that certain words should indicate a specific
language, boosting detection accuracy for domain-specific text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HintDictionary:
    """Runtime dictionary for word-to-language hints.

    Provides a way to define single-token words that should indicate
    specific languages. Useful for domain-specific vocabulary, chat
    slang, or abbreviations that standard detectors miss.

    Hints are checked during detection and boost confidence for matching
    languages. Supports both exact matching and fuzzy matching for typos.

    Attributes:
        case_sensitive: Whether word matching is case-sensitive.
            Defaults to False (case-insensitive).
        hint_confidence: Base confidence score (0.0-1.0) assigned to
            hint matches. Defaults to 0.8.

    Example:
        >>> hints = HintDictionary()
        >>> hints.add("bonjour", "fr")
        >>> hints.add("gracias", "es")
        >>> hints.lookup("Bonjour tout le monde")
        ('fr', 0.8)

        >>> # Load built-in chat/slang hints
        >>> hints = HintDictionary.default_short_words()
        >>> hints.lookup("thx")
        ('en', 0.8)
    """

    _words: dict[str, str] = field(default_factory=dict)
    case_sensitive: bool = False
    hint_confidence: float = 0.8
    """Base confidence to assign to hint matches."""

    def add(self, word: str, language: str) -> None:
        """Add a word-to-language hint.

        Args:
            word: Single token word to match. Will be normalized
                according to case_sensitive setting.
            language: ISO 639-1 language code (e.g., "en", "fr", "de").

        Raises:
            ValueError: If word is empty or contains spaces.

        Example:
            >>> hints = HintDictionary()
            >>> hints.add("merci", "fr")
            >>> hints.add("danke", "de")
        """
        normalized = self._normalize(word)
        if not normalized:
            raise ValueError("Word must be a non-empty single token")
        if " " in normalized:
            raise ValueError("Word must be a single token (no spaces)")
        self._words[normalized] = language.lower()

    def add_many(self, hints: dict[str, str]) -> None:
        """
        Add multiple word -> language hints.

        Args:
            hints: Dict mapping words to language codes
        """
        for word, lang in hints.items():
            self.add(word, lang)

    def remove(self, word: str) -> None:
        """Remove a hint."""
        normalized = self._normalize(word)
        self._words.pop(normalized, None)

    def get(self, word: str) -> str | None:
        """Get the language for a specific word."""
        normalized = self._normalize(word)
        return self._words.get(normalized)

    def lookup(self, text: str, fuzzy: bool = True) -> tuple[str, float] | None:
        """
        Look up any matching hints in the text.

        Args:
            text: Input text to search for hints
            fuzzy: Allow fuzzy matching for spelling mistakes (1 edit distance)

        Returns:
            Tuple of (language, confidence) if hint found, None otherwise
        """
        if not self._words:
            return None

        # Tokenize text
        tokens = re.findall(r"\b\w+\b", text)

        # Count hint matches per language
        matches: dict[str, int] = {}
        fuzzy_matches: dict[str, int] = {}

        for token in tokens:
            normalized = self._normalize(token)

            # Exact match first
            if normalized in self._words:
                lang = self._words[normalized]
                matches[lang] = matches.get(lang, 0) + 1
            elif fuzzy and len(normalized) >= 3:
                # Try fuzzy match for words 3+ chars
                fuzzy_lang = self._fuzzy_lookup(normalized)
                if fuzzy_lang:
                    fuzzy_matches[fuzzy_lang] = fuzzy_matches.get(fuzzy_lang, 0) + 1

        # Prefer exact matches
        if matches:
            best_lang = max(matches, key=lambda k: matches[k])
            match_boost = min(matches[best_lang] * 0.05, 0.15)
            confidence = min(self.hint_confidence + match_boost, 1.0)
            return best_lang, confidence

        # Fall back to fuzzy matches with lower confidence
        if fuzzy_matches:
            best_lang = max(fuzzy_matches, key=lambda k: fuzzy_matches[k])
            confidence = self.hint_confidence * 0.7  # Lower confidence for fuzzy
            return best_lang, confidence

        return None

    def _fuzzy_lookup(self, word: str) -> str | None:
        """Find a hint word using fast fuzzy matching."""
        try:
            from rapidfuzz import process, fuzz

            # Use rapidfuzz for fast matching
            result = process.extractOne(
                word,
                list(self._words.keys()),
                scorer=fuzz.ratio,
                score_cutoff=80,  # 80% similarity threshold
            )
            if result:
                matched_word, score, _ = result
                return self._words[matched_word]
        except ImportError:
            # Fallback to simple edit distance
            for hint_word, lang in self._words.items():
                if len(hint_word) >= 3 and abs(len(hint_word) - len(word)) <= 1:
                    if self._edit_distance_one(word, hint_word):
                        return lang
        return None

    @staticmethod
    def _edit_distance_one(s1: str, s2: str) -> bool:
        """Check if two strings are within edit distance 1."""
        if s1 == s2:
            return True
        if abs(len(s1) - len(s2)) > 1:
            return False
        if len(s1) == len(s2):
            return sum(c1 != c2 for c1, c2 in zip(s1, s2)) == 1
        longer, shorter = (s1, s2) if len(s1) > len(s2) else (s2, s1)
        i = j = diff = 0
        while i < len(longer) and j < len(shorter):
            if longer[i] != shorter[j]:
                if diff:
                    return False
                diff = 1
                i += 1
            else:
                i += 1
                j += 1
        return True

    def lookup_all(self, text: str) -> dict[str, float]:
        """
        Look up all matching hints in the text.

        Args:
            text: Input text to search for hints

        Returns:
            Dict mapping language codes to match scores
        """
        if not self._words:
            return {}

        tokens = re.findall(r"\b\w+\b", text)

        matches: dict[str, int] = {}
        total_matches = 0

        for token in tokens:
            normalized = self._normalize(token)
            if normalized in self._words:
                lang = self._words[normalized]
                matches[lang] = matches.get(lang, 0) + 1
                total_matches += 1

        if not matches or total_matches == 0:
            return {}

        # Convert to scores
        return {
            lang: (count / total_matches) * self.hint_confidence
            for lang, count in matches.items()
        }

    def merge(self, other: HintDictionary | None) -> HintDictionary:
        """
        Create a new dictionary merging this one with another.

        The other dictionary takes precedence for conflicts.

        Args:
            other: Another HintDictionary to merge with

        Returns:
            New merged HintDictionary
        """
        if other is None:
            return self

        merged = HintDictionary(
            case_sensitive=self.case_sensitive,
            hint_confidence=self.hint_confidence,
        )
        merged._words = {**self._words, **other._words}
        return merged

    def _normalize(self, word: str) -> str:
        """Normalize a word for lookup."""
        if self.case_sensitive:
            return word.strip()
        return word.strip().lower()

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        return self._normalize(word) in self._words

    def __iter__(self):
        return iter(self._words.items())

    def items(self):
        """Return word-language pairs."""
        return self._words.items()

    def words(self) -> list[str]:
        """Return all words in the dictionary."""
        return list(self._words.keys())

    def languages(self) -> set[str]:
        """Return all languages in the dictionary."""
        return set(self._words.values())

    def to_dict(self) -> dict[str, str]:
        """Export hints as a dictionary."""
        return dict(self._words)

    @classmethod
    def from_dict(cls, hints: dict[str, str], **kwargs) -> HintDictionary:
        """Create a HintDictionary from a dictionary."""
        d = cls(**kwargs)
        d.add_many(hints)
        return d

    # Module-level cache for default hints (loaded once)
    _default_hints_cache: HintDictionary | None = None

    @classmethod
    def default_short_words(cls) -> HintDictionary:
        """Create a dictionary with common unambiguous short words from JSON file.

        Results are cached at module level for performance.
        """
        if cls._default_hints_cache is not None:
            return cls._default_hints_cache

        import json
        from pathlib import Path

        data_file = Path(__file__).parent.parent / "data" / "default_hints.json"

        hints: dict[str, str] = {}
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for lang, words in data.items():
                    for word in words:
                        hints[word] = lang

        cls._default_hints_cache = cls.from_dict(hints)
        return cls._default_hints_cache
