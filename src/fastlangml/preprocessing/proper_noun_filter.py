"""Proper noun filtering for language detection.

Filters out proper nouns from text before language detection to avoid
biasing results towards languages that match names rather than the
actual content language.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import spacy


class ProperNounFilter:
    """
    Filter proper nouns from text before language detection.

    Uses a combination of:
    1. Capitalization heuristics (lightweight, no dependencies)
    2. Optional spaCy NER (most accurate, requires spaCy + model)
    """

    # Entity types to filter (PERSON, ORG, GPE are most common proper nouns)
    DEFAULT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}

    def __init__(
        self,
        strategy: Literal["remove", "mask", "none"] = "remove",
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
        entity_types: set[str] | None = None,
    ) -> None:
        """
        Args:
            strategy: How to handle proper nouns
                - "remove": Remove proper nouns from text
                - "mask": Replace with [NAME] placeholder
                - "none": Do not filter
            use_spacy: Use spaCy NER for accurate entity detection
            spacy_model: spaCy model to use (default: en_core_web_sm)
            entity_types: Entity types to filter. Default: PERSON, ORG, GPE, LOC, FAC, NORP
        """
        self._strategy = strategy
        self._use_spacy = use_spacy
        self._spacy_model_name = spacy_model
        self._entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self._nlp: spacy.Language | None = None

        if use_spacy:
            self._nlp = self._load_spacy_model()

        # Common title words to preserve (not proper nouns)
        self._common_words = {
            # English
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "shall", "can", "need", "dare", "ought",
            "used", "i", "you", "he", "she", "it", "we", "they", "what",
            "which", "who", "whom", "this", "that", "these", "those", "am",
            "and", "or", "but", "if", "then", "else", "when", "where", "why",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "also", "now", "here", "there",
            # Common words in other languages that might appear capitalized
            "der", "die", "das", "und", "oder",  # German
            "le", "la", "les", "et", "ou",  # French
            "el", "los", "las", "y", "o",  # Spanish
            "il", "lo", "gli", "e",  # Italian
        }

        # Pattern for potential proper nouns (capitalized words not at sentence start)
        self._proper_noun_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

    def _load_spacy_model(self) -> spacy.Language | None:
        """Load spaCy model for NER."""
        try:
            import spacy

            return spacy.load(self._spacy_model_name)
        except (ImportError, OSError):
            # spaCy not installed or model not found
            return None

    @property
    def spacy_available(self) -> bool:
        """Check if spaCy NER is available and loaded."""
        return self._nlp is not None

    def filter(self, text: str) -> str:
        """
        Filter proper nouns from text.

        Args:
            text: Input text

        Returns:
            Text with proper nouns removed/masked based on strategy
        """
        if self._strategy == "none":
            return text

        if self._use_spacy and self._nlp is not None:
            return self._filter_with_spacy(text)

        return self._filter_with_heuristics(text)

    def _filter_with_spacy(self, text: str) -> str:
        """
        spaCy NER-based proper noun filtering.

        Uses named entity recognition to identify and filter entities
        like PERSON, ORG, GPE (geopolitical entity), LOC, etc.
        """
        if self._nlp is None:
            return self._filter_with_heuristics(text)

        doc = self._nlp(text)
        result = text

        # Process entities in reverse order to preserve character positions
        for ent in reversed(doc.ents):
            if ent.label_ in self._entity_types:
                if self._strategy == "mask":
                    result = result[: ent.start_char] + "[NAME]" + result[ent.end_char :]
                else:  # remove
                    # Remove entity and cleanup extra spaces
                    result = result[: ent.start_char] + result[ent.end_char :]

        # Clean up multiple spaces
        result = re.sub(r"\s+", " ", result).strip()
        return result

    def _filter_with_heuristics(self, text: str) -> str:
        """
        Heuristic-based proper noun filtering.

        Strategy:
        1. Words starting with uppercase that are NOT at sentence start
        2. Sequences of capitalized words (multi-word names)
        3. Preserve common words that might be capitalized
        """
        sentences = self._split_sentences(text)
        filtered_sentences = []

        for sentence in sentences:
            words = sentence.split()
            filtered_words = []

            for i, word in enumerate(words):
                # Check if word is potentially a proper noun
                clean_word = re.sub(r"[^\w]", "", word).lower()

                is_first_word = i == 0
                is_capitalized = bool(word) and word[0].isupper()
                is_common = clean_word in self._common_words
                is_all_caps = word.isupper() and len(word) > 1
                is_numeric = any(c.isdigit() for c in word)

                # Keep word if:
                # - It's lowercase
                # - It's first word of sentence (capitalization expected)
                # - It's a common word
                # - It's an acronym/all caps (might be important)
                # - It contains numbers
                if (
                    not is_capitalized
                    or is_first_word
                    or is_common
                    or is_numeric
                ):
                    filtered_words.append(word)
                elif is_all_caps:
                    # Might be an acronym, keep it
                    filtered_words.append(word)
                elif self._strategy == "mask":
                    filtered_words.append("[NAME]")
                # else: remove (don't append)

            if filtered_words:
                filtered_sentences.append(" ".join(filtered_words))

        return " ".join(filtered_sentences)

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting."""
        # Split on common sentence endings
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def identify_proper_nouns(self, text: str) -> list[str]:
        """
        Return list of identified proper nouns (for debugging/inspection).

        Args:
            text: Input text

        Returns:
            List of identified proper nouns
        """
        if self._use_spacy and self._nlp is not None:
            doc = self._nlp(text)
            return [ent.text for ent in doc.ents if ent.label_ in self._entity_types]

        # Heuristic approach
        matches = self._proper_noun_pattern.findall(text)
        # Filter out sentence starters
        sentences = self._split_sentences(text)
        first_words = {s.split()[0] if s.split() else "" for s in sentences}

        return [m for m in matches if m not in first_words]
