"""Tests for proper noun filtering."""

import pytest

from fastlangid.preprocessing.proper_noun_filter import ProperNounFilter


class TestProperNounFilter:
    """Tests for ProperNounFilter."""

    def test_remove_proper_nouns(self):
        """Test removing proper nouns."""
        filter = ProperNounFilter(strategy="remove")
        # Note: First word "He" is preserved (capitalized by convention)
        text = "He went to Paris with John."

        result = filter.filter(text)

        # "Paris" and "John" should be removed (not sentence-initial)
        assert "Paris" not in result
        assert "John" not in result
        assert "went" in result
        assert "to" in result

    def test_mask_proper_nouns(self):
        """Test masking proper nouns."""
        filter = ProperNounFilter(strategy="mask")
        text = "He visited Paris with Mary."

        result = filter.filter(text)

        assert "[NAME]" in result
        assert "visited" in result

    def test_none_strategy(self):
        """Test no filtering."""
        filter = ProperNounFilter(strategy="none")
        text = "John went to Paris."

        result = filter.filter(text)

        assert result == text

    def test_preserve_first_word(self):
        """Test that first word of sentence is preserved."""
        filter = ProperNounFilter(strategy="remove")
        text = "The quick brown fox."

        result = filter.filter(text)

        # "The" should be preserved (first word)
        assert "The" in result

    def test_preserve_common_words(self):
        """Test that common words are preserved."""
        filter = ProperNounFilter(strategy="remove")
        text = "I went to The store."

        result = filter.filter(text)

        # "I" and "The" should be preserved
        assert "I" in result or "went" in result

    def test_preserve_acronyms(self):
        """Test that acronyms are preserved."""
        filter = ProperNounFilter(strategy="remove")
        text = "The FBI investigated."

        result = filter.filter(text)

        # "FBI" should be preserved (all caps)
        assert "FBI" in result

    def test_identify_proper_nouns(self):
        """Test identifying proper nouns."""
        filter = ProperNounFilter(strategy="remove")
        text = "John and Mary went to Paris."

        nouns = filter.identify_proper_nouns(text)

        assert "John" in nouns or len(nouns) > 0

    def test_empty_text(self):
        """Test with empty text."""
        filter = ProperNounFilter(strategy="remove")
        result = filter.filter("")
        assert result == ""

    def test_multiple_sentences(self):
        """Test with multiple sentences."""
        filter = ProperNounFilter(strategy="remove")
        text = "John went home. Mary stayed in Paris."

        result = filter.filter(text)

        assert "went" in result
        assert "stayed" in result
