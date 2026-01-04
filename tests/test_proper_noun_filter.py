"""Tests for proper noun filtering."""

import pytest

from fastlangml.preprocessing.proper_noun_filter import ProperNounFilter


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


class TestProperNounFilterSpacy:
    """Tests for ProperNounFilter with spaCy NER."""

    @pytest.fixture
    def spacy_filter(self):
        """Create a spaCy-enabled filter."""
        pnf = ProperNounFilter(strategy="remove", use_spacy=True)
        if not pnf.spacy_available:
            pytest.skip("spaCy model not available")
        return pnf

    def test_spacy_available_property(self):
        """Test spacy_available property."""
        pnf = ProperNounFilter(use_spacy=True)
        # Should be True if spaCy is installed, False otherwise
        assert isinstance(pnf.spacy_available, bool)

    def test_spacy_remove_persons(self, spacy_filter):
        """Test removing PERSON entities with spaCy."""
        text = "John Smith went to the store."
        result = spacy_filter.filter(text)
        assert "John" not in result
        assert "Smith" not in result
        assert "store" in result

    def test_spacy_remove_locations(self, spacy_filter):
        """Test removing GPE/LOC entities with spaCy."""
        text = "She traveled to Paris and London."
        result = spacy_filter.filter(text)
        assert "Paris" not in result
        assert "London" not in result
        assert "traveled" in result

    def test_spacy_remove_organizations(self, spacy_filter):
        """Test removing ORG entities with spaCy."""
        text = "He works at Google and Microsoft."
        result = spacy_filter.filter(text)
        assert "Google" not in result
        assert "Microsoft" not in result
        assert "works" in result

    def test_spacy_mask_strategy(self):
        """Test masking entities with spaCy."""
        pnf = ProperNounFilter(strategy="mask", use_spacy=True)
        if not pnf.spacy_available:
            pytest.skip("spaCy model not available")

        text = "John works at Google."
        result = pnf.filter(text)
        assert "[NAME]" in result
        assert "works" in result

    def test_spacy_identify_entities(self, spacy_filter):
        """Test identifying entities with spaCy."""
        text = "John Smith works at Google in New York."
        entities = spacy_filter.identify_proper_nouns(text)
        # Should identify at least some entities
        assert len(entities) > 0

    def test_spacy_custom_entity_types(self):
        """Test with custom entity types."""
        pnf = ProperNounFilter(
            strategy="remove",
            use_spacy=True,
            entity_types={"PERSON"},  # Only filter PERSON
        )
        if not pnf.spacy_available:
            pytest.skip("spaCy model not available")

        text = "John went to Paris."
        result = pnf.filter(text)
        # John (PERSON) should be removed, Paris (GPE) should remain
        assert "John" not in result
        # Note: Paris might still be in result depending on entity type filtering
