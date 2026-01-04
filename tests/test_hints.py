"""Tests for hint dictionary."""

import pytest

from fastlangid.hints.dictionary import HintDictionary


class TestHintDictionary:
    """Tests for HintDictionary."""

    def test_add_and_get(self):
        """Test adding and getting hints."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")
        hints.add("gracias", "es")

        assert hints.get("bonjour") == "fr"
        assert hints.get("gracias") == "es"
        assert hints.get("hello") is None

    def test_case_insensitive(self):
        """Test case insensitive lookup."""
        hints = HintDictionary(case_sensitive=False)
        hints.add("Bonjour", "fr")

        assert hints.get("bonjour") == "fr"
        assert hints.get("BONJOUR") == "fr"
        assert "Bonjour" in hints

    def test_case_sensitive(self):
        """Test case sensitive lookup."""
        hints = HintDictionary(case_sensitive=True)
        hints.add("Bonjour", "fr")

        assert hints.get("Bonjour") == "fr"
        assert hints.get("bonjour") is None

    def test_lookup_text(self):
        """Test looking up hints in text."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")
        hints.add("merci", "fr")

        result = hints.lookup("Bonjour, merci beaucoup!")
        assert result is not None
        assert result[0] == "fr"
        assert result[1] > 0

    def test_lookup_no_match(self):
        """Test lookup with no matches."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")

        result = hints.lookup("Hello world")
        assert result is None

    def test_remove(self):
        """Test removing hints."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")
        hints.remove("bonjour")

        assert hints.get("bonjour") is None
        assert len(hints) == 0

    def test_merge(self):
        """Test merging dictionaries."""
        hints1 = HintDictionary()
        hints1.add("bonjour", "fr")

        hints2 = HintDictionary()
        hints2.add("gracias", "es")
        hints2.add("bonjour", "en")  # Conflict

        merged = hints1.merge(hints2)

        assert merged.get("bonjour") == "en"  # hints2 takes precedence
        assert merged.get("gracias") == "es"

    def test_add_many(self):
        """Test adding multiple hints."""
        hints = HintDictionary()
        hints.add_many({"bonjour": "fr", "gracias": "es", "danke": "de"})

        assert len(hints) == 3
        assert hints.get("danke") == "de"

    def test_from_dict(self):
        """Test creating from dictionary."""
        hints = HintDictionary.from_dict({"hello": "en", "hola": "es"})

        assert hints.get("hello") == "en"
        assert hints.get("hola") == "es"

    def test_invalid_word(self):
        """Test adding invalid words."""
        hints = HintDictionary()

        with pytest.raises(ValueError):
            hints.add("", "fr")

        with pytest.raises(ValueError):
            hints.add("hello world", "en")  # Contains space

    def test_lookup_all(self):
        """Test looking up all matches."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")
        hints.add("hola", "es")

        result = hints.lookup_all("Bonjour! Hola!")
        assert "fr" in result
        assert "es" in result
