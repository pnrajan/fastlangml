"""Tests for language confusion resolution."""

from fastlangml.ensemble.confusion import (
    CONFUSED_PAIRS,
    ConfusionResolver,
    LanguageSimilarity,
)


class TestConfusionResolver:
    """Tests for ConfusionResolver."""

    def test_get_confused_pair_spanish_portuguese(self):
        """Test detecting Spanish-Portuguese confusion pair."""
        resolver = ConfusionResolver()
        pair = resolver.get_confused_pair({"es", "pt"})
        assert pair == frozenset({"es", "pt"})

    def test_get_confused_pair_scandinavian(self):
        """Test detecting Scandinavian confusion pair."""
        resolver = ConfusionResolver()
        pair = resolver.get_confused_pair({"no", "da"})
        assert pair == frozenset({"no", "da", "sv"})

    def test_get_confused_pair_not_confused(self):
        """Test that unrelated languages return None."""
        resolver = ConfusionResolver()
        pair = resolver.get_confused_pair({"en", "zh"})
        assert pair is None

    def test_resolve_spanish_portuguese(self):
        """Test resolving Spanish vs Portuguese."""
        resolver = ConfusionResolver()

        # Portuguese text with "tenho" (have)
        scores = {"es": 0.45, "pt": 0.42}
        adjusted = resolver.resolve("Eu tenho um problema", scores)

        # Portuguese should be boosted
        assert adjusted["pt"] > scores["pt"]

    def test_resolve_clear_winner(self):
        """Test that clear winners are not adjusted."""
        resolver = ConfusionResolver()

        # Clear Spanish winner
        scores = {"es": 0.85, "pt": 0.10}
        adjusted = resolver.resolve("Hola como estas", scores)

        # Should not change significantly
        assert adjusted["es"] == scores["es"]

    def test_resolve_no_features(self):
        """Test resolution with no discriminating features."""
        resolver = ConfusionResolver()

        # Use text with no Spanish or Portuguese words/characters
        scores = {"es": 0.45, "pt": 0.42}
        adjusted = resolver.resolve("xyz 123", scores)

        # Should return original scores since no features match
        assert adjusted == scores

    def test_get_discriminating_features(self):
        """Test getting discriminating features for language pair."""
        resolver = ConfusionResolver()

        features = resolver.get_discriminating_features("es", "pt")
        assert features is not None
        es_features, pt_features = features
        assert "pero" in es_features  # Spanish "but"
        assert "mas" in pt_features  # Portuguese "but"

    def test_get_discriminating_features_not_pair(self):
        """Test getting features for non-confused pair."""
        resolver = ConfusionResolver()
        features = resolver.get_discriminating_features("en", "zh")
        assert features is None


class TestLanguageSimilarity:
    """Tests for LanguageSimilarity."""

    def test_get_family_romance(self):
        """Test getting language family."""
        sim = LanguageSimilarity()
        assert sim.get_family("es") == "romance"
        assert sim.get_family("pt") == "romance"
        assert sim.get_family("fr") == "romance"

    def test_get_family_germanic(self):
        """Test Germanic family."""
        sim = LanguageSimilarity()
        assert sim.get_family("en") == "germanic"
        assert sim.get_family("de") == "germanic"

    def test_get_family_unknown(self):
        """Test unknown language."""
        sim = LanguageSimilarity()
        assert sim.get_family("xyz") is None

    def test_are_related_true(self):
        """Test related languages."""
        sim = LanguageSimilarity()
        assert sim.are_related("es", "pt") is True
        assert sim.are_related("en", "de") is True

    def test_are_related_false(self):
        """Test unrelated languages."""
        sim = LanguageSimilarity()
        assert sim.are_related("en", "zh") is False
        assert sim.are_related("es", "ru") is False

    def test_get_related_languages(self):
        """Test getting related languages."""
        sim = LanguageSimilarity()
        related = sim.get_related_languages("es")
        assert "pt" in related
        assert "fr" in related
        assert "es" not in related  # Should not include itself

    def test_similarity_score_same(self):
        """Test similarity score for same language."""
        sim = LanguageSimilarity()
        assert sim.similarity_score("en", "en") == 1.0

    def test_similarity_score_confused_pair(self):
        """Test similarity score for confused pair."""
        sim = LanguageSimilarity()
        score = sim.similarity_score("es", "pt")
        assert score == 0.9

    def test_similarity_score_same_family(self):
        """Test similarity score for same family."""
        sim = LanguageSimilarity()
        score = sim.similarity_score("en", "de")
        assert score == 0.6

    def test_similarity_score_different_families(self):
        """Test similarity score for different families."""
        sim = LanguageSimilarity()
        score = sim.similarity_score("en", "zh")
        assert score == 0.0


class TestConfusedPairs:
    """Tests for CONFUSED_PAIRS data."""

    def test_confused_pairs_structure(self):
        """Test that confused pairs have proper structure."""
        for pair, features in CONFUSED_PAIRS.items():
            assert isinstance(pair, frozenset)
            assert len(pair) >= 2
            assert isinstance(features, dict)

            for lang, words in features.items():
                assert lang in pair
                assert isinstance(words, list)
                assert len(words) > 0

    def test_spanish_portuguese_features(self):
        """Test Spanish-Portuguese discriminating features."""
        pair = frozenset({"es", "pt"})
        assert pair in CONFUSED_PAIRS

        features = CONFUSED_PAIRS[pair]
        assert "es" in features
        assert "pt" in features
        assert "pero" in features["es"]  # Spanish "but"
        assert "mas" in features["pt"]  # Portuguese "but"
