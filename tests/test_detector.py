"""Tests for main detector."""

import pytest

from fastlangml.context.conversation import ConversationContext
from fastlangml.detector import (
    DetectionConfig,
    FastLangDetector,
    FastLangDetectorBuilder,
)
from fastlangml.exceptions import NoBackendsAvailableError
from fastlangml.hints.dictionary import HintDictionary
from fastlangml.result import DetectionResult


class TestFastLangDetector:
    """Tests for FastLangDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector for testing."""
        # Skip if no backends available
        try:
            return FastLangDetector()
        except NoBackendsAvailableError:
            pytest.skip("No backends available")

    def test_detect_english(self, detector):
        """Test detecting English."""
        result = detector.detect("Hello, how are you today?")
        assert isinstance(result, DetectionResult)
        assert result.lang == "en"

    def test_detect_french(self, detector):
        """Test detecting French."""
        result = detector.detect("Bonjour, comment allez-vous?")
        assert isinstance(result, DetectionResult)
        assert result.lang == "fr"

    def test_detect_spanish(self, detector):
        """Test detecting Spanish."""
        result = detector.detect("Hola, como estas?")
        assert isinstance(result, DetectionResult)
        assert result.lang == "es"

    def test_detect_german(self, detector):
        """Test detecting German."""
        result = detector.detect("Guten Tag, wie geht es Ihnen?")
        assert isinstance(result, DetectionResult)
        assert result.lang == "de"

    def test_detect_top_k(self, detector):
        """Test getting top-k results."""
        result = detector.detect("Hello world", top_k=3)
        assert isinstance(result, DetectionResult)
        # top_k > 1 populates candidates list
        assert isinstance(result.candidates, list)

    def test_detect_short_mode(self, detector):
        """Test short mode for tiny strings."""
        result = detector.detect("Hi", mode="short")
        assert isinstance(result, DetectionResult)
        # Should return something (might be 'und' for too short)

    def test_detect_batch(self, detector):
        """Test batch detection."""
        texts = ["Hello", "Bonjour", "Hola"]
        results = detector.detect_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, DetectionResult) for r in results)

    def test_set_languages(self, detector):
        """Test setting allowed languages."""
        detector.set_languages(["en", "fr"])

        result = detector.detect("Hello world")
        assert isinstance(result, DetectionResult)
        assert result.lang in ["en", "fr", "und"]

    def test_add_hint(self, detector):
        """Test adding hints."""
        detector.add_hint("merci", "fr")
        assert "merci" in detector.hints

    def test_remove_hint(self, detector):
        """Test removing hints."""
        detector.add_hint("merci", "fr")
        detector.remove_hint("merci")
        assert "merci" not in detector.hints

    def test_with_context(self, detector):
        """Test detection with context."""
        context = ConversationContext()
        context.add_turn("Bonjour", detected_language="fr", confidence=0.9)
        context.add_turn("Comment ca va?", detected_language="fr", confidence=0.9)

        result = detector.detect("Tres bien!", context=context)
        assert isinstance(result, DetectionResult)
        # Context should help identify French
        assert result.lang is not None

    def test_with_hints(self, detector):
        """Test detection with hints."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")

        result = detector.detect("Bonjour!", hints=hints)
        assert isinstance(result, DetectionResult)
        assert result.lang == "fr"

    def test_empty_text(self, detector):
        """Test with empty text."""
        result = detector.detect("")
        assert isinstance(result, DetectionResult)
        assert result.lang == "und"
        assert result.reason is not None

    def test_available_backends(self, detector):
        """Test getting available backends."""
        backends = detector.available_backends
        assert isinstance(backends, list)
        assert len(backends) > 0


class TestFastLangDetectorBuilder:
    """Tests for FastLangDetectorBuilder."""

    def test_build_default(self):
        """Test building with defaults."""
        try:
            detector = FastLangDetectorBuilder().build()
            assert isinstance(detector, FastLangDetector)
        except NoBackendsAvailableError:
            pytest.skip("No backends available")

    def test_with_backends(self):
        """Test specifying backends."""
        builder = FastLangDetectorBuilder().with_backends("langdetect")
        try:
            detector = builder.build()
            assert "langdetect" in detector.available_backends
        except NoBackendsAvailableError:
            pytest.skip("langdetect not available")

    def test_with_voting_strategy(self):
        """Test setting voting strategy."""
        try:
            detector = (
                FastLangDetectorBuilder()
                .with_voting_strategy("hard")
                .build()
            )
            assert isinstance(detector, FastLangDetector)
        except NoBackendsAvailableError:
            pytest.skip("No backends available")

    def test_with_proper_noun_filtering(self):
        """Test setting proper noun filtering."""
        try:
            detector = (
                FastLangDetectorBuilder()
                .with_proper_noun_filtering("mask")
                .build()
            )
            assert isinstance(detector, FastLangDetector)
        except NoBackendsAvailableError:
            pytest.skip("No backends available")

    def test_with_hints(self):
        """Test providing hints."""
        hints = HintDictionary()
        hints.add("bonjour", "fr")

        try:
            detector = (
                FastLangDetectorBuilder()
                .with_hints(hints)
                .build()
            )
            assert "bonjour" in detector.hints
        except NoBackendsAvailableError:
            pytest.skip("No backends available")


class TestDetectionConfig:
    """Tests for DetectionConfig."""

    def test_defaults(self):
        """Test default configuration."""
        config = DetectionConfig()
        assert config.mode == "default"
        assert config.filter_proper_nouns is True
        assert config.use_script_filter is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = DetectionConfig(
            mode="short",
            thresholds={"short": 0.3, "default": 0.7, "long": 0.6},
            filter_proper_nouns=False,
        )
        assert config.mode == "short"
        assert config.thresholds["default"] == 0.7
        assert config.filter_proper_nouns is False


class TestDetectionResult:
    """Tests for DetectionResult."""

    def test_repr(self):
        """Test string representation."""
        result = DetectionResult(lang="en", confidence=0.95)
        repr_str = repr(result)
        assert "en" in repr_str
        assert "0.95" in repr_str

    def test_defaults(self):
        """Test default values."""
        result = DetectionResult(lang="en", confidence=0.9)
        assert result.reliable is True
        assert result.script is None
        assert result.candidates == []

    def test_und_repr(self):
        """Test 'und' result representation."""
        result = DetectionResult(lang="und", confidence=0.0, reason="too_little_text")
        repr_str = repr(result)
        assert "und" in repr_str
        assert "too_little_text" in repr_str
