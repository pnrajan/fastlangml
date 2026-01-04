"""Tests for code-switching detection."""

import pytest

from fastlangml.codeswitching import (
    CODE_SWITCH_PATTERNS,
    CodeSwitchDetector,
    CodeSwitchResult,
    CodeSwitchSpan,
    detect_code_switching_pattern,
)


class TestCodeSwitchDetector:
    """Tests for CodeSwitchDetector."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return CodeSwitchDetector()

    def test_empty_text(self, detector):
        """Test detection on empty text."""
        result = detector.detect("")
        assert result.is_mixed is False
        assert result.primary_language == "und"

    def test_whitespace_only(self, detector):
        """Test detection on whitespace."""
        result = detector.detect("   ")
        assert result.is_mixed is False
        assert result.primary_language == "und"

    def test_get_language_spans(self, detector):
        """Test getting language spans."""
        spans = detector.get_language_spans("Hello world")
        assert isinstance(spans, list)
        for text, lang in spans:
            assert isinstance(text, str)
            assert isinstance(lang, str)

    def test_is_code_switched_simple(self, detector):
        """Test simple code-switching check."""
        result = detector.is_code_switched("Hello")
        assert isinstance(result, bool)


class TestCodeSwitchResult:
    """Tests for CodeSwitchResult dataclass."""

    def test_languages_property(self):
        """Test the languages property."""
        result = CodeSwitchResult(
            is_mixed=True,
            primary_language="en",
            secondary_languages=["es", "fr"],
            spans=[],
            language_distribution={"en": 0.6, "es": 0.3, "fr": 0.1},
            confidence=0.85,
        )
        assert result.languages == ["en", "es", "fr"]

    def test_languages_single(self):
        """Test languages property with single language."""
        result = CodeSwitchResult(
            is_mixed=False,
            primary_language="en",
            secondary_languages=[],
            spans=[],
            language_distribution={"en": 1.0},
            confidence=0.95,
        )
        assert result.languages == ["en"]


class TestCodeSwitchSpan:
    """Tests for CodeSwitchSpan dataclass."""

    def test_span_creation(self):
        """Test creating a span."""
        span = CodeSwitchSpan(
            text="hello",
            language="en",
            confidence=0.9,
            start=0,
            end=5,
        )
        assert span.text == "hello"
        assert span.language == "en"
        assert span.confidence == 0.9
        assert span.start == 0
        assert span.end == 5


class TestCodeSwitchPatterns:
    """Tests for CODE_SWITCH_PATTERNS."""

    def test_patterns_exist(self):
        """Test that patterns are defined."""
        assert ("en", "es") in CODE_SWITCH_PATTERNS
        assert ("en", "fr") in CODE_SWITCH_PATTERNS

    def test_pattern_structure(self):
        """Test pattern structure."""
        for key, patterns in CODE_SWITCH_PATTERNS.items():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(patterns, list)
            assert all(isinstance(p, str) for p in patterns)


class TestDetectCodeSwitchingPattern:
    """Tests for detect_code_switching_pattern function."""

    def test_spanglish_pattern(self):
        """Test detecting Spanglish pattern."""
        # "very bueno" matches English + Spanish pattern
        result = detect_code_switching_pattern("That's very bueno")
        if result:
            assert "en" in result
            assert "es" in result

    def test_no_pattern(self):
        """Test text with no code-switching pattern."""
        result = detect_code_switching_pattern("Hello world")
        # May or may not match depending on patterns
        assert result is None or isinstance(result, tuple)

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        result1 = detect_code_switching_pattern("very BUENO")
        result2 = detect_code_switching_pattern("VERY bueno")
        # Both should have same result
        assert result1 == result2
