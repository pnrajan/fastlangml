"""Tests for custom backend registration via @backend decorator."""

import pytest

from fastlangml import (
    Backend,
    DetectionConfig,
    FastLangDetector,
    backend,
    list_registered_backends,
    register_backend,
    unregister_backend,
)
from fastlangml.backends import (
    _BACKEND_RELIABILITY,
    clear_registered_backends,
    create_backend,
    get_available_backends,
)
from fastlangml.backends.base import DetectionResult


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up custom backends before and after each test."""
    clear_registered_backends()
    yield
    clear_registered_backends()
    FastLangDetector.reset_default()


class TestBackendDecorator:
    """Tests for @backend decorator registration."""

    def test_decorator_registers_backend(self):
        """Test that @backend decorator registers the class."""

        @backend("test_backend", reliability=4)
        class TestBackend(Backend):
            @property
            def name(self) -> str:
                return "test_backend"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("test_backend", "en", 0.9)

            def supported_languages(self) -> set[str]:
                return {"en", "fr"}

        assert "test_backend" in list_registered_backends()
        assert _BACKEND_RELIABILITY.get("test_backend") == 4

    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""

        @backend("returnable")
        class ReturnableBackend(Backend):
            @property
            def name(self) -> str:
                return "returnable"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("returnable", "en", 0.8)

            def supported_languages(self) -> set[str]:
                return {"en"}

        # Can instantiate the class
        instance = ReturnableBackend()
        assert instance.name == "returnable"

    def test_decorated_backend_detects(self):
        """Test detection using a decorated backend."""

        @backend("french_detector", reliability=5)
        class FrenchDetector(Backend):
            @property
            def name(self) -> str:
                return "french_detector"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("french_detector", "fr", 0.95)

            def supported_languages(self) -> set[str]:
                return {"fr"}

        detector = FastLangDetector(
            config=DetectionConfig(backends=["french_detector"])
        )
        result = detector.detect("Bonjour")

        assert result.lang == "fr"
        assert result.backend == "french_detector"

    def test_decorator_default_reliability(self):
        """Test that default reliability is 3."""

        @backend("default_rel")
        class DefaultRelBackend(Backend):
            @property
            def name(self) -> str:
                return "default_rel"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("default_rel", "en", 0.5)

            def supported_languages(self) -> set[str]:
                return {"en"}

        assert _BACKEND_RELIABILITY.get("default_rel") == 3


class TestRegisterBackendFunction:
    """Tests for register_backend function."""

    def test_register_basic(self):
        """Test basic backend registration."""

        class SimpleBackend(Backend):
            @property
            def name(self) -> str:
                return "simple"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("simple", "en", 0.9)

            def supported_languages(self) -> set[str]:
                return {"en"}

        register_backend("simple", SimpleBackend)
        assert "simple" in list_registered_backends()

    def test_cannot_override_builtin(self):
        """Test that built-in backends cannot be overridden."""

        class FakeBackend(Backend):
            @property
            def name(self) -> str:
                return "fasttext"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("fasttext", "en", 0.9)

            def supported_languages(self) -> set[str]:
                return {"en"}

        with pytest.raises(ValueError, match="conflicts with built-in"):
            register_backend("fasttext", FakeBackend)

    def test_invalid_backend_class(self):
        """Test that non-Backend classes are rejected."""
        with pytest.raises(TypeError, match="must extend Backend"):
            register_backend("invalid", str)  # type: ignore

    def test_invalid_reliability(self):
        """Test that invalid reliability scores are rejected."""

        class ValidBackend(Backend):
            @property
            def name(self) -> str:
                return "valid"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("valid", "en", 0.9)

            def supported_languages(self) -> set[str]:
                return {"en"}

        with pytest.raises(ValueError, match="must be 1-5"):
            register_backend("valid", ValidBackend, reliability=0)

        with pytest.raises(ValueError, match="must be 1-5"):
            register_backend("valid", ValidBackend, reliability=10)


class TestUnregisterBackend:
    """Tests for unregister_backend function."""

    def test_unregister_existing(self):
        """Test unregistering an existing backend."""

        @backend("removable")
        class RemovableBackend(Backend):
            @property
            def name(self) -> str:
                return "removable"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("removable", "en", 0.9)

            def supported_languages(self) -> set[str]:
                return {"en"}

        assert "removable" in list_registered_backends()

        result = unregister_backend("removable")

        assert result is True
        assert "removable" not in list_registered_backends()

    def test_unregister_nonexistent(self):
        """Test unregistering a non-existent backend."""
        result = unregister_backend("nonexistent")
        assert result is False

    def test_cannot_unregister_builtin(self):
        """Test that built-in backends cannot be unregistered."""
        with pytest.raises(ValueError, match="Cannot unregister built-in"):
            unregister_backend("fasttext")


class TestCustomBackendInEnsemble:
    """Tests for using custom backends in ensemble detection."""

    def test_custom_backend_in_available_list(self):
        """Test that custom backend appears in available list."""

        @backend("available_test")
        class AvailableBackend(Backend):
            @property
            def name(self) -> str:
                return "available_test"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult("available_test", "en", 0.9)

            def supported_languages(self) -> set[str]:
                return {"en"}

        available = get_available_backends()
        assert "available_test" in available

    def test_ensemble_with_custom_backends(self):
        """Test ensemble voting with multiple custom backends."""

        @backend("english_voter", reliability=3)
        class EnglishVoter(Backend):
            @property
            def name(self) -> str:
                return "english_voter"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(
                    "english_voter", "en", 0.9,
                    all_probabilities={"en": 0.9, "fr": 0.1}
                )

            def supported_languages(self) -> set[str]:
                return {"en", "fr"}

        @backend("french_voter", reliability=5)
        class FrenchVoter(Backend):
            @property
            def name(self) -> str:
                return "french_voter"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                return DetectionResult(
                    "french_voter", "fr", 0.95,
                    all_probabilities={"fr": 0.95, "en": 0.05}
                )

            def supported_languages(self) -> set[str]:
                return {"en", "fr"}

        detector = FastLangDetector(
            config=DetectionConfig(
                backends=["english_voter", "french_voter"],
                voting_strategy="weighted",
            )
        )

        # French voter has higher reliability, should win
        result = detector.detect("Hello")
        assert result.backend == "ensemble"


class TestBackendInterface:
    """Tests verifying the Backend interface."""

    def test_batch_detection(self):
        """Test batch detection uses default implementation."""

        @backend("batch_test")
        class BatchBackend(Backend):
            @property
            def name(self) -> str:
                return "batch_test"

            @property
            def is_available(self) -> bool:
                return True

            def detect(self, text: str) -> DetectionResult:
                # Return different language based on text
                lang = "fr" if "bonjour" in text.lower() else "en"
                return DetectionResult("batch_test", lang, 0.9)

            def supported_languages(self) -> set[str]:
                return {"en", "fr"}

        backend_instance = create_backend("batch_test")
        results = backend_instance.detect_batch(["Hello", "Bonjour", "World"])

        assert len(results) == 3
        assert results[0].language == "en"
        assert results[1].language == "fr"
        assert results[2].language == "en"
