"""Lingua-based language detection backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastlangml.backends.base import Backend, DetectionResult

if TYPE_CHECKING:
    pass


class LinguaBackend(Backend):
    """
    Lingua-based language detection backend.

    Uses the lingua-language-detector package which provides highly accurate
    language detection, especially for short text and mixed-language content.

    Supports 75 languages with excellent accuracy for short texts.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        minimum_relative_distance: float = 0.0,
        preload_models: bool = False,
        low_accuracy_mode: bool = False,
    ) -> None:
        """
        Args:
            languages: Restrict detection to these ISO 639-1 codes (None = all)
            minimum_relative_distance: Minimum distance between top results (0.0-0.99)
            preload_models: Preload all language models (uses more memory but faster)
            low_accuracy_mode: Use low accuracy mode (faster but less accurate)
        """
        self._languages = languages
        self._min_distance = minimum_relative_distance
        self._preload = preload_models
        self._low_accuracy = low_accuracy_mode
        self._detector = None

    def _get_detector(self) -> object:
        if self._detector is None:
            from lingua import Language, LanguageDetectorBuilder

            if self._languages:
                # Map ISO 639-1 to Lingua Language enum
                lang_enums = []
                for code in self._languages:
                    code_upper = code.upper()
                    if hasattr(Language, code_upper):
                        lang_enums.append(getattr(Language, code_upper))
                if lang_enums:
                    builder = LanguageDetectorBuilder.from_languages(*lang_enums)
                else:
                    builder = LanguageDetectorBuilder.from_all_languages()
            else:
                builder = LanguageDetectorBuilder.from_all_languages()

            if self._min_distance > 0:
                builder = builder.with_minimum_relative_distance(self._min_distance)

            if self._preload:
                builder = builder.with_preloaded_language_models()

            if self._low_accuracy:
                builder = builder.with_low_accuracy_mode()

            self._detector = builder.build()

        return self._detector

    @property
    def name(self) -> str:
        return "lingua"

    @property
    def is_available(self) -> bool:
        try:
            from lingua import LanguageDetectorBuilder  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, text: str) -> DetectionResult:
        if not text.strip():
            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )

        try:
            detector = self._get_detector()
            confidence_values = detector.compute_language_confidence_values(text)

            all_probs: dict[str, float] = {}
            for cv in confidence_values:
                # Convert Lingua Language enum to ISO 639-1
                iso_code = cv.language.iso_code_639_1.name.lower()
                all_probs[iso_code] = cv.value

            if confidence_values:
                top = confidence_values[0]
                top_lang = top.language.iso_code_639_1.name.lower()
                return DetectionResult(
                    backend_name=self.name,
                    language=top_lang,
                    confidence=top.value,
                    all_probabilities=all_probs,
                    is_reliable=top.value > 0.5,
                )

            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )
        except Exception:
            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )

    def supported_languages(self) -> set[str]:
        """Lingua supports 75 languages."""
        return {
            "af",
            "sq",
            "ar",
            "hy",
            "az",
            "eu",
            "be",
            "bn",
            "nb",
            "bs",
            "bg",
            "ca",
            "zh",
            "hr",
            "cs",
            "da",
            "nl",
            "en",
            "eo",
            "et",
            "fi",
            "fr",
            "lg",
            "ka",
            "de",
            "el",
            "gu",
            "he",
            "hi",
            "hu",
            "is",
            "id",
            "ga",
            "it",
            "ja",
            "kk",
            "ko",
            "la",
            "lv",
            "lt",
            "mk",
            "ms",
            "mi",
            "mr",
            "mn",
            "nn",
            "fa",
            "pl",
            "pt",
            "pa",
            "ro",
            "ru",
            "sr",
            "sn",
            "sk",
            "sl",
            "so",
            "st",
            "es",
            "sw",
            "sv",
            "tl",
            "ta",
            "te",
            "th",
            "ts",
            "tn",
            "tr",
            "uk",
            "ur",
            "vi",
            "cy",
            "xh",
            "yo",
            "zu",
        }
