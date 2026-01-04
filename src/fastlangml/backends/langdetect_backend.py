"""Langdetect-based language detection backend."""

from __future__ import annotations

from fastlangml.backends.base import Backend, DetectionResult


class LangdetectBackend(Backend):
    """
    Langdetect-based language detection backend.

    Uses the langdetect package which is a port of Google's language-detection
    library from Java. Supports 55 languages.
    """

    def __init__(self, seed: int | None = 0) -> None:
        """
        Args:
            seed: Random seed for deterministic results (None for non-deterministic)
        """
        self._seed = seed
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized and self._seed is not None:
            from langdetect import DetectorFactory

            DetectorFactory.seed = self._seed
            self._initialized = True

    @property
    def name(self) -> str:
        return "langdetect"

    @property
    def is_available(self) -> bool:
        try:
            from langdetect import detect  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, text: str) -> DetectionResult:
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException

        self._ensure_initialized()

        if not text.strip():
            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )

        try:
            results = detect_langs(text)

            all_probs = {str(r.lang): r.prob for r in results}
            top_result = results[0]

            return DetectionResult(
                backend_name=self.name,
                language=str(top_result.lang),
                confidence=top_result.prob,
                all_probabilities=all_probs,
                is_reliable=top_result.prob > 0.5,
            )
        except LangDetectException:
            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )

    def supported_languages(self) -> set[str]:
        """Langdetect supports 55 languages."""
        return {
            "af",
            "ar",
            "bg",
            "bn",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "fa",
            "fi",
            "fr",
            "gu",
            "he",
            "hi",
            "hr",
            "hu",
            "id",
            "it",
            "ja",
            "kn",
            "ko",
            "lt",
            "lv",
            "mk",
            "ml",
            "mr",
            "ne",
            "nl",
            "no",
            "pa",
            "pl",
            "pt",
            "ro",
            "ru",
            "sk",
            "sl",
            "so",
            "sq",
            "sv",
            "sw",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "uk",
            "ur",
            "vi",
            "zh-cn",
            "zh-tw",
        }
