"""PyCLD3-based language detection backend."""

from __future__ import annotations

from fastlangml.backends.base import Backend, DetectionResult


class PyCLD3Backend(Backend):
    """
    PyCLD3-based language detection backend.

    Uses Google's Compact Language Detector v3 (CLD3) via pycld3 bindings.
    CLD3 uses a neural network model and is particularly good at detecting
    languages in short text.

    Supports 107 languages.
    """

    @property
    def name(self) -> str:
        return "pycld3"

    @property
    def is_available(self) -> bool:
        try:
            import cld3  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, text: str) -> DetectionResult:
        import cld3

        if not text.strip():
            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )

        try:
            # Get top prediction
            result = cld3.get_language(text)

            if result is None:
                return DetectionResult(
                    backend_name=self.name,
                    language="unknown",
                    confidence=0.0,
                    all_probabilities={},
                    is_reliable=False,
                )

            # CLD3 returns (language, probability, is_reliable, proportion)
            lang = result.language
            prob = result.probability
            is_reliable = result.is_reliable

            # Also get top N predictions for all_probabilities
            all_probs: dict[str, float] = {}
            try:
                top_n = cld3.get_frequent_languages(text, num_langs=5)
                for pred in top_n:
                    all_probs[pred.language] = pred.probability
            except Exception:
                all_probs[lang] = prob

            return DetectionResult(
                backend_name=self.name,
                language=lang,
                confidence=prob,
                all_probabilities=all_probs,
                is_reliable=is_reliable,
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
        """CLD3 supports 107 languages."""
        return {
            "af",
            "am",
            "ar",
            "az",
            "be",
            "bg",
            "bn",
            "bs",
            "ca",
            "ceb",
            "co",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "eo",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fil",
            "fr",
            "fy",
            "ga",
            "gd",
            "gl",
            "gu",
            "ha",
            "haw",
            "he",
            "hi",
            "hmn",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
            "ig",
            "is",
            "it",
            "iw",
            "ja",
            "jv",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "ku",
            "ky",
            "la",
            "lb",
            "lo",
            "lt",
            "lv",
            "mg",
            "mi",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "no",
            "ny",
            "pa",
            "pl",
            "ps",
            "pt",
            "ro",
            "ru",
            "sd",
            "si",
            "sk",
            "sl",
            "sm",
            "sn",
            "so",
            "sq",
            "sr",
            "st",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tr",
            "uk",
            "ur",
            "uz",
            "vi",
            "xh",
            "yi",
            "yo",
            "zh",
            "zu",
        }
