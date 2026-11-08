"""FastLangID-based language detection backend.

FastLangID provides better accuracy for CJK languages (Japanese, Korean,
Chinese, Cantonese) compared to standard FastText models.
"""

from __future__ import annotations

from fastlangml.backends.base import Backend, DetectionResult


class FastLangIDBackend(Backend):
    """
    FastLangID-based language detection backend.

    Uses the fastlangid package which provides improved accuracy for
    Japanese, Korean, Chinese, and Cantonese detection. Supports 177 languages.
    """

    def __init__(self) -> None:
        self._langid = None

    def _ensure_initialized(self) -> None:
        if self._langid is None:
            from fastlangid.langid import LID

            self._langid = LID()

    @property
    def name(self) -> str:
        return "fastlangid"

    @property
    def is_available(self) -> bool:
        try:
            from fastlangid.langid import LID  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, text: str) -> DetectionResult:
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
            # predict() returns a bare code by default. With prob=True it returns
            # (lang, p) or [(lang, p), ...] depending on the language, so ask for
            # k > 1 to always get a list.
            ranked = self._langid.predict(text, k=3, prob=True)
            if isinstance(ranked, tuple):
                ranked = [ranked]
            probabilities = {str(lang): float(p) for lang, p in ranked}
            lang, confidence = max(probabilities.items(), key=lambda item: item[1])

            return DetectionResult(
                backend_name=self.name,
                language=lang,
                confidence=confidence,
                all_probabilities=probabilities,
                is_reliable=confidence > 0.5,
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
        """FastLangID supports 177 languages including Cantonese."""
        return {
            "af",
            "am",
            "an",
            "ar",
            "as",
            "av",
            "az",
            "ba",
            "be",
            "bg",
            "bh",
            "bn",
            "bo",
            "br",
            "bs",
            "ca",
            "ce",
            "co",
            "cs",
            "cv",
            "cy",
            "da",
            "de",
            "dv",
            "el",
            "en",
            "eo",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fr",
            "fy",
            "ga",
            "gd",
            "gl",
            "gn",
            "gu",
            "gv",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "ia",
            "id",
            "ie",
            "io",
            "is",
            "it",
            "ja",
            "jv",
            "ka",
            "kk",
            "km",
            "kn",
            "ko",
            "ku",
            "kw",
            "ky",
            "la",
            "lb",
            "li",
            "lo",
            "lt",
            "lv",
            "mg",
            "mk",
            "ml",
            "mn",
            "mr",
            "ms",
            "mt",
            "my",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "or",
            "os",
            "pa",
            "pl",
            "ps",
            "pt",
            "qu",
            "rm",
            "ro",
            "ru",
            "sa",
            "sc",
            "sd",
            "si",
            "sk",
            "sl",
            "so",
            "sq",
            "sr",
            "su",
            "sv",
            "sw",
            "ta",
            "te",
            "tg",
            "th",
            "tk",
            "tl",
            "tr",
            "tt",
            "ug",
            "uk",
            "ur",
            "uz",
            "vi",
            "vo",
            "wa",
            "xh",
            "yi",
            "yo",
            "zh",
            "zh-yue",  # Cantonese
        }
