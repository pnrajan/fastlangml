"""Langid-based language detection backend."""

from __future__ import annotations

from fastlangml.backends.base import Backend, DetectionResult


class LangidBackend(Backend):
    """
    Langid-based language detection backend.

    Uses the langid.py package which is based on the paper
    "langid.py: An Off-the-shelf Language Identification Tool" (2012).

    NOTE: This backend is somewhat stale (last updated 2017) but is included
    for compatibility. For best results, prefer fasttext or lingua.

    Supports 97 languages.
    """

    def __init__(self, normalize_probs: bool = True) -> None:
        """
        Args:
            normalize_probs: Normalize probabilities to sum to 1.0
        """
        self._normalize = normalize_probs
        self._classifier = None

    def _get_classifier(self) -> object:
        if self._classifier is None:
            from langid.langid import LanguageIdentifier, model

            # The module-level langid.classify() returns an unnormalized log
            # score (around -50), not a probability. norm_probs=True makes
            # classify() return a real 0-1 probability.
            self._classifier = LanguageIdentifier.from_modelstring(
                model, norm_probs=self._normalize
            )
        return self._classifier

    @property
    def name(self) -> str:
        return "langid"

    @property
    def is_available(self) -> bool:
        try:
            import langid  # noqa: F401

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
            classifier = self._get_classifier()
            lang, confidence = classifier.classify(text)

            # Without normalization langid returns a log score; it is not a
            # probability, so treat it as no confidence information.
            confidence = min(max(float(confidence), 0.0), 1.0)

            return DetectionResult(
                backend_name=self.name,
                language=lang,
                confidence=confidence,
                all_probabilities={lang: confidence},
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
        """Langid supports 97 languages."""
        return {
            "af",
            "am",
            "an",
            "ar",
            "as",
            "az",
            "be",
            "bg",
            "bn",
            "br",
            "bs",
            "ca",
            "cs",
            "cy",
            "da",
            "de",
            "dz",
            "el",
            "en",
            "eo",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fo",
            "fr",
            "ga",
            "gl",
            "gu",
            "he",
            "hi",
            "hr",
            "ht",
            "hu",
            "hy",
            "id",
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
            "ky",
            "la",
            "lb",
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
            "nb",
            "ne",
            "nl",
            "nn",
            "no",
            "oc",
            "or",
            "pa",
            "pl",
            "ps",
            "pt",
            "qu",
            "ro",
            "ru",
            "rw",
            "se",
            "si",
            "sk",
            "sl",
            "sq",
            "sr",
            "sv",
            "sw",
            "ta",
            "te",
            "th",
            "tl",
            "tr",
            "ug",
            "uk",
            "ur",
            "vi",
            "vo",
            "wa",
            "xh",
            "zh",
            "zu",
        }
