"""FastText-based language detection backend."""

from __future__ import annotations

from fastlangml.backends.base import Backend, DetectionResult


class FastTextBackend(Backend):
    """
    FastText-based language detection backend.

    Uses the fasttext-langdetect package which wraps Facebook's FastText
    language identification model trained on Wikipedia data.

    Supports 176 languages with high accuracy.
    """

    def __init__(self, low_memory: bool = False) -> None:
        """
        Args:
            low_memory: Use low memory mode (slower but uses less RAM)
        """
        self._low_memory = low_memory

    @property
    def name(self) -> str:
        return "fasttext"

    @property
    def is_available(self) -> bool:
        try:
            from ftlangdetect import detect  # noqa: F401

            return True
        except ImportError:
            return False

    def detect(self, text: str) -> DetectionResult:
        from ftlangdetect import detect

        # Clean text (fasttext expects single line)
        clean_text = " ".join(text.split())

        if not clean_text:
            return DetectionResult(
                backend_name=self.name,
                language="unknown",
                confidence=0.0,
                all_probabilities={},
                is_reliable=False,
            )

        try:
            result = detect(text=clean_text, low_memory=self._low_memory)

            lang = result.get("lang", "unknown")
            score = float(result.get("score", 0.0))

            return DetectionResult(
                backend_name=self.name,
                language=lang,
                confidence=score,
                all_probabilities={lang: score},
                is_reliable=score > 0.5,
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
        """FastText supports 176 languages."""
        return {
            "af", "als", "am", "an", "ar", "arz", "as", "ast", "av", "az",
            "azb", "ba", "bar", "bcl", "be", "bg", "bh", "bn", "bo", "bpy",
            "br", "bs", "bxr", "ca", "cbk", "ce", "ceb", "ckb", "co", "cs",
            "cv", "cy", "da", "de", "diq", "dsb", "dty", "dv", "el", "eml",
            "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "frr", "fy",
            "ga", "gd", "gl", "gn", "gom", "gu", "gv", "he", "hi", "hif",
            "hr", "hsb", "ht", "hu", "hy", "ia", "id", "ie", "ilo", "io",
            "is", "it", "ja", "jbo", "jv", "ka", "kk", "km", "kn", "ko",
            "krc", "ku", "kv", "kw", "ky", "la", "lb", "lez", "li", "lmo",
            "lo", "lrc", "lt", "lv", "mai", "mg", "mhr", "min", "mk", "ml",
            "mn", "mr", "mrj", "ms", "mt", "mwl", "my", "myv", "mzn", "nah",
            "nap", "nds", "ne", "new", "nl", "nn", "no", "oc", "or", "os",
            "pa", "pam", "pfl", "pl", "pms", "pnb", "ps", "pt", "qu", "rm",
            "ro", "ru", "rue", "sa", "sah", "sc", "scn", "sco", "sd", "sh",
            "si", "sk", "sl", "so", "sq", "sr", "su", "sv", "sw", "ta",
            "te", "tg", "th", "tk", "tl", "tr", "tt", "tyv", "ug", "uk",
            "ur", "uz", "vec", "vep", "vi", "vls", "vo", "wa", "war", "wuu",
            "xal", "xmf", "yi", "yo", "yue", "zh",
        }
