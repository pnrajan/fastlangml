"""Built-in backend adapters return usable votes.

Each test is skipped when its backend isn't installed. CI installs
langdetect, langid, and lingua.
"""

from __future__ import annotations

import pytest

from fastlangml.backends import create_backend, get_available_backends

CLEAR_TEXT = {
    "en": "The weather is lovely today and we are going to the park.",
    "fr": "Le temps est magnifique aujourd'hui et nous allons au parc.",
    "es": "El tiempo es estupendo hoy y vamos a ir al parque.",
}


def _backend_or_skip(name: str):
    if name not in get_available_backends():
        pytest.skip(f"{name} not installed")
    return create_backend(name)


@pytest.mark.parametrize("name", ["langid", "langdetect", "lingua", "fasttext", "fastlangid"])
class TestBuiltinBackends:
    @pytest.mark.parametrize("lang", sorted(CLEAR_TEXT))
    def test_clear_sentence_gets_confident_vote(self, name, lang):
        # A confidence near 0 on clear text means the adapter is misreading the
        # library's score (langid used to report exp(-60) ~ 0 for everything).
        result = _backend_or_skip(name).detect(CLEAR_TEXT[lang])

        assert result.language == lang
        assert result.confidence > 0.5

    def test_confidence_is_a_probability(self, name):
        backend = _backend_or_skip(name)

        for text in [*CLEAR_TEXT.values(), "ok", "Hola amigo"]:
            assert 0.0 <= backend.detect(text).confidence <= 1.0

    def test_empty_text_is_unknown(self, name):
        result = _backend_or_skip(name).detect("   ")

        assert result.confidence == 0.0


def test_cantonese_tags_normalize_consistently():
    from fastlangml import normalize_lang_tag

    # fastText says "yue", fastlangid says "zh-yue"; both mean Cantonese.
    assert normalize_lang_tag("yue") == normalize_lang_tag("zh-yue") == "zh-Yue"


def test_default_hints_ship_with_package():
    # The JSON was once git-ignored, so installed copies got an empty dictionary.
    from fastlangml import FastLangDetector, HintDictionary

    hints = HintDictionary.default_short_words()
    assert len(hints) > 1000
    assert FastLangDetector(hints=hints).detect("thx") == "en"
