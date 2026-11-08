"""Core detector behavior, tested with scripted fake backends.

Real backends differ between machines and CI, so these tests register fake
backends whose votes are fixed per text. That keeps every assertion about
FastLangML's own logic (caching, context, hints, filtering, "und" handling)
rather than about what fastText or Lingua happen to answer.
"""

from __future__ import annotations

import pytest

from fastlangml import (
    Backend,
    ConversationContext,
    DetectionConfig,
    FastLangDetector,
    HintDictionary,
    Reasons,
    detect,
    register_backend,
)
from fastlangml.backends import clear_registered_backends
from fastlangml.backends.base import DetectionResult as Vote

LANGS = {"en", "fr", "es", "pt", "de", "it"}

# A near-tie between Spanish and Portuguese: Spanish wins by a hair on its own.
NEAR_TIE = "tengo uma pergunta"


class Script:
    """Per-backend scripted votes plus a log of every text each backend saw."""

    def __init__(self) -> None:
        self.votes: dict[str, dict[str, tuple[str, float]]] = {"a": {}, "b": {}}
        self.calls: list[tuple[str, str]] = []

    def set(self, text: str, a: tuple[str, float], b: tuple[str, float] | None = None) -> None:
        self.votes["a"][text] = a
        self.votes["b"][text] = b or a

    def texts_seen(self) -> set[str]:
        return {text for _, text in self.calls}


def _fake_backend(name: str, script: Script) -> type[Backend]:
    class FakeBackend(Backend):
        @property
        def name(self) -> str:
            return name

        @property
        def is_available(self) -> bool:
            return True

        def detect(self, text: str) -> Vote:
            script.calls.append((name, text))
            lang, conf = script.votes[name].get(text, ("en", 0.9))
            return Vote(name, lang, conf)

        def supported_languages(self) -> set[str]:
            return LANGS

    return FakeBackend


@pytest.fixture
def script():
    s = Script()
    clear_registered_backends()
    register_backend("a", _fake_backend("a", s))
    register_backend("b", _fake_backend("b", s))
    s.set(NEAR_TIE, a=("es", 0.80), b=("pt", 0.78))
    yield s
    clear_registered_backends()
    FastLangDetector.reset_default()


@pytest.fixture
def make_detector(script):
    """Build a detector that uses only the two fake backends."""

    def _make(**config) -> FastLangDetector:
        config.setdefault("backends", ["a", "b"])
        config.setdefault("backend_weights", {"a": 1.0, "b": 1.0})
        return FastLangDetector(config=DetectionConfig(**config))

    return _make


# --- Basic detection ---------------------------------------------------------


class TestBasicDetection:
    def test_agreeing_backends_give_reliable_result(self, script, make_detector):
        script.set("Bonjour tout le monde", ("fr", 0.9))
        result = make_detector().detect("Bonjour tout le monde")

        assert result == "fr"
        assert result.reliable
        assert result.backend == "ensemble"
        assert result.reason is None

    def test_weights_decide_disagreement(self, script, make_detector):
        script.set("hola olá", a=("es", 0.9), b=("pt", 0.9))

        assert make_detector(backend_weights={"a": 3.0, "b": 1.0}).detect("hola olá") == "es"
        assert make_detector(backend_weights={"a": 1.0, "b": 3.0}).detect("hola olá") == "pt"

    def test_top_k_returns_ranked_candidates(self, make_detector):
        result = make_detector().detect(NEAR_TIE, top_k=2)

        assert [c.lang for c in result.candidates] == ["es", "pt"]
        assert result.candidates[0].confidence >= result.candidates[1].confidence

    def test_allowed_langs_excludes_other_languages(self, script, make_detector):
        script.set("ciao amico mio", ("it", 0.9))
        detector = make_detector()

        assert detector.detect("ciao amico mio") == "it"
        assert detector.detect("ciao amico mio", allowed_langs=["en", "fr"]) == "und"

    def test_set_languages_restricts_output(self, make_detector):
        detector = make_detector()
        detector.set_languages(["pt"])

        assert detector.detect(NEAR_TIE) == "pt"


# --- Undetermined ("und") results -------------------------------------------


class TestUndetermined:
    @pytest.mark.parametrize(
        ("text", "reason"),
        [
            ("", Reasons.EMPTY_TEXT),
            ("   ", Reasons.EMPTY_TEXT),
            ("12345", Reasons.TOO_LITTLE_TEXT),
            ("x", Reasons.TOO_LITTLE_TEXT),
        ],
    )
    def test_unusable_input_returns_und_without_calling_backends(
        self, script, make_detector, text, reason
    ):
        result = make_detector().detect(text)

        assert result == "und"
        assert result.reason == reason
        assert not result.reliable
        assert script.calls == []

    def test_close_split_vote_below_threshold_is_disagreement(self, script, make_detector):
        script.set("mixed signal text", a=("es", 0.2), b=("pt", 0.2))
        detector = make_detector(thresholds={"short": 1.1, "default": 1.1, "long": 1.1})

        result = detector.detect("mixed signal text")

        assert result == "und"
        assert result.reason == Reasons.DISAGREEMENT

    def test_clear_winner_below_threshold_is_low_confidence(self, script, make_detector):
        script.set("fairly clear text", a=("fr", 0.9), b=("en", 0.1))
        detector = make_detector(thresholds={"short": 1.1, "default": 1.1, "long": 1.1})

        result = detector.detect("fairly clear text")

        assert result == "und"
        assert result.reason == Reasons.LOW_CONFIDENCE

    def test_top_confidence_is_normalized_to_one(self, make_detector):
        # Current behavior: scores are divided by the top score, so the winner
        # always reports confidence 1.0, even on a near-tie.
        assert make_detector().detect(NEAR_TIE).confidence == 1.0

    @pytest.mark.xfail(
        strict=True,
        reason="Known bug: top score is normalized to 1.0, so default thresholds "
        "(0.3-0.6) never mark a near-tie as unreliable.",
    )
    def test_near_tie_is_unreliable_with_default_thresholds(self, make_detector):
        assert not make_detector().detect(NEAR_TIE).reliable

    def test_script_short_circuit_skips_backends(self, script, make_detector):
        assert make_detector().detect("안녕하세요") == "ko"
        assert script.calls == []


# --- Result cache ------------------------------------------------------------


class TestCache:
    def test_repeated_text_is_served_from_cache(self, script, make_detector):
        detector = make_detector()
        first = detector.detect("Hello there friend")
        calls_after_first = len(script.calls)

        second = detector.detect("Hello there friend")

        assert second is first
        assert len(script.calls) == calls_after_first

    @pytest.mark.parametrize(
        "variant",
        [
            {"mode": "long"},
            {"top_k": 3},
            {"allowed_langs": ["en", "fr"]},
            {"filter_proper_nouns": False},
        ],
    )
    def test_cache_key_covers_call_options(self, script, make_detector, variant):
        detector = make_detector()
        detector.detect("Hello there friend")
        calls = len(script.calls)

        detector.detect("Hello there friend", **variant)

        assert len(script.calls) > calls, f"{variant} was served a cached result"

    def test_per_call_hints_do_not_leak_into_cache(self, make_detector):
        detector = make_detector()
        hints = HintDictionary()
        hints.add("tengo", "pt")

        assert detector.detect(NEAR_TIE, hints=hints) == "pt"
        assert detector.detect(NEAR_TIE) == "es"

    def test_cached_result_not_reused_for_per_call_hints(self, make_detector):
        detector = make_detector()
        hints = HintDictionary()
        hints.add("tengo", "pt")

        assert detector.detect(NEAR_TIE) == "es"
        assert detector.detect(NEAR_TIE, hints=hints) == "pt"

    def test_add_hint_invalidates_cached_results(self, make_detector):
        detector = make_detector()
        assert detector.detect(NEAR_TIE) == "es"

        detector.add_hint("pergunta", "pt")
        assert detector.detect(NEAR_TIE) == "pt"

        detector.remove_hint("pergunta")
        assert detector.detect(NEAR_TIE) == "es"

    def test_context_calls_bypass_cache(self, script, make_detector):
        detector = make_detector()
        detector.detect("Hello there friend", context=ConversationContext())

        assert detector.cache_stats["size"] == 0

        detector.detect("Hello there friend")
        calls = len(script.calls)
        detector.detect("Hello there friend", context=ConversationContext())

        assert len(script.calls) > calls, "context call was served from cache"


# --- Conversation context ----------------------------------------------------


def _portuguese_history() -> ConversationContext:
    return ConversationContext.from_history([("pt", 0.95), ("pt", 0.95)])


class TestContext:
    def test_context_breaks_near_tie(self, make_detector):
        detector = make_detector()

        assert detector.detect(NEAR_TIE) == "es"
        assert detector.detect(NEAR_TIE, context=_portuguese_history()) == "pt"

    def test_context_does_not_override_confident_detection(self, script, make_detector):
        script.set("Buenos días a todos", ("es", 0.95))

        result = make_detector().detect("Buenos días a todos", context=_portuguese_history())

        assert result == "es"

    def test_empty_context_changes_nothing(self, make_detector):
        assert make_detector().detect(NEAR_TIE, context=ConversationContext()) == "es"

    def test_detector_method_does_not_update_context_by_default(self, make_detector):
        ctx = ConversationContext()
        make_detector().detect(NEAR_TIE, context=ctx)

        assert len(ctx) == 0

    def test_detector_method_updates_context_when_asked(self, make_detector):
        ctx = ConversationContext()
        result = make_detector().detect(NEAR_TIE, context=ctx, auto_update=True)

        assert len(ctx) == 1
        assert ctx.last_turn.detected_language == result.lang

    def test_module_detect_updates_context_by_default(self, script, make_detector):
        FastLangDetector._default_instance = make_detector()
        script.set("Bonjour tout le monde", ("fr", 0.9))
        ctx = ConversationContext(max_turns=5)

        detect("Bonjour tout le monde", context=ctx)
        detect("Bonjour tout le monde", context=ctx)  # same text: no cache shortcut

        assert [t.detected_language for t in ctx.turns] == ["fr", "fr"]

    @pytest.mark.parametrize("text", ["", "12345", "?!"])
    def test_und_results_are_not_recorded(self, make_detector, text):
        ctx = _portuguese_history()

        result = make_detector().detect(text, context=ctx, auto_update=True)

        assert result == "und"
        assert [t.detected_language for t in ctx.turns] == ["pt", "pt"]

    def test_rejected_vote_und_is_not_recorded(self, script, make_detector):
        script.set("mixed signal text", a=("es", 0.2), b=("pt", 0.2))
        detector = make_detector(thresholds={"short": 0.99, "default": 1.1, "long": 1.1})
        ctx = _portuguese_history()

        detector.detect("mixed signal text", context=ctx, auto_update=True)

        assert len(ctx) == 2


# --- Hints and proper nouns --------------------------------------------------


class TestHints:
    def test_detector_hint_overrides_short_text(self, script, make_detector):
        script.set("zzzz", ("en", 0.4))
        detector = make_detector()
        detector.add_hint("zzzz", "fr")

        assert detector.detect("zzzz", mode="short") == "fr"

    def test_constructor_hints_are_used(self, script):
        hints = HintDictionary()
        hints.add("pergunta", "pt")
        detector = FastLangDetector(
            config=DetectionConfig(backends=["a", "b"], backend_weights={"a": 1.0, "b": 1.0}),
            hints=hints,
        )

        assert detector.detect(NEAR_TIE) == "pt"


class TestProperNounFilter:
    TEXT = "I love Paris and Berlin"

    def test_backends_see_text_without_names(self, script, make_detector):
        make_detector().detect(self.TEXT, filter_proper_nouns=True)

        assert self.TEXT not in script.texts_seen()
        assert all("Paris" not in t and "Berlin" not in t for t in script.texts_seen())

    def test_filter_can_be_disabled_per_call(self, script, make_detector):
        make_detector().detect(self.TEXT, filter_proper_nouns=False)

        assert script.texts_seen() == {self.TEXT}
