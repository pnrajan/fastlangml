"""Tests for context serialization."""

from __future__ import annotations

from fastlangml.context import ConversationContext, ConversationTurn


class TestConversationTurnSerialization:
    """Tests for ConversationTurn to_dict/from_dict."""

    def test_to_dict(self) -> None:
        turn = ConversationTurn(
            text="Bonjour!",
            detected_language="fr",
            confidence=0.95,
            timestamp=1234567890.0,
        )
        data = turn.to_dict()
        assert data["text"] == "Bonjour!"
        assert data["detected_language"] == "fr"
        assert data["confidence"] == 0.95
        assert data["timestamp"] == 1234567890.0

    def test_from_dict(self) -> None:
        data = {
            "text": "Hello",
            "detected_language": "en",
            "confidence": 0.9,
            "timestamp": 1234567890.0,
        }
        turn = ConversationTurn.from_dict(data)
        assert turn.text == "Hello"
        assert turn.detected_language == "en"
        assert turn.confidence == 0.9
        assert turn.timestamp == 1234567890.0

    def test_from_dict_with_defaults(self) -> None:
        data = {"text": "Hi"}
        turn = ConversationTurn.from_dict(data)
        assert turn.text == "Hi"
        assert turn.detected_language is None
        assert turn.confidence == 0.0
        assert turn.timestamp > 0

    def test_round_trip(self) -> None:
        original = ConversationTurn("Test", "en", 0.8, 123.0)
        restored = ConversationTurn.from_dict(original.to_dict())
        assert original.text == restored.text
        assert original.detected_language == restored.detected_language
        assert original.confidence == restored.confidence
        assert original.timestamp == restored.timestamp


class TestConversationContextFromHistory:
    """Tests for ConversationContext.from_history() stateless pattern."""

    def test_from_history_tuples(self) -> None:
        ctx = ConversationContext.from_history([("fr", 0.95), ("fr", 0.9)])
        assert len(ctx) == 2
        assert ctx.dominant_language == "fr"

    def test_from_history_dicts(self) -> None:
        ctx = ConversationContext.from_history(
            [
                {"lang": "es", "confidence": 0.9},
                {"lang": "es", "confidence": 0.85},
            ]
        )
        assert len(ctx) == 2
        assert ctx.dominant_language == "es"

    def test_from_history_mixed_keys(self) -> None:
        # Supports both "lang" and "detected_language" keys
        ctx = ConversationContext.from_history(
            [
                {"detected_language": "de", "confidence": 0.9},
            ]
        )
        assert ctx.dominant_language == "de"

    def test_from_history_empty(self) -> None:
        ctx = ConversationContext.from_history([])
        assert len(ctx) == 0
        assert ctx.dominant_language is None

    def test_from_history_with_config(self) -> None:
        ctx = ConversationContext.from_history(
            [("en", 0.9)],
            max_turns=10,
            decay_factor=0.8,
        )
        assert ctx.max_turns == 10
        assert ctx.decay_factor == 0.8

    def test_from_history_skips_empty_lang(self) -> None:
        ctx = ConversationContext.from_history(
            [
                {"lang": "", "confidence": 0.9},
                {"lang": "fr", "confidence": 0.8},
            ]
        )
        assert len(ctx) == 1
        assert ctx.dominant_language == "fr"


class TestConversationContextSerialization:
    """Tests for ConversationContext to_dict/from_dict."""

    def test_to_dict_empty(self) -> None:
        ctx = ConversationContext(max_turns=5, decay_factor=0.8)
        data = ctx.to_dict()
        assert data["max_turns"] == 5
        assert data["decay_factor"] == 0.8
        assert data["turns"] == []

    def test_to_dict_with_turns(self) -> None:
        ctx = ConversationContext()
        ctx.add_turn("Hello", "en", 0.9)
        ctx.add_turn("Bonjour", "fr", 0.95)
        data = ctx.to_dict()
        assert len(data["turns"]) == 2
        assert data["turns"][0]["text"] == "Hello"
        assert data["turns"][1]["text"] == "Bonjour"

    def test_from_dict(self) -> None:
        data = {
            "max_turns": 10,
            "decay_factor": 0.7,
            "turns": [
                {"text": "Hi", "detected_language": "en", "confidence": 0.9, "timestamp": 100.0}
            ],
        }
        ctx = ConversationContext.from_dict(data)
        assert ctx.max_turns == 10
        assert ctx.decay_factor == 0.7
        assert len(ctx) == 1
        assert ctx.dominant_language == "en"

    def test_from_dict_with_defaults(self) -> None:
        data = {"turns": []}
        ctx = ConversationContext.from_dict(data)
        assert ctx.max_turns == 2
        assert ctx.decay_factor == 0.9
        assert len(ctx) == 0

    def test_round_trip(self) -> None:
        original = ConversationContext(max_turns=5, decay_factor=0.85)
        original.add_turn("Test", "en", 0.9)
        original.add_turn("Prueba", "es", 0.85)

        restored = ConversationContext.from_dict(original.to_dict())
        assert len(restored) == 2
        assert restored.max_turns == 5
        assert restored.decay_factor == 0.85
        assert restored.dominant_language == original.dominant_language
