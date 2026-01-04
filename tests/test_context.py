"""Tests for conversation context."""


from fastlangml.context.conversation import ConversationContext


class TestConversationContext:
    """Tests for ConversationContext."""

    def test_add_turn(self):
        """Test adding conversation turns."""
        context = ConversationContext()
        context.add_turn("Bonjour!", detected_language="fr", confidence=0.95)

        assert len(context) == 1
        assert context.last_turn is not None
        assert context.last_turn.detected_language == "fr"

    def test_dominant_language(self):
        """Test dominant language detection."""
        context = ConversationContext()
        context.add_turn("Hello", detected_language="en", confidence=0.9)
        context.add_turn("Bonjour", detected_language="fr", confidence=0.95)
        context.add_turn("Comment ca va?", detected_language="fr", confidence=0.9)

        assert context.dominant_language == "fr"

    def test_language_distribution(self):
        """Test language distribution calculation."""
        context = ConversationContext()
        context.add_turn("Hello", detected_language="en", confidence=0.9)
        context.add_turn("Bonjour", detected_language="fr", confidence=0.9)

        dist = context.language_distribution
        assert "en" in dist
        assert "fr" in dist
        assert abs(sum(dist.values()) - 1.0) < 0.01  # Should sum to 1

    def test_recency_weighting(self):
        """Test that recent turns are weighted higher."""
        context = ConversationContext(decay_factor=0.5)

        # Add old turn
        context.add_turn("Hello", detected_language="en", confidence=0.9)
        # Add recent turn
        context.add_turn("Bonjour", detected_language="fr", confidence=0.9)

        dist = context.language_distribution
        # French should be weighted higher (more recent)
        assert dist["fr"] > dist["en"]

    def test_language_streak(self):
        """Test language streak detection."""
        context = ConversationContext(max_turns=5)  # Need more turns for streak test
        context.add_turn("Hello", detected_language="en", confidence=0.9)
        context.add_turn("How are you?", detected_language="en", confidence=0.9)
        context.add_turn("Nice to meet you", detected_language="en", confidence=0.9)

        lang, streak = context.get_language_streak()
        assert lang == "en"
        assert streak == 3

    def test_context_boost(self):
        """Test context boost calculation."""
        context = ConversationContext()
        context.add_turn("Bonjour", detected_language="fr", confidence=0.9)
        context.add_turn("Comment ca va?", detected_language="fr", confidence=0.9)

        boost = context.get_context_boost("fr")
        assert boost > 0

        no_boost = context.get_context_boost("es")
        assert no_boost == 0

    def test_max_turns(self):
        """Test max turns limit."""
        context = ConversationContext(max_turns=3)

        for i in range(5):
            context.add_turn(f"Text {i}", detected_language="en", confidence=0.9)

        assert len(context) == 3

    def test_clear(self):
        """Test clearing context."""
        context = ConversationContext()
        context.add_turn("Hello", detected_language="en", confidence=0.9)
        context.clear()

        assert len(context) == 0
        assert context.dominant_language is None

    def test_empty_context(self):
        """Test empty context."""
        context = ConversationContext()

        assert len(context) == 0
        assert context.dominant_language is None
        assert context.language_distribution == {}
        assert context.get_language_streak() == (None, 0)
