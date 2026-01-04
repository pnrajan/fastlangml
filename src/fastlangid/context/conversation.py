"""Conversation context management for language detection.

Tracks language detection history across conversation turns to improve
accuracy for subsequent messages.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    text: str
    detected_language: str | None = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConversationContext:
    """Manages conversation history for context-aware language detection.

    Tracks detected languages across conversation turns to improve accuracy
    for subsequent messages. More recent turns are weighted higher using
    exponential decay, making recent language choices more influential.

    This is the core feature that differentiates FastLangID from traditional
    language detectors - it uses conversation history to resolve ambiguity
    in short or unclear messages.

    Attributes:
        max_turns: Maximum number of turns to remember. Older turns are
            automatically discarded. Defaults to 20.
        decay_factor: Exponential decay factor for recency weighting (0-1).
            Higher values give more weight to recent turns. Defaults to 0.9.

    Example:
        >>> context = ConversationContext(max_turns=10)
        >>> context.add_turn("Bonjour!", detected_language="fr", confidence=0.95)
        >>> context.add_turn("Comment ca va?", detected_language="fr", confidence=0.90)
        >>> context.dominant_language
        'fr'
        >>> context.language_distribution
        {'fr': 1.0}

    Note:
        Context should be created per conversation/session. Do not share
        context across different users or conversations.
    """

    max_turns: int = 2
    """Maximum number of turns to remember. Default is 2 for short conversations."""

    decay_factor: float = 0.9
    """Decay factor for recency weighting (0-1). Higher = more recency bias."""

    _turns: deque[ConversationTurn] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._turns = deque(maxlen=self.max_turns)

    def add_turn(
        self,
        text: str,
        detected_language: str | None = None,
        confidence: float = 0.0,
    ) -> None:
        """Add a conversation turn to the context history.

        Records a message and its detected language for use in future
        detection. Call this after each message in a conversation to
        build up context that improves accuracy on ambiguous messages.

        Args:
            text: The text content of the message.
            detected_language: ISO 639-1 language code (e.g., "en", "fr").
                Use None or "und" if detection failed.
            confidence: Confidence score from 0.0 to 1.0 for the detection.
                Higher confidence turns have more influence on context.

        Example:
            >>> context = ConversationContext()
            >>> result = detector.detect("Bonjour!")
            >>> context.add_turn("Bonjour!", result.lang, result.confidence)
        """
        turn = ConversationTurn(
            text=text,
            detected_language=detected_language,
            confidence=confidence,
            timestamp=time.time(),
        )
        self._turns.append(turn)

    @property
    def turns(self) -> list[ConversationTurn]:
        """Get all turns (oldest first, most recent last)."""
        return list(self._turns)

    @property
    def last_turn(self) -> ConversationTurn | None:
        """Get the most recent turn."""
        return self._turns[-1] if self._turns else None

    @property
    def dominant_language(self) -> str | None:
        """
        Get the most likely language based on conversation history.

        Uses weighted voting where recent turns have higher weight.
        """
        dist = self.language_distribution
        if not dist:
            return None
        return max(dist, key=lambda k: dist[k])

    @property
    def language_distribution(self) -> dict[str, float]:
        """
        Get weighted language distribution across conversation.

        Returns:
            Dict mapping language code to weight (sums to 1.0)
        """
        if not self._turns:
            return {}

        # Calculate weighted counts
        weighted_counts: dict[str, float] = {}
        total_weight = 0.0

        turns_list = list(self._turns)
        for i, turn in enumerate(turns_list):
            if turn.detected_language and turn.detected_language != "unknown":
                # More recent turns get higher weight (exponential decay)
                recency_weight = self.decay_factor ** (len(turns_list) - 1 - i)
                weight = recency_weight * turn.confidence

                lang = turn.detected_language
                if lang not in weighted_counts:
                    weighted_counts[lang] = 0.0
                weighted_counts[lang] += weight
                total_weight += weight

        # Normalize to probabilities
        if total_weight > 0:
            return {lang: weight / total_weight for lang, weight in weighted_counts.items()}
        return {}

    def get_language_streak(self) -> tuple[str | None, int]:
        """
        Get the current language streak (consecutive same-language turns).

        Returns:
            Tuple of (language, streak_count) or (None, 0)
        """
        if not self._turns:
            return None, 0

        turns_reversed = list(reversed(self._turns))
        if not turns_reversed[0].detected_language:
            return None, 0

        current_lang = turns_reversed[0].detected_language
        streak = 0

        for turn in turns_reversed:
            if turn.detected_language == current_lang:
                streak += 1
            else:
                break

        return current_lang, streak

    def get_context_boost(self, language: str) -> float:
        """
        Get a confidence boost for a language based on conversation context.

        Args:
            language: Language code to check

        Returns:
            Boost value between 0.0 and 0.3
        """
        dist = self.language_distribution
        if language not in dist:
            return 0.0

        # Base boost from distribution
        base_boost = dist[language] * 0.15

        # Streak bonus
        streak_lang, streak_count = self.get_language_streak()
        if streak_lang == language and streak_count >= 2:
            streak_bonus = min(streak_count * 0.02, 0.1)
            base_boost += streak_bonus

        return min(base_boost, 0.3)

    def clear(self) -> None:
        """Clear conversation history."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[ConversationTurn]:
        return iter(self._turns)

    def __bool__(self) -> bool:
        return len(self._turns) > 0
