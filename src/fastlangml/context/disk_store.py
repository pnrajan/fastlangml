"""Disk-based context storage using diskcache."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from fastlangml.context.conversation import ConversationContext


class DiskContextStore:
    """Disk-based context storage using diskcache.

    Requires: pip install fastlangml[diskcache]

    Stores only essential data (lang, confidence) for efficiency.

    Args:
        directory: Path to cache directory.
        ttl_seconds: Time-to-live in seconds. None = no expiration.
        max_turns: Max history to keep. Default: 5.

    Example:
        >>> store = DiskContextStore("./contexts", ttl_seconds=3600)
        >>> with store.session(session_id) as ctx:
        ...     result = detect(text, context=ctx, auto_update=True)
    """

    def __init__(
        self,
        directory: str,
        ttl_seconds: int | None = None,
        max_turns: int = 5,
    ) -> None:
        from diskcache import Cache

        self._cache: Any = Cache(directory)
        self._ttl = ttl_seconds
        self._max_turns = max_turns

    def save(self, session_id: str, context: ConversationContext) -> None:
        """Save context (stores only lang/confidence)."""
        history = [
            (t.detected_language, t.confidence)
            for t in context.turns
            if t.detected_language
        ][-self._max_turns:]
        self._cache.set(session_id, history, expire=self._ttl)

    def load(self, session_id: str) -> ConversationContext | None:
        """Load context. Returns None if not found."""
        history = self._cache.get(session_id)
        if history is None:
            return None
        return ConversationContext.from_history(history, max_turns=self._max_turns)

    def delete(self, session_id: str) -> bool:
        """Delete context."""
        return self._cache.delete(session_id)

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return session_id in self._cache

    @contextmanager
    def session(self, session_id: str) -> Iterator[ConversationContext]:
        """Context manager that auto-saves on exit."""
        ctx = self.load(session_id) or ConversationContext(max_turns=self._max_turns)
        try:
            yield ctx
        finally:
            self.save(session_id, ctx)
