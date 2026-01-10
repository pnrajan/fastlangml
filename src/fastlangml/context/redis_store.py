"""Redis-based context storage."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastlangml.context.conversation import ConversationContext


class RedisContextStore:
    """Redis-based context storage.

    Requires: pip install fastlangml[redis]

    Stores only essential data (lang, confidence) for efficiency.

    Args:
        redis_url: Redis connection URL.
        ttl_seconds: Time-to-live in seconds. None = no expiration.
        max_turns: Max history to keep. Default: 5.
        prefix: Key prefix. Default: "fastlangml:ctx:"

    Example:
        >>> store = RedisContextStore("redis://localhost:6379", ttl_seconds=1800)
        >>> with store.session(session_id) as ctx:
        ...     result = detect(text, context=ctx, auto_update=True)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int | None = None,
        max_turns: int = 5,
        prefix: str = "fastlangml:ctx:",
    ) -> None:
        import redis

        self._client: Any = redis.from_url(redis_url)
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def save(self, session_id: str, context: ConversationContext) -> None:
        """Save context (stores only lang/confidence)."""
        history = [
            (t.detected_language, t.confidence) for t in context.turns if t.detected_language
        ][-self._max_turns :]
        data = json.dumps(history)
        if self._ttl:
            self._client.setex(self._key(session_id), self._ttl, data)
        else:
            self._client.set(self._key(session_id), data)

    def load(self, session_id: str) -> ConversationContext | None:
        """Load context. Returns None if not found."""
        data = self._client.get(self._key(session_id))
        if data is None:
            return None
        history = json.loads(data)
        return ConversationContext.from_history(history, max_turns=self._max_turns)

    def delete(self, session_id: str) -> bool:
        """Delete context."""
        return self._client.delete(self._key(session_id)) > 0

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self._client.exists(self._key(session_id)) > 0

    @contextmanager
    def session(self, session_id: str) -> Iterator[ConversationContext]:
        """Context manager that auto-saves on exit."""
        ctx = self.load(session_id) or ConversationContext(max_turns=self._max_turns)
        try:
            yield ctx
        finally:
            self.save(session_id, ctx)
