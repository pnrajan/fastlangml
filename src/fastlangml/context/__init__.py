"""Conversation context management for improved detection accuracy."""

from fastlangml.context.conversation import ConversationContext, ConversationTurn

# Optional stores (require extra deps)
try:
    from fastlangml.context.disk_store import DiskContextStore
except ImportError:
    DiskContextStore = None  # type: ignore[misc, assignment]

try:
    from fastlangml.context.redis_store import RedisContextStore
except ImportError:
    RedisContextStore = None  # type: ignore[misc, assignment]

__all__ = [
    "ConversationContext",
    "ConversationTurn",
    "DiskContextStore",
    "RedisContextStore",
]
