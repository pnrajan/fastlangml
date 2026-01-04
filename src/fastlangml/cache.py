"""Caching utilities for language detection results.

Uses cachetools for efficient LRU caching of detection results,
especially useful for repeated short strings like "ok", "thanks", etc.
"""

from __future__ import annotations

from typing import Any

from cachetools import LRUCache

# Default cache for detection results
_default_cache: LRUCache[str, Any] = LRUCache(maxsize=1000)


def get_cache(max_size: int = 1000) -> LRUCache[str, Any]:
    """Get or create an LRU cache.

    Args:
        max_size: Maximum number of entries to cache.

    Returns:
        LRUCache instance.
    """
    return LRUCache(maxsize=max_size)


def clear_default_cache() -> None:
    """Clear the default detection cache."""
    _default_cache.clear()


# Export for convenience
__all__ = ["LRUCache", "get_cache", "clear_default_cache"]
