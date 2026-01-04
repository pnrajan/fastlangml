"""Caching utilities for language detection results.

Uses cachetools for efficient LRU caching of detection results,
especially useful for repeated short strings like "ok", "thanks", etc.
"""

from __future__ import annotations

from typing import TypeVar

from cachetools import LRUCache as _LRUCache

T = TypeVar("T")


class DetectionCache:
    """Thread-safe LRU cache for detection results using cachetools.

    Args:
        max_size: Maximum number of entries to cache.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: _LRUCache[str, T] = _LRUCache(maxsize=max_size)
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> T | None:
        """Get a cached result.

        Args:
            key: Cache key (typically the normalized input text + mode).

        Returns:
            Cached result or None if not found.
        """
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def put(self, key: str, value: T) -> None:
        """Cache a result.

        Args:
            key: Cache key.
            value: Result to cache.
        """
        self._cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int | float]:
        """Cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


# Alias for backward compatibility
LRUCache = DetectionCache
