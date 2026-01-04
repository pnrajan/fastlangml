"""Result dataclasses for language detection outputs.

Provides a consistent schema for detection results across all backends
and detection modes, with support for unknown/abstain handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """A candidate language detection result."""

    lang: str
    """Normalized language tag (e.g., 'en', 'es', 'zh-Hans')."""

    confidence: float
    """Confidence score from 0.0 to 1.0."""

    backend_votes: dict[str, float] = field(default_factory=dict)
    """Mapping of backend name to its confidence for this language."""


@dataclass
class DetectionResult:
    """Result of language detection.

    This is the primary return type for all detection operations.
    Uses BCP-47-like language tags with 'und' for undetermined.

    Attributes:
        lang: Normalized language tag. 'und' for unknown/abstained.
        confidence: Overall confidence score from 0.0 to 1.0.
        reliable: Whether the detection is considered reliable.
        script: Detected Unicode script (Latin, Cyrillic, Arabic, Han, etc.).
        backend: The backend(s) used for detection.
        candidates: Top-k candidates when requested.
        reason: Explanation when lang='und' (e.g., 'too_little_text').
        meta: Additional metadata (timings, raw scores, etc.).
    """

    lang: str
    """Normalized language tag (BCP-47-like). 'und' for undetermined."""

    confidence: float
    """Overall confidence score from 0.0 to 1.0."""

    reliable: bool = True
    """Whether the detection is considered reliable."""

    script: str | None = None
    """Detected Unicode script (e.g., 'Latin', 'Cyrillic', 'Han')."""

    backend: str = "unknown"
    """Backend used: 'cld3', 'lingua', 'langid', 'fasttext', 'ensemble'."""

    candidates: list[Candidate] = field(default_factory=list)
    """Top-k candidate languages when top_k > 1."""

    reason: str | None = None
    """Reason for 'und' (unknown): 'too_little_text', 'non_linguistic',
    'low_confidence', 'disagreement', 'no_backend'."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Additional metadata: timings, raw backend scores, etc."""

    def __repr__(self) -> str:
        if self.lang == "und":
            return f"DetectionResult(lang='und', reason='{self.reason}')"
        return f"DetectionResult(lang='{self.lang}', confidence={self.confidence:.4f})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "lang": self.lang,
            "confidence": round(self.confidence, 4),
            "reliable": self.reliable,
        }

        if self.script:
            result["script"] = self.script

        result["backend"] = self.backend

        if self.candidates:
            result["candidates"] = [
                {
                    "lang": c.lang,
                    "confidence": round(c.confidence, 4),
                    "backend_votes": {k: round(v, 4) for k, v in c.backend_votes.items()},
                }
                for c in self.candidates
            ]

        if self.reason:
            result["reason"] = self.reason

        if self.meta:
            result["meta"] = self.meta

        return result


# Reason constants for unknown results
class Reasons:
    """Standard reasons for returning 'und' (unknown)."""

    TOO_LITTLE_TEXT = "too_little_text"
    """Text has too few letters for reliable detection."""

    NON_LINGUISTIC = "non_linguistic"
    """Text is mostly non-linguistic (emoji, numbers, punctuation)."""

    LOW_CONFIDENCE = "low_confidence"
    """Confidence is below the threshold."""

    DISAGREEMENT = "disagreement"
    """Backends disagree significantly."""

    NO_BACKEND = "no_backend"
    """No backend is available."""

    EMPTY_TEXT = "empty_text"
    """Text is empty or whitespace-only."""
