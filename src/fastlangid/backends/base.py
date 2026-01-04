"""Abstract base class for language detection backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    """Result from a single backend detection."""

    backend_name: str
    language: str
    confidence: float
    all_probabilities: dict[str, float] = field(default_factory=dict)
    is_reliable: bool = True  # Backend-specific reliability flag

    def __post_init__(self) -> None:
        # Normalize language code to lowercase
        self.language = self.language.lower() if self.language else "unknown"
        # Ensure all_probabilities includes the top result
        if self.language != "unknown" and self.language not in self.all_probabilities:
            self.all_probabilities[self.language] = self.confidence


class Backend(ABC):
    """Abstract base class for language detection backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are installed."""
        pass

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """
        Detect language of text.

        Args:
            text: Preprocessed text to analyze

        Returns:
            DetectionResult with language code, confidence, and all probabilities
        """
        pass

    @abstractmethod
    def supported_languages(self) -> set[str]:
        """Return set of ISO 639-1 codes this backend supports."""
        pass

    def detect_batch(self, texts: list[str]) -> list[DetectionResult]:
        """
        Detect language for multiple texts.

        Default implementation calls detect() for each text.
        Backends can override for optimized batch processing.

        Args:
            texts: List of texts to analyze

        Returns:
            List of DetectionResults
        """
        return [self.detect(text) for text in texts]
