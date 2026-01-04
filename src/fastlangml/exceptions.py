"""Custom exceptions for fastlang."""

from __future__ import annotations


class FastLangError(Exception):
    """Base exception for fastlang."""

    pass


class DetectionError(FastLangError):
    """Raised when language detection fails."""

    pass


class BackendUnavailableError(FastLangError):
    """Raised when a requested backend is not installed."""

    def __init__(self, message: str, backend_name: str | None = None) -> None:
        super().__init__(message)
        self.backend_name = backend_name


class BackendError(FastLangError):
    """Raised when a backend encounters an error during detection."""

    def __init__(self, message: str, backend_name: str | None = None) -> None:
        super().__init__(message)
        self.backend_name = backend_name


class ConfigurationError(FastLangError):
    """Raised for configuration issues."""

    pass


class HintError(FastLangError):
    """Raised for hint dictionary issues."""

    pass


class NoBackendsAvailableError(FastLangError):
    """Raised when no detection backends are available."""

    pass
