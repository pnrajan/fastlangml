"""Backend implementations for language detection.

Supports built-in backends and custom backends via decorator registration.

Example using decorator:
    >>> from fastlangml import backend, Backend
    >>> from fastlangml.backends.base import DetectionResult
    >>>
    >>> @backend("mybackend", reliability=4)
    ... class MyBackend(Backend):
    ...     @property
    ...     def name(self) -> str:
    ...         return "mybackend"
    ...
    ...     @property
    ...     def is_available(self) -> bool:
    ...         return True
    ...
    ...     def detect(self, text: str) -> DetectionResult:
    ...         return DetectionResult("mybackend", "en", 0.9)
    ...
    ...     def supported_languages(self) -> set[str]:
    ...         return {"en", "fr", "es"}
    >>>
    >>> # Now use it
    >>> from fastlangml import detect
    >>> detect("Hello", backends=["mybackend"]).lang
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from fastlangml.backends.base import Backend, DetectionResult
from fastlangml.exceptions import BackendUnavailableError

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=type[Backend])


# =============================================================================
# Backend Registry
# =============================================================================

_CUSTOM_BACKEND_REGISTRY: dict[str, type[Backend]] = {}

_BACKEND_RELIABILITY: dict[str, int] = {
    "fasttext": 5,
    "fastlangid": 5,
    "lingua": 4,
    "langid": 3,
    "pycld3": 3,
    "langdetect": 2,
}

_BUILTIN_BACKEND_NAMES = frozenset({"fasttext", "fastlangid", "langdetect", "lingua", "pycld3", "langid"})


def backend(name: str, reliability: int = 3) -> Callable[[T], T]:
    """Decorator to register a custom backend.

    Use this decorator on a class extending Backend to register it
    for use in FastLangID's ensemble detection.

    Args:
        name: Unique identifier for the backend.
        reliability: Reliability score 1-5 (higher = more weight in voting).

    Returns:
        Decorator function that registers the class.

    Example:
        >>> from fastlangml import backend, Backend
        >>> from fastlangml.backends.base import DetectionResult
        >>>
        >>> @backend("openai", reliability=5)
        ... class OpenAIBackend(Backend):
        ...     '''Custom backend using OpenAI API.'''
        ...
        ...     @property
        ...     def name(self) -> str:
        ...         return "openai"
        ...
        ...     @property
        ...     def is_available(self) -> bool:
        ...         try:
        ...             import openai
        ...             return True
        ...         except ImportError:
        ...             return False
        ...
        ...     def detect(self, text: str) -> DetectionResult:
        ...         # Your OpenAI API call here
        ...         lang, conf = self._detect_with_openai(text)
        ...         return DetectionResult(self.name, lang, conf)
        ...
        ...     def supported_languages(self) -> set[str]:
        ...         return {"en", "fr", "de", "es", "zh", "ja", "ko"}
        >>>
        >>> # Use in detection
        >>> from fastlangml import FastLangDetector, DetectionConfig
        >>> detector = FastLangDetector(
        ...     config=DetectionConfig(backends=["openai", "fasttext"])
        ... )
    """
    def decorator(cls: T) -> T:
        register_backend(name, cls, reliability)
        return cls
    return decorator


def register_backend(
    name: str,
    backend_class: type[Backend],
    reliability: int = 3,
) -> None:
    """Register a custom backend (function form).

    Prefer using the @backend decorator for cleaner code.

    Args:
        name: Unique identifier for the backend.
        backend_class: Class extending Backend.
        reliability: Reliability score 1-5.

    Raises:
        ValueError: If name conflicts with built-in or reliability out of range.
        TypeError: If backend_class doesn't extend Backend.
    """
    if name in _BUILTIN_BACKEND_NAMES:
        raise ValueError(
            f"Cannot register '{name}': conflicts with built-in backend. "
            f"Built-in: {sorted(_BUILTIN_BACKEND_NAMES)}"
        )

    if not isinstance(backend_class, type) or not issubclass(backend_class, Backend):
        raise TypeError(
            f"backend_class must extend Backend, got {type(backend_class)}"
        )

    if not isinstance(reliability, int) or not 1 <= reliability <= 5:
        raise ValueError(f"reliability must be 1-5, got {reliability}")

    _CUSTOM_BACKEND_REGISTRY[name] = backend_class
    _BACKEND_RELIABILITY[name] = reliability


def unregister_backend(name: str) -> bool:
    """Remove a registered custom backend.

    Args:
        name: Backend name to remove.

    Returns:
        True if removed, False if not found.
    """
    if name in _BUILTIN_BACKEND_NAMES:
        raise ValueError(f"Cannot unregister built-in backend: {name}")

    removed = _CUSTOM_BACKEND_REGISTRY.pop(name, None) is not None
    _BACKEND_RELIABILITY.pop(name, None)
    return removed


def list_registered_backends() -> list[str]:
    """List custom registered backends."""
    return list(_CUSTOM_BACKEND_REGISTRY.keys())


def clear_registered_backends() -> None:
    """Remove all custom backends (useful for testing)."""
    for name in list(_CUSTOM_BACKEND_REGISTRY.keys()):
        unregister_backend(name)


# =============================================================================
# Backend Factory
# =============================================================================

def _get_backend_class(name: str) -> type[Backend]:
    """Get backend class by name (lazy import to avoid loading unavailable deps)."""
    # Check custom registry first
    if name in _CUSTOM_BACKEND_REGISTRY:
        return _CUSTOM_BACKEND_REGISTRY[name]

    # Built-in backends (lazy import)
    if name == "fasttext":
        from fastlangml.backends.fasttext_backend import FastTextBackend
        return FastTextBackend
    elif name == "fastlangid":
        from fastlangml.backends.fastlangid_backend import FastLangIDBackend
        return FastLangIDBackend
    elif name == "langdetect":
        from fastlangml.backends.langdetect_backend import LangdetectBackend
        return LangdetectBackend
    elif name == "lingua":
        from fastlangml.backends.lingua_backend import LinguaBackend
        return LinguaBackend
    elif name == "pycld3":
        from fastlangml.backends.pycld3_backend import PyCLD3Backend
        return PyCLD3Backend
    elif name == "langid":
        from fastlangml.backends.langid_backend import LangidBackend
        return LangidBackend
    else:
        available = sorted(_BUILTIN_BACKEND_NAMES | set(_CUSTOM_BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown backend: {name}. Available: {available}")


# For backward compatibility
BACKEND_NAMES = _BUILTIN_BACKEND_NAMES
BACKEND_RELIABILITY = _BACKEND_RELIABILITY


def create_backend(name: str, **kwargs: object) -> Backend:
    """
    Create a backend instance by name.

    Args:
        name: Backend name ("fasttext", "langdetect", "lingua", "pycld3", "langid")
        **kwargs: Backend-specific configuration

    Raises:
        BackendUnavailableError: If backend dependencies not installed
        ValueError: If backend name is unknown
    """
    backend_class = _get_backend_class(name)
    backend = backend_class(**kwargs)  # type: ignore[arg-type]

    if not backend.is_available:
        install_cmd = f"pip install fastlang[{name}]"
        raise BackendUnavailableError(
            f"Backend '{name}' is not available. Install with: {install_cmd}",
            backend_name=name,
        )

    return backend


def get_available_backends() -> list[str]:
    """Return list of available backend names (built-in + custom)."""
    available = []

    # Check built-in backends
    for name in _BUILTIN_BACKEND_NAMES:
        try:
            backend_class = _get_backend_class(name)
            backend_instance = backend_class()
            if backend_instance.is_available:
                available.append(name)
        except Exception:
            pass

    # Check custom backends
    for name, backend_class in _CUSTOM_BACKEND_REGISTRY.items():
        try:
            backend_instance = backend_class()
            if backend_instance.is_available:
                available.append(name)
        except Exception:
            pass

    return sorted(available, key=lambda x: _BACKEND_RELIABILITY.get(x, 0), reverse=True)


def get_backend_reliability(name: str) -> int:
    """Get reliability score for a backend (higher = more reliable)."""
    return BACKEND_RELIABILITY.get(name, 0)


__all__ = [
    # Base classes
    "Backend",
    "DetectionResult",
    # Factory functions
    "create_backend",
    "get_available_backends",
    "get_backend_reliability",
    # Custom backend registration
    "backend",  # Decorator (preferred)
    "register_backend",
    "unregister_backend",
    "list_registered_backends",
    "clear_registered_backends",
    # Constants
    "BACKEND_NAMES",
    "BACKEND_RELIABILITY",
]
