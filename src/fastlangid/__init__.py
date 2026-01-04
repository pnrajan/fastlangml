"""FastLangID - Fast, accurate language detection with multiple backends.

A production-first language detection toolkit optimized for short text that provides:
- Stable, deterministic API
- Unknown/abstain support (returns 'und' with reason when uncertain)
- Top-k predictions with consistent confidence semantics
- Multi-backend ensembling (fasttext, langdetect, lingua, pycld3, langid)
- Batch + caching for throughput
- Benchmark harness to compare backends

Quick Start:
    >>> from fastlangid import detect
    >>> result = detect("Bonjour le monde")
    >>> result.lang
    'fr'
    >>> result.confidence
    0.95

    >>> from fastlangid import detect
    >>> result = detect("Hello world", top_k=3)
    >>> result.candidates
    [Candidate(lang='en', confidence=0.95), ...]

Ensemble Usage:
    >>> from fastlangid import ensemble
    >>> result = ensemble(
    ...     "Bonjour",
    ...     backends=["fasttext", "lingua"],
    ...     weights={"fasttext": 0.7, "lingua": 0.3},
    ... )
    >>> result.lang
    'fr'

Advanced Usage:
    >>> from fastlangid import FastLangDetector
    >>> detector = FastLangDetector()
    >>> detector.add_hint("merci", "fr")
    >>> result = detector.detect("Merci beaucoup!")
    >>> result.lang
    'fr'
"""

from __future__ import annotations

from fastlangid.detector import (
    DetectionConfig,
    FastLangDetector,
    FastLangDetectorBuilder,
    detect,
    detect_batch,
    ensemble,
)
from fastlangid.result import (
    Candidate,
    DetectionResult,
    Reasons,
)
from fastlangid.context.conversation import ConversationContext, ConversationTurn
from fastlangid.hints.dictionary import HintDictionary
from fastlangid.hints.persistence import HintPersistence
from fastlangid.exceptions import (
    FastLangError,
    DetectionError,
    BackendUnavailableError,
    BackendError,
    ConfigurationError,
    HintError,
    NoBackendsAvailableError,
)
from fastlangid.preprocessing.script_filter import Script, detect_script
from fastlangid.backends import (
    get_available_backends,
    backend,
    register_backend,
    unregister_backend,
    list_registered_backends,
    Backend,
)
from fastlangid.normalize import (
    compute_text_stats,
    normalize_text,
    normalize_lang_tag,
)
from fastlangid.ensemble.voting import (
    VotingStrategy,
    HardVoting,
    SoftVoting,
    WeightedVoting,
    ConsensusVoting,
)

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("fastlangid")
except Exception:
    __version__ = "0.1.0"  # Fallback for development

__all__ = [
    # Version
    "__version__",
    # Main API
    "detect",
    "detect_batch",
    "ensemble",
    # Result types
    "DetectionResult",
    "Candidate",
    "Reasons",
    # Detector
    "FastLangDetector",
    "FastLangDetectorBuilder",
    "DetectionConfig",
    # Context
    "ConversationContext",
    "ConversationTurn",
    # Hints
    "HintDictionary",
    "HintPersistence",
    # Script detection
    "Script",
    "detect_script",
    # Normalization
    "compute_text_stats",
    "normalize_text",
    "normalize_lang_tag",
    # Utilities
    "get_available_backends",
    # Custom backend registration
    "backend",  # Decorator (preferred)
    "register_backend",
    "unregister_backend",
    "list_registered_backends",
    "Backend",
    # Voting strategies (for customization)
    "VotingStrategy",
    "HardVoting",
    "SoftVoting",
    "WeightedVoting",
    "ConsensusVoting",
    # Exceptions
    "FastLangError",
    "DetectionError",
    "BackendUnavailableError",
    "BackendError",
    "ConfigurationError",
    "HintError",
    "NoBackendsAvailableError",
]
