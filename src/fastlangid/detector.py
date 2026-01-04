"""Main language detector with ensemble support and context awareness.

The core detection engine that combines multiple backends with voting,
proper noun filtering, script detection, and runtime hints.

Implements the fastlangid spec with:
- Deterministic API
- Unknown/abstain support with reasons
- Top-k predictions with consistent confidence semantics
- Multi-backend ensembling
- Batch + caching for throughput
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal, overload

from fastlangid.backends import (
    Backend,
    DetectionResult as BackendResult,
    create_backend,
    get_available_backends,
    get_backend_reliability,
)
from fastlangid.cache import DetectionCache
from fastlangid.context.conversation import ConversationContext
from fastlangid.ensemble.voting import (
    TieBreaker,
    VotingStrategy,
    create_voting_strategy,
)
from fastlangid.exceptions import NoBackendsAvailableError
from fastlangid.hints.dictionary import HintDictionary
from fastlangid.normalize import (
    compute_text_stats,
    is_linguistic,
    is_sufficient_length,
    normalize_lang_tag,
    normalize_text,
)
from fastlangid.preprocessing.proper_noun_filter import ProperNounFilter
from fastlangid.preprocessing.script_filter import ScriptFilter, detect_script
from fastlangid.result import Candidate, DetectionResult, Reasons


@dataclass
class DetectionConfig:
    """Configuration for language detection."""

    # Detection mode settings
    mode: Literal["short", "default", "long"] = "default"
    """Detection mode: 'short' for tiny strings, 'default' for normal text."""

    # Backend settings
    backend: str = "ensemble"
    """Backend to use: 'ensemble', 'cld3', 'lingua', 'langid', 'fasttext'."""

    backends: list[str] = field(default_factory=list)
    """Backend names to use for ensemble. Empty = auto-detect available."""

    backend_weights: dict[str, float] = field(default_factory=dict)
    """Weights for each backend in voting."""

    voting_strategy: Literal["hard", "soft", "weighted", "consensus"] = "weighted"
    """Ensemble voting strategy name (used if custom_voting not provided)."""

    custom_voting: VotingStrategy | None = None
    """Custom voting strategy instance. If provided, overrides voting_strategy."""

    # Thresholds per mode
    min_chars: dict[str, int] = field(
        default_factory=lambda: {"short": 2, "default": 3, "long": 3}
    )
    """Minimum character count per mode."""

    min_letters: dict[str, int] = field(
        default_factory=lambda: {"short": 1, "default": 2, "long": 2}
    )
    """Minimum letter count per mode."""

    min_letter_ratio: float = 0.3
    """Minimum ratio of letters to total characters."""

    thresholds: dict[str, float] = field(
        default_factory=lambda: {"short": 0.3, "default": 0.5, "long": 0.6}
    )
    """Confidence thresholds per mode."""

    # Unknown handling
    unknown_policy: dict[str, bool | str] = field(
        default_factory=lambda: {"allow": True, "label": "und"}
    )
    """Unknown handling policy."""

    # Preprocessing
    normalize: bool = True
    """Whether to apply unicode normalization."""

    strip_noise: bool = True
    """Whether to strip URLs, emails, etc."""

    filter_proper_nouns: bool = True
    """Filter proper nouns before detection."""

    proper_noun_strategy: Literal["remove", "mask", "none"] = "remove"
    """How to handle proper nouns."""

    use_script_filter: bool = True
    """Use Unicode script detection for pre-filtering."""

    # Caching
    cache_size: int = 1000
    """LRU cache size for repeated strings."""

    # Context and hints
    context_weight: float = 0.2
    """How much conversation context influences detection."""

    hint_weight: float = 0.3
    """How much user hints influence detection."""


class FastLangDetector:
    """
    Main language detection class with ensemble support.

    Implements the fastlangid spec with:
    - Deterministic, stable output
    - Unknown/abstain support with reasons
    - Top-k predictions
    - Multi-backend ensembling
    - Batch + caching

    Example:
        >>> from fastlangid import FastLangDetector
        >>> detector = FastLangDetector()
        >>> result = detector.detect("Bonjour le monde")
        >>> result.lang
        'fr'
        >>> result.confidence
        0.95
    """

    _default_instance: FastLangDetector | None = None

    def __init__(
        self,
        config: DetectionConfig | None = None,
        hints: HintDictionary | None = None,
    ) -> None:
        """Initialize the detector.

        Args:
            config: Detection configuration
            hints: Pre-populated hint dictionary
        """
        self._config = config or DetectionConfig()
        self._hints = hints or HintDictionary()
        self._allowed_languages: set[str] | None = None

        # Initialize components
        self._backends = self._initialize_backends()
        self._proper_noun_filter = ProperNounFilter(
            strategy=self._config.proper_noun_strategy
        )
        self._script_filter = ScriptFilter()
        self._voting = self._create_voting_strategy()
        self._cache: DetectionCache = DetectionCache(self._config.cache_size)

        # Persistent thread pool for parallel backend calls (lazy init)
        self._executor: ThreadPoolExecutor | None = None

    @classmethod
    def default(cls) -> FastLangDetector:
        """Get or create a default detector instance (singleton)."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def reset_default(cls) -> None:
        """Reset the default instance."""
        cls._default_instance = None

    def _initialize_backends(self) -> list[Backend]:
        """Initialize requested backends."""
        backend_names = self._config.backends or get_available_backends()

        if not backend_names:
            # Return empty list - we'll handle this gracefully
            return []

        backends = []
        for name in backend_names:
            try:
                backend = create_backend(name)
                backends.append(backend)
            except Exception:
                continue

        # Sort by reliability
        backends.sort(key=lambda b: get_backend_reliability(b.name), reverse=True)
        return backends

    def _create_voting_strategy(self) -> VotingStrategy:
        """Create the voting strategy (uses custom if provided)."""
        if self._config.custom_voting is not None:
            return self._config.custom_voting
        return create_voting_strategy(
            self._config.voting_strategy,
            default_weights=self._config.backend_weights,
        )

    def set_languages(self, allowed: list[str] | None) -> None:
        """Set allowed languages for detection."""
        self._allowed_languages = set(allowed) if allowed else None

    def add_hint(self, word: str, language: str) -> None:
        """Add a single-token word -> language hint."""
        self._hints.add(word, language)

    def remove_hint(self, word: str) -> None:
        """Remove a hint."""
        self._hints.remove(word)

    def _update_context_if_needed(
        self,
        text: str,
        result: DetectionResult,
        context: ConversationContext | None,
        auto_update: bool,
    ) -> None:
        """Update context with detection result if auto_update is enabled."""
        if auto_update and context is not None:
            context.add_turn(text, result.lang, result.confidence)

    def detect(
        self,
        text: str,
        *,
        top_k: int = 1,
        mode: Literal["short", "default", "long"] | None = None,
        allowed_langs: list[str] | None = None,
        filter_proper_nouns: bool | None = None,
        context: ConversationContext | None = None,
        hints: HintDictionary | None = None,
        auto_update: bool = False,
    ) -> DetectionResult:
        """Detect the language of the given text.

        Analyzes the input text using configured backends and returns a
        detection result with language code, confidence score, and metadata.

        Args:
            text: Input text to analyze. Can be any length, but accuracy
                improves with longer text.
            top_k: Number of language candidates to include in result.
                Defaults to 1 (only top result).
            mode: Detection mode affecting thresholds and behavior.
                - "short": Optimized for chat messages (≤10 chars)
                - "default": Standard detection
                - "long": Stricter thresholds for documents
            allowed_langs: Restrict detection to these ISO 639-1 codes.
                If None, all languages are considered.
            filter_proper_nouns: Override proper noun filtering.
                If None, uses detector's default setting.
            context: Conversation context for multi-turn accuracy.
                Previous turns influence ambiguous detections.
            hints: Additional word-to-language hints merged with
                detector's built-in hints.
            auto_update: If True, automatically update the context with
                this detection result. Saves you from calling
                context.add_turn() manually after each detection.

        Returns:
            DetectionResult containing:
                - lang: ISO 639-1 code or "und" for undetermined
                - confidence: Score from 0.0 to 1.0
                - reliable: True if confidence exceeds threshold
                - reason: Explanation when lang is "und"
                - candidates: Top-k language candidates (if top_k > 1)
                - script: Detected Unicode script (Latin, Cyrillic, etc.)

        Example:
            >>> detector = FastLangDetector()
            >>> result = detector.detect("Bonjour le monde")
            >>> result.lang
            'fr'
            >>> result.confidence
            0.95
        """
        start_time = time.perf_counter()
        effective_mode = mode or self._config.mode

        if allowed_langs:
            effective_langs_set: set[str] | None = set(allowed_langs)
        else:
            effective_langs_set = self._allowed_languages

        # Check cache (include allowed_langs for correctness)
        langs_key = ",".join(sorted(effective_langs_set)) if effective_langs_set else ""
        cache_key = f"{text}|{effective_mode}|{top_k}|{langs_key}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Normalize text
        processed = text
        if self._config.normalize:
            processed = normalize_text(text, strip_noise=self._config.strip_noise)

        # Check for empty/whitespace
        if not processed or not processed.strip():
            result = DetectionResult(
                lang="und",
                confidence=0.0,
                reliable=False,
                reason=Reasons.EMPTY_TEXT,
                backend="none",
                meta={"elapsed_ms": (time.perf_counter() - start_time) * 1000},
            )
            return result

        # Check sufficient length
        min_chars = self._config.min_chars.get(effective_mode, 3)
        min_letters = self._config.min_letters.get(effective_mode, 2)
        sufficient, reason = is_sufficient_length(processed, min_chars, min_letters)
        if not sufficient:
            result = DetectionResult(
                lang="und",
                confidence=0.0,
                reliable=False,
                reason=reason,
                backend="none",
                meta={"elapsed_ms": (time.perf_counter() - start_time) * 1000},
            )
            return result

        # Check linguistic content
        linguistic, reason = is_linguistic(processed, self._config.min_letter_ratio)
        if not linguistic:
            result = DetectionResult(
                lang="und",
                confidence=0.0,
                reliable=False,
                reason=reason,
                backend="none",
                meta={"elapsed_ms": (time.perf_counter() - start_time) * 1000},
            )
            return result

        # Check backends available
        if not self._backends:
            result = DetectionResult(
                lang="und",
                confidence=0.0,
                reliable=False,
                reason=Reasons.NO_BACKEND,
                backend="none",
                meta={
                    "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                    "error": "No backends available. Install: pip install fastlangid[all]",
                },
            )
            return result

        # Apply proper noun filter
        detection_text = processed
        if filter_proper_nouns if filter_proper_nouns is not None else self._config.filter_proper_nouns:
            detection_text = self._proper_noun_filter.filter(processed)
            if not detection_text.strip():
                detection_text = processed

        # Detect script
        stats = compute_text_stats(detection_text)
        detected_script = stats.script

        # Script-based filtering
        script_languages = None
        if self._config.use_script_filter:
            script_languages = self._script_filter.filter_languages(
                detection_text, effective_langs_set
            )

        # Check hints
        merged_hints = self._hints.merge(hints) if hints else self._hints
        hint_scores = merged_hints.lookup_all(detection_text)

        # Run backends
        backend_results: list[BackendResult] = []
        backend_name = "ensemble" if len(self._backends) > 1 else self._backends[0].name if self._backends else "none"

        if len(self._backends) > 1:
            # Lazy init persistent executor
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=len(self._backends))

            def run_backend(backend: Backend) -> BackendResult | None:
                try:
                    return backend.detect(detection_text)
                except Exception:
                    return None

            futures = {self._executor.submit(run_backend, b): b for b in self._backends}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    backend_results.append(result)
        else:
            for backend in self._backends:
                try:
                    result = backend.detect(detection_text)
                    backend_results.append(result)
                except Exception:
                    continue

        if not backend_results:
            result = DetectionResult(
                lang="und",
                confidence=0.0,
                reliable=False,
                reason=Reasons.NO_BACKEND,
                backend=backend_name,
                script=detected_script,
                meta={"elapsed_ms": (time.perf_counter() - start_time) * 1000},
            )
            return result

        # Ensemble voting
        ensemble_scores = self._voting.vote(
            backend_results, weights=self._config.backend_weights or None
        )

        # Apply tie-breaking if needed
        if self._needs_tiebreak(ensemble_scores):
            tie_breaker = TieBreaker(
                script_languages=script_languages,
                allowed_languages=effective_langs_set,
            )
            ensemble_scores = tie_breaker.resolve(backend_results)

        # Apply hint boosting (stronger for short text mode)
        if hint_scores:
            # For very short text, hints are more reliable than backends
            text_len = len(text.strip())
            if mode == "short" and text_len <= 5:
                # Very short text: hints dominate
                for lang, score in hint_scores.items():
                    ensemble_scores[lang] = ensemble_scores.get(lang, 0.0) + score * 2.0
            else:
                hint_multiplier = 1.5 if mode == "short" else 1.0
                for lang, score in hint_scores.items():
                    current = ensemble_scores.get(lang, 0.0)
                    ensemble_scores[lang] = current + score * self._config.hint_weight * hint_multiplier

        # Apply context
        if context:
            for lang in ensemble_scores:
                boost = context.get_context_boost(lang)
                ensemble_scores[lang] += boost * self._config.context_weight

        # Filter by allowed languages
        if effective_langs_set:
            ensemble_scores = {
                lang: score
                for lang, score in ensemble_scores.items()
                if lang in effective_langs_set
            }

        # Normalize scores
        if ensemble_scores:
            max_score = max(ensemble_scores.values())
            if max_score > 0:
                ensemble_scores = {
                    lang: min(score / max_score, 1.0)
                    for lang, score in ensemble_scores.items()
                }

        # Sort and build candidates
        sorted_results = sorted(
            ensemble_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Check for disagreement
        if len(sorted_results) >= 2:
            diff = sorted_results[0][1] - sorted_results[1][1]
            if diff < 0.1 and sorted_results[0][1] < self._config.thresholds.get(effective_mode, 0.5):
                result = DetectionResult(
                    lang="und",
                    confidence=sorted_results[0][1],
                    reliable=False,
                    reason=Reasons.DISAGREEMENT,
                    backend=backend_name,
                    script=detected_script,
                    candidates=[
                        Candidate(
                            lang=normalize_lang_tag(lang),
                            confidence=score,
                            backend_votes={r.backend_name: r.confidence for r in backend_results if r.language == lang},
                        )
                        for lang, score in sorted_results[:top_k]
                    ],
                    meta={
                        "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                        "backend_results": [(r.backend_name, r.language, r.confidence) for r in backend_results],
                    },
                )
                self._cache.put(cache_key, result)
                self._update_context_if_needed(text, result, context, auto_update)
                return result

        if not sorted_results:
            result = DetectionResult(
                lang="und",
                confidence=0.0,
                reliable=False,
                reason=Reasons.LOW_CONFIDENCE,
                backend=backend_name,
                script=detected_script,
                meta={"elapsed_ms": (time.perf_counter() - start_time) * 1000},
            )
            self._update_context_if_needed(text, result, context, auto_update)
            return result

        # Get top result
        top_lang, top_score = sorted_results[0]
        threshold = self._config.thresholds.get(effective_mode, 0.5)

        # Check confidence threshold
        if top_score < threshold and self._config.unknown_policy.get("allow", True):
            result = DetectionResult(
                lang="und",
                confidence=top_score,
                reliable=False,
                reason=Reasons.LOW_CONFIDENCE,
                backend=backend_name,
                script=detected_script,
                candidates=[
                    Candidate(
                        lang=normalize_lang_tag(lang),
                        confidence=score,
                        backend_votes={},
                    )
                    for lang, score in sorted_results[:top_k]
                ],
                meta={
                    "elapsed_ms": (time.perf_counter() - start_time) * 1000,
                    "backend_results": [(r.backend_name, r.language, r.confidence) for r in backend_results],
                },
            )
            self._cache.put(cache_key, result)
            self._update_context_if_needed(text, result, context, auto_update)
            return result

        # Build successful result
        candidates = [
            Candidate(
                lang=normalize_lang_tag(lang),
                confidence=round(score, 4),
                backend_votes={r.backend_name: round(r.confidence, 4) for r in backend_results if r.language == lang},
            )
            for lang, score in sorted_results[:top_k]
        ]

        result = DetectionResult(
            lang=normalize_lang_tag(top_lang),
            confidence=round(top_score, 4),
            reliable=top_score >= threshold,
            script=detected_script,
            backend=backend_name,
            candidates=candidates if top_k > 1 else [],
            meta={
                "elapsed_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "backend_results": [(r.backend_name, r.language, round(r.confidence, 4)) for r in backend_results],
            },
        )

        self._cache.put(cache_key, result)
        self._update_context_if_needed(text, result, context, auto_update)
        return result

    def detect_batch(
        self,
        texts: list[str],
        *,
        top_k: int = 1,
        mode: Literal["short", "default", "long"] | None = None,
        allowed_langs: list[str] | None = None,
        max_workers: int | None = None,
    ) -> list[DetectionResult]:
        """Detect language for multiple texts.

        Args:
            texts: List of texts to analyze.
            top_k: Number of candidates per result.
            mode: Detection mode.
            allowed_langs: Restrict to these languages.
            max_workers: Max parallel workers (ignored, uses detector's pool).

        Returns:
            List of DetectionResult objects.
        """
        if not texts:
            return []

        if len(texts) <= 2:
            return [
                self.detect(text, top_k=top_k, mode=mode, allowed_langs=allowed_langs)
                for text in texts
            ]

        def detect_single(idx_text: tuple[int, str]) -> tuple[int, DetectionResult]:
            idx, text = idx_text
            result = self.detect(text, top_k=top_k, mode=mode, allowed_langs=allowed_langs)
            return idx, result

        results: list[DetectionResult | None] = [None] * len(texts)

        # Reuse detector's executor or create one for batch
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=max(len(self._backends), 4))

        futures = {
            self._executor.submit(detect_single, (i, text)): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

        return [r for r in results if r is not None]

    def _needs_tiebreak(self, scores: dict[str, float]) -> bool:
        """Check if tie-breaking is needed."""
        if len(scores) < 2:
            return False
        sorted_scores = sorted(scores.values(), reverse=True)
        return (sorted_scores[0] - sorted_scores[1]) < 0.1

    @property
    def available_backends(self) -> list[str]:
        """Get names of initialized backends."""
        return [b.name for b in self._backends]

    @property
    def hints(self) -> HintDictionary:
        """Get the hint dictionary."""
        return self._hints

    @property
    def allowed_languages(self) -> set[str] | None:
        """Get allowed languages."""
        return self._allowed_languages

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return self._cache.stats


class FastLangDetectorBuilder:
    """Builder pattern for constructing FastLangDetector instances."""

    def __init__(self) -> None:
        self._config = DetectionConfig()
        self._hints: HintDictionary | None = None

    def with_backends(self, *backends: str) -> FastLangDetectorBuilder:
        """Specify which backends to use."""
        self._config.backends = list(backends)
        return self

    def with_backend(self, backend: str) -> FastLangDetectorBuilder:
        """Use a single backend."""
        self._config.backend = backend
        if backend != "ensemble":
            self._config.backends = [backend]
        return self

    def with_weights(self, weights: dict[str, float]) -> FastLangDetectorBuilder:
        """Set weights for each backend."""
        self._config.backend_weights = weights
        return self

    def with_voting_strategy(
        self, strategy: Literal["hard", "soft", "weighted", "consensus"]
    ) -> FastLangDetectorBuilder:
        """Set the ensemble voting strategy."""
        self._config.voting_strategy = strategy
        return self

    def with_mode(
        self, mode: Literal["short", "default", "long"]
    ) -> FastLangDetectorBuilder:
        """Set the default detection mode."""
        self._config.mode = mode
        return self

    def with_thresholds(self, thresholds: dict[str, float]) -> FastLangDetectorBuilder:
        """Set confidence thresholds per mode."""
        self._config.thresholds = thresholds
        return self

    def with_cache_size(self, size: int) -> FastLangDetectorBuilder:
        """Set cache size."""
        self._config.cache_size = size
        return self

    def with_normalization(self, enabled: bool = True) -> FastLangDetectorBuilder:
        """Enable/disable text normalization."""
        self._config.normalize = enabled
        return self

    def with_proper_noun_filtering(
        self, strategy: Literal["remove", "mask", "none"]
    ) -> FastLangDetectorBuilder:
        """Configure proper noun handling."""
        self._config.proper_noun_strategy = strategy
        self._config.filter_proper_nouns = strategy != "none"
        return self

    def with_script_filter(self, enabled: bool = True) -> FastLangDetectorBuilder:
        """Enable or disable script-based pre-filtering."""
        self._config.use_script_filter = enabled
        return self

    def with_hints(self, hints: HintDictionary) -> FastLangDetectorBuilder:
        """Provide a pre-populated hint dictionary."""
        self._hints = hints
        return self

    def with_context_weight(self, weight: float) -> FastLangDetectorBuilder:
        """Set how much conversation context influences detection."""
        self._config.context_weight = weight
        return self

    def with_hint_weight(self, weight: float) -> FastLangDetectorBuilder:
        """Set how much user hints influence detection."""
        self._config.hint_weight = weight
        return self

    def build(self) -> FastLangDetector:
        """Build and return the configured FastLangDetector."""
        return FastLangDetector(self._config, self._hints)


# Convenience functions for module-level API
def detect(
    text: str,
    *,
    context: ConversationContext | None = None,
    mode: Literal["short", "default", "long"] = "short",
    top_k: int = 1,
    allowed_langs: list[str] | None = None,
    auto_update: bool = True,
) -> DetectionResult:
    """Detect language of text in one line.

    This is the primary API for FastLangID. Pass conversation context
    to improve accuracy on ambiguous short messages.

    Args:
        text: Input text to analyze.
        context: Conversation history for context-aware detection.
            Pass this to improve accuracy on ambiguous messages like
            "ok", "yes", or single words.
        mode: Detection mode:
            - "short": Optimized for chat messages (default)
            - "default": Standard detection
            - "long": Stricter thresholds for documents
        top_k: Number of candidates to return.
        allowed_langs: Restrict to these ISO 639-1 codes.
        auto_update: Automatically update context with detection result.
            Defaults to True for convenience.

    Returns:
        DetectionResult with lang, confidence, reliable, etc.

    Example:
        >>> from fastlangid import detect, ConversationContext
        >>> context = ConversationContext()
        >>> detect("Bonjour!", context=context).lang  # Context auto-updated
        'fr'
        >>> detect("Merci", context=context).lang     # Uses previous context
        'fr'
    """
    detector = FastLangDetector.default()
    return detector.detect(
        text,
        top_k=top_k,
        mode=mode,
        context=context,
        allowed_langs=allowed_langs,
        auto_update=auto_update,
    )


def detect_batch(
    texts: list[str],
    *,
    top_k: int = 1,
    mode: Literal["short", "default", "long"] = "short",
    allowed_langs: list[str] | None = None,
) -> list[DetectionResult]:
    """Detect language for multiple texts.

    Args:
        texts: List of texts to analyze.
        top_k: Number of candidates per result.
        mode: Detection mode.
        allowed_langs: Restrict to these languages.

    Returns:
        List of DetectionResult objects.
    """
    detector = FastLangDetector.default()
    return detector.detect_batch(texts, top_k=top_k, mode=mode, allowed_langs=allowed_langs)


def ensemble(
    text: str,
    *,
    backends: list[str] | None = None,
    weights: dict[str, float] | None = None,
    strategy: Literal["hard", "soft", "weighted", "consensus"] = "weighted",
    top_k: int = 1,
    mode: Literal["short", "default", "long"] = "short",
) -> DetectionResult:
    """Detect language using explicit ensemble configuration.

    This function provides fine-grained control over the ensemble
    detection process, allowing you to specify backends, weights,
    and voting strategy.

    Args:
        text: Input text to analyze.
        backends: List of backend names to use. Defaults to all available.
        weights: Dict mapping backend name to weight (0.0-1.0).
        strategy: Voting strategy ('weighted', 'soft', 'hard', 'consensus').
        top_k: Number of candidates to return.
        mode: Detection mode ('short', 'default', 'long').

    Returns:
        DetectionResult with lang, confidence, reliable, candidates, etc.

    Example:
        >>> from fastlangid import ensemble
        >>> result = ensemble(
        ...     "Bonjour le monde",
        ...     backends=["fasttext", "lingua"],
        ...     weights={"fasttext": 0.7, "lingua": 0.3},
        ...     strategy="weighted",
        ... )
        >>> result.lang
        'fr'
    """
    config = DetectionConfig(
        backends=backends or [],
        backend_weights=weights or {},
        voting_strategy=strategy,
        mode=mode,
    )
    detector = FastLangDetector(config)
    return detector.detect(text, top_k=top_k, mode=mode)
