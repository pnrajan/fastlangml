"""Ensemble voting strategies for combining multiple backend results.

Provides various strategies for combining language detection results from
multiple backends, including tie-breaking logic based on reliability,
confidence, script matching, and allowed languages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastlangml.backends import BACKEND_RELIABILITY

if TYPE_CHECKING:
    from fastlangml.backends.base import DetectionResult


@dataclass
class TieBreaker:
    """
    Tie-breaking logic for resolving disagreements between backends.

    Priority order:
    1. Reliability flags (is_reliable from backend)
    2. Backend reliability ranking (fasttext > lingua > pycld3 > langdetect > langid)
    3. Confidence scores
    4. Script match (if script_languages provided)
    5. Allowed languages (if provided)
    """

    script_languages: set[str] | None = None
    """Languages matching the detected script (for tie-breaking)."""

    allowed_languages: set[str] | None = None
    """User-specified allowed languages (for filtering)."""

    def resolve(self, results: list[DetectionResult]) -> dict[str, float]:
        """
        Resolve disagreement between backend results.

        Args:
            results: List of detection results from different backends

        Returns:
            Dict mapping language codes to adjusted confidence scores
        """
        if not results:
            return {}

        # Group results by language
        lang_results: dict[str, list[DetectionResult]] = {}
        for r in results:
            if r.language != "unknown":
                if r.language not in lang_results:
                    lang_results[r.language] = []
                lang_results[r.language].append(r)

        if not lang_results:
            return {}

        # Calculate scores for each language
        scores: dict[str, float] = {}

        for lang, lang_res in lang_results.items():
            score = 0.0

            # Factor 1: Number of backends agreeing
            agreement_score = len(lang_res) / len(results)
            score += agreement_score * 0.3

            # Factor 2: Average confidence
            avg_confidence = sum(r.confidence for r in lang_res) / len(lang_res)
            score += avg_confidence * 0.25

            # Factor 3: Reliability flags
            reliable_count = sum(1 for r in lang_res if r.is_reliable)
            reliability_score = reliable_count / len(lang_res)
            score += reliability_score * 0.2

            # Factor 4: Backend reliability ranking
            max_backend_reliability = max(
                BACKEND_RELIABILITY.get(r.backend_name, 0) for r in lang_res
            )
            backend_score = max_backend_reliability / 5.0  # Normalize to 0-1
            score += backend_score * 0.15

            # Factor 5: Script match bonus
            if self.script_languages and lang in self.script_languages:
                score += 0.05

            # Factor 6: Allowed languages filter
            if self.allowed_languages and lang not in self.allowed_languages:
                score *= 0.1  # Heavy penalty for non-allowed

            scores[lang] = min(score, 1.0)  # Cap at 1.0

        return scores


class VotingStrategy(ABC):
    """Abstract base for ensemble voting strategies."""

    @abstractmethod
    def vote(
        self,
        results: list[DetectionResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Combine results from multiple backends.

        Args:
            results: List of detection results from different backends
            weights: Optional dict mapping backend name to weight (0.0-1.0).
                     Weights are normalized automatically.

        Returns:
            Dict mapping language codes to combined confidence scores
        """
        pass


class HardVoting(VotingStrategy):
    """
    Majority voting - each backend gets one vote.

    The final language is the one with the most votes.
    Simple but effective when backends have similar accuracy.
    """

    def vote(
        self,
        results: list[DetectionResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not results:
            return {}

        # Count votes (only from reliable results if possible)
        reliable_results = [r for r in results if r.is_reliable]
        voting_results = reliable_results if reliable_results else results

        # If weights provided, use weighted voting instead of simple count
        if weights:
            weighted_votes: dict[str, float] = {}
            total_weight = 0.0
            for r in voting_results:
                if r.language != "unknown":
                    w = weights.get(r.backend_name, 1.0)
                    weighted_votes[r.language] = weighted_votes.get(r.language, 0.0) + w
                    total_weight += w
            if total_weight > 0:
                return {lang: score / total_weight for lang, score in weighted_votes.items()}
            return {}

        votes = Counter(r.language for r in voting_results if r.language != "unknown")

        if not votes:
            return {}

        total_votes = sum(votes.values())

        # Convert to probabilities
        return {lang: count / total_votes for lang, count in votes.items()}


class SoftVoting(VotingStrategy):
    """
    Average probability voting - average confidence across backends.

    Takes the mean of all backend probabilities for each language.
    Better than hard voting when confidence scores are well-calibrated.
    """

    def vote(
        self,
        results: list[DetectionResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not results:
            return {}

        # If weights provided, use weighted average
        if weights:
            weighted_probs: dict[str, float] = {}
            total_weight = sum(weights.get(r.backend_name, 1.0) for r in results)
            if total_weight == 0:
                total_weight = 1.0

            for result in results:
                w = weights.get(result.backend_name, 1.0) / total_weight
                for lang, prob in result.all_probabilities.items():
                    weighted_probs[lang] = weighted_probs.get(lang, 0.0) + prob * w

            return weighted_probs

        # Aggregate all probabilities
        all_langs: dict[str, list[float]] = {}

        for result in results:
            for lang, prob in result.all_probabilities.items():
                if lang not in all_langs:
                    all_langs[lang] = []
                all_langs[lang].append(prob)

        # Average probabilities (filling in 0 for backends that didn't report)
        averaged = {}
        for lang, probs in all_langs.items():
            averaged[lang] = sum(probs) / len(results)

        return averaged


@dataclass
class WeightedVoting(VotingStrategy):
    """
    Weighted soft voting - backends have different influence.

    More accurate backends can be given higher weights.
    Recommended for production use.
    """

    default_weights: dict[str, float] = field(default_factory=dict)
    """Default weights to use if not provided at vote time."""

    use_reliability_weights: bool = True
    """If True, use built-in reliability rankings when weights not specified."""

    def vote(
        self,
        results: list[DetectionResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not results:
            return {}

        # Use provided weights, fall back to default_weights, then reliability
        effective_weights = weights or self.default_weights

        # Determine weights for each backend
        backend_weights = {}
        for result in results:
            if result.backend_name in effective_weights:
                backend_weights[result.backend_name] = effective_weights[result.backend_name]
            elif self.use_reliability_weights:
                # Use reliability ranking as default weight
                backend_weights[result.backend_name] = BACKEND_RELIABILITY.get(
                    result.backend_name, 1.0
                )
            else:
                backend_weights[result.backend_name] = 1.0

        # Normalize weights
        total_weight = sum(backend_weights.values())
        if total_weight == 0:
            total_weight = 1.0
        normalized_weights = {k: v / total_weight for k, v in backend_weights.items()}

        # Aggregate weighted probabilities
        weighted_probs: dict[str, float] = {}

        for result in results:
            weight = normalized_weights[result.backend_name]

            # Apply reliability penalty if result is unreliable
            if not result.is_reliable:
                weight *= 0.5

            for lang, prob in result.all_probabilities.items():
                if lang not in weighted_probs:
                    weighted_probs[lang] = 0.0
                weighted_probs[lang] += prob * weight

        return weighted_probs


@dataclass
class ConsensusVoting(VotingStrategy):
    """
    Require agreement between backends.

    Only returns a result if a minimum number of backends agree.
    Most conservative strategy, useful when certainty is important.
    """

    min_agreement: int = 2
    """Minimum number of backends that must agree on the language."""

    fallback_strategy: VotingStrategy | None = None
    """Strategy to use if no consensus is reached. Defaults to SoftVoting."""

    def vote(
        self,
        results: list[DetectionResult],
        weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        if not results:
            return {}

        # Count top-1 predictions
        top_predictions = Counter(
            r.language for r in results if r.language != "unknown"
        )

        # Filter to languages meeting agreement threshold
        agreed = {
            lang: count / len(results)
            for lang, count in top_predictions.items()
            if count >= self.min_agreement
        }

        if agreed:
            return agreed

        # Fallback if no consensus (pass weights through)
        fallback = self.fallback_strategy or SoftVoting()
        return fallback.vote(results, weights=weights)


def create_voting_strategy(
    strategy_name: str,
    default_weights: dict[str, float] | None = None,
    min_agreement: int = 2,
) -> VotingStrategy:
    """
    Create a voting strategy by name.

    Args:
        strategy_name: One of "hard", "soft", "weighted", "consensus"
        default_weights: Default backend weights (can be overridden at vote time)
        min_agreement: Minimum agreement for consensus voting

    Returns:
        VotingStrategy instance
    """
    if strategy_name == "hard":
        return HardVoting()
    elif strategy_name == "soft":
        return SoftVoting()
    elif strategy_name == "weighted":
        return WeightedVoting(default_weights=default_weights or {})
    elif strategy_name == "consensus":
        return ConsensusVoting(min_agreement=min_agreement)
    else:
        raise ValueError(
            f"Unknown voting strategy: {strategy_name}. "
            f"Available: hard, soft, weighted, consensus"
        )


__all__ = [
    "VotingStrategy",
    "HardVoting",
    "SoftVoting",
    "WeightedVoting",
    "ConsensusVoting",
    "TieBreaker",
    "create_voting_strategy",
]
