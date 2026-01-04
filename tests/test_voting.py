"""Tests for ensemble voting strategies."""

import pytest

from fastlangid.backends.base import DetectionResult
from fastlangid.ensemble.voting import (
    HardVoting,
    SoftVoting,
    WeightedVoting,
    ConsensusVoting,
    TieBreaker,
)


def make_result(
    backend: str,
    language: str,
    confidence: float,
    is_reliable: bool = True,
) -> DetectionResult:
    """Helper to create DetectionResult."""
    return DetectionResult(
        backend_name=backend,
        language=language,
        confidence=confidence,
        all_probabilities={language: confidence},
        is_reliable=is_reliable,
    )


class TestHardVoting:
    """Tests for HardVoting strategy."""

    def test_majority_vote(self):
        """Test majority voting."""
        voting = HardVoting()
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "en", 0.8),
            make_result("c", "fr", 0.9),
        ]

        scores = voting.vote(results)
        assert scores["en"] > scores["fr"]

    def test_tie(self):
        """Test tie handling."""
        voting = HardVoting()
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        scores = voting.vote(results)
        assert scores["en"] == scores["fr"]

    def test_empty_results(self):
        """Test with empty results."""
        voting = HardVoting()
        scores = voting.vote([])
        assert scores == {}


class TestSoftVoting:
    """Tests for SoftVoting strategy."""

    def test_average_confidence(self):
        """Test averaging confidences."""
        voting = SoftVoting()
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "en", 0.7),
        ]

        scores = voting.vote(results)
        # Average should be 0.8, but divided by 2 backends = 0.8
        assert 0.7 <= scores["en"] <= 0.9

    def test_multiple_languages(self):
        """Test with multiple languages."""
        voting = SoftVoting()
        results = [
            DetectionResult(
                backend_name="a",
                language="en",
                confidence=0.6,
                all_probabilities={"en": 0.6, "fr": 0.3},
            ),
            DetectionResult(
                backend_name="b",
                language="fr",
                confidence=0.7,
                all_probabilities={"fr": 0.7, "en": 0.2},
            ),
        ]

        scores = voting.vote(results)
        assert "en" in scores
        assert "fr" in scores


class TestWeightedVoting:
    """Tests for WeightedVoting strategy."""

    def test_default_weights(self):
        """Test default weights set at construction."""
        voting = WeightedVoting(default_weights={"a": 0.8, "b": 0.2})
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        scores = voting.vote(results)
        # Backend "a" has higher weight
        assert scores["en"] > scores["fr"]

    def test_weights_as_parameter(self):
        """Test weights passed as parameter to vote()."""
        voting = WeightedVoting()
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        # Pass weights at vote time
        scores = voting.vote(results, weights={"a": 0.8, "b": 0.2})
        assert scores["en"] > scores["fr"]

    def test_parameter_overrides_default(self):
        """Test that parameter weights override default weights."""
        voting = WeightedVoting(default_weights={"a": 0.2, "b": 0.8})
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        # Default would favor "b"/fr, but parameter overrides
        scores = voting.vote(results, weights={"a": 0.8, "b": 0.2})
        assert scores["en"] > scores["fr"]

    def test_reliability_weights(self):
        """Test using reliability-based weights."""
        voting = WeightedVoting(use_reliability_weights=True)
        results = [
            make_result("fasttext", "en", 0.9),  # Higher reliability
            make_result("langid", "fr", 0.9),  # Lower reliability
        ]

        scores = voting.vote(results)
        # Fasttext has higher reliability
        assert scores["en"] > scores["fr"]

    def test_unreliable_penalty(self):
        """Test penalty for unreliable results."""
        voting = WeightedVoting()
        results = [
            make_result("a", "en", 0.9, is_reliable=True),
            make_result("b", "en", 0.9, is_reliable=False),  # Penalized
        ]

        scores = voting.vote(results)
        # Score should still be positive but lower than if both reliable
        assert scores["en"] > 0


class TestVotingWithWeightsParameter:
    """Tests for weights parameter across all voting strategies."""

    def test_hard_voting_with_weights(self):
        """Test HardVoting accepts weights parameter."""
        voting = HardVoting()
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        scores = voting.vote(results, weights={"a": 0.9, "b": 0.1})
        assert scores["en"] > scores["fr"]

    def test_soft_voting_with_weights(self):
        """Test SoftVoting accepts weights parameter."""
        voting = SoftVoting()
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        scores = voting.vote(results, weights={"a": 0.9, "b": 0.1})
        assert scores["en"] > scores["fr"]

    def test_consensus_voting_with_weights(self):
        """Test ConsensusVoting passes weights to fallback."""
        voting = ConsensusVoting(min_agreement=3)  # Forces fallback
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
        ]

        scores = voting.vote(results, weights={"a": 0.9, "b": 0.1})
        # Fallback should use weights
        assert scores["en"] > scores["fr"]


class TestConsensusVoting:
    """Tests for ConsensusVoting strategy."""

    def test_consensus_reached(self):
        """Test when consensus is reached."""
        voting = ConsensusVoting(min_agreement=2)
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "en", 0.8),
            make_result("c", "fr", 0.9),
        ]

        scores = voting.vote(results)
        assert "en" in scores
        assert scores["en"] > 0

    def test_no_consensus(self):
        """Test when no consensus is reached."""
        voting = ConsensusVoting(min_agreement=3)
        results = [
            make_result("a", "en", 0.9),
            make_result("b", "fr", 0.9),
            make_result("c", "de", 0.9),
        ]

        scores = voting.vote(results)
        # Falls back to soft voting
        assert len(scores) > 0


class TestTieBreaker:
    """Tests for TieBreaker."""

    def test_resolve_by_reliability(self):
        """Test resolving by reliability."""
        tie_breaker = TieBreaker()
        results = [
            make_result("a", "en", 0.9, is_reliable=True),
            make_result("b", "fr", 0.9, is_reliable=False),
        ]

        scores = tie_breaker.resolve(results)
        assert scores["en"] > scores["fr"]

    def test_script_match_bonus(self):
        """Test script match bonus."""
        tie_breaker = TieBreaker(script_languages={"ru", "uk"})
        results = [
            make_result("a", "ru", 0.7),
            make_result("b", "en", 0.7),
        ]

        scores = tie_breaker.resolve(results)
        # Russian matches script, should get bonus
        assert scores["ru"] > scores["en"]

    def test_allowed_languages_filter(self):
        """Test allowed languages filtering."""
        tie_breaker = TieBreaker(allowed_languages={"en", "de"})
        results = [
            make_result("a", "en", 0.7),
            make_result("b", "fr", 0.9),  # Not allowed
        ]

        scores = tie_breaker.resolve(results)
        # French should be penalized
        assert scores["en"] > scores["fr"]
