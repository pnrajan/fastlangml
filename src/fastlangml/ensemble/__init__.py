"""Ensemble voting strategies for combining multiple backend results."""

from fastlangml.ensemble.voting import (
    ConsensusVoting,
    HardVoting,
    SoftVoting,
    TieBreaker,
    VotingStrategy,
    WeightedVoting,
)

__all__ = [
    "VotingStrategy",
    "HardVoting",
    "SoftVoting",
    "WeightedVoting",
    "ConsensusVoting",
    "TieBreaker",
]
