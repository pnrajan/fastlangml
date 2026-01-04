"""Ensemble voting strategies for combining multiple backend results."""

from fastlangid.ensemble.voting import (
    VotingStrategy,
    HardVoting,
    SoftVoting,
    WeightedVoting,
    ConsensusVoting,
    TieBreaker,
)

__all__ = [
    "VotingStrategy",
    "HardVoting",
    "SoftVoting",
    "WeightedVoting",
    "ConsensusVoting",
    "TieBreaker",
]
