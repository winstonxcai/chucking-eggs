"""Distributed actor-learner orchestration.

Process modules:
- ``train``            — top-level orchestrator (entry point via ``main``)
- ``learner``          — unified ``learner_loop`` + ``LearnerProtocol``
- ``learners/``        — concrete learner implementations and loss buckets
- ``weights``          — atomic weight publish + read helpers
- ``worker``           — per-actor subprocess; ``actor_loop`` target
- ``actor/``           — rollout, actor runtime setup, opponents, sample packing
"""

from .actor import (
    LaneConfig,
    SeatPolicy,
    all_latest_seats,
    argmax_q,
    play_episode,
    play_episodes_batched,
    select_legal,
)
from .train import main, train

__all__ = [
    "argmax_q",
    "LaneConfig",
    "main",
    "play_episode",
    "play_episodes_batched",
    "SeatPolicy",
    "all_latest_seats",
    "select_legal",
    "train",
]
