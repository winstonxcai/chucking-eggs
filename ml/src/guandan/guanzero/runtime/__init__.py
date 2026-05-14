"""Distributed actor-learner orchestration.

The five process modules that compose a guanzero training run:
- ``train``           — top-level orchestrator (entry point via ``main``)
- ``learner``         — gradient step + weight publish; ``learner_loop`` target
- ``worker``          — per-actor subprocess; ``actor_loop`` target
- ``actor``           — single-episode self-play helpers used by the worker
- ``inference_server``— optional shared-GPU inference server
"""

from .actor import argmax_q, argmax_q_role, play_episode, select_legal
from .train import main, train

__all__ = [
    "argmax_q",
    "argmax_q_role",
    "main",
    "play_episode",
    "select_legal",
    "train",
]
