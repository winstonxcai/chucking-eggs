"""Rule-based heuristic agent for baseline evaluation."""

from __future__ import annotations

import random

from .cards import BOMB_TYPES, ComboType
from .combos import Combo


def heuristic_play(legal_moves: list[Combo], is_leading: bool) -> Combo:
    """Pick a move using simple heuristics.

    Leading: play the smallest single to shed cards. Avoid bombs unless
    hand is nearly empty.
    Following: play the smallest legal move that beats the trick. Pass
    if no good option.
    """
    if is_leading:
        return _lead(legal_moves)
    return _follow(legal_moves)


def _lead(legal_moves: list[Combo]) -> Combo:
    """When leading, play smallest non-bomb combo. Prefer singles."""
    non_bombs = [m for m in legal_moves if m.type not in BOMB_TYPES]
    if non_bombs:
        # Prefer singles, then pairs, then by card count ascending
        non_bombs.sort(key=lambda m: (m.type, m.key))
        return non_bombs[0]
    # Only bombs available — play smallest bomb
    bombs = [m for m in legal_moves if m.type in BOMB_TYPES]
    bombs.sort(key=lambda m: (m.type, m.key))
    return bombs[0]


def _follow(legal_moves: list[Combo]) -> Combo:
    """When following, play smallest beater or pass."""
    passes = [m for m in legal_moves if m.type == ComboType.PASS]
    non_pass = [m for m in legal_moves if m.type != ComboType.PASS]

    if not non_pass:
        return passes[0] if passes else legal_moves[0]

    # Prefer non-bomb responses
    non_bombs = [m for m in non_pass if m.type not in BOMB_TYPES]
    if non_bombs:
        non_bombs.sort(key=lambda m: m.key)
        return non_bombs[0]

    # Only bombs can beat — use smallest, but prefer to pass if possible
    bombs = sorted(non_pass, key=lambda m: (m.type, m.key))
    # Pass unless hand is small (≤ 5 cards worth keeping a bomb)
    if passes:
        return passes[0]
    return bombs[0]
