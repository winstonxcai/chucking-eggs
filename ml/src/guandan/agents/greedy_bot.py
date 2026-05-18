"""GreedyBot — always plays the cheapest legal option.

No strategy, no partner awareness, no bombing. Just economy of force.
"""

from __future__ import annotations

from ..cards import BOMB_TYPES, ComboType, Rank, level_order_key
from ..combos import generate_all_leads, generate_responses
from .base import Agent

# Types where level_order_key applies for comparison.
_LEVEL_ORDER_TYPES = frozenset({
    ComboType.SINGLE, ComboType.PAIR,
    ComboType.TRIPLE, ComboType.FULL_HOUSE,
})


class GreedyBot(Agent):
    """Always play the cheapest legal option. No strategy, just economy."""

    label = "Greedy"
    description = "Plays it safe."
    source = "In-house"
    color = "#1f77b4"
    sample_tag = 6

    def __init__(self, level_rank: int = Rank.TWO):
        self.level_rank = level_rank

    def act(self, env, player: int):
        if env.current_trick is None:
            return self._lead(env, player)
        else:
            return self._follow(env, player)

    def _lead(self, env, player: int):
        """Lead with the smallest, weakest combo."""
        all_leads = generate_all_leads(env.hands[player], self.level_rank)

        # Filter out bombs (save them)
        non_bomb = [m for m in all_leads
                    if m.type not in BOMB_TYPES and m.type != ComboType.PASS]
        if not non_bomb:
            non_bomb = [m for m in all_leads if m.type != ComboType.PASS]

        # Sort: fewest cards first, weakest rank first
        non_bomb.sort(key=lambda m: (
            len(m.cards),
            level_order_key(m.key, self.level_rank),
        ))
        return non_bomb[0]

    def _follow(self, env, player: int):
        """Play cheapest beat, or pass."""
        trick = env.current_trick
        responses = generate_responses(env.hands[player], self.level_rank, trick)

        beats = [m for m in responses if m.type != ComboType.PASS]

        if not beats:
            return _get_pass(responses)

        # Among non-bomb beats, pick cheapest
        non_bomb_beats = [m for m in beats if m.type not in BOMB_TYPES]
        if non_bomb_beats:
            if trick.type in _LEVEL_ORDER_TYPES:
                non_bomb_beats.sort(
                    key=lambda m: level_order_key(m.key, self.level_rank))
            else:
                non_bomb_beats.sort(key=lambda m: m.key)
            return non_bomb_beats[0]

        # Only bombs available — pass instead of wasting them
        return _get_pass(responses)


def _get_pass(responses: list) -> object:
    """Find the PASS action in responses."""
    for c in responses:
        if c.type == ComboType.PASS:
            return c
    return responses[-1]
