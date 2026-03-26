"""Determinization for imperfect-information search.

Samples consistent opponent/partner hands given what the searching player
can observe: their own hand, all played cards, and card counts.

Uses an isolated RNG to avoid polluting the game's random state.
"""

from __future__ import annotations

import copy
import random as _random_mod

from ..cards import make_deck
from ..game import GuanDanEnv

# Isolated RNG so search doesn't affect game randomness
_search_rng = _random_mod.Random()


def sample_consistent_deal(env: GuanDanEnv, player: int) -> GuanDanEnv:
    """Deep-copy env with reshuffled unknown cards among other players.

    Known (from player's perspective):
    - Own hand: env.hands[player]
    - All played cards: env.played[0..3]
    - Card counts per player: len(env.hands[p])

    Unknown: the specific cards in other players' remaining hands.

    Procedure:
    1. Pool all cards not in our hand and not yet played
    2. Shuffle the pool (isolated RNG)
    3. Deal correct number of cards to each other player
    """
    world = copy.deepcopy(env)

    # Collect all known cards (our hand + everything played)
    known = set(world.hands[player])
    for p in range(4):
        known |= world.played[p]

    # Unknown pool: full deck minus known cards
    pool = [c for c in make_deck() if c not in known]
    _search_rng.shuffle(pool)

    # Deal to each other player, preserving their hand size
    idx = 0
    for p in range(4):
        if p == player:
            continue
        n = len(env.hands[p])  # use original sizes
        world.hands[p] = set(pool[idx : idx + n])
        idx += n

    return world
