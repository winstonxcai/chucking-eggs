"""Combo classification, generation, comparison, and move generation.

Move generation (``generate_all_leads`` and ``generate_responses``) is
delegated to the ``guandan_rs`` Rust extension. Build with::

    cd ml/src/guandan_rs && maturin develop --release
"""

from __future__ import annotations

from .cards import (
    BOMB_TYPES,
    Card,
    ComboType,
    level_order_key,
)
from guandan_rs import (
    generate_all_leads as _rs_leads,
    generate_responses as _rs_responses,
)


class Combo:
    __slots__ = ("type", "key", "cards", "length", "wild_count")

    def __init__(
        self,
        combo_type: ComboType,
        key_rank: int,
        cards: list[Card],
        length: int = 0,
        wild_count: int = 0,
    ):
        self.type = combo_type
        self.key = key_rank          # primary rank for comparison
        self.cards = tuple(cards)    # actual cards played
        self.length = length         # for straights (always 5 for now)
        self.wild_count = wild_count

    def beats(self, other: Combo | None, level_rank: int) -> bool:
        """Can this combo beat *other*? (other=None means free lead.)"""
        if other is None:
            return True

        my_bomb = self.type in BOMB_TYPES
        their_bomb = other.type in BOMB_TYPES

        # Any bomb beats any non-bomb
        if my_bomb and not their_bomb:
            return True
        if not my_bomb and their_bomb:
            return False

        # Both bombs
        if my_bomb and their_bomb:
            if self.type != other.type:
                return self.type > other.type
            # Same bomb tier
            if self.type == ComboType.STRAIGHT_FLUSH:
                return self.key > other.key  # natural order
            if self.type == ComboType.BOMB_JOKER:
                return False  # only one four-joker bomb exists
            # N-of-a-kind: level order
            return level_order_key(self.key, level_rank) > level_order_key(
                other.key, level_rank
            )

        # Neither is bomb: must match type
        if self.type != other.type:
            return False

        # Straights, tubes, plates: natural order
        if self.type in (ComboType.STRAIGHT, ComboType.TUBE, ComboType.PLATE):
            return self.key > other.key

        # Singles, pairs, triples, full houses: level order
        return level_order_key(self.key, level_rank) > level_order_key(
            other.key, level_rank
        )

    def __repr__(self) -> str:
        return f"Combo({self.type.name}, key={self.key}, cards={len(self.cards)}, wilds={self.wild_count})"


def _rs_tuple_to_combo(t: tuple) -> Combo:
    """Convert Rust (combo_type, key, cards, length, wild_count) tuple to Combo."""
    combo_type, key, cards_raw, length, wild_count = t
    cards = [Card(r, s, d) for r, s, d in cards_raw]
    return Combo(ComboType(combo_type), key, cards, length, wild_count)


def generate_all_leads(hand: set[Card], level_rank: int) -> list[Combo]:
    """Generate every legal combo from hand (free lead)."""
    hand_tuples = [(c.rank, c.suit, c.deck) for c in hand]
    return [_rs_tuple_to_combo(t) for t in _rs_leads(hand_tuples, level_rank)]


def generate_responses(
    hand: set[Card], level_rank: int, trick: Combo
) -> list[Combo]:
    """Generate all combos that beat the current trick, plus PASS."""
    hand_tuples = [(c.rank, c.suit, c.deck) for c in hand]
    trick_tuple = (
        int(trick.type), trick.key,
        [(c.rank, c.suit, c.deck) for c in trick.cards],
        trick.length, trick.wild_count,
    )
    return [
        _rs_tuple_to_combo(t)
        for t in _rs_responses(hand_tuples, level_rank, trick_tuple)
    ]
