"""Strategic deduplication of legal moves for GuanZero ML consumers.

The game engine enumerates every physical-card selection as a separate Combo
(e.g., A♠+A♥ and A♣+A♦ are distinct). For training, suit-variants of the same
logical play are strategically identical — the encoder is suit-agnostic —
so keeping them dilutes labels without adding information.

``dedup_strategic`` collapses those variants, retaining the first-seen
representative, before encoding or action selection. The engine and web backend
are unaffected; this wrapper is applied only at ML call sites.
"""

from __future__ import annotations

from typing import Iterable

from ..cards import ComboType
from ..combos import Combo

# Combo types where suit composition does not affect strength under Combo.beats().
# STRAIGHT_FLUSH and BOMB_JOKER are excluded: their suit composition is intrinsic
# to identity (SF suits are distinct), so they keep the full card-tuple key.
_SUIT_AGNOSTIC = frozenset({
    ComboType.SINGLE, ComboType.PAIR, ComboType.TRIPLE,
    ComboType.FULL_HOUSE, ComboType.STRAIGHT,
    ComboType.TUBE, ComboType.PLATE,
    ComboType.BOMB_4, ComboType.BOMB_5, ComboType.BOMB_6,
    ComboType.BOMB_7, ComboType.BOMB_8, ComboType.BOMB_9, ComboType.BOMB_10,
})


def strategic_key(c: Combo) -> tuple:
    """Equivalence key under which suit-variants of the same play collapse.

    PASS, STRAIGHT_FLUSH, and BOMB_JOKER preserve full card identity so
    beats() semantics are not disturbed.
    """
    if c.type in _SUIT_AGNOSTIC:
        ranks = tuple(sorted(card.rank for card in c.cards))
        return (c.type, c.key, c.length, c.wild_count, ranks)
    card_ids = tuple(sorted((card.rank, card.suit) for card in c.cards))
    return (c.type, c.key, c.length, c.wild_count, card_ids)


def dedup_strategic(legal: Iterable[Combo]) -> list[Combo]:
    """Collapse suit-variants of strategically-equivalent plays.

    First-seen variant survives; insertion order is preserved.
    """
    out: dict[tuple, Combo] = {}
    for c in legal:
        out.setdefault(strategic_key(c), c)
    return list(out.values())


__all__ = ["dedup_strategic", "strategic_key"]
