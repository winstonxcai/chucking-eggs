"""Strategic deduplication of legal moves for ML consumers.

The engine (`combos.py`, `GuanDanEnv.legal_moves`) enumerates every physical-card
selection as a separate `Combo`. The web backend depends on this — players click on
specific physical cards. But for ML, suit-variants of the same logical play are
strategically identical (the encoder is suit-agnostic via `[15,4]` clipping in
`cards_to_matrix`), and enumeration dilutes labels and PPO log-probs.

This helper is applied at ML call sites only. Engine and web behavior are unchanged.

See `ml/LOGBOOK.md` §39 (architecture-ceiling investigation) for context.
"""

from __future__ import annotations

from typing import Iterable

from ..cards import ComboType
from ..combos import Combo

# Combo types where suit doesn't change strength under `Combo.beats()`. For
# STRAIGHT_FLUSH and BOMB_JOKER, suit composition is intrinsic to identity and
# we keep the full card-tuple key so distinct SF suits stay distinct.
_SUIT_AGNOSTIC = frozenset({
    ComboType.SINGLE, ComboType.PAIR, ComboType.TRIPLE,
    ComboType.FULL_HOUSE, ComboType.STRAIGHT,
    ComboType.TUBE, ComboType.PLATE,
    ComboType.BOMB_4, ComboType.BOMB_5, ComboType.BOMB_6,
    ComboType.BOMB_7, ComboType.BOMB_8, ComboType.BOMB_9, ComboType.BOMB_10,
})


def strategic_key(c: Combo) -> tuple:
    """Equivalence key under which suit-variants of the same play collapse.

    PASS, STRAIGHT_FLUSH, BOMB_JOKER preserve full card identity so beats()
    semantics are preserved.
    """
    if c.type in _SUIT_AGNOSTIC:
        ranks = tuple(sorted(card.rank for card in c.cards))
        return (c.type, c.key, c.length, c.wild_count, ranks)
    card_ids = tuple(sorted((card.rank, card.suit) for card in c.cards))
    return (c.type, c.key, c.length, c.wild_count, card_ids)


def dedup_strategic(legal: Iterable[Combo]) -> list[Combo]:
    """Collapse suit-variants of strategically-equivalent plays.

    First-seen variant survives. Engine is untouched — this is a wrapper at
    ML call sites; the web backend keeps the un-deduped engine output.
    """
    seen: set = set()
    out: list[Combo] = []
    for c in legal:
        k = strategic_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out
