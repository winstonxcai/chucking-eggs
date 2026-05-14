"""Strategic deduplication and legal-move selection for GuanZero ML consumers.

The game engine enumerates every physical-card selection as a separate Combo
(e.g., A♠+A♥ and A♣+A♦ are distinct). For training, suit-variants of the same
logical play are strategically identical — the encoder is suit-agnostic —
so keeping them dilutes labels and produces redundant NN evaluations.

``dedup_strategic`` collapses those variants and ``select_legal`` is the
canonical entry point for actors. Both delegate to the Rust ``guandan_rs``
extension so the Python episode loop and the Rust rollout loop share a single
implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import guandan_rs

from ...cards import ComboType
from ...combos import Combo, _rs_tuple_to_combo

if TYPE_CHECKING:
    from ...game import GuanDanEnv

_PASS = Combo(ComboType.PASS, 0, [], 0, 0)


def _combo_to_tuple(c: Combo) -> tuple:
    return (
        int(c.type),
        c.key,
        [(card.rank, card.suit, card.deck) for card in c.cards],
        c.length,
        c.wild_count,
    )


def dedup_strategic(legal: Iterable[Combo]) -> list[Combo]:
    """Collapse suit-variants of strategically-equivalent plays; first-seen wins."""
    tuples = [_combo_to_tuple(c) for c in legal]
    return [_rs_tuple_to_combo(t) for t in guandan_rs.dedup_strategic(tuples)]


def select_legal(env: "GuanDanEnv", player: int) -> list[Combo]:
    """Return the deduplicated legal moves for ``player``.

    Appends an explicit PASS move if the player must respond but PASS is
    absent from the raw legal-move set (can happen with strict-response rules).
    """
    legal = dedup_strategic(env.legal_moves(player))
    if not env.is_leading() and not any(m.type == ComboType.PASS for m in legal):
        legal.append(_PASS)
    return legal


__all__ = ["dedup_strategic", "select_legal"]
