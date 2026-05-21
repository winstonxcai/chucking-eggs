"""Strategic deduplication and legal-move selection for Dart ML consumers.

The game engine enumerates every physical-card selection as a separate Combo
(e.g., A♠+A♥ and A♣+A♦ are distinct). For training, suit-variants of the same
logical play are strategically identical — the encoder is suit-agnostic —
so keeping them dilutes labels and produces redundant NN evaluations.

``dedup_strategic`` collapses those variants and ``select_legal`` is the
canonical entry point for actors. The ``guandan_rs`` package uses the optional
native extension when available and falls back to Python otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import guandan_rs

from ...cards import ComboType
from ...combos import Combo, _rs_tuple_to_combo

if TYPE_CHECKING:
    from ...game import GuanDanEnv

logger = logging.getLogger(__name__)
_warned_python_select_legal = False
_PASS = Combo(ComboType.PASS, 0, [], 0, 0)


def _combo_to_tuple(c: Combo) -> tuple:
    return (
        int(c.type),
        c.key,
        [(card.rank, card.suit, card.deck) for card in c.cards],
        c.length,
        c.wild_count,
    )


def _card_to_tuple(card) -> tuple[int, int, int]:
    return (card.rank, card.suit, card.deck)


def dedup_strategic(legal: Iterable[Combo]) -> list[Combo]:
    """Collapse suit-variants of strategically-equivalent plays; first-seen wins."""
    tuples = [_combo_to_tuple(c) for c in legal]
    return [_rs_tuple_to_combo(t) for t in guandan_rs.dedup_strategic(tuples)]


def select_legal(env: GuanDanEnv, player: int) -> list[Combo]:
    """Return the deduplicated legal moves for ``player``.

    Appends an explicit PASS move if the player must respond but PASS is
    absent from the raw legal-move set (can happen with strict-response rules).
    """
    global _warned_python_select_legal
    if not getattr(guandan_rs, "HAS_NATIVE", False) and not _warned_python_select_legal:
        logger.warning("guandan_rs native extension unavailable; using slower Python legal-move path")
        _warned_python_select_legal = True

    if all(hasattr(env, name) for name in ("hands", "level_rank", "current_trick")):
        hand = [_card_to_tuple(card) for card in env.hands[player]]
        trick = None if env.current_trick is None else _combo_to_tuple(env.current_trick)
        return [
            _rs_tuple_to_combo(t)
            for t in guandan_rs.select_legal(hand, int(env.level_rank), trick)
        ]

    legal = dedup_strategic(env.legal_moves(player))
    if not env.is_leading() and not any(m.type == ComboType.PASS for m in legal):
        legal.append(_PASS)
    return legal


__all__ = ["dedup_strategic", "select_legal"]
