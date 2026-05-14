"""Shared encoder utilities for GuanZero.

This module is the home of code shared between the M0/M1 base encoder
(``encoding.base_encoder``) and the M3 role-aware encoder
(``encoding.role_encoder``):

- channel constants (``HISTORY_LEN``, ``LEVEL_DIM``, ``RANK_BUCKETS``, ``BEHAVIOR_DIM``)
- ``_multi_hot`` helper
- coordination behavior-flag logic (state-level and action-level variants)
- re-exports of ``CARD_ID_DIM``, ``card_to_id``, ``id_to_card`` from ``cards``

Encoder-specific channel schemas and ``StateActionEncoder``/``RoleAwareStateActionEncoder``
classes live in the per-encoder modules under ``encoding/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ...cards import CARD_ID_DIM, Card, ComboType, card_to_id, id_to_card
from ...combos import Combo
from ...game import GuanDanEnv


# ─── Channel constants ─────────────────────────────────────

HISTORY_LEN = 20
LEVEL_DIM = 13           # ranks 2..A (level rank can only be one of these)
RANK_BUCKETS = 27        # 0..25 + jokers; one-hot count over 0..26 capped at 26
BEHAVIOR_DIM = 9


# ─── Shared multi-hot helper ───────────────────────────────

def _multi_hot(cards: Iterable[Card]) -> np.ndarray:
    """108-dim float32 multi-hot over physical card ids.

    Engine-cached ``env.hand_multihot`` / ``env.played_multihot`` cover the
    hottest paths; this helper remains for one-off card-set encodings
    (e.g. candidate actions, history rows).
    """
    out = np.zeros(CARD_ID_DIM, dtype=np.float32)
    for c in cards:
        out[card_to_id(c)] = 1.0
    return out


# ─── Coordination behavior eligibility ─────────────────────

@dataclass(frozen=True, slots=True)
class _BehaviorEligibility:
    partner: int
    min_opp: int
    partner_hand_size: int
    is_leading: bool
    can_coop: bool
    can_dwarf: bool
    can_assist: bool


def _is_highest_rank(combo: Combo, level_rank: int) -> bool:
    if combo.type != ComboType.TRIPLE:
        return False
    return all(c.rank == level_rank for c in combo.cards)


def _behavior_eligibility(
    env: GuanDanEnv,
    player: int,
    legal_moves: list[Combo],
) -> _BehaviorEligibility:
    """Precompute coordination-flag predicates shared by all candidate actions."""
    partner = (player + 2) % 4
    opp_left = (player + 1) % 4
    opp_right = (player - 1) % 4
    is_leading = env.current_trick is None

    active_opp_sizes = [
        len(env.hands[opp_left]) if not env.is_out[opp_left] else None,
        len(env.hands[opp_right]) if not env.is_out[opp_right] else None,
    ]
    active_opp_sizes = [s for s in active_opp_sizes if s is not None]
    min_opp = min(active_opp_sizes) if active_opp_sizes else 27

    can_coop = (
        not is_leading
        and env.current_trick is not None
        and env.trick_winner == partner
        and any(m.type != ComboType.PASS for m in legal_moves)
    )
    can_dwarf = is_leading and any(
        m.type != ComboType.PASS and len(m.cards) > min_opp
        for m in legal_moves
    )
    partner_hand_size = len(env.hands[partner])
    can_assist = is_leading and any(
        m.type != ComboType.PASS
        and len(m.cards) < partner_hand_size
        and not _is_highest_rank(m, env.level_rank)
        for m in legal_moves
    )
    return _BehaviorEligibility(
        partner=partner,
        min_opp=min_opp,
        partner_hand_size=partner_hand_size,
        is_leading=is_leading,
        can_coop=can_coop,
        can_dwarf=can_dwarf,
        can_assist=can_assist,
    )


def compute_state_behavior_flags(
    env: GuanDanEnv,
    player: int,
    legal_moves: list[Combo],
) -> np.ndarray:
    """Paper-faithful state-level coordination flags. Layout (9 dims):

      [0:3] cooperating [N/A, has_doing_option, has_refusing_option]
      [3:6] dwarfing    [N/A, has_doing_option, has_refusing_option]
      [6:9] assisting   [N/A, has_doing_option, has_refusing_option]

    Both 'doing' and 'refusing' bits can co-fire when the player faces a real
    coordination choice. Computed once per decision (no action conditioning).
    """
    flags = np.zeros(BEHAVIOR_DIM, dtype=np.float32)
    elig = _behavior_eligibility(env, player, legal_moves)

    if not elig.can_coop:
        flags[0] = 1.0
    else:
        has_pass = any(m.type == ComboType.PASS for m in legal_moves)
        has_non_pass = any(m.type != ComboType.PASS for m in legal_moves)
        if has_pass:
            flags[1] = 1.0
        if has_non_pass:
            flags[2] = 1.0

    if not elig.can_dwarf:
        flags[3] = 1.0
    else:
        for m in legal_moves:
            is_dwarfing = m.type != ComboType.PASS and len(m.cards) > elig.min_opp
            if is_dwarfing:
                flags[4] = 1.0
            else:
                flags[5] = 1.0
            if flags[4] and flags[5]:
                break

    if not elig.can_assist:
        flags[6] = 1.0
    else:
        for m in legal_moves:
            is_assisting = (
                m.type != ComboType.PASS
                and len(m.cards) < elig.partner_hand_size
                and not _is_highest_rank(m, env.level_rank)
            )
            if is_assisting:
                flags[7] = 1.0
            else:
                flags[8] = 1.0
            if flags[7] and flags[8]:
                break

    return flags


__all__ = [
    "BEHAVIOR_DIM",
    "CARD_ID_DIM",
    "HISTORY_LEN",
    "LEVEL_DIM",
    "RANK_BUCKETS",
    "card_to_id",
    "id_to_card",
    "compute_state_behavior_flags",
]
