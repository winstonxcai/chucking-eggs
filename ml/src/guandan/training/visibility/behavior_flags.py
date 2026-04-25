"""Per-action team-coordination behavior flags.

Resurrected from commit 3237fa8 (Apr 10, 2026 — archived with Tier 1 work).
Three 3-dim axes encode team-coordination intent for the (player, action)
pair: cooperating, dwarfing, assisting. Each axis is one-hot over
[not_applicable, doing_it, refusing].

Used by the team-aware encoder to give the policy net an explicit per-action
signal for partner-coordination decisions.
"""

from __future__ import annotations

import numpy as np

from ...cards import ComboType
from ...game import GuanDanEnv

FLAG_DIM = 9


def _is_highest_rank(combo, level_rank: int) -> bool:
    """Triple of the level-rank card — the highest possible lead."""
    if combo.type != ComboType.TRIPLE:
        return False
    return all(c.rank == level_rank for c in combo.cards)


def compute_behavior_flags(
    env: GuanDanEnv,
    player: int,
    action,
    legal_moves: list,
) -> np.ndarray:
    """9-dim flags for (env, player, action). Layout:

      [0:3] cooperating [N/A, doing, refusing]
      [3:6] dwarfing    [N/A, doing, refusing]
      [6:9] assisting   [N/A, doing, refusing]
    """
    flags = np.zeros(FLAG_DIM, dtype=np.float32)
    partner = (player + 2) % 4
    opp_left = (player + 1) % 4
    opp_right = (player - 1) % 4

    is_pass = action.type == ComboType.PASS
    is_leading = env.current_trick is None

    # ── Cooperating: partner is winning the trick and we have a beat ─────
    can_coop = (
        not is_leading
        and env.current_trick is not None
        and env.trick_winner == partner
        and any(m.type != ComboType.PASS for m in legal_moves)
    )
    if not can_coop:
        flags[0] = 1.0
    elif is_pass:
        flags[1] = 1.0
    else:
        flags[2] = 1.0

    # ── Dwarfing: leading with a combo larger than min active opponent ───
    active_opp_sizes = [
        len(env.hands[opp_left]) if not env.is_out[opp_left] else None,
        len(env.hands[opp_right]) if not env.is_out[opp_right] else None,
    ]
    active_opp_sizes = [s for s in active_opp_sizes if s is not None]
    min_opp = min(active_opp_sizes) if active_opp_sizes else 27

    can_dwarf = is_leading and any(
        m.type != ComboType.PASS and len(m.cards) > min_opp
        for m in legal_moves
    )

    action_card_count = len(action.cards) if not is_pass else 0

    if not can_dwarf:
        flags[3] = 1.0
    elif not is_pass and action_card_count > min_opp:
        flags[4] = 1.0
    else:
        flags[5] = 1.0

    # ── Assisting: leading small non-strong combo to help partner shed ───
    partner_hand_size = len(env.hands[partner])
    can_assist = is_leading and any(
        m.type != ComboType.PASS
        and len(m.cards) < partner_hand_size
        and not _is_highest_rank(m, env.level_rank)
        for m in legal_moves
    )

    if not can_assist:
        flags[6] = 1.0
    elif (
        not is_pass
        and action_card_count < partner_hand_size
        and not _is_highest_rank(action, env.level_rank)
    ):
        flags[7] = 1.0
    else:
        flags[8] = 1.0

    return flags
