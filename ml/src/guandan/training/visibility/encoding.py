"""Tier 1 encoding: standard state + partner's hand inserted at position 60.

encode_state_tier1 wraps encode_state via np.insert — no reimplementation.
This guarantees bit-identical output for the non-partner portion, which is
required for the zero-init checkpoint expansion trick to work.
"""

from __future__ import annotations

import numpy as np

from ...game import GuanDanEnv
from ..encoding import STATE_DIM, cards_to_matrix, encode_state

# Partner hand is 60 dims (15 ranks x 4 suits count matrix, values 0/1/2).
PARTNER_HAND_DIM = 60
PARTNER_INSERT_POS = 60  # after own hand (positions 0:60)

STATE_DIM_TIER1 = STATE_DIM + PARTNER_HAND_DIM  # 417 + 60 = 477


def encode_state_tier1(env: GuanDanEnv, player: int) -> np.ndarray:
    """State features with partner hand visibility. 477 dims.

    Wraps encode_state (417) and inserts partner's hand (60) at position 60,
    right after the player's own hand.
    """
    base = encode_state(env, player)  # [417]
    partner = env.partner(player)
    partner_hand = cards_to_matrix(env.hands[partner]).flatten()  # [60]
    return np.insert(base, PARTNER_INSERT_POS, partner_hand)  # [477]
