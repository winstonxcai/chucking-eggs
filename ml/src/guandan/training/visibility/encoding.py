"""Tier 1 encoding: standard state + partner's hand inserted at position 60.

encode_state_tier1 wraps encode_state via np.insert — no reimplementation.
This guarantees bit-identical output for the non-partner portion, which is
required for the zero-init checkpoint expansion trick to work.
"""

from __future__ import annotations

import numpy as np

from ...cards import BOMB_TYPES, Rank, is_wild
from ...game import GuanDanEnv
from ..encoding import (
    INITIAL_HAND_SIZE,
    NUM_COMBO_TYPES,
    NUM_RANKS,
    STATE_DIM,
    _FULL_DECK,
    _rank_index,
    cards_to_matrix,
    encode_state,
)

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


# ─── Team-centric encoding (one-entity framing) ─────────────────────────
#
# Under full partner visibility, the network should see the team as one
# entity. The encoding is invariant to which teammate is currently acting:
# calling encode_state_tier1_team(env, 0) and encode_state_tier1_team(env, 2)
# produces identical output except for the acting_seat_flag.
#
# Seat convention (team {0, 2} vs opponents {1, 3}):
#   lo  = 0  (low-index teammate, fixed)
#   hi  = 2  (high-index teammate, fixed)
#   oL  = 1  (left opponent, fixed)
#   oR  = 3  (right opponent, fixed)
#
# Layout (480 dims):
#   [  0: 60] team_hand_lo         (60)
#   [ 60:120] team_hand_hi         (60)
#   [120:180] team_played_lo       (60)
#   [180:240] team_played_hi       (60)
#   [240:300] played_opp_l         (60)
#   [300:360] played_opp_r         (60)
#   [360:420] remaining            (60)
#   [420:424] counts [lo,hi,oL,oR] (4)
#   [424:428] out_flags            (4)
#   [428:430] acting_seat_flag     (2) ← [1,0] for lo acts, [0,1] for hi acts
#   [430:433] wilds_in_team        (3) ← [lo>=1, hi>=1, total>=2]
#   [433:446] level_oh             (13)
#   [446:447] is_leader            (1)
#   [447:464] trick_type_oh        (17)
#   [464:479] trick_key_oh         (15)
#   [479:480] trick_is_bomb        (1)

STATE_DIM_TIER1_TEAM = 480
ACTING_FLAG_START = 428
ACTING_FLAG_END = 430


def encode_state_tier1_team(env: GuanDanEnv, player: int) -> np.ndarray:
    """Team-centric tier 1 state encoding. 480 dims.

    Invariant across teammates: encode_state_tier1_team(env, 0) and
    encode_state_tier1_team(env, 2) produce the same output except for
    the acting_seat_flag at positions [428:430].

    Assumes team = {0, 2}, opponents = {1, 3}.
    """
    LO, HI, OL, OR = 0, 2, 1, 3

    hand_lo = cards_to_matrix(env.hands[LO]).flatten()        # 60
    hand_hi = cards_to_matrix(env.hands[HI]).flatten()        # 60
    played_lo = cards_to_matrix(env.played[LO]).flatten()     # 60
    played_hi = cards_to_matrix(env.played[HI]).flatten()     # 60
    played_opl = cards_to_matrix(env.played[OL]).flatten()    # 60
    played_opr = cards_to_matrix(env.played[OR]).flatten()    # 60

    # Unknown cards (not in any team hand, not played)
    all_played: set = set()
    for p in range(4):
        all_played |= env.played[p]
    known = env.hands[LO] | env.hands[HI] | all_played
    unknown = [c for c in _FULL_DECK if c not in known]
    remaining = cards_to_matrix(unknown).flatten()            # 60

    counts = np.array([
        len(env.hands[LO]) / INITIAL_HAND_SIZE,
        len(env.hands[HI]) / INITIAL_HAND_SIZE,
        len(env.hands[OL]) / INITIAL_HAND_SIZE,
        len(env.hands[OR]) / INITIAL_HAND_SIZE,
    ], dtype=np.float32)                                      # 4

    out_flags = np.array([
        float(env.is_out[LO]),
        float(env.is_out[HI]),
        float(env.is_out[OL]),
        float(env.is_out[OR]),
    ], dtype=np.float32)                                      # 4

    # Acting seat flag: [1,0] if lo acts, [0,1] if hi acts
    acting_flag = np.zeros(2, dtype=np.float32)
    if player == LO:
        acting_flag[0] = 1.0
    elif player == HI:
        acting_flag[1] = 1.0
    # else: called during opponent turn — flag stays zero (shouldn't happen
    # in normal trainer use, but don't crash)

    # Wilds across team
    wilds_lo = sum(1 for c in env.hands[LO] if is_wild(c, env.level_rank))
    wilds_hi = sum(1 for c in env.hands[HI] if is_wild(c, env.level_rank))
    wild_flags = np.array([
        float(wilds_lo >= 1),
        float(wilds_hi >= 1),
        float(wilds_lo + wilds_hi >= 2),
    ], dtype=np.float32)                                      # 3

    # Level rank one-hot
    level_oh = np.zeros(13, dtype=np.float32)
    level_oh[env.level_rank - 2] = 1.0                        # 13

    # Current trick info (absolute, not acting-seat relative)
    is_leader = np.array([float(env.current_trick is None)], dtype=np.float32)  # 1
    trick_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)  # 17
    trick_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)      # 15
    trick_is_bomb = np.array([0.0], dtype=np.float32)         # 1

    if env.current_trick is not None:
        trick_type_oh[env.current_trick.type] = 1.0
        if 2 <= env.current_trick.key <= 17:
            trick_key_oh[_rank_index(env.current_trick.key)] = 1.0
        trick_is_bomb[0] = float(env.current_trick.type in BOMB_TYPES)

    return np.concatenate([
        hand_lo, hand_hi, played_lo, played_hi, played_opl, played_opr,  # 360
        remaining,                                                        # 60
        counts, out_flags,                                                # 8
        acting_flag,                                                      # 2
        wild_flags,                                                       # 3
        level_oh,                                                         # 13
        is_leader, trick_type_oh, trick_key_oh, trick_is_bomb,            # 34
    ])
    # Total: 360 + 60 + 8 + 2 + 3 + 13 + 34 = 480
