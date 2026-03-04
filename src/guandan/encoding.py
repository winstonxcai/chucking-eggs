"""State and action encoding for the Q-network."""

from __future__ import annotations

import numpy as np

from .cards import BOMB_TYPES, Card, ComboType, Rank, is_wild, make_deck
from .combos import Combo
from .game import GuanDanEnv

NUM_RANKS = 15  # 2..A(14), BJ, RJ → indices 0..14
NUM_COLS = 4    # one per suit
NUM_COMBO_TYPES = 17  # ComboType enum values 0..16


def _rank_index(rank: int) -> int:
    """Map card rank to matrix row index (0-14)."""
    if rank <= 14:
        return rank - 2  # 2→0, 3→1, ..., A(14)→12
    if rank == Rank.BLACK_JOKER:
        return 13
    if rank == Rank.RED_JOKER:
        return 14
    raise ValueError(f"Unknown rank {rank}")


def cards_to_matrix(cards) -> np.ndarray:
    """Cards → [15, 4] count matrix (suit-level counts, 0/1/2)."""
    mat = np.zeros((NUM_RANKS, NUM_COLS), dtype=np.float32)
    for c in cards:
        ri = _rank_index(c.rank)
        si = c.suit if c.rank <= 14 else (0 if c.rank == Rank.BLACK_JOKER else 1)
        mat[ri, si] = min(mat[ri, si] + 1, 2)
    return mat


# Pre-build the full deck for unknown-card computation
_FULL_DECK = make_deck()


def encode_state(env: GuanDanEnv, player: int) -> np.ndarray:
    """State features for player (relative perspective). ~417 dims."""
    partner = env.partner(player)
    opp_l = (player - 1) % 4  # left opponent
    opp_r = (player + 1) % 4  # right opponent

    hand = cards_to_matrix(env.hands[player]).flatten()              # 60
    played_me = cards_to_matrix(env.played[player]).flatten()        # 60
    played_par = cards_to_matrix(env.played[partner]).flatten()      # 60
    played_opl = cards_to_matrix(env.played[opp_l]).flatten()        # 60
    played_opr = cards_to_matrix(env.played[opp_r]).flatten()        # 60

    # Cards unaccounted for (not in our hand, not played by anyone)
    all_played: set[Card] = set()
    for p in range(4):
        all_played |= env.played[p]
    known = env.hands[player] | all_played
    unknown = [c for c in _FULL_DECK if c not in known]
    remaining = cards_to_matrix(unknown).flatten()                   # 60

    # Card counts (normalized, relative order: me, right, partner, left)
    counts = np.array(
        [len(env.hands[(player + i) % 4]) / 27.0 for i in range(4)],
        dtype=np.float32,
    )                                                                 # 4

    # Out flags (relative order)
    out_flags = np.array(
        [float(env.is_out[(player + i) % 4]) for i in range(4)],
        dtype=np.float32,
    )                                                                 # 4

    # Level rank (one-hot over 13 normal ranks: 2..A)
    level_oh = np.zeros(13, dtype=np.float32)
    level_oh[env.level_rank - 2] = 1.0                               # 13

    # Wild cards in hand
    wilds_in_hand = sum(1 for c in env.hands[player] if is_wild(c, env.level_rank))
    wild_flags = np.array(
        [float(wilds_in_hand >= 1), float(wilds_in_hand >= 2)],
        dtype=np.float32,
    )                                                                 # 2

    # Current trick info
    is_leader = np.array([float(env.current_trick is None)], dtype=np.float32)  # 1
    trick_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)      # 17
    trick_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)             # 15
    trick_is_bomb = np.array([0.0], dtype=np.float32)                # 1

    if env.current_trick is not None:
        trick_type_oh[env.current_trick.type] = 1.0
        if 2 <= env.current_trick.key <= 17:
            trick_key_oh[_rank_index(env.current_trick.key)] = 1.0
        trick_is_bomb[0] = float(env.current_trick.type in BOMB_TYPES)

    return np.concatenate([
        hand, played_me, played_par, played_opl, played_opr,  # 300
        remaining, counts, out_flags,                          # 68
        level_oh, wild_flags, is_leader,                       # 16
        trick_type_oh, trick_key_oh, trick_is_bomb,            # 33
    ])
    # Total: 417


STATE_DIM = 417


def encode_action(combo: Combo) -> np.ndarray:
    """Action features. ~97 dims."""
    cards_played = cards_to_matrix(combo.cards).flatten()              # 60
    combo_type_oh = np.zeros(NUM_COMBO_TYPES, dtype=np.float32)       # 17
    combo_type_oh[combo.type] = 1.0
    combo_key_oh = np.zeros(NUM_RANKS, dtype=np.float32)              # 15
    if 2 <= combo.key <= 17:
        combo_key_oh[_rank_index(combo.key)] = 1.0
    num_cards = np.array([len(combo.cards) / 10.0], dtype=np.float32) # 1
    is_bomb = np.array(
        [float(combo.type in BOMB_TYPES)], dtype=np.float32
    )                                                                  # 1
    wild_oh = np.zeros(3, dtype=np.float32)                           # 3
    wild_oh[min(combo.wild_count, 2)] = 1.0

    return np.concatenate([
        cards_played, combo_type_oh, combo_key_oh,
        num_cards, is_bomb, wild_oh,
    ])
    # Total: 97


ACTION_DIM = 97
